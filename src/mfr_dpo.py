"""QLoRA DPO training, replay training, and preference-margin scoring."""

from __future__ import annotations

import copy
import math
from pathlib import Path
import time

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, get_linear_schedule_with_warmup

MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
ANCHOR_METHODS = ("dapr", "dapr_c", "copr_adapted")


def load_model(model_name=MODEL_NAME, lora_r=16, adapter_path=None, revision=None,
               lora_alpha=None, lora_dropout=0.05):
    """Load the frozen 4-bit base model and either a fresh or saved LoRA adapter."""
    tokenizer = AutoTokenizer.from_pretrained(model_name, revision=revision)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    compute_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        revision=revision,
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=True,
        ),
        device_map={"": 0},
    )
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(
        model,
        use_gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
    )
    if adapter_path:
        model = PeftModel.from_pretrained(model, adapter_path, is_trainable=True)
    else:
        lora_alpha = 2 * lora_r if lora_alpha is None else lora_alpha
        model = get_peft_model(model, LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            task_type="CAUSAL_LM",
        ))
    return model, tokenizer


def chat_prompt(tokenizer, prompt):
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True
    )


def make_batch(tokenizer, rows, max_tokens=1024, device="cuda"):
    """Build a padded pair batch; chosen sequences precede rejected sequences."""
    seqs = []
    for key in ("chosen", "rejected"):
        for row in rows:
            prompt_ids = tokenizer(chat_prompt(tokenizer, row["prompt"]), add_special_tokens=False)["input_ids"]
            answer_ids = tokenizer(row[key] + tokenizer.eos_token, add_special_tokens=False)["input_ids"]
            ids = (prompt_ids + answer_ids)[:max_tokens]
            mask = ([0] * len(prompt_ids) + [1] * len(answer_ids))[:max_tokens]
            seqs.append((ids, mask))
    width = max(len(ids) for ids, _ in seqs)
    pad = tokenizer.pad_token_id
    return {
        "input_ids": torch.tensor([ids + [pad] * (width - len(ids)) for ids, _ in seqs], device=device),
        "attention_mask": torch.tensor(
            [[1] * len(ids) + [0] * (width - len(ids)) for ids, _ in seqs], device=device
        ),
        "response_mask": torch.tensor([mask + [0] * (width - len(mask)) for _, mask in seqs], device=device),
        "n_pairs": len(rows),
        "pair_ids": [row.get("id") for row in rows],
        "is_replay": [bool(row.get("_is_replay", False)) for row in rows],
    }


def pair_length(row):
    return row["prompt_tokens"] + max(row["chosen_tokens"], row["rejected_tokens"])


def response_token_logprobs(model, batch):
    """Return response-token log probabilities, their sequence ids, and sequence lengths."""
    logits = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]).logits[:, :-1, :]
    targets = batch["input_ids"][:, 1:]
    mask = batch["response_mask"][:, 1:].bool()
    token_logp = -F.cross_entropy(logits[mask].float(), targets[mask], reduction="none")
    seq_index = mask.nonzero()[:, 0]
    return token_logp, seq_index, mask.sum(-1)


def response_logprobs(model, batch):
    """Return summed response log probabilities and response lengths per sequence."""
    token_logp, seq_index, lengths = response_token_logprobs(model, batch)
    sums = torch.zeros(batch["input_ids"].shape[0], device=token_logp.device, dtype=token_logp.dtype)
    return sums.index_add(0, seq_index, token_logp), lengths.float()


def _cached_reference(batch, reference_cache, device):
    cache = reference_cache if reference_cache.index.name == "id" else reference_cache.set_index("id", drop=False)
    missing = [pair_id for pair_id in batch["pair_ids"] if pair_id not in cache.index]
    if missing:
        raise KeyError(f"reference cache is missing pair IDs: {missing[:3]}")
    chosen = cache.loc[batch["pair_ids"], "chosen_logp"].to_numpy(float)
    rejected = cache.loc[batch["pair_ids"], "rejected_logp"].to_numpy(float)
    return torch.tensor(np.concatenate([chosen, rejected]), device=device, dtype=torch.float32)


def policy_and_reference(model, batch, reference_cache=None):
    """Return policy and frozen-base log probabilities, using a cache when supplied."""
    policy, lengths = response_logprobs(model, batch)
    if reference_cache is None:
        with torch.no_grad(), model.disable_adapter():
            reference, _ = response_logprobs(model, batch)
    else:
        reference = _cached_reference(batch, reference_cache, policy.device)
    return policy, reference, lengths


def policy_minus_reference(model, batch, reference_cache=None):
    policy, reference, lengths = policy_and_reference(model, batch, reference_cache)
    return policy - reference, lengths


def _huber(values, delta=1.0):
    absolute = values.abs()
    return torch.where(absolute <= delta, 0.5 * values.square(), delta * (absolute - 0.5 * delta))


def _anchor_regularizer(method, policy, token_logp, seq_index, batch, anchors, huber_delta=1.0):
    """Return a replay-only peak anchor penalty and transparent diagnostics."""
    if method not in ANCHOR_METHODS:
        zero = policy.new_zeros(())
        return zero, {
            "anchor_loss": 0.0, "chosen_violation_rate": 0.0,
            "rejected_violation_rate": 0.0, "huber_cap_rate": 0.0,
            "anchor_common_shift": 0.0, "anchored_pairs": 0,
        }
    if anchors is None:
        raise ValueError(f"{method} requires peak preference anchors")

    n = batch["n_pairs"]
    replay_indices = [
        index for index, is_replay in enumerate(batch.get("is_replay", [False] * n)) if is_replay
    ]
    if not replay_indices:
        zero = policy.new_zeros(())
        return zero, {
            "anchor_loss": 0.0, "chosen_violation_rate": 0.0,
            "rejected_violation_rate": 0.0, "huber_cap_rate": 0.0,
            "anchor_common_shift": 0.0, "anchored_pairs": 0,
        }

    penalties, chosen_rates, rejected_rates, cap_rates, shifts = [], [], [], [], []
    for index in replay_indices:
        pair_id = batch["pair_ids"][index]
        if pair_id not in anchors:
            raise KeyError(f"peak preference anchor missing for replay pair {pair_id!r}")
        saved = anchors[pair_id]
        current_chosen = token_logp[seq_index == index]
        current_rejected = token_logp[seq_index == n + index]
        anchor_chosen = torch.as_tensor(
            saved["chosen"], device=current_chosen.device, dtype=current_chosen.dtype
        )
        anchor_rejected = torch.as_tensor(
            saved["rejected"], device=current_rejected.device, dtype=current_rejected.dtype
        )
        if current_chosen.numel() != anchor_chosen.numel() or current_rejected.numel() != anchor_rejected.numel():
            raise ValueError(
                f"anchor/token length mismatch for {pair_id}: "
                f"chosen {anchor_chosen.numel()} != {current_chosen.numel()}, "
                f"rejected {anchor_rejected.numel()} != {current_rejected.numel()}"
            )

        if method == "copr_adapted":
            current_log_distribution = F.log_softmax(
                torch.stack([policy[index], policy[n + index]]), dim=0
            )
            target = torch.as_tensor(
                [saved["chosen_sum"], saved["rejected_sum"]],
                device=policy.device, dtype=policy.dtype,
            )
            target_log_distribution = F.log_softmax(target, dim=0)
            penalty = 0.5 * (current_log_distribution - target_log_distribution).square().sum()
            penalties.append(penalty)
            chosen_rates.append(float(current_log_distribution[0] < target_log_distribution[0]))
            rejected_rates.append(float(current_log_distribution[1] > target_log_distribution[1]))
            cap_rates.append(0.0)
            shifts.append(0.0)
            continue

        chosen_delta = current_chosen - anchor_chosen
        rejected_delta = current_rejected - anchor_rejected
        if method == "dapr_c":
            common_shift = torch.cat([chosen_delta, rejected_delta]).mean().detach()
        else:
            common_shift = policy.new_zeros(())
        chosen_violation = F.relu(common_shift - chosen_delta)
        rejected_violation = F.relu(rejected_delta - common_shift)
        violations = torch.cat([chosen_violation, rejected_violation])
        penalties.append(_huber(chosen_violation, huber_delta).sum()
                         + _huber(rejected_violation, huber_delta).sum())
        chosen_rates.append((chosen_violation > 0).float().mean().item())
        rejected_rates.append((rejected_violation > 0).float().mean().item())
        cap_rates.append((violations > huber_delta).float().mean().item())
        shifts.append(common_shift.item())

    regularizer = torch.stack(penalties).mean()
    return regularizer, {
        "anchor_loss": regularizer.detach().item(),
        "chosen_violation_rate": float(np.mean(chosen_rates)),
        "rejected_violation_rate": float(np.mean(rejected_rates)),
        "huber_cap_rate": float(np.mean(cap_rates)),
        "anchor_common_shift": float(np.mean(shifts)),
        "anchored_pairs": len(replay_indices),
    }


def dpo_loss(model, batch, beta, reference_cache=None, method="none", anchors=None,
             anchor_strength=0.1, huber_delta=1.0, return_diagnostics=False):
    """Summed-log-probability DPO, optionally with a replay-only peak constraint."""
    n = batch["n_pairs"]
    if method in ANCHOR_METHODS:
        token_logp, seq_index, lengths = response_token_logprobs(model, batch)
        policy = torch.zeros(
            batch["input_ids"].shape[0], device=token_logp.device, dtype=token_logp.dtype
        ).index_add(0, seq_index, token_logp)
        if reference_cache is None:
            with torch.no_grad(), model.disable_adapter():
                reference, _ = response_logprobs(model, batch)
        else:
            reference = _cached_reference(batch, reference_cache, policy.device)
        ratio = policy - reference
    else:
        ratio, _ = policy_minus_reference(model, batch, reference_cache)
        policy = token_logp = seq_index = None
    logits = beta * (ratio[:n] - ratio[n:])
    base_loss = -F.logsigmoid(logits).mean()
    if method in ANCHOR_METHODS:
        regularizer, diagnostics = _anchor_regularizer(
            method, policy, token_logp, seq_index, batch, anchors, huber_delta
        )
    else:
        regularizer = base_loss.new_zeros(())
        diagnostics = {
            "anchor_loss": 0.0, "chosen_violation_rate": 0.0,
            "rejected_violation_rate": 0.0, "huber_cap_rate": 0.0,
            "anchor_common_shift": 0.0, "anchored_pairs": 0,
        }
    loss = base_loss + float(anchor_strength) * regularizer
    diagnostics.update({
        "dpo_loss": base_loss.detach().item(),
        "weighted_anchor_loss": float(anchor_strength) * diagnostics["anchor_loss"],
    })
    result = (loss, (logits > 0).float().mean().item())
    return (*result, diagnostics) if return_diagnostics else result


@torch.no_grad()
def score_pairs(model, tokenizer, df, beta=0.1, batch_size=4, max_tokens=1024,
                desc="scoring", reference_cache=None):
    """Score relative-to-base and absolute policy preference margins for every pair."""
    was_training = model.training
    model.eval()
    rows = df.to_dict("records")
    order = sorted(range(len(rows)), key=lambda i: pair_length(rows[i]) if "prompt_tokens" in rows[i] else 0)
    columns = {name: [0.0] * len(rows) for name in
               ("margin", "margin_sum", "policy_margin", "policy_margin_sum")}
    for start in tqdm(
        range(0, len(rows), batch_size), desc=desc, unit="batch", leave=True, dynamic_ncols=True
    ):
        indices = order[start:start + batch_size]
        batch = make_batch(tokenizer, [rows[i] for i in indices], max_tokens)
        policy, reference, lengths = policy_and_reference(model, batch, reference_cache)
        n = len(indices)
        ratio = policy - reference
        values = {
            "margin": beta * (ratio[:n] / lengths[:n] - ratio[n:] / lengths[n:]),
            "margin_sum": beta * (ratio[:n] - ratio[n:]),
            "policy_margin": (policy[:n] / lengths[:n] - policy[n:] / lengths[n:]),
            "policy_margin_sum": policy[:n] - policy[n:],
        }
        for name, tensor in values.items():
            for j, index in enumerate(indices):
                columns[name][index] = tensor[j].item()
    if was_training:
        model.train()
    return pd.DataFrame(columns, index=df["id"].values)


def score_margins(model, tokenizer, df, beta=0.1, batch_size=4, max_tokens=1024,
                  reference_cache=None):
    return score_pairs(model, tokenizer, df, beta, batch_size, max_tokens,
                       reference_cache=reference_cache)["margin"]


@torch.no_grad()
def score_preference_anchors(model, tokenizer, df, batch_size=4, max_tokens=1024,
                             desc="saving peak anchors"):
    """Capture exact peak-time response-token log probabilities for DAPR/COPR-adapted."""
    was_training = model.training
    model.eval()
    rows = df.to_dict("records")
    order = sorted(
        range(len(rows)), key=lambda i: pair_length(rows[i]) if "prompt_tokens" in rows[i] else 0
    )
    anchors = {}
    for start in tqdm(
        range(0, len(rows), batch_size), desc=desc, unit="batch", leave=True,
        dynamic_ncols=True,
    ):
        indices = order[start:start + batch_size]
        batch = make_batch(tokenizer, [rows[i] for i in indices], max_tokens)
        token_logp, seq_index, _ = response_token_logprobs(model, batch)
        n = len(indices)
        for local, original in enumerate(indices):
            chosen = token_logp[seq_index == local].detach().cpu().numpy().astype(np.float16)
            rejected = token_logp[seq_index == n + local].detach().cpu().numpy().astype(np.float16)
            anchors[rows[original]["id"]] = {
                "chosen": chosen,
                "rejected": rejected,
                "chosen_sum": float(chosen.astype(np.float32).sum()),
                "rejected_sum": float(rejected.astype(np.float32).sum()),
            }
    if was_training:
        model.train()
    return anchors


def save_preference_anchors(anchors, path):
    """Save a no-pickle compressed anchor sidecar suitable for audited stage resume."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ids = sorted(anchors)

    def flatten(key):
        arrays = [np.asarray(anchors[pair_id][key], dtype=np.float16) for pair_id in ids]
        offsets = np.zeros(len(arrays) + 1, dtype=np.int64)
        if arrays:
            offsets[1:] = np.cumsum([array.size for array in arrays])
            values = np.concatenate(arrays)
        else:
            values = np.array([], dtype=np.float16)
        return values, offsets

    chosen, chosen_offsets = flatten("chosen")
    rejected, rejected_offsets = flatten("rejected")
    np.savez_compressed(
        path,
        ids=np.asarray(ids, dtype=str),
        chosen=chosen,
        chosen_offsets=chosen_offsets,
        rejected=rejected,
        rejected_offsets=rejected_offsets,
        chosen_sum=np.asarray([anchors[pair_id]["chosen_sum"] for pair_id in ids], dtype=np.float32),
        rejected_sum=np.asarray([anchors[pair_id]["rejected_sum"] for pair_id in ids], dtype=np.float32),
    )


def load_preference_anchors(path):
    """Load anchors written by :func:`save_preference_anchors` without pickle."""
    with np.load(path, allow_pickle=False) as saved:
        ids = saved["ids"].astype(str).tolist()
        chosen, chosen_offsets = saved["chosen"], saved["chosen_offsets"]
        rejected, rejected_offsets = saved["rejected"], saved["rejected_offsets"]
        chosen_sum, rejected_sum = saved["chosen_sum"], saved["rejected_sum"]
        return {
            pair_id: {
                "chosen": chosen[chosen_offsets[index]:chosen_offsets[index + 1]].copy(),
                "rejected": rejected[rejected_offsets[index]:rejected_offsets[index + 1]].copy(),
                "chosen_sum": float(chosen_sum[index]),
                "rejected_sum": float(rejected_sum[index]),
            }
            for index, pair_id in enumerate(ids)
        }


def summarize(scores):
    """Aggregate pair scores. Accuracy is the percentage whose chosen answer has margin > 0."""
    out = {
        "accuracy": round(100 * (scores["margin"] > 0).mean(), 1),
        "accuracy_sum": round(100 * (scores["margin_sum"] > 0).mean(), 1),
        "mean_margin": round(scores["margin"].mean(), 5),
        "mean_margin_sum": round(scores["margin_sum"].mean(), 4),
    }
    if "policy_margin" in scores:
        out.update({
            "policy_accuracy": round(100 * (scores["policy_margin"] > 0).mean(), 1),
            "policy_accuracy_sum": round(100 * (scores["policy_margin_sum"] > 0).mean(), 1),
            "mean_policy_margin": round(scores["policy_margin"].mean(), 5),
            "mean_policy_margin_sum": round(scores["policy_margin_sum"].mean(), 4),
        })
    return out


def _train_microbatches(model, tokenizer, pairs, optimizer, params, beta, micro_batch, max_tokens,
                        reference_cache=None, method="none", anchors=None,
                        anchor_strength=0.1, huber_delta=1.0):
    optimizer.zero_grad()
    step_loss, step_acc = 0.0, 0.0
    step_diagnostics = {
        "dpo_loss": 0.0, "anchor_loss": 0.0, "weighted_anchor_loss": 0.0,
        "chosen_violation_rate": 0.0, "rejected_violation_rate": 0.0,
        "huber_cap_rate": 0.0, "anchor_common_shift": 0.0, "anchored_pairs": 0.0,
    }
    total_replay = sum(bool(pair.get("_is_replay", False)) for pair in pairs)
    for start in range(0, len(pairs), micro_batch):
        chunk = pairs[start:start + micro_batch]
        batch = make_batch(tokenizer, chunk, max_tokens)
        weight = len(chunk) / len(pairs)
        chunk_replay = sum(bool(pair.get("_is_replay", False)) for pair in chunk)
        replay_weight = chunk_replay / total_replay if total_replay else 0.0
        # The step objective is mean DPO over every pair plus alpha times mean anchor loss over
        # replay pairs. Compensating here prevents the 10% replay share from silently shrinking
        # alpha by another factor of ten during gradient accumulation.
        effective_anchor_strength = (
            float(anchor_strength) * replay_weight / weight if replay_weight else 0.0
        )
        result = dpo_loss(
            model, batch, beta, reference_cache=reference_cache, method=method,
            anchors=anchors, anchor_strength=effective_anchor_strength, huber_delta=huber_delta,
            return_diagnostics=True,
        )
        if len(result) == 2:  # CPU bookkeeping tests replace dpo_loss with a two-value stand-in.
            loss, accuracy = result
            diagnostics = step_diagnostics
        else:
            loss, accuracy, diagnostics = result
        (loss * weight).backward()
        step_loss += loss.item() * weight
        step_acc += accuracy * weight
        step_diagnostics["dpo_loss"] += float(diagnostics.get("dpo_loss", 0.0)) * weight
        for key in (
            "anchor_loss", "chosen_violation_rate", "rejected_violation_rate",
            "huber_cap_rate", "anchor_common_shift",
        ):
            step_diagnostics[key] += float(diagnostics.get(key, 0.0)) * replay_weight
        step_diagnostics["weighted_anchor_loss"] = (
            float(anchor_strength) * step_diagnostics["anchor_loss"]
        )
        step_diagnostics["anchored_pairs"] += float(diagnostics.get("anchored_pairs", 0.0))
    torch.nn.utils.clip_grad_norm_(params, 1.0)
    optimizer.step()
    return step_loss, step_acc, step_diagnostics


def _peak_gb():
    return torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else 0.0


def _counterfactual_projected_scores(
    model, tokenizer, buffer_rows, future_steps, optimizer, scheduler, params,
    beta, micro_batch, max_tokens, score_batch_size, reference_cache, label="CPMR",
):
    """Measure old-pair margins before and after a reversible new-data-only lookahead.

    The virtual optimizer steps use the real upcoming examples, optimizer state, learning-rate
    schedule, and model dropout. Parameters, optimizer, scheduler, and random-number state are then
    restored so the lookahead cannot change the actual experiment trajectory.
    """
    current = score_pairs(
        model, tokenizer, buffer_rows, beta=beta, batch_size=score_batch_size,
        max_tokens=max_tokens, desc=f"{label}: current buffer", reference_cache=reference_cache,
    )
    parameter_state = [parameter.detach().clone() for parameter in params]
    optimizer_state = copy.deepcopy(optimizer.state_dict())
    scheduler_state = copy.deepcopy(scheduler.state_dict())
    cpu_rng_state = torch.get_rng_state()
    cuda_rng_state = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    was_training = getattr(model, "training", True)

    try:
        model.train()
        # The caller passes one list per upcoming optimizer step. Keeping each step separate is
        # essential: one large batch would not reproduce the scheduled new-task trajectory.
        for pairs in tqdm(
            future_steps, desc=f"{label}: virtual lookahead", unit="step", leave=True,
            dynamic_ncols=True,
        ):
            _train_microbatches(
                model, tokenizer, pairs, optimizer, params, beta, micro_batch, max_tokens,
                reference_cache,
            )
            scheduler.step()
        projected = score_pairs(
            model, tokenizer, buffer_rows, beta=beta, batch_size=score_batch_size,
            max_tokens=max_tokens, desc=f"{label}: projected buffer", reference_cache=reference_cache,
        )
    finally:
        with torch.no_grad():
            for parameter, saved in zip(params, parameter_state):
                parameter.copy_(saved)
        optimizer.load_state_dict(optimizer_state)
        scheduler.load_state_dict(scheduler_state)
        optimizer.zero_grad()
        torch.set_rng_state(cpu_rng_state)
        if cuda_rng_state is not None:
            torch.cuda.set_rng_state_all(cuda_rng_state)
        model.train(was_training)

    return current, projected


def train_stage(model, tokenizer, df, beta=0.1, lr=1e-4, pairs_per_step=18, micro_batch=2,
                max_tokens=1024, seed=0, desc="training", reference_cache=None):
    """Train for one pass over all new pairs and return optimizer-step history."""
    torch.manual_seed(seed)
    rows = df.sample(frac=1, random_state=seed).to_dict("records")
    n_steps = math.ceil(len(rows) / pairs_per_step)
    params = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=lr)
    scheduler = get_linear_schedule_with_warmup(optimizer, max(1, n_steps // 20), n_steps)
    model.train()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    started, history = time.time(), []
    bar = tqdm(range(n_steps), desc=desc, unit="step", leave=True, dynamic_ncols=True)
    for step in bar:
        pairs = rows[step * pairs_per_step:(step + 1) * pairs_per_step]
        if "prompt_tokens" in pairs[0]:
            pairs = sorted(pairs, key=pair_length)
        loss, accuracy, _ = _train_microbatches(
            model, tokenizer, pairs, optimizer, params, beta, micro_batch, max_tokens, reference_cache
        )
        scheduler.step()
        history.append({"step": step + 1, "loss": loss, "train_acc": accuracy,
                        "n_new": len(pairs), "n_replay": 0,
                        "examples_seen": sum(item["n_new"] for item in history) + len(pairs),
                        "minutes": (time.time() - started) / 60, "scoring_minutes": 0.0,
                        "peak_gpu_gb": _peak_gb()})
        bar.set_postfix(loss=f"{loss:.4f}", accuracy=f"{100 * accuracy:.1f}%")
    print(f"Done: {len(rows)} pairs in {(time.time() - started) / 60:.1f} min, peak {_peak_gb():.1f} GB")
    return pd.DataFrame(history)


def train_stage_replay(model, tokenizer, df, buffer=None, method="none", beta=0.1, lr=1e-4,
                       new_per_step=18, old_per_step=2, refreshes=5, micro_batch=2,
                       max_tokens=1024, seed=0, max_share_per_dataset=None, score_batch_size=4,
                       desc="training", progress_path=None, save_every=25, reference_cache=None,
                       fmcr_velocity_decay=0.5, fmcr_forecast_horizon=1.0,
                       anchors=None, anchor_strength=0.1, huber_delta=1.0,
                       mir_lookahead_steps=1):
    """Train once over new data with auditable replay; default batches are exactly 90/10 new/old."""
    import mfr_replay
    from mfr_utils import seed_everything

    seed_everything(seed)
    rng = np.random.default_rng(seed)
    rows = df.sample(frac=1, random_state=seed).to_dict("records")
    n_steps = math.ceil(len(rows) / new_per_step)
    interval_len = max(1, math.ceil(n_steps / max(1, refreshes)))
    replaying = method != "none" and buffer is not None and len(buffer) and old_per_step > 0
    params = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=lr)
    scheduler = get_linear_schedule_with_warmup(optimizer, max(1, n_steps // 20), n_steps)
    model.train()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    started, scoring_seconds = time.time(), 0.0
    history, replay_log = [], []
    plan = pd.DataFrame()
    plan_at = 0
    bar = tqdm(range(n_steps), desc=desc, unit="step", leave=True, dynamic_ncols=True)
    for step in bar:
        interval = step // interval_len
        if replaying and step % interval_len == 0:
            if method in ("cpmr", "mir_dpo"):
                refresh_started = time.time()
                upcoming_steps = []
                lookahead = interval_len if method == "cpmr" else int(mir_lookahead_steps)
                if lookahead <= 0:
                    raise ValueError("mir_lookahead_steps must be positive")
                for future_step in range(step, min(step + lookahead, n_steps)):
                    future_pairs = rows[
                        future_step * new_per_step:(future_step + 1) * new_per_step
                    ]
                    if future_pairs and "prompt_tokens" in future_pairs[0]:
                        future_pairs = sorted(future_pairs, key=pair_length)
                    upcoming_steps.append(future_pairs)
                current_scores, projected_scores = _counterfactual_projected_scores(
                    model, tokenizer, buffer.rows(), upcoming_steps, optimizer, scheduler, params,
                    beta, micro_batch, max_tokens, score_batch_size, reference_cache,
                    label="CPMR" if method == "cpmr" else "MIR-DPO",
                )
                if method == "cpmr" and step > 0:
                    # These are observed margins after the previous interval's real training. The
                    # counterfactual projection assumed no replay, so their difference is a useful
                    # intervention diagnostic rather than a conventional forecast error.
                    for event in replay_log:
                        if event["interval"] != interval:
                            continue
                        next_margin = float(current_scores.loc[event["id"], "margin"])
                        event["next_margin"] = next_margin
                        event["observed_minus_projected"] = (
                            next_margin - float(event["projected_margin"])
                        )
                if method == "mir_dpo" and step > 0:
                    for event in replay_log:
                        if event["interval"] != interval:
                            continue
                        next_margin_sum = float(current_scores.loc[event["id"], "margin_sum"])
                        next_loss = float(np.logaddexp(0.0, -next_margin_sum))
                        event["next_margin"] = float(current_scores.loc[event["id"], "margin"])
                        event["next_dpo_loss"] = next_loss
                        event["observed_minus_projected_loss"] = (
                            next_loss - float(event["projected_dpo_loss"])
                        )
                if method == "cpmr":
                    buffer.set_counterfactual_scores(
                        current_scores["margin"], projected_scores["margin"]
                    )
                else:
                    buffer.set_mir_scores(current_scores, projected_scores)
                scoring_seconds += time.time() - refresh_started

            # FMCR must score at step zero to initialize absolute policy margins. This also resets
            # velocities at each task boundary so a trend from the previous task is never projected
            # into the new one. Other margin methods retain the original four-refresh schedule.
            should_score = (method not in ("cpmr", "mir_dpo") and mfr_replay.needs_refresh(method)
                            and (step > 0 or method == "fmcr"))
            if should_score:
                refresh_started = time.time()
                scores = score_pairs(
                    model, tokenizer, buffer.rows(), beta=beta, batch_size=score_batch_size,
                    max_tokens=max_tokens, desc="refreshing buffer", reference_cache=reference_cache
                )
                if method == "fmcr":
                    if step > 0:
                        # Attach the next observed state to the previous interval's selections.
                        # The last interval has no in-stage successor and remains intentionally NA.
                        for event in replay_log:
                            if event["interval"] != interval:
                                continue
                            pair_id = event["id"]
                            next_margin = float(scores.loc[pair_id, "margin"])
                            next_policy = float(scores.loc[pair_id, "policy_margin"])
                            event["next_margin"] = next_margin
                            event["next_policy_margin"] = next_policy
                            event["actual_relative_nonpositive"] = bool(next_margin <= 0)
                            event["actual_policy_nonpositive"] = bool(next_policy <= 0)
                            label = event.get("risk_label")
                            if label == "forecast_policy_crossing":
                                event["forecast_correct"] = bool(next_policy <= 0)
                            elif label == "forecast_relative_crossing":
                                event["forecast_correct"] = bool(next_margin <= 0)
                            else:
                                event["forecast_correct"] = np.nan
                            event["policy_recovered"] = bool(
                                event.get("current_policy_margin", np.inf) <= 0 and next_policy > 0
                            )
                    buffer.set_forecast_scores(
                        scores, velocity_decay=fmcr_velocity_decay, initialize=(step == 0)
                    )
                else:
                    buffer.set_current(scores["margin"])
                scoring_seconds += time.time() - refresh_started
            plan = mfr_replay.plan_interval_details(
                buffer, method, interval_len * old_per_step, rng, max_share_per_dataset,
                plan_index=interval, forecast_horizon=fmcr_forecast_horizon,
            )
            if method == "fmcr" and progress_path:
                refresh_path = Path(progress_path).parent / f"fmcr_refresh_{interval + 1}.csv"
                buffer.rows().to_csv(refresh_path, index=False)
            if method == "cpmr" and progress_path:
                refresh_path = Path(progress_path).parent / f"cpmr_refresh_{interval + 1}.csv"
                buffer.rows().to_csv(refresh_path, index=False)
            if method == "mir_dpo" and progress_path:
                refresh_path = Path(progress_path).parent / f"mir_dpo_refresh_{interval + 1}.csv"
                buffer.rows().to_csv(refresh_path, index=False)
            plan_at = 0

        new_pairs = rows[step * new_per_step:(step + 1) * new_per_step]
        # A final partial new-data batch receives proportionally fewer replay rows. With 2,000
        # examples this gives 222 old rows: the closest possible integer budget to 10% of 2,222.
        step_old = min(old_per_step, int(round(len(new_pairs) * old_per_step / new_per_step)))
        selected = plan.iloc[plan_at:plan_at + step_old] if len(plan) else plan
        plan_at += step_old
        old_ids = selected["id"].tolist() if len(selected) else []
        marked_new = [{**pair, "_is_replay": False} for pair in new_pairs]
        marked_old = [{**pair, "_is_replay": True} for pair in buffer.get(old_ids)] if old_ids else []
        pairs = marked_new + marked_old
        for _, item in selected.iterrows():
            replay_log.append({"step": step + 1, "interval": interval + 1, **item.to_dict()})
        if "prompt_tokens" in pairs[0]:
            pairs = sorted(pairs, key=pair_length)
        loss, accuracy, diagnostics = _train_microbatches(
            model, tokenizer, pairs, optimizer, params, beta, micro_batch, max_tokens,
            reference_cache, method=method, anchors=anchors,
            anchor_strength=anchor_strength, huber_delta=huber_delta,
        )
        scheduler.step()
        previous_examples = history[-1]["examples_seen"] if history else 0
        history.append({
            "step": step + 1, "loss": loss, "train_acc": accuracy,
            "n_new": len(new_pairs), "n_replay": len(old_ids),
            "examples_seen": previous_examples + len(pairs),
            "minutes": (time.time() - started) / 60,
            "scoring_minutes": scoring_seconds / 60,
            "peak_gpu_gb": _peak_gb(),
            **diagnostics,
        })
        bar.set_postfix(loss=f"{loss:.4f}", accuracy=f"{100 * accuracy:.1f}%",
                        replay=len(old_ids))
        if progress_path and (step + 1) % save_every == 0:
            pd.DataFrame(history).to_csv(progress_path, index=False)
    elapsed = (time.time() - started) / 60
    print(f"Done: {len(rows)} new + {len(replay_log)} replayed in {elapsed:.1f} min "
          f"({scoring_seconds / 60:.1f} min scoring), peak {_peak_gb():.1f} GB")
    columns = ["step", "interval", *mfr_replay.PLAN_COLUMNS]
    if replay_log:
        replay_frame = pd.DataFrame(replay_log)
    else:
        replay_frame = pd.DataFrame(columns=columns)
    return pd.DataFrame(history), replay_frame
