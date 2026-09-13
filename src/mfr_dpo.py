"""QLoRA DPO training, replay training, and preference-margin scoring."""

from __future__ import annotations

import math
import time

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, get_linear_schedule_with_warmup

MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"


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
    }


def pair_length(row):
    return row["prompt_tokens"] + max(row["chosen_tokens"], row["rejected_tokens"])


def response_logprobs(model, batch):
    """Return summed response log probabilities and response lengths per sequence."""
    logits = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]).logits[:, :-1, :]
    targets = batch["input_ids"][:, 1:]
    mask = batch["response_mask"][:, 1:].bool()
    token_logp = -F.cross_entropy(logits[mask].float(), targets[mask], reduction="none")
    seq_index = mask.nonzero()[:, 0]
    sums = torch.zeros(mask.shape[0], device=token_logp.device, dtype=token_logp.dtype)
    return sums.index_add(0, seq_index, token_logp), mask.sum(-1).float()


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


def dpo_loss(model, batch, beta, reference_cache=None):
    """Standard summed-log-probability DPO loss."""
    n = batch["n_pairs"]
    ratio, _ = policy_minus_reference(model, batch, reference_cache)
    logits = beta * (ratio[:n] - ratio[n:])
    return -F.logsigmoid(logits).mean(), (logits > 0).float().mean().item()


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
                        reference_cache=None):
    optimizer.zero_grad()
    step_loss, step_acc = 0.0, 0.0
    for start in range(0, len(pairs), micro_batch):
        chunk = pairs[start:start + micro_batch]
        batch = make_batch(tokenizer, chunk, max_tokens)
        if reference_cache is None:
            loss, accuracy = dpo_loss(model, batch, beta)
        else:
            loss, accuracy = dpo_loss(model, batch, beta, reference_cache=reference_cache)
        weight = len(chunk) / len(pairs)
        (loss * weight).backward()
        step_loss += loss.item() * weight
        step_acc += accuracy * weight
    torch.nn.utils.clip_grad_norm_(params, 1.0)
    optimizer.step()
    return step_loss, step_acc


def _peak_gb():
    return torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else 0.0


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
        loss, accuracy = _train_microbatches(
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
                       desc="training", progress_path=None, save_every=25, reference_cache=None):
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
            if step > 0 and method in ("mfr", "lowest_margin"):
                refresh_started = time.time()
                scores = score_pairs(
                    model, tokenizer, buffer.rows(), beta=beta, batch_size=score_batch_size,
                    max_tokens=max_tokens, desc="refreshing buffer", reference_cache=reference_cache
                )
                buffer.set_current(scores["margin"])
                scoring_seconds += time.time() - refresh_started
            plan = mfr_replay.plan_interval_details(
                buffer, method, interval_len * old_per_step, rng, max_share_per_dataset
            )
            plan_at = 0

        new_pairs = rows[step * new_per_step:(step + 1) * new_per_step]
        # A final partial new-data batch receives proportionally fewer replay rows. With 2,000
        # examples this gives 222 old rows: the closest possible integer budget to 10% of 2,222.
        step_old = min(old_per_step, int(round(len(new_pairs) * old_per_step / new_per_step)))
        selected = plan.iloc[plan_at:plan_at + step_old] if len(plan) else plan
        plan_at += step_old
        old_ids = selected["id"].tolist() if len(selected) else []
        pairs = new_pairs + (buffer.get(old_ids) if old_ids else [])
        for _, item in selected.iterrows():
            replay_log.append({"step": step + 1, "interval": interval + 1, **item.to_dict()})
        if "prompt_tokens" in pairs[0]:
            pairs = sorted(pairs, key=pair_length)
        loss, accuracy = _train_microbatches(
            model, tokenizer, pairs, optimizer, params, beta, micro_batch, max_tokens, reference_cache
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
        })
        bar.set_postfix(loss=f"{loss:.4f}", accuracy=f"{100 * accuracy:.1f}%",
                        replay=len(old_ids))
        if progress_path and (step + 1) % save_every == 0:
            pd.DataFrame(history).to_csv(progress_path, index=False)
    elapsed = (time.time() - started) / 60
    print(f"Done: {len(rows)} new + {len(replay_log)} replayed in {elapsed:.1f} min "
          f"({scoring_seconds / 60:.1f} min scoring), peak {_peak_gb():.1f} GB")
    columns = ["step", "interval", "id", "dataset", "selection_score", "selection_rank", "method"]
    return pd.DataFrame(history), pd.DataFrame(replay_log, columns=columns)
