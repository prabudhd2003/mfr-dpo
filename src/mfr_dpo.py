"""DPO training and margin scoring for Qwen2.5-1.5B-Instruct with 4-bit QLoRA.

One LoRA adapter sits on top of the frozen 4-bit base model.
    policy    = base model + adapter (what we train)
    reference = base model with the adapter switched off (frozen, no extra memory)

The same function (response_logprobs) is used for training (the DPO loss) and for
measuring how much the model prefers chosen over rejected (the margin).
"""

import math
import time

import pandas as pd
import torch
import torch.nn.functional as F
from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, get_linear_schedule_with_warmup

MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"


# ----------------------------------------------------------------------------
# Model
# ----------------------------------------------------------------------------

def load_model(model_name=MODEL_NAME, lora_r=16, adapter_path=None):
    """Load the base model in 4-bit and add a LoRA adapter.

    adapter_path=None  -> a fresh adapter (starts with no preference at all)
    adapter_path="..." -> continue from an adapter saved earlier with model.save_pretrained(...)
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    compute_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
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
        model, use_gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},  # the newer, recommended checkpointing
    )
    if adapter_path:
        model = PeftModel.from_pretrained(model, adapter_path, is_trainable=True)
    else:
        model = get_peft_model(model, LoraConfig(
            r=lora_r,
            lora_alpha=2 * lora_r,
            lora_dropout=0.05,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            task_type="CAUSAL_LM",
        ))
    return model, tokenizer


# ----------------------------------------------------------------------------
# Turning pairs into tensors
# ----------------------------------------------------------------------------

def chat_prompt(tokenizer, prompt):
    """The prompt exactly as the model sees it (Qwen chat template, ready for the answer)."""
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True
    )


def make_batch(tokenizer, rows, max_tokens=1024, device="cuda"):
    """Build one padded batch: all chosen sequences first, then all rejected ones.

    response_mask is 1 on the answer tokens and 0 on the prompt and padding,
    so only the answer counts when we add up log-probabilities.
    """
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
        "attention_mask": torch.tensor([[1] * len(ids) + [0] * (width - len(ids)) for ids, _ in seqs], device=device),
        "response_mask": torch.tensor([m + [0] * (width - len(m)) for _, m in seqs], device=device),
        "n_pairs": len(rows),
    }


def pair_length(row):
    """Tokens in a pair's longer sequence (columns come from the saved splits)."""
    return row["prompt_tokens"] + max(row["chosen_tokens"], row["rejected_tokens"])


# ----------------------------------------------------------------------------
# Log-probabilities, DPO loss, margins
# ----------------------------------------------------------------------------

def response_logprobs(model, batch):
    """For each sequence: (sum of log-probs of the answer tokens, number of answer tokens).

    Only answer positions go through the softmax, so prompt and padding tokens cost no extra memory.
    """
    logits = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]).logits
    logits = logits[:, :-1, :]                       # token t predicts token t+1
    targets = batch["input_ids"][:, 1:]
    mask = batch["response_mask"][:, 1:].bool()

    token_logp = -F.cross_entropy(logits[mask].float(), targets[mask], reduction="none")
    seq_index = mask.nonzero()[:, 0]                 # which sequence each answer token belongs to
    sums = torch.zeros(mask.shape[0], device=token_logp.device, dtype=token_logp.dtype)
    sums = sums.index_add(0, seq_index, token_logp)
    return sums, mask.sum(-1).float()


def policy_minus_reference(model, batch):
    """log p_policy(answer) - log p_reference(answer), plus answer lengths."""
    policy, lengths = response_logprobs(model, batch)
    with torch.no_grad(), model.disable_adapter():
        reference, _ = response_logprobs(model, batch)
    return policy - reference, lengths


def dpo_loss(model, batch, beta):
    """Standard DPO loss. Returns (loss, share of pairs where chosen already beats rejected)."""
    n = batch["n_pairs"]
    ratio, _ = policy_minus_reference(model, batch)
    logits = beta * (ratio[:n] - ratio[n:])          # chosen minus rejected
    return -F.logsigmoid(logits).mean(), (logits > 0).float().mean().item()


@torch.no_grad()
def score_pairs(model, tokenizer, df, beta=0.1, batch_size=4, max_tokens=1024, desc="scoring"):
    """Two margins for every pair in df (one row per pair, indexed by pair id).

    margin      length-normalized: beta * (per-token log-ratio of chosen - of rejected).
                The one our proposal uses for MFR.
    margin_sum  standard DPO score: beta * (summed log-ratio of chosen - of rejected).
                The one the training loss uses.
    Both are > 0 when the trained model prefers the chosen answer more than the base model did.
    A fresh adapter changes nothing, so every margin is exactly 0 before training.
    """
    was_training = model.training
    model.eval()
    rows = df.to_dict("records")
    order = sorted(range(len(rows)), key=lambda i: pair_length(rows[i]) if "prompt_tokens" in rows[i] else 0)
    margin, margin_sum = [0.0] * len(rows), [0.0] * len(rows)

    for start in tqdm(range(0, len(rows), batch_size), desc=desc, unit="batch", leave=False):
        idx = order[start:start + batch_size]
        batch = make_batch(tokenizer, [rows[i] for i in idx], max_tokens)
        ratio, lengths = policy_minus_reference(model, batch)
        n = len(idx)
        norm = (beta * (ratio[:n] / lengths[:n] - ratio[n:] / lengths[n:])).tolist()
        total = (beta * (ratio[:n] - ratio[n:])).tolist()
        for j, i in enumerate(idx):
            margin[i], margin_sum[i] = norm[j], total[j]

    if was_training:
        model.train()
    return pd.DataFrame({"margin": margin, "margin_sum": margin_sum}, index=df["id"].values)


def score_margins(model, tokenizer, df, beta=0.1, batch_size=4, max_tokens=1024):
    """Length-normalized margin only (a Series indexed by pair id). Kept for notebook 03."""
    return score_pairs(model, tokenizer, df, beta, batch_size, max_tokens)["margin"]


def summarize(scores):
    """One row of numbers from score_pairs output."""
    return {
        "accuracy": round(100 * (scores["margin"] > 0).mean(), 1),         # % of pairs, normalized margin
        "accuracy_sum": round(100 * (scores["margin_sum"] > 0).mean(), 1),  # % of pairs, standard DPO score
        "mean_margin": round(scores["margin"].mean(), 5),
        "mean_margin_sum": round(scores["margin_sum"].mean(), 4),
    }


# ----------------------------------------------------------------------------
# Training
# ----------------------------------------------------------------------------

def train_stage(model, tokenizer, df, beta=0.1, lr=5e-5, pairs_per_step=16, micro_batch=2,
                max_tokens=1024, seed=0, desc="training", log_every=None):
    """Train on every pair in df once (one epoch). Returns a table with one row per optimizer step.

    pairs_per_step: pairs per optimizer step (gradient accumulation over micro-batches).
    micro_batch:    pairs on the GPU at once. Lower it to 1 if you run out of memory.
    Within each step, pairs are grouped by length so short pairs aren't padded to long ones
    (this only changes speed, not the result of the step).
    """
    torch.manual_seed(seed)
    rows = df.sample(frac=1, random_state=seed).to_dict("records")
    n_steps = math.ceil(len(rows) / pairs_per_step)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=lr)
    scheduler = get_linear_schedule_with_warmup(optimizer, max(1, n_steps // 20), n_steps)

    model.train()
    torch.cuda.reset_peak_memory_stats()
    start = time.time()
    history = []
    bar = tqdm(range(n_steps), desc=desc, unit="step")
    for step in bar:
        pairs = rows[step * pairs_per_step:(step + 1) * pairs_per_step]
        if "prompt_tokens" in pairs[0]:
            pairs = sorted(pairs, key=pair_length)
        optimizer.zero_grad()
        step_loss, step_acc = 0.0, 0.0
        for i in range(0, len(pairs), micro_batch):
            chunk = pairs[i:i + micro_batch]
            loss, acc = dpo_loss(model, make_batch(tokenizer, chunk, max_tokens), beta)
            weight = len(chunk) / len(pairs)
            (loss * weight).backward()
            step_loss += loss.item() * weight
            step_acc += acc * weight
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        optimizer.step()
        scheduler.step()

        history.append({"step": step + 1, "loss": step_loss, "train_acc": step_acc,
                        "minutes": (time.time() - start) / 60})
        recent = history[-10:]
        bar.set_postfix(loss=f"{sum(h['loss'] for h in recent) / len(recent):.3f}",
                        acc=f"{sum(h['train_acc'] for h in recent) / len(recent):.2f}")

    minutes = (time.time() - start) / 60
    peak_gb = torch.cuda.max_memory_allocated() / 1e9
    print(f"Done: {len(rows)} pairs in {minutes:.1f} min, peak GPU memory {peak_gb:.1f} GB")
    return pd.DataFrame(history)
