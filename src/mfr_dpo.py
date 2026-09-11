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
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, get_linear_schedule_with_warmup

MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"


# ----------------------------------------------------------------------------
# Model
# ----------------------------------------------------------------------------

def load_model(model_name=MODEL_NAME, lora_r=16):
    """Load the base model in 4-bit and add a fresh LoRA adapter."""
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


# ----------------------------------------------------------------------------
# Log-probabilities, DPO loss, margins
# ----------------------------------------------------------------------------

def response_logprobs(model, batch):
    """For each sequence: (sum of log-probs of the answer tokens, number of answer tokens)."""
    logits = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]).logits
    logits = logits[:, :-1, :]                       # token t predicts token t+1
    targets = batch["input_ids"][:, 1:]
    mask = batch["response_mask"][:, 1:].float()
    token_logp = -F.cross_entropy(logits.float().transpose(1, 2), targets, reduction="none")
    return (token_logp * mask).sum(-1), mask.sum(-1)


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
def score_margins(model, tokenizer, df, beta=0.1, batch_size=4, max_tokens=1024):
    """Length-normalized margin for every pair in df (a Series indexed by pair id).

    margin = beta * (per-token log-ratio of chosen - per-token log-ratio of rejected)
    > 0 means the trained model prefers the chosen answer more than the base model did.
    A fresh adapter changes nothing, so every margin is exactly 0 before training.
    """
    was_training = model.training
    model.eval()
    rows = df.to_dict("records")
    margins = []
    for i in range(0, len(rows), batch_size):
        chunk = rows[i:i + batch_size]
        batch = make_batch(tokenizer, chunk, max_tokens)
        ratio, lengths = policy_minus_reference(model, batch)
        per_token = ratio / lengths
        n = len(chunk)
        margins += (beta * (per_token[:n] - per_token[n:])).tolist()
    if was_training:
        model.train()
    return pd.Series(margins, index=df["id"].values, name="margin")


# ----------------------------------------------------------------------------
# Training
# ----------------------------------------------------------------------------

def train_stage(model, tokenizer, df, beta=0.1, lr=5e-5, pairs_per_step=16, micro_batch=2,
                max_tokens=1024, seed=0, log_every=10):
    """Train on every pair in df once (one epoch). Returns a table with one row per optimizer step.

    pairs_per_step: pairs per optimizer step (gradient accumulation over micro-batches).
    micro_batch:    pairs on the GPU at once. Lower it if you run out of memory.
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
    for step in range(n_steps):
        pairs = rows[step * pairs_per_step:(step + 1) * pairs_per_step]
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
        if (step + 1) % log_every == 0 or step + 1 == n_steps:
            print(f"step {step + 1:4d}/{n_steps} | loss {step_loss:.3f} | batch acc {step_acc:.2f} "
                  f"| {history[-1]['minutes']:.1f} min")

    minutes = (time.time() - start) / 60
    peak_gb = torch.cuda.max_memory_allocated() / 1e9
    print(f"\nDone: {len(rows)} pairs in {minutes:.1f} min, peak GPU memory {peak_gb:.1f} GB")
    return pd.DataFrame(history)
