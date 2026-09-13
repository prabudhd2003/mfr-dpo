"""Locked test scoring, deterministic generation, and external evaluator adapters."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import torch
from tqdm.auto import tqdm

from mfr_dpo import chat_prompt, score_pairs, summarize
from mfr_utils import save_json_atomic


def assert_test_ready(run_dir):
    """Prevent accidental final-test evaluation on unfinished experiment runs."""
    marker = Path(run_dir) / "COMPLETE.json"
    if not marker.exists():
        raise RuntimeError(f"{run_dir} is incomplete; final test evaluation is locked")
    return True


def score_checkpoint(model, tokenizer, splits, output_dir, **score_kwargs):
    """Score all three held-out sets and save aggregate plus pair-level results."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_rows = []
    for dataset, df in splits.items():
        scores = score_pairs(model, tokenizer, df, desc=f"test: {dataset}", **score_kwargs)
        scores.insert(0, "id", df["id"].values)
        scores.to_csv(output_dir / f"{dataset}_test_scores.csv", index=False)
        summary_rows.append({"eval_set": dataset, **summarize(scores)})
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(output_dir / "test_results.csv", index=False)
    return summary


@torch.no_grad()
def generate_responses(model, tokenizer, prompts, batch_size=4, max_new_tokens=512,
                       do_sample=False, seed=0):
    """Generate reproducible responses and retain prompt IDs for paired comparison."""
    torch.manual_seed(seed)
    model.eval()
    rows = prompts.to_dict("records")
    output = []
    old_cache = getattr(model.config, "use_cache", None)
    old_padding_side = tokenizer.padding_side
    tokenizer.padding_side = "left"
    model.config.use_cache = True
    for start in tqdm(range(0, len(rows), batch_size), desc="generating", leave=False):
        chunk = rows[start:start + batch_size]
        rendered = [chat_prompt(tokenizer, row["prompt"]) for row in chunk]
        inputs = tokenizer(rendered, return_tensors="pt", padding=True).to(model.device)
        generated = model.generate(
            **inputs, max_new_tokens=max_new_tokens, do_sample=do_sample,
            pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id,
        )
        prompt_width = inputs["input_ids"].shape[1]
        for row, tokens in zip(chunk, generated):
            response = tokenizer.decode(tokens[prompt_width:], skip_special_tokens=True).strip()
            output.append({"id": row.get("id"), "prompt": row["prompt"], "response": response})
    model.config.use_cache = old_cache
    tokenizer.padding_side = old_padding_side
    return pd.DataFrame(output)


def save_generations(generations, output_path, metadata=None):
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    generations.to_json(path, orient="records", lines=True, force_ascii=False)
    save_json_atomic({"created_at_utc": datetime.now(timezone.utc).isoformat(),
                      "rows": len(generations), **(metadata or {})}, path.with_suffix(".manifest.json"))


def run_evaluator(generations, evaluator, output_path=None):
    """Run any evaluator callable(response row -> dict) without coupling training to one package."""
    rows = []
    for record in generations.to_dict("records"):
        result = evaluator(record)
        rows.append({"id": record.get("id"), **result})
    scores = pd.DataFrame(rows)
    if output_path:
        scores.to_csv(output_path, index=False)
    return scores


def write_evaluation_manifest(path, **metadata):
    save_json_atomic({"evaluated_at_utc": datetime.now(timezone.utc).isoformat(), **metadata}, path)
