#!/usr/bin/env python3
"""Gradient conflict between behaviors at a stage boundary (plan analysis A3).

Loads the checkpoint a run starts stage k from (the end of stage k-1), computes the mean DPO-loss
gradient over the LoRA parameters for a fixed sample of train pairs from each behavior, and
reports the pairwise cosine similarities. A negative cosine between the upcoming behavior and an
earlier one means a step on the new behavior locally increases the old behavior's loss -- the
mechanism behind forgetting (e.g. Helpful vs Safe).

Output: RUN_DIR/analysis/gradient_conflict_stage<k>.csv
"""

import argparse
import json
import sys
from itertools import combinations
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd
import torch

import mfr_data
import mfr_dpo
from mfr_analysis import stage_folder
from mfr_utils import buffer_seed, load_protocol, save_json_atomic, seed_everything

BEHAVIORS = ("helpful", "safe", "quality")


def mean_gradient(model, tokenizer, frame, beta, max_tokens, micro_batch=2):
    """Mean DPO-loss gradient over `frame`, flattened over trainable LoRA parameters."""
    named = mfr_dpo._trainable_lora_parameters(model)
    model.zero_grad(set_to_none=True)
    rows = frame.to_dict("records")
    for start in range(0, len(rows), micro_batch):
        chunk = rows[start:start + micro_batch]
        batch = mfr_dpo.make_batch(tokenizer, chunk, max_tokens)
        loss, _ = mfr_dpo.dpo_loss(model, batch, beta)
        (loss * len(chunk) / len(rows)).backward()
    flat = torch.cat([p.grad.detach().float().flatten() if p.grad is not None
                      else torch.zeros(p.numel(), device=p.device) for p in named.values()])
    model.zero_grad(set_to_none=True)
    return flat


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--stage", type=int, required=True, help="Stage about to start (2 or 3)")
    parser.add_argument("--pairs", type=int, default=64)
    parser.add_argument("--protocol", default=str(ROOT / "configs" / "experiment_protocol.json"))
    args = parser.parse_args()

    run_dir = Path(args.run_dir).expanduser().resolve()
    settings = json.loads((run_dir / "settings.json").read_text(encoding="utf-8"))
    order = settings["order"]
    if not 2 <= args.stage <= len(order):
        raise ValueError(f"--stage must be between 2 and {len(order)}")
    protocol = load_protocol(args.protocol)
    adapter = Path(stage_folder(str(run_dir), order, args.stage - 1, settings.get("stage1_from")))
    seed_everything(settings["seed"])
    model, tokenizer = mfr_dpo.load_model(
        protocol["model_name"], protocol["lora_r"], adapter_path=adapter,
        revision=protocol["model_revision"], lora_alpha=protocol["lora_alpha"],
        lora_dropout=protocol["lora_dropout"],
    )
    model.eval()  # no dropout: gradients are deterministic for a fixed sample
    splits = mfr_data.load_splits(ROOT / protocol["data_dir"])
    gradients = {}
    for behavior in BEHAVIORS:
        sample = splits[behavior]["train"].sample(
            n=args.pairs, random_state=buffer_seed(settings["seed"], behavior) + 7919)
        print(f"gradient: {behavior} ({len(sample)} pairs)", flush=True)
        gradients[behavior] = mean_gradient(model, tokenizer, sample, protocol["beta"],
                                            protocol["max_tokens"], protocol["micro_batch"])
    rows = []
    for left, right in combinations(BEHAVIORS, 2):
        cosine = torch.nn.functional.cosine_similarity(gradients[left], gradients[right], dim=0)
        rows.append({"run_name": settings["run_name"], "method": settings["method"],
                     "order_id": settings["order_id"], "seed": settings["seed"],
                     "stage": args.stage, "upcoming": order[args.stage - 1],
                     "behavior_a": left, "behavior_b": right, "cosine": float(cosine),
                     "pairs_per_behavior": args.pairs})
    out = run_dir / "analysis"
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out / f"gradient_conflict_stage{args.stage}.csv", index=False)
    save_json_atomic({"adapter": str(adapter), "pairs": args.pairs},
                     out / f"gradient_conflict_stage{args.stage}.manifest.json")
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()
