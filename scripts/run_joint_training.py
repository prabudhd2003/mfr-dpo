#!/usr/bin/env python3
"""Train the offline joint-access DPO reference on all three behaviors at once."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import mfr_cache
import mfr_data
import mfr_dpo
from mfr_utils import (file_sha256, load_protocol, mark_run_complete, run_info,
                       save_json_atomic, seed_everything, stage_seed)


DATASETS = ("helpful", "safe", "quality")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir", required=True,
        help="CARC artifact root; joint runs are written below OUTPUT_DIR/joint_runs",
    )
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--reference-cache", required=True)
    parser.add_argument(
        "--protocol", default=str(ROOT / "configs" / "experiment_protocol.json"),
        help="Authoritative protocol JSON (default: primary Qwen protocol)",
    )
    return parser.parse_args()


def score_validation(model, tokenizer, splits, stage, trained_on, run_meta, folder,
                     protocol, reference_cache):
    folder.mkdir(parents=True, exist_ok=True)
    rows = []
    for dataset in DATASETS:
        scores = mfr_dpo.score_pairs(
            model, tokenizer, splits[dataset]["val"], beta=protocol["beta"],
            max_tokens=protocol["max_tokens"], reference_cache=reference_cache,
            desc=f"joint stage {stage}: {dataset} validation",
        )
        scores.insert(0, "id", splits[dataset]["val"]["id"].values)
        scores.to_csv(folder / f"margins_{dataset}_val.csv", index=False)
        rows.append({
            **run_meta,
            "stage": stage,
            "trained_on": trained_on,
            "eval_set": dataset,
            **mfr_dpo.summarize(scores),
        })
    return pd.DataFrame(rows)


def main():
    args = parse_args()
    protocol = load_protocol(args.protocol)
    joint_seeds = protocol["joint_seeds"]
    if args.seed not in joint_seeds:
        raise ValueError(f"joint seed {args.seed} is not configured; choose one of {joint_seeds}")

    artifact_root = Path(args.output_dir).expanduser().resolve()
    run_name = f"{protocol['data_version']}_joint_s{args.seed}"
    run_dir = artifact_root / "joint_runs" / run_name
    complete_path = run_dir / "COMPLETE.json"
    if complete_path.exists():
        print(f"Skipping completed joint run: {run_name}", flush=True)
        return
    if run_dir.exists() and any(run_dir.iterdir()):
        raise RuntimeError(
            f"incomplete joint run already exists: {run_dir}. Inspect or archive it before rerunning."
        )

    current_info = run_info(ROOT)
    if current_info.get("git_dirty"):
        raise RuntimeError("refusing to start a scientific run from an uncommitted Git checkout")

    data_dir = ROOT / protocol["data_dir"]
    manifest_path = data_dir / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"missing frozen data manifest: {manifest_path}")
    manifest_hash = file_sha256(manifest_path)
    print("Loading and verifying the frozen data...", flush=True)
    splits = mfr_data.load_splits(data_dir)
    joint_train = mfr_data.joint_training_frame(splits, DATASETS)
    expected_rows = protocol["train_pairs"] * len(DATASETS)
    if len(joint_train) != expected_rows:
        raise ValueError(f"joint training expected {expected_rows} rows, found {len(joint_train)}")

    reference_cache = mfr_cache.load_reference_cache(args.reference_cache, {
        "protocol_version": protocol["protocol_version"],
        "data_version": protocol["data_version"],
        "model_revision": protocol["model_revision"],
        "data_manifest_sha256": manifest_hash,
    }).set_index("id", drop=False)

    pairs_per_step = protocol["new_per_step"]
    training_seed = stage_seed(args.seed, 1)
    settings = {
        "run_name": run_name,
        "training_regime": "offline_joint_access",
        "method": "joint",
        "order_id": 0,
        "order": None,
        "seed": args.seed,
        "joint_seeds": joint_seeds,
        "datasets": list(DATASETS),
        "pairs_per_dataset": protocol["train_pairs"],
        "total_train_pairs": len(joint_train),
        "pairs_per_step": pairs_per_step,
        "optimizer_steps": math.ceil(len(joint_train) / pairs_per_step),
        "shuffle": "one_global_seeded_shuffle_without_replacement",
        "training_seed": training_seed,
        "protocol_version": protocol["protocol_version"],
        "data_version": protocol["data_version"],
        "data_manifest_sha256": manifest_hash,
        "model_name": protocol["model_name"],
        "model_revision": protocol["model_revision"],
        "lr": protocol["learning_rate"],
        "beta": protocol["beta"],
        "max_tokens": protocol["max_tokens"],
        "lora_r": protocol["lora_r"],
        "lora_alpha": protocol["lora_alpha"],
        "lora_dropout": protocol["lora_dropout"],
        "epochs": protocol["epochs"],
        "micro_batch": protocol["micro_batch"],
        **current_info,
    }
    run_dir.mkdir(parents=True, exist_ok=True)
    settings_path = run_dir / "settings.json"
    save_json_atomic(settings, settings_path)

    seed_everything(args.seed)
    print(f"Loading {protocol['model_name']} with a fresh LoRA adapter...", flush=True)
    model, tokenizer = mfr_dpo.load_model(
        protocol["model_name"], protocol["lora_r"], revision=protocol["model_revision"],
        lora_alpha=protocol["lora_alpha"], lora_dropout=protocol["lora_dropout"],
    )
    print("Model loaded.", flush=True)
    canary = mfr_cache.verify_reference_cache(
        model, tokenizer, splits["helpful"]["val"], reference_cache,
        beta=protocol["beta"], max_tokens=protocol["max_tokens"],
    )
    settings["reference_cache_canary"] = canary
    save_json_atomic(settings, settings_path)
    print("Reference cache canary passed:", canary, flush=True)

    run_meta = {
        "run_name": run_name,
        "order_id": 0,
        "method": "joint",
        "seed": args.seed,
        "protocol_version": protocol["protocol_version"],
        "data_version": protocol["data_version"],
    }
    base_dir = run_dir / "stage0_base"
    results = score_validation(
        model, tokenizer, splits, 0, "base", run_meta, base_dir, protocol, reference_cache
    )

    print(
        f"Training jointly on {len(joint_train)} pairs: "
        f"{protocol['train_pairs']} from each behavior",
        flush=True,
    )
    joint_dir = run_dir / "joint_all"
    joint_dir.mkdir(parents=True, exist_ok=True)
    history = mfr_dpo.train_stage(
        model, tokenizer, joint_train, beta=protocol["beta"], lr=protocol["learning_rate"],
        pairs_per_step=pairs_per_step, micro_batch=protocol["micro_batch"],
        max_tokens=protocol["max_tokens"], seed=training_seed,
        desc=f"joint seed {args.seed}: helpful + safe + quality",
        reference_cache=reference_cache,
    )
    history.to_csv(joint_dir / "history.csv", index=False)
    model.save_pretrained(joint_dir)
    final_results = score_validation(
        model, tokenizer, splits, 1, "joint", run_meta, joint_dir, protocol, reference_cache
    )
    results = pd.concat([results, final_results], ignore_index=True)
    results.to_csv(run_dir / "results.csv", index=False)

    expected = [
        settings_path,
        run_dir / "results.csv",
        joint_dir / "history.csv",
        joint_dir / "adapter_config.json",
        *[base_dir / f"margins_{dataset}_val.csv" for dataset in DATASETS],
        *[joint_dir / f"margins_{dataset}_val.csv" for dataset in DATASETS],
    ]
    mark_run_complete(run_dir, expected)
    print(results.to_string(index=False))
    print(f"Complete: {run_dir}", flush=True)


if __name__ == "__main__":
    main()
