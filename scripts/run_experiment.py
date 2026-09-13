#!/usr/bin/env python3
"""Run or resume one frozen-protocol continual DPO experiment."""

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import mfr_cache
import mfr_data
import mfr_dpo
from mfr_replay import ReplayBuffer
from mfr_utils import (file_sha256, load_protocol, mark_run_complete, run_info, save_json_atomic,
                       seed_everything, stage_seed, validate_resume_settings, validate_stage1_source)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--drive-dir", required=True, help="Shared mfr-dpo folder containing runs/")
    parser.add_argument("--order", type=int, choices=(1, 2), required=True)
    parser.add_argument("--method", choices=("none", "random", "lowest_margin", "mfr"), required=True)
    parser.add_argument("--seed", type=int, choices=(0, 1), required=True)
    parser.add_argument("--start-stage", type=int, choices=(1, 2, 3), default=1)
    parser.add_argument("--stage1-from", help="Compatible completed run directory whose stage 1 should be reused")
    parser.add_argument("--reference-cache", help="Optional CSV made by build_reference_cache.py")
    return parser.parse_args()


def resolve_source_run(recorded, run_dir):
    """Use the recorded source path or the equivalent sibling under this Drive mount."""
    source = Path(recorded)
    if source.exists():
        return source
    sibling = run_dir.parent / source.name
    return sibling if sibling.exists() else source


def score_sets(model, tokenizer, splits, stage, trained_on, run_meta, run_dir, protocol, reference_cache):
    rows = []
    for dataset in ("safe", "helpful", "quality"):
        scores = mfr_dpo.score_pairs(
            model, tokenizer, splits[dataset]["val"], beta=protocol["beta"],
            max_tokens=protocol["max_tokens"], reference_cache=reference_cache,
            desc=f"stage {stage}: {dataset} validation",
        )
        scores.insert(0, "id", splits[dataset]["val"]["id"].values)
        folder = run_dir / ("stage0_base" if stage == 0 else f"stage{stage}_{trained_on}")
        folder.mkdir(parents=True, exist_ok=True)
        scores.to_csv(folder / f"margins_{dataset}_val.csv", index=False)
        rows.append({**run_meta, "stage": stage, "trained_on": trained_on, "eval_set": dataset,
                     **mfr_dpo.summarize(scores)})
    return pd.DataFrame(rows)


def update_buffer(model, tokenizer, buffer, stage, dataset, train_df, stage_dir, protocol, reference_cache):
    candidates = buffer.candidates(dataset, train_df)
    peak = mfr_dpo.score_pairs(
        model, tokenizer, candidates, beta=protocol["beta"], max_tokens=protocol["max_tokens"],
        reference_cache=reference_cache, desc=f"stage {stage}: buffer peak"
    )["margin"]
    buffer.add_stage(dataset, candidates, peak)
    older_rows = buffer.rows(exclude_dataset=dataset)
    if len(older_rows):
        current = mfr_dpo.score_pairs(
            model, tokenizer, older_rows, beta=protocol["beta"], max_tokens=protocol["max_tokens"],
            reference_cache=reference_cache, desc=f"stage {stage}: older buffer state"
        )["margin"]
        buffer.set_current(current)
    buffer.to_csv(stage_dir / "buffer.csv")
    return buffer


def main():
    args = parse_args()
    protocol = load_protocol(ROOT / "configs" / "experiment_protocol.json")
    order = protocol["orders"][str(args.order)]
    run_name = f"v2_o{args.order}_{args.method}_s{args.seed}"
    run_dir = Path(args.drive_dir) / "runs" / run_name
    data_dir = ROOT / protocol["data_dir"]
    manifest_path = data_dir / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Run scripts/prepare_data.py first; missing {manifest_path}")
    manifest_hash = file_sha256(manifest_path)
    splits = mfr_data.load_splits(data_dir)  # also verifies every JSONL against the manifest
    reference_cache = None
    if args.reference_cache:
        reference_cache = mfr_cache.load_reference_cache(args.reference_cache, {
            "protocol_version": protocol["protocol_version"], "data_version": protocol["data_version"],
            "model_revision": protocol["model_revision"], "data_manifest_sha256": manifest_hash,
        })
        reference_cache = reference_cache.set_index("id", drop=False)
    settings = {
        "run_name": run_name, "order_id": args.order, "order": order, "method": args.method,
        "seed": args.seed, "protocol_version": protocol["protocol_version"],
        "data_version": protocol["data_version"], "data_manifest_sha256": manifest_hash,
        "model_name": protocol["model_name"], "model_revision": protocol["model_revision"],
        "lr": protocol["learning_rate"], "beta": protocol["beta"],
        "new_per_step": protocol["new_per_step"], "old_per_step": protocol["old_per_step"],
        "max_tokens": protocol["max_tokens"], "buffer_size": protocol["buffer_size"],
        "refreshes": protocol["refreshes"], "stage1_from": args.stage1_from,
        "stage1_run_name": Path(args.stage1_from).name if args.stage1_from else None,
        "lora_r": protocol["lora_r"], "lora_alpha": protocol["lora_alpha"],
        "lora_dropout": protocol["lora_dropout"], "epochs": protocol["epochs"],
    }
    current_info = run_info(ROOT)
    if current_info.get("git_dirty"):
        raise RuntimeError("refusing to start/resume a scientific run from an uncommitted Git checkout")
    if (run_dir / "COMPLETE.json").exists():
        raise RuntimeError(f"run is already complete and will not be overwritten: {run_dir}")
    if args.start_stage == 1 and (run_dir / "results.csv").exists():
        raise RuntimeError("results already exist; set --start-stage to the first unfinished stage")
    run_dir.mkdir(parents=True, exist_ok=True)
    current_record = {**settings, **current_info}
    existing_settings = run_dir / "settings.json"
    if existing_settings.exists():
        with open(existing_settings, encoding="utf-8") as handle:
            old = json.load(handle)
        validate_resume_settings(current_record, old)
        resume_event = {
            "resumed": current_info.get("started"), "git_commit": current_info.get("git_commit"),
            "gpu": current_info.get("gpu"), "packages": current_info.get("packages"),
        }
        saved_settings = {**old, "resume_events": [*old.get("resume_events", []), resume_event]}
    else:
        saved_settings = current_record
    save_json_atomic(saved_settings, existing_settings)
    results = pd.DataFrame()
    buffer = ReplayBuffer(protocol["buffer_size"], args.seed)
    first_stage = args.start_stage
    previous_adapter = None
    if args.start_stage > 1:
        if args.start_stage == 2 and args.stage1_from:
            source_run = resolve_source_run(args.stage1_from, run_dir)
            if not (source_run / "COMPLETE.json").exists():
                raise RuntimeError(f"stage-1 source run is not complete: {source_run}")
            with open(source_run / "settings.json", encoding="utf-8") as handle:
                source_settings = json.load(handle)
            validate_stage1_source(current_record, source_settings)
            previous_adapter = source_run / f"stage1_{order[0]}"
            buffer = ReplayBuffer.from_csv(previous_adapter / "buffer.csv", protocol["buffer_size"], args.seed)
            source_results = pd.read_csv(source_run / "results.csv")
            results = source_results[source_results["stage"] <= 1].copy()
            results["run_name"], results["method"] = run_name, args.method
        else:
            previous_adapter = run_dir / f"stage{args.start_stage - 1}_{order[args.start_stage - 2]}"
            buffer = ReplayBuffer.from_csv(previous_adapter / "buffer.csv", protocol["buffer_size"], args.seed)
            results = pd.read_csv(run_dir / "results.csv")
            results = results[results["stage"] < args.start_stage]
    elif args.stage1_from:
        source_run = resolve_source_run(args.stage1_from, run_dir)
        if not (source_run / "COMPLETE.json").exists():
            raise RuntimeError(f"stage-1 source run is not complete: {source_run}")
        with open(source_run / "settings.json", encoding="utf-8") as handle:
            source_settings = json.load(handle)
        validate_stage1_source(current_record, source_settings)
        previous_adapter = source_run / f"stage1_{order[0]}"
        if not previous_adapter.exists():
            raise FileNotFoundError(previous_adapter)
        buffer = ReplayBuffer.from_csv(previous_adapter / "buffer.csv", protocol["buffer_size"], args.seed)
        source_results = pd.read_csv(source_run / "results.csv")
        results = source_results[source_results["stage"] <= 1].copy()
        results["run_name"], results["method"] = run_name, args.method
        first_stage = 2

    seed_everything(args.seed)
    model, tokenizer = mfr_dpo.load_model(
        protocol["model_name"], protocol["lora_r"], adapter_path=previous_adapter,
        revision=protocol["model_revision"],
        lora_alpha=protocol["lora_alpha"], lora_dropout=protocol["lora_dropout"],
    )
    if reference_cache is not None:
        canary = mfr_cache.verify_reference_cache(
            model, tokenizer, splits["helpful"]["val"], reference_cache,
            beta=protocol["beta"], max_tokens=protocol["max_tokens"],
        )
        with open(existing_settings, encoding="utf-8") as handle:
            saved_settings = json.load(handle)
        saved_settings["reference_cache_canary"] = canary
        save_json_atomic(saved_settings, existing_settings)
        print("Reference cache canary passed:", canary)
    run_meta = {"run_name": run_name, "order_id": args.order, "method": args.method, "seed": args.seed,
                "protocol_version": protocol["protocol_version"], "data_version": protocol["data_version"]}
    if first_stage == 1:
        results = score_sets(model, tokenizer, splits, 0, "base", run_meta, run_dir, protocol, reference_cache)

    for stage in range(first_stage, 4):
        dataset = order[stage - 1]
        stage_dir = run_dir / f"stage{stage}_{dataset}"
        stage_dir.mkdir(parents=True, exist_ok=True)
        history, replay_log = mfr_dpo.train_stage_replay(
            model, tokenizer, splits[dataset]["train"], buffer=buffer, method=args.method,
            beta=protocol["beta"], lr=protocol["learning_rate"],
            new_per_step=protocol["new_per_step"], old_per_step=protocol["old_per_step"],
            refreshes=protocol["refreshes"], micro_batch=protocol["micro_batch"],
            max_tokens=protocol["max_tokens"], seed=stage_seed(args.seed, stage),
            max_share_per_dataset=protocol["max_share_per_dataset"],
            desc=f"stage {stage}: {dataset}", progress_path=stage_dir / "history.csv",
            reference_cache=reference_cache,
        )
        model.save_pretrained(stage_dir)
        history.to_csv(stage_dir / "history.csv", index=False)
        replay_log.to_csv(stage_dir / "replay_log.csv", index=False)
        stage_results = score_sets(
            model, tokenizer, splits, stage, dataset, run_meta, run_dir, protocol, reference_cache
        )
        results = pd.concat([results, stage_results], ignore_index=True)
        results.to_csv(run_dir / "results.csv", index=False)
        if stage < len(order):
            buffer = update_buffer(
                model, tokenizer, buffer, stage, dataset, splits[dataset]["train"], stage_dir,
                protocol, reference_cache,
            )

    expected = [run_dir / "settings.json", run_dir / "results.csv"] + [
        run_dir / f"stage{stage}_{dataset}" / "history.csv" for stage, dataset in enumerate(order, 1)
    ]
    if args.stage1_from:
        expected = [path for path in expected if "stage1_" not in str(path)]
    mark_run_complete(run_dir, expected)
    print(results.tail(9).to_string(index=False))
    print(f"Complete: {run_dir}")


if __name__ == "__main__":
    main()
