#!/usr/bin/env python3
"""Score the locked preference test sets at every stage checkpoint of one completed run.

Retention, final-task acquisition and the final three-behavior average all need test scores
at each stage, not only at the end. Outputs go to RUN_DIR/locked_test/:

    stage<k>_<dataset>/<eval>_test_scores.csv   pair-level margins
    stage<k>_<dataset>/test_results.csv         per-behavior summary for that checkpoint
    results_test.csv                            one row per (stage, eval_set), like results.csv
    manifest.json                               adapter hashes, protocol, code commit

A borrowed Stage-1 checkpoint is identical to the `none` run's Stage 1, so its scores are copied
from that run's locked_test folder when present (score the `none` runs first).
Joint-training runs (order = null) have a single checkpoint, scored as stage 1 "joint".
"""

import argparse
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd

import mfr_data
import mfr_dpo
import mfr_eval
from mfr_analysis import resolve_borrowed_run
from mfr_utils import file_sha256, load_protocol, run_info, save_json_atomic

DATASETS = ("helpful", "safe", "quality")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument(
        "--protocol", default=str(ROOT / "configs" / "experiment_protocol.json"),
        help="Protocol the run was trained under (Qwen default; second_model_protocol.json for Llama)",
    )
    parser.add_argument(
        "--confirm-final-evaluation", action="store_true",
        help="Required acknowledgement that method selection is frozen (docs/FREEZE.md)",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def checkpoints(run_dir, settings):
    """[(stage, trained_on, adapter_dir, borrowed_source_run_or_None)] in training order."""
    if settings.get("order") is None:
        return [(1, "joint", run_dir / "joint_all", None)]
    order = settings["order"]
    items = []
    for stage, dataset in enumerate(order, 1):
        direct = run_dir / f"stage{stage}_{dataset}"
        if (direct / "adapter_config.json").exists():
            items.append((stage, dataset, direct, None))
        elif stage == 1 and settings.get("stage1_from"):
            source = Path(resolve_borrowed_run(str(run_dir), settings["stage1_from"]))
            items.append((stage, dataset, source / f"stage1_{dataset}", source))
        else:
            raise FileNotFoundError(f"missing checkpoint for stage {stage}: {direct}")
    return items


def adapter_hash(adapter):
    for name in ("adapter_model.safetensors", "adapter_model.bin"):
        if (adapter / name).exists():
            return file_sha256(adapter / name)
    raise FileNotFoundError(f"no adapter weights in {adapter}")


def complete(folder):
    return all((folder / f"{d}_test_scores.csv").exists() for d in DATASETS) and (
        folder / "test_results.csv").exists()


def main():
    args = parse_args()
    if not args.confirm_final_evaluation:
        raise RuntimeError(
            "locked-test scoring is disabled until method selection is frozen (docs/FREEZE.md); "
            "rerun with --confirm-final-evaluation"
        )
    run_dir = Path(args.run_dir).expanduser().resolve()
    mfr_eval.assert_test_ready(run_dir)
    settings_path = run_dir / "settings.json"
    settings = json.loads(settings_path.read_text(encoding="utf-8"))
    protocol_path = Path(args.protocol).expanduser().resolve()
    protocol = load_protocol(protocol_path)
    for key in ("protocol_version", "data_version", "model_name", "model_revision"):
        if settings.get(key) != protocol.get(key):
            raise ValueError(
                f"run {key}={settings.get(key)!r} does not match protocol {protocol.get(key)!r}; "
                "pass the protocol the run was trained with"
            )
    out_root = run_dir / "locked_test"
    if (out_root / "results_test.csv").exists() and not args.overwrite:
        raise FileExistsError(f"{out_root} already has results; use --overwrite intentionally")

    splits = mfr_data.load_splits(ROOT / protocol["data_dir"])
    test = {dataset: splits[dataset]["test"] for dataset in DATASETS}
    rows, manifest = [], []
    for stage, trained_on, adapter, borrowed in checkpoints(run_dir, settings):
        folder = out_root / f"stage{stage}_{trained_on}"
        source_folder = (borrowed / "locked_test" / f"stage1_{trained_on}") if borrowed else None
        if source_folder is not None and complete(source_folder) and not args.overwrite:
            shutil.copytree(source_folder, folder, dirs_exist_ok=True)
            summary = pd.read_csv(folder / "test_results.csv")
            how = f"copied from {borrowed.name}"
        else:
            print(f"Scoring stage {stage} ({trained_on}) from {adapter} ...", flush=True)
            model, tokenizer = mfr_dpo.load_model(
                protocol["model_name"], protocol["lora_r"], adapter_path=adapter,
                revision=protocol["model_revision"], lora_alpha=protocol["lora_alpha"],
                lora_dropout=protocol["lora_dropout"],
            )
            summary = mfr_eval.score_checkpoint(
                model, tokenizer, test, folder, beta=protocol["beta"],
                max_tokens=protocol["max_tokens"],
            )
            del model
            mfr_dpo.torch.cuda.empty_cache()
            how = "scored"
        for record in summary.to_dict("records"):
            rows.append({
                "run_name": settings["run_name"], "order_id": settings.get("order_id"),
                "method": settings["method"], "seed": settings["seed"],
                "stage": stage, "trained_on": trained_on, **record,
            })
        manifest.append({"stage": stage, "trained_on": trained_on, "adapter": str(adapter),
                         "adapter_sha256": adapter_hash(adapter), "how": how})
        print(f"  stage {stage}: {how}", flush=True)

    results = pd.DataFrame(rows)
    results.to_csv(out_root / "results_test.csv", index=False)
    save_json_atomic({
        "run_name": settings["run_name"],
        "settings_sha256": file_sha256(settings_path),
        "protocol": str(protocol_path),
        "protocol_sha256": file_sha256(protocol_path),
        "checkpoints": manifest,
        **run_info(ROOT),
    }, out_root / "manifest.json")
    print(results.to_string(index=False))


if __name__ == "__main__":
    main()
