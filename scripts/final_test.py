#!/usr/bin/env python3
"""Run the one-time locked preference test evaluation for a chosen final run."""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import mfr_data
import mfr_dpo
import mfr_eval
from mfr_utils import load_protocol


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    args = parser.parse_args()
    run_dir = Path(args.run_dir)
    mfr_eval.assert_test_ready(run_dir)
    with open(run_dir / "settings.json", encoding="utf-8") as handle:
        settings = json.load(handle)
    protocol = load_protocol(ROOT / "configs" / "experiment_protocol.json")
    if settings.get("protocol_version") != protocol["protocol_version"]:
        raise ValueError("only v2 runs may be used for final test evaluation")
    final_dataset = settings["order"][-1]
    adapter = run_dir / f"stage3_{final_dataset}"
    print(f"Loading final checkpoint for {settings['run_name']}...", flush=True)
    model, tokenizer = mfr_dpo.load_model(
        protocol["model_name"], protocol["lora_r"], adapter_path=adapter,
        revision=protocol["model_revision"],
        lora_alpha=protocol["lora_alpha"], lora_dropout=protocol["lora_dropout"],
    )
    splits = mfr_data.load_splits(ROOT / protocol["data_dir"])
    test = {dataset: parts["test"] for dataset, parts in splits.items()}
    print("Model loaded. Scoring the three locked test sets...", flush=True)
    print(mfr_eval.score_checkpoint(
        model, tokenizer, test, run_dir / "final_test", beta=protocol["beta"],
        max_tokens=protocol["max_tokens"]
    ).to_string(index=False))


if __name__ == "__main__":
    main()
