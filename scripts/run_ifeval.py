#!/usr/bin/env python3
"""Evaluate one completed final adapter on IFEval with logged samples."""

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mfr_eval import assert_test_ready
from mfr_utils import load_protocol


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    args = parser.parse_args()
    run_dir = Path(args.run_dir)
    assert_test_ready(run_dir)
    with open(run_dir / "settings.json", encoding="utf-8") as handle:
        settings = json.load(handle)
    protocol = load_protocol(ROOT / "configs" / "experiment_protocol.json")
    if settings.get("protocol_version") != protocol["protocol_version"]:
        raise ValueError("IFEval final reporting accepts protocol-v2 runs only")
    adapter = run_dir / f"stage3_{settings['order'][-1]}"
    output = run_dir / "generation" / "ifeval"
    output.mkdir(parents=True, exist_ok=True)
    model_args = (
        f"pretrained={protocol['model_name']},revision={protocol['model_revision']},"
        f"peft={adapter},load_in_4bit=True"
    )
    command = [
        "lm-eval", "run", "--model", "hf", "--model_args", model_args,
        "--tasks", "ifeval", "--apply_chat_template", "--batch_size", "4",
        "--device", "cuda:0", "--output_path", str(output), "--log_samples",
    ]
    print("Running IFEval for", settings["run_name"])
    subprocess.run(command, check=True)
    print("Saved IFEval results and per-prompt samples to", output)


if __name__ == "__main__":
    main()
