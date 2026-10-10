#!/usr/bin/env python3
"""Score generated PKU-SafeRLHF, XSTest and HarmBench responses with WildGuard.

Works on any generation folder: final_eval (default), post_stage_safe (behavioral forgetting) or
smoke (--smoke, testing only). Takes many --run-dir values; WildGuard loads once per job and runs
that already have a summary are skipped.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd

from mfr_eval import assert_test_ready
from mfr_generation_eval import (
    evaluation_manifest,
    generation_subdir,
    load_evaluation_protocol,
    parse_wildguard_output,
    read_jsonl,
    summarize_wildguard,
    wildguard_prompt,
)
from mfr_utils import file_sha256, save_json_atomic
from mfr_vllm import evaluator, vllm_version


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", nargs="+", required=True)
    parser.add_argument("--checkpoint", choices=("final", "post_stage"), default="final")
    parser.add_argument("--behavior", default="safe",
                        help="With --checkpoint post_stage: the behavior folder to grade (safe)")
    parser.add_argument("--smoke", action="store_true", help="Grade generation/smoke/ (testing only)")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    behavior = args.behavior if args.checkpoint == "post_stage" else None
    subdir = "smoke" if args.smoke else generation_subdir(args.checkpoint, behavior)
    jobs = []
    for run in args.run_dir:
        run_dir = Path(run).expanduser().resolve()
        assert_test_ready(run_dir)
        generation_dir = run_dir / "generation" / subdir
        candidates = {"safe_test": generation_dir / "safe_test.jsonl",
                      "xstest": generation_dir / "xstest.jsonl",
                      "harmbench": generation_dir / "harmbench.jsonl"}
        if not candidates["safe_test"].exists():
            raise FileNotFoundError(f"generate responses first; missing {candidates['safe_test']}")
        if (generation_dir / "safety" / "summary.csv").exists() and not args.overwrite:
            print(f"Skipping {run_dir.name}: already graded", flush=True)
            continue
        jobs.append((run_dir, generation_dir,
                     {suite: path for suite, path in candidates.items() if path.exists()}))
    if not jobs:
        print("Nothing to grade.")
        return

    protocol_path = ROOT / "configs" / "evaluation_protocol.json"
    protocol = load_evaluation_protocol(protocol_path)
    config = protocol["safety_evaluator"]
    print(f"Loading WildGuard {config['model_name']}@{config['model_revision']} "
          f"({config.get('engine', 'vllm')})...", flush=True)
    classify = evaluator(config)

    for run_dir, generation_dir, inputs in jobs:
        if args.checkpoint == "final" and not args.smoke and len(inputs) < 3:
            print(f"Note: {run_dir.name}: grading only {sorted(inputs)}", flush=True)
        output_dir = generation_dir / "safety"
        output_dir.mkdir(parents=True, exist_ok=True)
        summaries = []
        for suite, input_path in inputs.items():
            generations = read_jsonl(input_path)
            prompts = [wildguard_prompt(row.prompt, row.response)
                       for row in generations.itertuples(index=False)]
            raw_outputs = classify(prompts, f"WildGuard {run_dir.name}: {suite}")
            parsed = pd.DataFrame([parse_wildguard_output(text) for text in raw_outputs])
            keep = [column for column in ("id", "prompt", "response", "run_name", "method",
                                          "xstest_label", "xstest_type", "focus", "note", "category")
                    if column in generations]
            scored = pd.concat([generations[keep].reset_index(drop=True), parsed], axis=1)
            scored["wildguard_output"] = raw_outputs
            output_path = output_dir / f"wildguard_{suite}.csv"
            scored.to_csv(output_path, index=False)
            summary_row = summarize_wildguard(scored, "xstest" if suite == "xstest" else "safety_test")
            summary_row["suite"] = suite  # safe_test / xstest / harmbench (harm_rate_pct = ASR)
            summaries.append(summary_row)
            evaluation_manifest(
                output_path, [input_path, protocol_path],
                evaluator_model=config["model_name"], evaluator_revision=config["model_revision"],
                evaluation_version=protocol["evaluation_version"],
                engine=config.get("engine", "vllm"), vllm_version=vllm_version(),
            )
        summary = pd.DataFrame(summaries)
        summary.to_csv(output_dir / "summary.csv", index=False)
        save_json_atomic({
            "evaluation_version": protocol["evaluation_version"],
            "evaluator_model": config["model_name"],
            "evaluator_revision": config["model_revision"],
            "engine": config.get("engine", "vllm"),
            "evaluation_protocol_sha256": file_sha256(protocol_path),
            "rows": summaries,
        }, output_dir / "summary.json")
        print(summary.round(2).to_string(index=False), flush=True)
        print(f"Saved safety evaluation to {output_dir}", flush=True)


if __name__ == "__main__":
    main()
