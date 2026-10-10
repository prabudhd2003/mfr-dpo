#!/usr/bin/env python3
"""Run position-controlled Prometheus comparisons for Helpful and Quality generations.

Two modes
  --candidate-run A --baseline-run B      two methods in the same order-seed cell (final checkpoints)
  --within-run RUN --behaviors helpful    behavioral forgetting: the final checkpoint (candidate)
                                          against the checkpoint right after that behavior was
                                          learned (baseline), on the same prompts
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd

from mfr_eval import assert_test_ready
from mfr_generation_eval import (
    evaluation_manifest,
    generate_classifier_text,
    generation_subdir,
    load_evaluation_protocol,
    load_quantized_model,
    parse_prometheus_choice,
    prometheus_prompt,
    read_jsonl,
    reconcile_pairwise_choices,
    summarize_pairwise,
)
from mfr_utils import save_json_atomic


def slug(value):
    return re.sub(r"[^a-zA-Z0-9_.-]+", "_", str(value))


def load_behavior(run_dir, behavior, subdir="final_eval"):
    path = run_dir / "generation" / subdir / f"{behavior}_test.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"generate {behavior} responses first: {path}")
    frame = read_jsonl(path)
    required = {"id", "prompt", "response", "chosen"}
    if missing := required - set(frame):
        raise ValueError(f"{path} is missing {sorted(missing)}")
    return frame, path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-run")
    parser.add_argument("--baseline-run")
    parser.add_argument("--within-run", help="Behavioral-forgetting mode: final vs post-stage")
    parser.add_argument("--behaviors", default="helpful,quality")
    parser.add_argument("--output-dir")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    within = args.within_run is not None
    if within == bool(args.candidate_run or args.baseline_run):
        raise ValueError("use either --within-run, or both --candidate-run and --baseline-run")
    if within:
        candidate_run = baseline_run = Path(args.within_run).expanduser().resolve()
    else:
        if not (args.candidate_run and args.baseline_run):
            raise ValueError("--candidate-run and --baseline-run are both required")
        candidate_run = Path(args.candidate_run).expanduser().resolve()
        baseline_run = Path(args.baseline_run).expanduser().resolve()
    assert_test_ready(candidate_run)
    assert_test_ready(baseline_run)
    candidate_settings = json.loads((candidate_run / "settings.json").read_text(encoding="utf-8"))
    baseline_settings = json.loads((baseline_run / "settings.json").read_text(encoding="utf-8"))
    behaviors = [item.strip() for item in args.behaviors.split(",") if item.strip()]
    if not behaviors or set(behaviors) - {"helpful", "quality"}:
        raise ValueError("--behaviors must contain helpful and/or quality")
    if not within and (candidate_settings["order_id"], candidate_settings["seed"]) != (
        baseline_settings["order_id"], baseline_settings["seed"]
    ) and not {"base", "joint"} & {candidate_settings["method"], baseline_settings["method"]}:
        raise ValueError("pairwise runs must use the same order and seed")
    candidate_label = candidate_settings["method"] + ("@final" if within else "")
    baseline_label = baseline_settings["method"] + ("@post_stage" if within else "")
    default_dir = (candidate_run / "generation" / "judges" / "final_vs_post_stage" if within else
                   candidate_run / "generation" / "final_eval" / "judges" /
                   f"vs_{slug(baseline_settings['run_name'])}")
    output_dir = Path(args.output_dir).expanduser().resolve() if args.output_dir else default_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "summary.csv"
    if summary_path.exists() and not args.overwrite:
        raise FileExistsError(f"{summary_path} already exists; use --overwrite intentionally")

    protocol_path = ROOT / "configs" / "evaluation_protocol.json"
    protocol = load_evaluation_protocol(protocol_path)
    config = protocol["pairwise_judge"]
    print(f"Loading pairwise judge {config['model_name']}...", flush=True)
    model, tokenizer = load_quantized_model(config["model_name"], config["model_revision"])

    summaries = []
    for behavior in behaviors:
        candidate, candidate_path = load_behavior(candidate_run, behavior)
        baseline, baseline_path = load_behavior(
            baseline_run, behavior,
            generation_subdir("post_stage", behavior) if within else "final_eval",
        )
        paired = candidate[["id", "prompt", "chosen", "response"]].merge(
            baseline[["id", "prompt", "chosen", "response"]], on="id", how="inner",
            suffixes=("_candidate", "_baseline"), validate="one_to_one",
        )
        if not (paired["prompt_candidate"] == paired["prompt_baseline"]).all():
            raise ValueError(f"prompt mismatch in {behavior} generations")
        if not (paired["chosen_candidate"] == paired["chosen_baseline"]).all():
            raise ValueError(f"reference-response mismatch in {behavior} generations")

        use_reference = bool(config.get("use_reference", False))
        forward_prompts = [prometheus_prompt(
            row.prompt_candidate, row.response_candidate, row.response_baseline, behavior,
            reference=row.chosen_candidate if use_reference else None,
        ) for row in paired.itertuples(index=False)]
        reverse_prompts = [prometheus_prompt(
            row.prompt_candidate, row.response_baseline, row.response_candidate, behavior,
            reference=row.chosen_candidate if use_reference else None,
        ) for row in paired.itertuples(index=False)]
        forward_text = generate_classifier_text(
            model, tokenizer, forward_prompts, config["batch_size"], config["max_new_tokens"],
            f"judge {behavior}: candidate=A", raw_prompts=True,
        )
        reverse_text = generate_classifier_text(
            model, tokenizer, reverse_prompts, config["batch_size"], config["max_new_tokens"],
            f"judge {behavior}: candidate=B", raw_prompts=True,
        )
        forward_choices = [parse_prometheus_choice(text) for text in forward_text]
        reverse_choices = [parse_prometheus_choice(text) for text in reverse_text]
        resolved = [reconcile_pairwise_choices(a, b) for a, b in zip(forward_choices, reverse_choices)]
        scored = pd.DataFrame({
            "id": paired["id"],
            "behavior": behavior,
            "candidate_method": candidate_label,
            "baseline_method": baseline_label,
            "forward_choice": forward_choices,
            "reverse_choice": reverse_choices,
            "final_choice": [item[0] for item in resolved],
            "position_consistent": [item[1] for item in resolved],
            "forward_judgment": forward_text,
            "reverse_judgment": reverse_text,
        })
        output_path = output_dir / f"{behavior}.csv"
        scored.to_csv(output_path, index=False)
        summary = summarize_pairwise(scored, candidate_label, baseline_label)
        summary["behavior"] = behavior
        summaries.append(summary)
        evaluation_manifest(
            output_path, [candidate_path, baseline_path, protocol_path],
            evaluator_model=config["model_name"], evaluator_revision=config["model_revision"],
            evaluation_version=protocol["evaluation_version"], score_both_orders=True,
        )

    summary = pd.DataFrame(summaries)
    summary.to_csv(summary_path, index=False)
    save_json_atomic({"rows": summaries, "evaluation_version": protocol["evaluation_version"]},
                     output_dir / "summary.json")
    print(summary.round(2).to_string(index=False), flush=True)
    print(f"Saved pairwise judgments to {output_dir}", flush=True)


if __name__ == "__main__":
    main()
