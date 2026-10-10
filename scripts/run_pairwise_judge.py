#!/usr/bin/env python3
"""Run position-controlled Prometheus comparisons for Helpful and Quality generations.

Two modes
  --candidate-run A --baseline-run B      two methods in the same order-seed cell (final checkpoints)
  --within-run RUN --behaviors helpful    behavioral forgetting: the final checkpoint (candidate)
                                          against the checkpoint right after that behavior was
                                          learned (baseline), on the same prompts

Every option takes several values (candidates and baselines are paired by position), so one job
judges many comparisons with a single judge load. Comparisons with a summary are skipped.
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
    generation_subdir,
    load_evaluation_protocol,
    parse_prometheus_choice,
    prometheus_prompt,
    read_jsonl,
    reconcile_pairwise_choices,
    summarize_pairwise,
)
from mfr_utils import save_json_atomic
from mfr_vllm import evaluator, vllm_version


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


def plan_comparison(candidate_run, baseline_run, within, behaviors, output_dir, overwrite):
    assert_test_ready(candidate_run)
    assert_test_ready(baseline_run)
    candidate_settings = json.loads((candidate_run / "settings.json").read_text(encoding="utf-8"))
    baseline_settings = json.loads((baseline_run / "settings.json").read_text(encoding="utf-8"))
    if not within and (candidate_settings["order_id"], candidate_settings["seed"]) != (
        baseline_settings["order_id"], baseline_settings["seed"]
    ) and not {"base", "joint"} & {candidate_settings["method"], baseline_settings["method"]}:
        raise ValueError(f"{candidate_run.name} vs {baseline_run.name}: pairwise runs must use the "
                         "same order and seed")
    default_dir = (candidate_run / "generation" / "judges" / "final_vs_post_stage" if within else
                   candidate_run / "generation" / "final_eval" / "judges" /
                   f"vs_{slug(baseline_settings['run_name'])}")
    out = Path(output_dir).expanduser().resolve() if output_dir else default_dir
    if (out / "summary.csv").exists() and not overwrite:
        print(f"Skipping {out}: already judged", flush=True)
        return None
    return {
        "candidate_run": candidate_run, "baseline_run": baseline_run, "within": within,
        "behaviors": behaviors, "output_dir": out,
        "candidate_label": candidate_settings["method"] + ("@final" if within else ""),
        "baseline_label": baseline_settings["method"] + ("@post_stage" if within else ""),
    }


def judge_comparison(job, judge, config, protocol, protocol_path):
    output_dir = job["output_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)
    candidate_label, baseline_label = job["candidate_label"], job["baseline_label"]
    use_reference = bool(config.get("use_reference", False))
    summaries = []
    for behavior in job["behaviors"]:
        candidate, candidate_path = load_behavior(job["candidate_run"], behavior)
        baseline, baseline_path = load_behavior(
            job["baseline_run"], behavior,
            generation_subdir("post_stage", behavior) if job["within"] else "final_eval",
        )
        paired = candidate[["id", "prompt", "chosen", "response"]].merge(
            baseline[["id", "prompt", "chosen", "response"]], on="id", how="inner",
            suffixes=("_candidate", "_baseline"), validate="one_to_one",
        )
        if not (paired["prompt_candidate"] == paired["prompt_baseline"]).all():
            raise ValueError(f"prompt mismatch in {behavior} generations")
        if not (paired["chosen_candidate"] == paired["chosen_baseline"]).all():
            raise ValueError(f"reference-response mismatch in {behavior} generations")
        forward_prompts = [prometheus_prompt(
            row.prompt_candidate, row.response_candidate, row.response_baseline, behavior,
            reference=row.chosen_candidate if use_reference else None,
        ) for row in paired.itertuples(index=False)]
        reverse_prompts = [prometheus_prompt(
            row.prompt_candidate, row.response_baseline, row.response_candidate, behavior,
            reference=row.chosen_candidate if use_reference else None,
        ) for row in paired.itertuples(index=False)]
        # one call for both orders keeps the GPU busy; split afterwards
        texts = judge(forward_prompts + reverse_prompts, f"judge {output_dir.name} {behavior}")
        forward_text, reverse_text = texts[:len(paired)], texts[len(paired):]
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
            engine=config.get("engine", "vllm"), vllm_version=vllm_version(),
        )
    summary = pd.DataFrame(summaries)
    summary.to_csv(output_dir / "summary.csv", index=False)
    save_json_atomic({"rows": summaries, "evaluation_version": protocol["evaluation_version"]},
                     output_dir / "summary.json")
    print(summary.round(2).to_string(index=False), flush=True)
    print(f"Saved pairwise judgments to {output_dir}", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-run", nargs="+")
    parser.add_argument("--baseline-run", nargs="+")
    parser.add_argument("--within-run", nargs="+", help="Behavioral-forgetting mode: final vs post-stage")
    parser.add_argument("--behaviors", default="helpful,quality")
    parser.add_argument("--output-dir", help="Only with a single comparison")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    behaviors = [item.strip() for item in args.behaviors.split(",") if item.strip()]
    if not behaviors or set(behaviors) - {"helpful", "quality"}:
        raise ValueError("--behaviors must contain helpful and/or quality")
    resolve = lambda values: [Path(v).expanduser().resolve() for v in values or []]
    if args.within_run:
        if args.candidate_run or args.baseline_run:
            raise ValueError("use either --within-run, or --candidate-run with --baseline-run")
        pairs = [(run, run, True) for run in resolve(args.within_run)]
    else:
        candidates, baselines = resolve(args.candidate_run), resolve(args.baseline_run)
        if not candidates or len(candidates) != len(baselines):
            raise ValueError("--candidate-run and --baseline-run need the same number of runs")
        pairs = [(c, b, False) for c, b in zip(candidates, baselines)]
    if args.output_dir and len(pairs) != 1:
        raise ValueError("--output-dir works only with one comparison")
    jobs = [job for c, b, within in pairs
            if (job := plan_comparison(c, b, within, behaviors, args.output_dir, args.overwrite))]
    if not jobs:
        print("Nothing to judge.")
        return

    protocol_path = ROOT / "configs" / "evaluation_protocol.json"
    protocol = load_evaluation_protocol(protocol_path)
    config = protocol["pairwise_judge"]
    print(f"Loading pairwise judge {config['model_name']} ({config.get('engine', 'vllm')})...", flush=True)
    judge = evaluator(config)
    for job in jobs:
        judge_comparison(job, judge, config, protocol, protocol_path)


if __name__ == "__main__":
    main()
