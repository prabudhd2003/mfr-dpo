#!/usr/bin/env python3
"""Validate the pairwise judge against human preference labels before using it (plan §5).

Runs the frozen judge (same prompt, settings and position control as run_pairwise_judge.py) on
the human-labeled validation pairs of Helpful and Quality: chosen vs rejected, in both A/B
orders. Agreement = the position-consistent verdict picks the human-chosen answer. The judge is
accepted only if agreement >= judge_validation.minimum_position_consistent_agreement_pct.
Validation pairs were used for method development, never the locked test.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd

import mfr_data
from mfr_generation_eval import (
    evaluation_manifest,
    generate_classifier_text,
    load_evaluation_protocol,
    load_quantized_model,
    parse_prometheus_choice,
    prometheus_prompt,
    reconcile_pairwise_choices,
)
from mfr_utils import file_sha256, load_protocol, save_json_atomic


def summarize_validation(frame, threshold):
    """Agreement with human labels; A = human-chosen in the forward order."""
    n = len(frame)
    consistent = frame["position_consistent"].astype(bool)
    agree = consistent & frame["final_choice"].eq("A")
    rng = np.random.default_rng(544)
    values = agree.to_numpy(float)
    boot = [rng.choice(values, size=n, replace=True).mean() for _ in range(10000)] if n else [np.nan]
    agreement = 100.0 * float(agree.mean()) if n else np.nan
    return {
        "pairs": int(n),
        "position_consistent_pct": 100.0 * float(consistent.mean()) if n else np.nan,
        "agreement_pct": agreement,
        "agreement_ci_low_pct": 100.0 * float(np.quantile(boot, 0.025)),
        "agreement_ci_high_pct": 100.0 * float(np.quantile(boot, 0.975)),
        "agreement_among_consistent_pct": (100.0 * float(agree[consistent].mean())
                                           if consistent.any() else np.nan),
        "threshold_pct": threshold,
        "accepted": bool(agreement >= threshold) if n else False,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True, help="Artifact root; writes judge_validation/")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    evaluation_path = ROOT / "configs" / "evaluation_protocol.json"
    evaluation = load_evaluation_protocol(evaluation_path)
    config, check = evaluation["pairwise_judge"], evaluation["judge_validation"]
    out = Path(args.output_dir).expanduser().resolve() / "judge_validation"
    out.mkdir(parents=True, exist_ok=True)
    if (out / "summary.csv").exists() and not args.overwrite:
        raise FileExistsError(f"{out / 'summary.csv'} exists; use --overwrite intentionally")

    protocol = load_protocol(ROOT / "configs" / "experiment_protocol.json")
    splits = mfr_data.load_splits(ROOT / protocol["data_dir"])
    use_reference = bool(config.get("use_reference", False))
    if use_reference:
        raise ValueError("judge validation requires the no-reference prompt (use_reference=false)")
    model, tokenizer = load_quantized_model(config["model_name"], config["model_revision"])

    rows = []
    for behavior in check["behaviors"]:
        pairs = splits[behavior][check["split"]].sort_values("id")
        forward = [prometheus_prompt(r.prompt, r.chosen, r.rejected, behavior)
                   for r in pairs.itertuples(index=False)]
        reverse = [prometheus_prompt(r.prompt, r.rejected, r.chosen, behavior)
                   for r in pairs.itertuples(index=False)]
        f_text = generate_classifier_text(model, tokenizer, forward, config["batch_size"],
                                          config["max_new_tokens"], f"validate {behavior}: chosen=A",
                                          raw_prompts=True)
        r_text = generate_classifier_text(model, tokenizer, reverse, config["batch_size"],
                                          config["max_new_tokens"], f"validate {behavior}: chosen=B",
                                          raw_prompts=True)
        f_choice = [parse_prometheus_choice(t) for t in f_text]
        r_choice = [parse_prometheus_choice(t) for t in r_text]
        resolved = [reconcile_pairwise_choices(a, b) for a, b in zip(f_choice, r_choice)]
        frame = pd.DataFrame({
            "id": pairs["id"].values, "behavior": behavior,
            "forward_choice": f_choice, "reverse_choice": r_choice,
            "final_choice": [x[0] for x in resolved], "position_consistent": [x[1] for x in resolved],
            "forward_judgment": f_text, "reverse_judgment": r_text,
        })
        path = out / f"{behavior}.csv"
        frame.to_csv(path, index=False)
        evaluation_manifest(path, [evaluation_path], evaluator_model=config["model_name"],
                            evaluator_revision=config["model_revision"])
        rows.append({"behavior": behavior, **summarize_validation(
            frame, check["minimum_position_consistent_agreement_pct"])})
        rows_all = pd.concat([pd.read_csv(out / f"{b}.csv") for b in
                              [r["behavior"] for r in rows]], ignore_index=True)
    rows.append({"behavior": "overall", **summarize_validation(
        rows_all, check["minimum_position_consistent_agreement_pct"])})
    summary = pd.DataFrame(rows)
    summary.to_csv(out / "summary.csv", index=False)
    save_json_atomic({"rows": rows, "judge": config["model_name"],
                      "judge_revision": config["model_revision"],
                      "evaluation_protocol_sha256": file_sha256(evaluation_path)},
                     out / "summary.json")
    print(summary.round(1).to_string(index=False))
    if not rows[-1]["accepted"]:
        print("\nJUDGE NOT ACCEPTED: below the predeclared agreement threshold. Do not run "
              "run_pairwise_judge.py; switch to the fallback judge in docs/FREEZE.md.", flush=True)


if __name__ == "__main__":
    main()
