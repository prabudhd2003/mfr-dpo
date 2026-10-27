#!/usr/bin/env python3
"""Summarize a complete, explicitly paired PID validation experiment (CPU only)."""

import argparse
import itertools
import json
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from mfr_analysis import cost_table, load_runs, summary_table


def summarize(output_dir, methods, orders, seeds):
    expected = {f"v2_o{o}_{m}_s{s}" for o, m, s in itertools.product(orders, methods, seeds)}
    runs = load_runs(output_dir, quiet=True, completed_only=True)
    runs = runs[runs["run_name"].isin(expected)].copy()
    missing = expected - set(runs["run_name"])
    if missing:
        raise ValueError("missing completed runs: " + ", ".join(sorted(missing)))
    fingerprints = set()
    gains = {}
    for folder in runs["folder"].unique():
        settings = json.loads((Path(folder) / "settings.json").read_text(encoding="utf-8"))
        fingerprints.add(settings["scientific_code_sha256"])
        gains[settings["method"]] = settings.get("pid_profile")
    if len(fingerprints) != 1:
        raise ValueError("paired gain search requires one scientific code fingerprint")
    cells = summary_table(runs)
    if len(cells) != len(expected):
        raise ValueError("a completed run lacks final-stage validation results")
    last = runs[runs["stage"] == runs["order"].map(len)]
    final = last.groupby("run_name", observed=True)["accuracy"].mean().rename("final_average")
    cells = cells.merge(final, on="run_name").merge(
        cost_table(runs)[["run_name", "minutes", "scoring_minutes"]], on="run_name")
    for baseline in ("random", "lowest_margin"):
        reference = cells[cells["method"] == baseline][
            ["order_id", "seed", "mean_retention_change", "final_new_stage_score"]]
        if len(reference) != len(orders) * len(seeds):
            raise ValueError(f"include {baseline} for every paired cell")
        reference = reference.rename(columns={
            "mean_retention_change": f"retention_{baseline}",
            "final_new_stage_score": f"new_score_{baseline}"})
        cells = cells.merge(reference, on=["order_id", "seed"], validate="many_to_one")
        cells[f"retention_delta_{baseline}"] = (
            cells["mean_retention_change"] - cells[f"retention_{baseline}"])
    cells["new_score_delta_random"] = cells["final_new_stage_score"] - cells["new_score_random"]
    metrics = ["mean_retention_change", "final_average", "final_new_stage_score", "minutes",
               "scoring_minutes", "retention_delta_random", "retention_delta_lowest_margin",
               "new_score_delta_random"]
    summary = cells.groupby("method", observed=True)[metrics].mean().reset_index()
    summary["cells"] = len(orders) * len(seeds)
    summary["gains"] = summary["method"].astype(str).map(
        lambda method: json.dumps(gains[method], sort_keys=True) if gains[method] else "")
    protocol = json.loads((ROOT / "configs" / "experiment_protocol.json").read_text())
    tolerance = protocol["new_learning_tolerance_points"]
    summary["passes_random_gate"] = (
        (summary["retention_delta_random"] > 0) &
        (summary["new_score_delta_random"] >= -tolerance))
    return cells, summary.sort_values("mean_retention_change", ascending=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--methods", required=True, help="Comma-separated, including random and lowest_margin")
    parser.add_argument("--orders", default="1,3")
    parser.add_argument("--seeds", default="0")
    parser.add_argument("--label", default="pd_screen")
    parser.add_argument("--suggest-integral", action="store_true",
                        help="Print only six I-gain profile names for the two best admissible P/PD settings")
    args = parser.parse_args()
    if not args.label.replace("_", "").replace("-", "").isalnum():
        raise ValueError("label must contain letters, numbers, underscores, or hyphens")
    cells, summary = summarize(
        Path(args.output_dir), args.methods.split(","),
        [int(x) for x in args.orders.split(",")], [int(x) for x in args.seeds.split(",")])
    if args.suggest_integral:
        protocol = json.loads((ROOT / "configs" / "experiment_protocol.json").read_text())
        eligible = summary[
            summary["method"].astype(str).isin(["mfr", "pid_i000_d025", "pid_i000_d050", "pid_i000_d100"]) &
            (summary["new_score_delta_random"] >= -protocol["new_learning_tolerance_points"])
        ].head(2)
        if len(eligible) < 2:
            raise ValueError("fewer than two P/PD settings pass the new-learning tolerance; review the pilot")
        profiles = []
        for method in eligible["method"].astype(str):
            derivative = "d000" if method == "mfr" else method.rsplit("_", 1)[1]
            profiles += [f"pid_i{i}_{derivative}" for i in ("002", "010", "030")]
        print(",".join(profiles))
        return
    destination = Path(args.output_dir) / "pid_analysis"
    destination.mkdir(parents=True, exist_ok=True)
    cells.to_csv(destination / f"{args.label}_cells.csv", index=False)
    summary.to_csv(destination / f"{args.label}_summary.csv", index=False)
    print(summary.to_string(index=False))
    print(f"Saved validation summaries in {destination}")


if __name__ == "__main__":
    main()
