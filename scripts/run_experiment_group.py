#!/usr/bin/env python3
"""Run every requested method for one order/seed cell inside one GPU job."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--order", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument(
        "--methods",
        help="Comma-separated methods. Default: every core and secondary method in the protocol.",
    )
    parser.add_argument("--reference-cache", required=True)
    return parser.parse_args()


def load_protocol():
    with open(ROOT / "configs" / "experiment_protocol.json", encoding="utf-8") as handle:
        return json.load(handle)


def requested_methods(protocol, value):
    configured = protocol.get("methods", []) + protocol.get("secondary_methods", [])
    methods = configured if not value else [item.strip() for item in value.split(",") if item.strip()]
    if not methods:
        raise ValueError("at least one method is required")
    if len(methods) != len(set(methods)):
        raise ValueError(f"duplicate methods are not allowed: {methods}")
    unknown = [method for method in methods if method not in configured]
    if unknown:
        raise ValueError(
            f"methods {unknown} are not configured; available methods are {configured}. "
            "Add a future method to the protocol and src/mfr_replay.py first."
        )
    return methods


def first_unfinished_stage(run_dir, initial_stage, n_stages):
    if (run_dir / "COMPLETE.json").exists():
        return None
    result_path = run_dir / "results.csv"
    if not result_path.exists():
        return initial_stage

    import pandas as pd

    results = pd.read_csv(result_path)
    if results.empty or "stage" not in results:
        return initial_stage
    finished = int(results["stage"].max())
    if finished >= n_stages:
        raise RuntimeError(
            f"{run_dir} has final-stage results but no COMPLETE.json; inspect it before resubmitting"
        )
    return max(initial_stage, finished + 1)


def run_one(output_dir, cache, order, seed, method, stage1_source, n_stages, data_version):
    run_name = f"{data_version}_o{order}_{method}_s{seed}"
    run_dir = output_dir / "runs" / run_name
    initial_stage = 1 if method == "none" else 2
    start_stage = first_unfinished_stage(run_dir, initial_stage, n_stages)
    if start_stage is None:
        print(f"Skipping completed run: {run_name}", flush=True)
        return

    command = [
        sys.executable,
        "-u",
        str(ROOT / "scripts" / "run_experiment.py"),
        "--output-dir",
        str(output_dir),
        "--order",
        str(order),
        "--method",
        method,
        "--seed",
        str(seed),
        "--start-stage",
        str(start_stage),
        "--reference-cache",
        str(cache),
    ]
    if method != "none":
        command.extend(["--stage1-from", str(stage1_source)])

    print("\n" + "=" * 78, flush=True)
    print(f"Running {run_name} from stage {start_stage}", flush=True)
    print("=" * 78, flush=True)
    subprocess.run(command, cwd=ROOT, check=True)


def main():
    args = parse_args()
    protocol = load_protocol()
    order_key = str(args.order)
    if order_key not in protocol["orders"]:
        raise ValueError(f"unknown order {args.order}; configured orders: {list(protocol['orders'])}")
    if args.seed not in protocol.get("seeds", []):
        raise ValueError(f"unknown seed {args.seed}; configured seeds: {protocol.get('seeds', [])}")

    output_dir = Path(args.output_dir).expanduser().resolve()
    cache = Path(args.reference_cache).expanduser().resolve()
    if not cache.exists() or not cache.with_suffix(".manifest.json").exists():
        raise FileNotFoundError(
            f"reference cache is incomplete: {cache}. Submit the cache job first."
        )
    output_dir.mkdir(parents=True, exist_ok=True)

    methods = requested_methods(protocol, args.methods)
    data_version = protocol["data_version"]
    none_run = output_dir / "runs" / f"{data_version}_o{args.order}_none_s{args.seed}"

    # Stage 1 contains no replay, so run it once with `none` and reuse it for every replay method.
    if "none" in methods:
        run_one(
            output_dir, cache, args.order, args.seed, "none", none_run,
            len(protocol["orders"][order_key]), data_version,
        )
    if not (none_run / "COMPLETE.json").exists():
        raise RuntimeError(
            f"the completed no-replay source is required before replay methods: {none_run}"
        )

    for method in methods:
        if method == "none":
            continue
        run_one(
            output_dir, cache, args.order, args.seed, method, none_run,
            len(protocol["orders"][order_key]), data_version,
        )

    print(f"\nAll requested methods finished for order {args.order}, seed {args.seed}.", flush=True)


if __name__ == "__main__":
    main()
