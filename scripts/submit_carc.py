#!/usr/bin/env python3
"""Submit reference-cache and experiment-group jobs to USC CARC Slurm."""

from __future__ import annotations

import argparse
import json
import shlex
import shutil
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
WORKER = ROOT / "scripts" / "carc_job.sh"
GPU_CHOICES = {
    "l40s": ("l40s", None),
    "a100-any": ("a100", None),
    "a100-40gb": ("a100", "a100-40gb"),
    "a100-80gb": ("a100", "a100-80gb"),
}


def add_common(parser, default_time):
    parser.add_argument("--output-dir", required=True, help="Absolute CARC artifact directory")
    parser.add_argument("--account", help="CARC project account; omit to use your default account")
    parser.add_argument(
        "--conda-env", default=str(ROOT / ".conda" / "envs" / "mfr-dpo"),
        help="Conda environment name or absolute prefix",
    )
    parser.add_argument(
        "--gpu", choices=tuple(GPU_CHOICES), default="l40s",
        help="GPU request; L40S is the project default",
    )
    parser.add_argument("--time", default=default_time, help="Slurm time limit")
    parser.add_argument("--dry-run", action="store_true", help="Print the sbatch command only")


def parse_args():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="action", required=True)

    cache = subparsers.add_parser("cache", help="Build the reference cache once")
    add_common(cache, "02:00:00")

    group = subparsers.add_parser("group", help="Run all requested methods for one order and seed")
    add_common(group, "04:00:00")
    group.add_argument("--order", type=int, required=True)
    group.add_argument("--seed", type=int, required=True)
    group.add_argument(
        "--methods",
        help="Comma-separated override; default is all five methods in the protocol",
    )
    return parser.parse_args()


def protocol():
    with open(ROOT / "configs" / "experiment_protocol.json", encoding="utf-8") as handle:
        return json.load(handle)


def validate_group(args, settings):
    if str(args.order) not in settings["orders"]:
        raise ValueError(f"unknown order {args.order}; configured orders: {list(settings['orders'])}")
    if args.seed not in settings.get("seeds", []):
        raise ValueError(f"unknown seed {args.seed}; configured seeds: {settings.get('seeds', [])}")
    configured = settings.get("methods", []) + settings.get("secondary_methods", [])
    methods = configured if not args.methods else [m.strip() for m in args.methods.split(",") if m.strip()]
    unknown = [method for method in methods if method not in configured]
    if unknown:
        raise ValueError(f"unknown methods {unknown}; configured methods: {configured}")
    return methods


def main():
    args = parse_args()
    settings = protocol()
    output_dir = Path(args.output_dir).expanduser()
    if not output_dir.is_absolute():
        raise ValueError("--output-dir must be an absolute CARC path, preferably under /project2 or /scratch")
    output_dir.mkdir(parents=True, exist_ok=True)
    log_dir = output_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    gpu_type, constraint = GPU_CHOICES[args.gpu]

    if args.action == "group":
        methods = validate_group(args, settings)
        method_text = ",".join(methods)
        job_name = f"mfr-o{args.order}-s{args.seed}"
        worker_args = [
            "group", str(ROOT), str(output_dir), args.conda_env, gpu_type,
            str(args.order), str(args.seed), method_text,
        ]
    else:
        job_name = "mfr-cache"
        worker_args = ["cache", str(ROOT), str(output_dir), args.conda_env, gpu_type]

    command = [
        "sbatch",
        f"--job-name={job_name}",
        "--partition=gpu",
        "--nodes=1",
        "--ntasks=1",
        "--cpus-per-task=8",
        "--mem=64G",
        f"--time={args.time}",
        f"--gpus-per-task={gpu_type}:1",
        f"--output={log_dir}/%x_%j.out",
    ]
    if constraint:
        command.append(f"--constraint={constraint}")
    if args.account:
        command.append(f"--account={args.account}")
    command.extend([str(WORKER), *worker_args])

    print("Submission command:")
    print(shlex.join(command))
    if args.dry_run:
        return
    if shutil.which("sbatch") is None:
        raise RuntimeError("sbatch was not found; run this command on a USC CARC login node")
    subprocess.run(command, check=True, text=True)
    print(f"Log pattern: {log_dir}/{job_name}_<job-id>.out")


if __name__ == "__main__":
    main()
