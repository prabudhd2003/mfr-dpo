#!/usr/bin/env python3
"""Submit pinned Llama-3.2-3B replication jobs without changing the Qwen artifact tree."""

from __future__ import annotations

import argparse
import json
import shlex
import shutil
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
WORKER = ROOT / "scripts" / "carc_second_model_job.sh"
PROTOCOL = ROOT / "configs" / "second_model_protocol.json"
GPU_CHOICES = {"l40s": ("l40s", None), "a100-any": ("a100", None)}


def add_common(parser, default_time):
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--account")
    parser.add_argument("--conda-env", default=str(ROOT / ".conda" / "envs" / "mfr-dpo"))
    parser.add_argument("--gpu", choices=GPU_CHOICES, default="l40s")
    parser.add_argument("--time", default=default_time)
    parser.add_argument("--dry-run", action="store_true")


def parse_args():
    parser = argparse.ArgumentParser()
    actions = parser.add_subparsers(dest="action", required=True)
    cache = actions.add_parser("cache")
    add_common(cache, "03:00:00")
    group = actions.add_parser("group")
    add_common(group, "04:00:00")
    group.add_argument("--order", type=int, required=True)
    group.add_argument("--seed", type=int, required=True)
    group.add_argument(
        "--methods", required=True,
        help="Explicit comma-separated confirmation methods; include none on a new order/seed",
    )
    joint = actions.add_parser("joint", help="One seed of the Llama joint-training reference")
    add_common(joint, "04:00:00")
    joint.add_argument("--seed", type=int, required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    settings = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    output_dir = Path(args.output_dir).expanduser().resolve()
    if not output_dir.is_absolute():
        raise ValueError("--output-dir must be an absolute, separate second-model artifact path")
    output_dir.mkdir(parents=True, exist_ok=True)
    log_dir = output_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    gpu_type, constraint = GPU_CHOICES[args.gpu]
    base_args = [str(ROOT), str(output_dir), args.conda_env, gpu_type, str(PROTOCOL)]

    if args.action == "cache":
        job_name = "mfr2-cache"
        worker_args = ["cache", *base_args]
    elif args.action == "joint":
        if args.seed not in settings["joint_seeds"]:
            raise ValueError(f"unknown joint seed {args.seed}; configured: {settings['joint_seeds']}")
        job_name = f"mfr2-joint-s{args.seed}"
        worker_args = ["joint", *base_args, str(args.seed)]
    else:
        if str(args.order) not in settings["orders"]:
            raise ValueError(f"unknown order {args.order}")
        methods = [item.strip() for item in args.methods.split(",") if item.strip()]
        configured = settings["methods"] + settings["secondary_methods"]
        if unknown := [item for item in methods if item not in configured]:
            raise ValueError(f"unknown methods {unknown}; configured methods: {configured}")
        if args.seed not in settings["seeds"]:
            raise ValueError(f"unknown seed {args.seed}; transfer seeds: {settings['seeds']}")
        if transfer := settings.get("transfer_methods"):
            if outside := [item for item in methods if item not in transfer]:
                raise ValueError(f"{outside} are not in the frozen transfer grid {transfer}")
        job_name = f"mfr2-o{args.order}-s{args.seed}"
        worker_args = ["group", *base_args, str(args.order), str(args.seed), ",".join(methods)]

    command = [
        "sbatch", f"--job-name={job_name}", "--partition=gpu", "--nodes=1", "--ntasks=1",
        "--cpus-per-task=8", "--mem=64G", f"--time={args.time}",
        f"--gpus-per-task={gpu_type}:1", f"--output={log_dir}/%x_%j.out",
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
        raise RuntimeError("sbatch was not found; run this on a CARC login node")
    result = subprocess.run(command, check=True, text=True, capture_output=True)
    print(result.stdout.strip())
    print(f"Log pattern: {log_dir}/{job_name}_<job-id>.out")


if __name__ == "__main__":
    main()
