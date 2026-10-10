#!/usr/bin/env python3
"""Submit evaluation jobs to CARC. Run-level actions accept many --run-dir values per job."""

from __future__ import annotations

import argparse
import shlex
import shutil
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
WORKER = ROOT / "scripts" / "carc_eval_job.sh"
GPU_CHOICES = {"l40s": ("l40s", None), "a100-any": ("a100", None)}
TRAIN_ENV = ROOT / ".conda" / "envs" / "mfr-dpo"
EVAL_ENV = ROOT / ".conda" / "envs" / "mfr-eval"  # vLLM + lm-eval (docs/RUNBOOK.md step 0b)
TRAIN_ACTIONS = {"gradient-conflict", "locked-test"}


def add_common(parser, default_time):
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--account")
    parser.add_argument("--conda-env", help="Default: mfr-dpo for locked-test and gradient-conflict, "
                                            "mfr-eval for everything else")
    parser.add_argument("--gpu", choices=GPU_CHOICES, default="l40s")
    parser.add_argument("--time", default=default_time)
    parser.add_argument("--dry-run", action="store_true")


def parse_args():
    parser = argparse.ArgumentParser()
    actions = parser.add_subparsers(dest="action", required=True)
    qwen = str(ROOT / "configs" / "experiment_protocol.json")

    base = actions.add_parser("vllm-base", help="Dequantize the training base for vLLM (once per model)")
    add_common(base, "01:00:00")
    base.add_argument("--protocol", default=qwen)

    parity = actions.add_parser("parity", help="HF vs vLLM greedy parity on validation prompts")
    add_common(parity, "01:00:00")
    parity.add_argument("--run-dir", required=True)
    parity.add_argument("--protocol", default=qwen)

    generate = actions.add_parser("generate", help="Generate responses (final or post-stage)")
    add_common(generate, "04:00:00")
    target = generate.add_mutually_exclusive_group(required=True)
    target.add_argument("--run-dir", nargs="+")
    target.add_argument("--base-model", action="store_true")
    generate.add_argument("--protocol", default=qwen)
    generate.add_argument("--checkpoint", choices=("final", "post_stage"), default="final")
    generate.add_argument("--behavior", choices=("helpful", "safe", "quality"))
    generate.add_argument("--suites", default="test,xstest,harmbench")
    generate.add_argument("--max-prompts", type=int)
    generate.add_argument("--confirm-final-evaluation", action="store_true", required=True)

    safety = actions.add_parser("safety", help="WildGuard grading of generated responses")
    add_common(safety, "04:00:00")
    safety.add_argument("--run-dir", nargs="+", required=True)
    safety.add_argument("--checkpoint", choices=("final", "post_stage"), default="final")
    safety.add_argument("--behavior", default="safe")

    ifeval = actions.add_parser("ifeval")
    add_common(ifeval, "06:00:00")
    ifeval.add_argument("--run-dir", nargs="+", required=True)
    ifeval.add_argument("--confirm-final-evaluation", action="store_true", required=True)

    judge = actions.add_parser("judge", help="Prometheus pairwise judging")
    add_common(judge, "04:00:00")
    judge.add_argument("--candidate-run", nargs="+")
    judge.add_argument("--baseline-run", nargs="+")
    judge.add_argument("--within-run", nargs="+", help="final vs post-stage of each run")
    judge.add_argument("--behaviors", default="helpful,quality")

    validate = actions.add_parser("validate-judge", help="Judge agreement with human labels (val)")
    add_common(validate, "01:00:00")

    conflict = actions.add_parser("gradient-conflict", help="Behavior gradient cosines (A3)")
    add_common(conflict, "01:00:00")
    conflict.add_argument("--run-dir", required=True)
    conflict.add_argument("--stage", type=int, required=True)
    conflict.add_argument("--pairs", type=int, default=64)
    conflict.add_argument("--protocol", default=qwen)

    locked = actions.add_parser("locked-test", help="Locked preference test at every stage")
    add_common(locked, "02:00:00")
    locked.add_argument("--run-dir", required=True)
    locked.add_argument("--protocol", default=qwen)
    locked.add_argument("--confirm-final-evaluation", action="store_true", required=True)
    return parser.parse_args()


def absolute_existing(path, description):
    value = Path(path).expanduser().resolve()
    if not value.exists():
        raise FileNotFoundError(f"{description} does not exist: {value}")
    return value


def runs(paths, description="run directory"):
    resolved = [str(absolute_existing(path, description)) for path in paths]
    name = Path(resolved[0]).name + (f"+{len(resolved) - 1}" if len(resolved) > 1 else "")
    return resolved, name


def main():
    args = parse_args()
    output_dir = Path(args.output_dir).expanduser().resolve()
    if not output_dir.is_absolute():
        raise ValueError("--output-dir must be absolute")
    log_dir = output_dir / "logs" / "evaluation"
    log_dir.mkdir(parents=True, exist_ok=True)
    gpu_type, constraint = GPU_CHOICES[args.gpu]

    if args.action == "vllm-base":
        forwarded = ["--protocol", str(absolute_existing(args.protocol, "protocol"))]
        name = Path(args.protocol).stem
    elif args.action == "parity":
        run = absolute_existing(args.run_dir, "run directory")
        forwarded = ["--run-dir", str(run), "--protocol", str(absolute_existing(args.protocol, "protocol"))]
        name = run.name
    elif args.action == "generate":
        if args.base_model:
            forwarded = ["--base-model", "--output-dir", str(output_dir)]
            name = "base"
        else:
            paths, name = runs(args.run_dir)
            forwarded = ["--run-dir", *paths]
        forwarded += ["--protocol", str(absolute_existing(args.protocol, "protocol")),
                      "--checkpoint", args.checkpoint, "--suites", args.suites]
        if args.behavior:
            forwarded += ["--behavior", args.behavior]
            name += f"-{args.behavior}"
        if args.max_prompts is not None:
            forwarded += ["--max-prompts", str(args.max_prompts)]
    elif args.action == "safety":
        paths, name = runs(args.run_dir)
        forwarded = ["--run-dir", *paths, "--checkpoint", args.checkpoint, "--behavior", args.behavior]
    elif args.action == "ifeval":
        paths, name = runs(args.run_dir)
        forwarded = ["--run-dir", *paths]
    elif args.action == "judge":
        if args.within_run:
            paths, name = runs(args.within_run)
            forwarded, name = ["--within-run", *paths], f"within-{name}"
        else:
            if not (args.candidate_run and args.baseline_run):
                raise ValueError("judge needs --within-run or both --candidate-run and --baseline-run")
            candidates, name = runs(args.candidate_run, "candidate run")
            baselines, _ = runs(args.baseline_run, "baseline run")
            if len(candidates) != len(baselines):
                raise ValueError("--candidate-run and --baseline-run need the same number of runs")
            forwarded = ["--candidate-run", *candidates, "--baseline-run", *baselines]
        forwarded += ["--behaviors", args.behaviors]
    elif args.action == "validate-judge":
        forwarded, name = [], "judge"
    elif args.action == "gradient-conflict":
        run = absolute_existing(args.run_dir, "run directory")
        forwarded = ["--run-dir", str(run), "--stage", str(args.stage), "--pairs", str(args.pairs),
                     "--protocol", str(absolute_existing(args.protocol, "protocol"))]
        name = f"{run.name}-s{args.stage}"
    else:  # locked-test
        run = absolute_existing(args.run_dir, "run directory")
        forwarded = ["--run-dir", str(run), "--protocol", str(absolute_existing(args.protocol, "protocol"))]
        name = run.name
    conda_env = args.conda_env or str(TRAIN_ENV if args.action in TRAIN_ACTIONS else EVAL_ENV)
    worker_args = [args.action, str(ROOT), str(output_dir), conda_env, gpu_type, *forwarded]
    job_name = f"eval-{args.action}-{name}"[:100]

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
