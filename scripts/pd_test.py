#!/usr/bin/env python3
"""Evaluate the frozen pid_i000_d100 final adapters on the locked v2 test split.

CPU commands: check, submit --dry-run, submit, summary. Only `run` loads a model,
and it requires a Slurm GPU allocation. Defaults cover orders 1-4 and seeds 0-1.
No training, configuration edits, or evaluations of other gain profiles occur.
The project protocol keeps retention a validation-only metric: this evaluates
only the final adapters, with 300 test pairs for each of three behaviors.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import csv
from datetime import datetime, timezone
import gc
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import tempfile


ROOT = Path(__file__).resolve().parents[1]
METHOD = "pid_i000_d100"
GAINS = {"kp": 1.0, "ki": 0.0, "kd": 1.0}
TOPICS = ("helpful", "safe", "quality")
SCORING_KEYS = ("protocol_version", "data_version", "model_name", "model_revision",
                "beta", "max_tokens", "lora_r", "lora_alpha", "lora_dropout")
MARGIN_COLUMNS = ("margin", "margin_sum", "policy_margin", "policy_margin_sum")
DEFAULT_ENV = "/project2/xiangren_1987/group26/.conda/envs/mfr-dpo"
WORKER = r'''#!/bin/bash
set -euo pipefail
umask 002
REPO_DIR="$1"
OUTPUT_DIR="$2"
CONDA_ENV="$3"
ORDER="$4"
module purge
module load conda
eval "$(conda shell.bash hook)"
conda activate "$CONDA_ENV"
export HF_HOME="$OUTPUT_DIR/huggingface"
export HF_HUB_CACHE="$HF_HOME/hub"
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
cd "$REPO_DIR"
exec "$CONDA_ENV/bin/python" -u scripts/pd_test.py run --output-dir "$OUTPUT_DIR" --order "$ORDER"
'''


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def digest_object(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def atomic_text(path, text):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix="." + path.name, dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary, 0o660)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def atomic_json(path, value):
    atomic_text(path, json.dumps(value, indent=2, sort_keys=True) + "\n")


def atomic_csv(path, rows):
    import io
    stream = io.StringIO(newline="")
    fields = list(dict.fromkeys(key for row in rows for key in row))
    writer = csv.DictWriter(stream, fieldnames=fields)
    writer.writeheader()
    writer.writerows(rows)
    atomic_text(path, stream.getvalue())


def read_csv(path):
    with Path(path).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


@contextmanager
def lock(path):
    import fcntl
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError(f"Another process holds {path}; do not submit a duplicate job") from error
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def science_hash(repo):
    sys.path.insert(0, str(repo / "src"))
    from mfr_utils import scientific_code_sha256
    return scientific_code_sha256(repo)


def git_commit(repo):
    result = subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"],
                            capture_output=True, text=True, check=False)
    return result.stdout.strip() or "unknown"


def integer_list(value):
    result = sorted(set(int(part.strip()) for part in value.split(",")))
    if not result or min(result) < 0:
        raise ValueError("orders/seeds must be nonempty comma-separated nonnegative integers")
    return result


def make_plan(repo, output, orders, seeds, batch_size):
    protocol_path = repo / "configs" / "experiment_protocol.json"
    protocol = read_json(protocol_path)
    if protocol.get("pid_replay_profiles", {}).get(METHOD) != GAINS:
        raise ValueError(f"Protocol must still define {METHOD} as {GAINS}")
    if batch_size < 1:
        raise ValueError("batch size must be positive")
    data = (repo / protocol["data_dir"]).resolve()
    manifest = read_json(data / "manifest.json")
    manifest_hash = sha(data / "manifest.json")
    current_code = science_hash(repo)
    tests = {}
    for topic in TOPICS:
        path = data / f"{topic}_test.jsonl"
        record = manifest["files"][path.name]
        with path.open("rb") as handle:
            count = sum(bool(line.strip()) for line in handle)
        if count != 300 or record["rows"] != 300 or sha(path) != record["sha256"]:
            raise ValueError(f"Frozen 300-pair test file failed manifest verification: {path}")
        tests[topic] = {"rows": count, "sha256": record["sha256"]}
    runs = []
    for order in orders:
        if str(order) not in protocol["orders"]:
            raise ValueError(f"Unknown order {order}")
        for seed in seeds:
            name = f"v2_o{order}_{METHOD}_s{seed}"
            folder = output / "runs" / name
            if not (folder / "COMPLETE.json").is_file():
                raise ValueError(f"Training run is not complete: {folder}")
            settings_path = folder / "settings.json"
            settings = read_json(settings_path)
            expected = {"run_name": name, "method": METHOD, "order_id": order, "seed": seed,
                        "order": protocol["orders"][str(order)], "pid_profile": GAINS,
                        "data_manifest_sha256": manifest_hash, "scientific_code_sha256": current_code,
                        **{key: protocol[key] for key in SCORING_KEYS}}
            mismatches = {key: (settings.get(key), value) for key, value in expected.items()
                          if settings.get(key) != value}
            if mismatches:
                raise ValueError(f"Training/scoring provenance mismatch in {name}: {mismatches}. "
                                 "Use the unchanged training/scoring code and protocol.")
            checkpoint = folder / f"stage3_{settings['order'][-1]}"
            config = checkpoint / "adapter_config.json"
            weights = [checkpoint / name for name in ("adapter_model.safetensors", "adapter_model.bin")
                       if (checkpoint / name).is_file()]
            if not config.is_file() or not weights or any(path.stat().st_size == 0 for path in weights):
                raise ValueError(f"Final saved adapter is missing or empty: {checkpoint}")
            runs.append({"run_name": name, "run_dir": str(folder), "checkpoint": str(checkpoint),
                         "order_id": order, "seed": seed, "order": settings["order"],
                         "settings_sha256": sha(settings_path),
                         "checkpoint_files": {path.name: sha(path) for path in [config, *weights]},
                         "training_git_commit": settings.get("git_commit", "unknown"),
                         **{key: settings[key] for key in SCORING_KEYS}})
    return {"schema": "pd-locked-final-test-v1", "method": METHOD, "gains": GAINS,
            "orders": orders, "seeds": seeds, "batch_size": batch_size, "data_dir": str(data),
            "data_manifest_sha256": manifest_hash, "test_files": tests,
            "scientific_code_sha256": current_code, "evaluator_sha256": sha(Path(__file__)),
            "runs": runs}


def plan_path(output):
    return output / "pd_test" / "PLAN.json"


def load_plan(output):
    envelope = read_json(plan_path(output))
    if envelope.get("plan_sha256") != digest_object(envelope["plan"]):
        raise ValueError("Locked test plan hash does not match its contents")
    return envelope["plan"], envelope["plan_sha256"]


def check_existing_plan(output, candidate):
    if plan_path(output).exists():
        saved, _ = load_plan(output)
        if saved != candidate:
            raise ValueError("An evaluation plan is already frozen and differs from this request; "
                             "do not change gains, inputs, evaluation code, or scope after testing begins")


def completed(run, plan_id):
    folder = Path(run["run_dir"]) / "final_test"
    marker = folder / "PD_TEST_COMPLETE.json"
    if not marker.exists():
        return False
    record = read_json(marker)
    if record.get("plan_sha256") != plan_id:
        raise ValueError(f"Completed test belongs to another plan: {folder}")
    expected = {f"{topic}_test_scores.csv" for topic in TOPICS} | {"test_results.csv", "pd_test_manifest.json"}
    if set(record.get("files", {})) != expected:
        raise ValueError(f"Incomplete result manifest: {marker}")
    for name, expected_hash in record["files"].items():
        if not (folder / name).is_file() or sha(folder / name) != expected_hash:
            raise ValueError(f"Completed test output changed or is missing: {folder / name}")
    return True


def check_legacy_results(plan, plan_id):
    for run in plan["runs"]:
        folder = Path(run["run_dir"]) / "final_test"
        if completed(run, plan_id):
            continue
        files = list(folder.glob("*_test_scores.csv")) + list(folder.glob("test_results.csv"))
        if files and not (folder / "pd_test_manifest.json").exists():
            raise ValueError(f"Existing test scores found without this launcher's provenance: {folder}. "
                             "Use/inspect the team's existing scores instead of rescoring or deleting them.")


def worker_command(repo, output, env, order):
    return [str(output / "pd_test" / "worker.sh"), str(repo), str(output), str(env), str(order)]


def sbatch_command(repo, output, env, account, time_limit, order, plan_id):
    name = f"pdtest-{plan_id[:6]}-o{order}"
    return ["sbatch", "--parsable", f"--job-name={name}", "--partition=gpu", "--nodes=1",
            "--ntasks=1", "--cpus-per-task=8", "--mem=64G", f"--time={time_limit}",
            "--gpus-per-task=l40s:1", f"--output={output}/logs/%x_%j.out", f"--account={account}",
            *worker_command(repo, output, env, order)]


def submit(args, plan, output):
    plan_id = digest_object(plan)
    pending = [order for order in plan["orders"]
               if not all(completed(run, plan_id) for run in plan["runs"] if run["order_id"] == order)]
    if args.dry_run:
        for order in pending:
            print(shlex.join(sbatch_command(ROOT, output, args.conda_env, args.account, args.time, order, plan_id)))
        print(f"DRY RUN: {len(pending)} job(s); no submission or test scoring; plan not written.")
        return
    if shutil.which("sbatch") is None or shutil.which("squeue") is None:
        raise RuntimeError("Submit from a CARC login node with sbatch and squeue available")
    env = Path(args.conda_env).expanduser()
    if not env.is_absolute() or not (env / "bin" / "python").is_file():
        raise ValueError("--conda-env must be an existing absolute environment prefix with bin/python")
    args.conda_env = str(env.resolve())
    output.joinpath("logs").mkdir(parents=True, exist_ok=True)
    with lock(output / "pd_test" / "submit.lock"):
        check_existing_plan(output, plan)
        if not plan_path(output).exists():
            atomic_json(plan_path(output), {"plan": plan, "plan_sha256": plan_id,
                        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
                        "evaluation_git_commit": git_commit(ROOT)})
        worker = output / "pd_test" / "worker.sh"
        if worker.exists() and worker.read_text() != WORKER:
            raise ValueError(f"Existing worker script differs: {worker}")
        if not worker.exists():
            atomic_text(worker, WORKER)
            worker.chmod(0o770)
        active = subprocess.run(["squeue", "--me", "--noheader", "--format=%j"],
                                check=True, capture_output=True, text=True).stdout.splitlines()
        for order in pending:
            command = sbatch_command(ROOT, output, args.conda_env, args.account, args.time, order, plan_id)
            name = next(item.split("=", 1)[1] for item in command if item.startswith("--job-name="))
            if name in {item.strip() for item in active}:
                print(f"Already queued/running: {name}; skipping", flush=True)
                continue
            result = subprocess.run(command, check=True, capture_output=True, text=True)
            job_id = result.stdout.strip().split(";", 1)[0]
            print(f"Submitted order {order}: {job_id}; log {output}/logs/{name}_{job_id}.out", flush=True)
            atomic_json(output / "pd_test" / f"submission_{job_id}.json",
                        {"job_id": job_id, "order": order, "plan_sha256": plan_id, "command": command})


def validate_score_rows(rows, expected_ids):
    ids = [str(row.get("id", "")) for row in rows]
    if len(ids) != len(expected_ids) or len(set(ids)) != len(ids) or set(ids) != set(expected_ids):
        raise ValueError("Test score IDs/count do not match the frozen test set")
    for row in rows:
        for column in MARGIN_COLUMNS:
            if not math.isfinite(float(row[column])):
                raise ValueError(f"Non-finite {column} for test pair {row['id']}")


def statistics(topic, rows):
    count = len(rows)
    if count == 0:
        raise ValueError("Cannot summarize an empty test set")
    correct = sum(float(row["margin"]) > 0 for row in rows)
    correct_sum = sum(float(row["margin_sum"]) > 0 for row in rows)
    return {"eval_set": topic, "n_pairs": count, "n_correct": correct,
            "n_correct_sum": correct_sum,
            "accuracy": 100 * correct / count, "accuracy_sum": 100 * correct_sum / count,
            "mean_margin": sum(float(row["margin"]) for row in rows) / count,
            "mean_margin_sum": sum(float(row["margin_sum"]) for row in rows) / count,
            "policy_accuracy": 100 * sum(float(row["policy_margin"]) > 0 for row in rows) / count,
            "policy_accuracy_sum": 100 * sum(float(row["policy_margin_sum"]) > 0 for row in rows) / count,
            "mean_policy_margin": sum(float(row["policy_margin"]) for row in rows) / count,
            "mean_policy_margin_sum": sum(float(row["policy_margin_sum"]) for row in rows) / count}


def score_run(run, plan, plan_id, splits, dpo, torch):
    if completed(run, plan_id):
        print(f"Skipping completed test: {run['run_name']}", flush=True)
        return
    folder = Path(run["run_dir"]) / "final_test"
    provenance = {"plan_sha256": plan_id, "run": run, "batch_size": plan["batch_size"],
                  "test_files": plan["test_files"], "primary_metric": "100 * mean(margin > 0)",
                  "reference": "same frozen base, evaluated with the adapter disabled"}
    metadata = folder / "pd_test_manifest.json"
    if metadata.exists():
        if read_json(metadata) != provenance:
            raise ValueError(f"Partial test has incompatible provenance: {folder}")
    else:
        if list(folder.glob("*.csv")):
            raise ValueError(f"Existing CSVs lack evaluation provenance: {folder}")
        atomic_json(metadata, provenance)
    model = tokenizer = None
    summaries = []
    try:
        for topic in TOPICS:
            df = splits[topic]["test"]
            expected = [str(value) for value in df["id"]]
            path = folder / f"{topic}_test_scores.csv"
            if path.exists():
                scores = read_csv(path)
                validate_score_rows(scores, expected)
                print(f"Reusing saved scores: {run['run_name']} / {topic}", flush=True)
            else:
                if model is None:
                    print(f"Loading final adapter: {run['checkpoint']}", flush=True)
                    model, tokenizer = dpo.load_model(
                        run["model_name"], run["lora_r"], adapter_path=run["checkpoint"],
                        revision=run["model_revision"], lora_alpha=run["lora_alpha"],
                        lora_dropout=run["lora_dropout"])
                    model.eval()
                with torch.inference_mode():
                    result = dpo.score_pairs(model, tokenizer, df, beta=run["beta"],
                                            max_tokens=run["max_tokens"], batch_size=plan["batch_size"],
                                            desc=f"{run['run_name']}: {topic} test")
                scores = [dict(id=pair_id, **record) for pair_id, record in zip(expected, result.to_dict("records"))]
                validate_score_rows(scores, expected)
                atomic_csv(path, scores)
            summaries.append(statistics(topic, scores))
        atomic_csv(folder / "test_results.csv", summaries)
        files = [f"{topic}_test_scores.csv" for topic in TOPICS] + ["test_results.csv", "pd_test_manifest.json"]
        atomic_json(folder / "PD_TEST_COMPLETE.json",
                    {"plan_sha256": plan_id, "run_name": run["run_name"],
                     "completed_at_utc": datetime.now(timezone.utc).isoformat(),
                     "evaluation_git_commit": git_commit(ROOT),
                     "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
                     "python": sys.version.split()[0],
                     "packages": {name: package_version(name) for name in
                                  ("torch", "transformers", "peft", "bitsandbytes", "pandas", "numpy")},
                     "files": {name: sha(folder / name) for name in files}})
        print(f"Completed test: {run['run_name']}\n" + json.dumps(summaries, indent=2), flush=True)
    finally:
        del model, tokenizer
        gc.collect()
        torch.cuda.empty_cache()


def package_version(name):
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "not installed"


def run_order(args, output):
    plan, plan_id = load_plan(output)
    if args.order not in plan["orders"]:
        raise ValueError(f"Order {args.order} is not in the frozen test plan")
    candidate = make_plan(ROOT, output, plan["orders"], plan["seeds"], plan["batch_size"])
    if candidate != plan:
        raise ValueError("Input checkpoints, data, settings or evaluator changed after the plan was frozen")
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("GPU evaluation must run inside the submitted Slurm job, not on a login node")
    with lock(output / "pd_test" / f"order_{args.order}.lock"):
        sys.path.insert(0, str(ROOT / "src"))
        import torch
        import mfr_data
        import mfr_dpo
        if not torch.cuda.is_available() or "L40S" not in torch.cuda.get_device_name(0).upper():
            raise RuntimeError("An allocated L40S CUDA GPU is required")
        print(f"GPU: {torch.cuda.get_device_name(0)}; Slurm job: {os.environ['SLURM_JOB_ID']}", flush=True)
        splits = mfr_data.load_splits(plan["data_dir"])
        for run in plan["runs"]:
            if run["order_id"] == args.order:
                score_run(run, plan, plan_id, splits, mfr_dpo, torch)
        print(f"Order {args.order}: all requested PD tests complete", flush=True)


def summarize(output):
    plan, plan_id = load_plan(output)
    cells, topics = [], []
    for run in plan["runs"]:
        if not completed(run, plan_id):
            raise ValueError(f"Test not yet complete: {run['run_name']}; inspect its Slurm log")
        records = read_csv(Path(run["run_dir"]) / "final_test" / "test_results.csv")
        if len(records) != 3 or {row["eval_set"] for row in records} != set(TOPICS):
            raise ValueError(f"Expected exactly three test-set summaries for {run['run_name']}")
        by_topic = {row["eval_set"]: row for row in records}
        for topic, row in by_topic.items():
            if int(row["n_pairs"]) != 300:
                raise ValueError(f"Wrong test denominator for {run['run_name']} / {topic}")
            topics.append({"run_name": run["run_name"], "order_id": run["order_id"],
                           "seed": run["seed"], "method": METHOD, **row})
        accuracy = {topic: float(by_topic[topic]["accuracy"]) for topic in TOPICS}
        cells.append({"run_name": run["run_name"], "order_id": run["order_id"], "seed": run["seed"],
                      "method": METHOD, **{f"{topic}_accuracy": accuracy[topic] for topic in TOPICS},
                      "final_average": sum(accuracy.values()) / 3,
                      "final_new_stage_score": accuracy[run["order"][-1]]})
    metrics = [f"{topic}_accuracy" for topic in TOPICS] + ["final_average", "final_new_stage_score"]
    summary = {"method": METHOD, "cells": len(cells), "kp": 1, "ki": 0, "kd": 1,
               **{key: sum(row[key] for row in cells) / len(cells) for key in metrics}}
    destination = output / "pd_test"
    with lock(destination / "summary.lock"):
        atomic_csv(destination / "pd_test_cells.csv", cells)
        atomic_csv(destination / "pd_test_summary.csv", [summary])
        atomic_csv(destination / "pd_test_by_topic.csv", topics)
    print(json.dumps(summary, indent=2))
    print(f"Saved test summaries in {destination}")
    print("Accuracy is the frozen-reference preference-margin metric; retention remains validation-only.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    for action in ("check", "submit", "run", "summary"):
        command = sub.add_parser(action)
        command.add_argument("--output-dir", required=True)
        if action in ("check", "submit"):
            command.add_argument("--orders", default="1,2,3,4")
            command.add_argument("--seeds", default="0,1")
            command.add_argument("--batch-size", type=int, default=4)
        if action == "submit":
            command.add_argument("--account", default="xiangren_1987")
            command.add_argument("--conda-env", default=DEFAULT_ENV)
            command.add_argument("--time", default="03:00:00")
            command.add_argument("--dry-run", action="store_true")
        if action == "run":
            command.add_argument("--order", type=int, required=True)
    args = parser.parse_args()
    os.umask(0o002)
    output = Path(args.output_dir).expanduser().resolve()
    try:
        if args.action in ("check", "submit"):
            plan = make_plan(ROOT, output, integer_list(args.orders), integer_list(args.seeds), args.batch_size)
            check_existing_plan(output, plan)
            check_legacy_results(plan, digest_object(plan))
            if args.action == "check":
                print(f"READY: {len(plan['runs'])} completed {METHOD} runs; gains={GAINS}")
                print(f"Orders={plan['orders']}; seeds={plan['seeds']}; 300 pairs per behavior; 900 per checkpoint")
                print("Checkpoint, settings, frozen data and scientific-code hashes verified. No model loaded.")
                for run in plan["runs"]:
                    print(run["run_name"], "DONE" if completed(run, digest_object(plan)) else "TO SCORE")
            else:
                submit(args, plan, output)
        elif args.action == "run":
            run_order(args, output)
        else:
            summarize(output)
    except (ValueError, RuntimeError, OSError, KeyError, subprocess.CalledProcessError) as error:
        parser.exit(2, f"ERROR: {error}\n")


if __name__ == "__main__":
    main()
