"""Seeding, frozen-protocol loading, hashing, and run bookkeeping."""

import hashlib
import importlib.metadata as meta
import json
import os
from pathlib import Path
import random
import subprocess
import tempfile


def seed_everything(seed):
    """Seed Python, NumPy and Torch without forcing slow deterministic kernels."""
    seed = int(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass
    return seed


def stage_seed(seed, stage):
    """Stable seed for one stage of one run."""
    return int(seed) * 1000 + int(stage)


def buffer_seed(seed, dataset):
    """Stable method-independent seed for one behavior's buffer candidates."""
    codes = {"helpful": 1, "safe": 2, "quality": 3}
    return int(seed) * 100 + codes.get(dataset, 9)


def load_protocol(path="configs/experiment_protocol.json"):
    """Load and minimally validate the authoritative experiment configuration."""
    with open(path) as stream:
        protocol = json.load(stream)
    required = {"protocol_version", "data_version", "model_name", "model_revision", "orders",
                "lora_r", "lora_alpha", "lora_dropout", "epochs", "learning_rate", "beta",
                "new_per_step", "old_per_step", "buffer_size", "refreshes"}
    missing = required - set(protocol)
    if missing:
        raise ValueError(f"protocol is missing {sorted(missing)}")
    replay_share = protocol["old_per_step"] / (protocol["new_per_step"] + protocol["old_per_step"])
    if abs(replay_share - 0.10) > 1e-12:
        raise ValueError(f"protocol replay share is {replay_share:.3%}, expected 10%")
    if protocol["epochs"] != 1:
        raise ValueError("this experiment runner supports exactly one epoch; set epochs to 1")
    if protocol["lora_alpha"] <= 0 or not 0 <= protocol["lora_dropout"] < 1:
        raise ValueError("invalid LoRA alpha/dropout in protocol")
    return protocol


def file_sha256(path, chunk_size=1024 * 1024):
    """SHA-256 digest of a local file."""
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def save_json_atomic(value, path):
    """Write JSON through a temporary file in the destination directory."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w") as stream:
            json.dump(value, stream, indent=2, sort_keys=True)
            stream.write("\n")
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def validate_stage1_source(settings, source_settings):
    """Reject a borrowed stage-1 checkpoint produced by an incompatible run."""
    keys = ("protocol_version", "data_version", "data_manifest_sha256", "model_name", "model_revision",
            "order_id", "seed", "lr", "beta", "new_per_step", "old_per_step", "max_tokens",
            "lora_r", "lora_alpha", "lora_dropout", "epochs", "git_commit")
    mismatches = {key: (settings.get(key), source_settings.get(key))
                  for key in keys if settings.get(key) != source_settings.get(key)}
    if mismatches:
        detail = ", ".join(f"{key}: {left!r} != {right!r}"
                           for key, (left, right) in mismatches.items())
        raise ValueError(f"incompatible stage-1 source: {detail}")
    return True


def validate_resume_settings(settings, saved):
    """Reject a resume that would combine different code or scientific settings in one run."""
    keys = ("run_name", "order_id", "order", "method", "seed", "protocol_version", "data_version",
            "data_manifest_sha256", "model_name", "model_revision", "lr", "beta", "new_per_step",
            "old_per_step", "max_tokens", "buffer_size", "refreshes", "lora_r", "lora_alpha",
            "lora_dropout", "epochs", "stage1_run_name", "git_commit")
    mismatches = {key: (settings.get(key), saved.get(key))
                  for key in keys if settings.get(key) != saved.get(key)}
    if mismatches:
        detail = ", ".join(f"{key}: {left!r} != {right!r}"
                           for key, (left, right) in mismatches.items())
        raise ValueError(f"incompatible resume: {detail}")
    return True


def mark_run_complete(run_dir, expected_paths):
    """Create a completion marker only after all required run files exist."""
    missing = [str(path) for path in expected_paths if not Path(path).exists()]
    if missing:
        raise FileNotFoundError(f"cannot complete run; missing {missing}")
    save_json_atomic({"complete": True, "files": [str(Path(path).name) for path in expected_paths]},
                     Path(run_dir) / "COMPLETE.json")


def run_info(repo_dir="."):
    """Machine, package, and repository provenance for a run."""
    info = {
        "git_commit": _git_commit(repo_dir, short=False),
        "git_dirty": _git_dirty(repo_dir),
        "gpu": _gpu_name(),
        "packages": _versions(),
        "deterministic_gpu_kernels": False,
    }
    try:
        import datetime
        info["started"] = datetime.datetime.now().astimezone().isoformat(timespec="seconds")
    except Exception:
        pass
    return info


def _git_commit(repo_dir, short=True):
    try:
        command = ["git", "-C", str(repo_dir), "rev-parse"]
        if short:
            command.append("--short")
        command.append("HEAD")
        result = subprocess.run(command,
                                capture_output=True, text=True, timeout=15)
        return result.stdout.strip() or "unknown"
    except Exception:
        return "unknown"


def _git_dirty(repo_dir):
    try:
        result = subprocess.run(["git", "-C", str(repo_dir), "status", "--porcelain"],
                                capture_output=True, text=True, timeout=15)
        return bool(result.stdout.strip())
    except Exception:
        return None


def _gpu_name():
    try:
        import torch
        return torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
    except Exception:
        return "unknown"


def _versions():
    import sys
    out = {"python": sys.version.split()[0]}
    for package in ("torch", "transformers", "peft", "bitsandbytes", "accelerate", "datasets",
                    "pandas", "numpy"):
        try:
            out[package] = meta.version(package)
        except Exception:
            out[package] = "not installed"
    return out
