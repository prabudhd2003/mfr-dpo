"""Seeding and run bookkeeping. Import this in every notebook that trains anything.

Seeding rules:
  1. seed_everything(SEED) BEFORE load_model, so a fresh LoRA adapter always starts the same way.
  2. seed_everything(stage_seed(SEED, stage)) at the start of every stage, so a stage does not depend
     on what ran before it in the same session (resuming reproduces an uninterrupted run).
  3. Anything random in our own code uses an explicit generator seeded from those numbers.
"""

import os
import random
import subprocess


def seed_everything(seed):
    """Seed python, numpy and torch (CPU + GPU)."""
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
    """Seed for one stage of a run: seed 0 gives 1, 2, 3; seed 1 gives 1001, 1002, 1003."""
    return int(seed) * 1000 + int(stage)


def buffer_seed(seed, dataset):
    """Seed for choosing which pairs of `dataset` go into the memory buffer.

    Depends only on the run seed and the dataset, never on the replay method,
    so every method stores exactly the same pairs (that is what makes the comparison paired).
    """
    codes = {"helpful": 1, "safe": 2, "quality": 3}
    return int(seed) * 100 + codes.get(dataset, 9)


def run_info(repo_dir="."):
    """Everything needed to reproduce a run later: code version, GPU, package versions."""
    info = {"git_commit": _git_commit(repo_dir), "gpu": _gpu_name(), "packages": _versions()}
    try:
        import datetime
        info["started"] = datetime.datetime.now().isoformat(timespec="seconds")
    except Exception:
        pass
    return info


def _git_commit(repo_dir):
    try:
        out = subprocess.run(["git", "-C", repo_dir, "rev-parse", "--short", "HEAD"],
                             capture_output=True, text=True, timeout=15)
        return out.stdout.strip() or "unknown"
    except Exception:
        return "unknown"


def _gpu_name():
    try:
        import torch
        return torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
    except Exception:
        return "unknown"


def _versions():
    import importlib.metadata as meta
    import sys
    out = {"python": sys.version.split()[0]}
    for package in ("torch", "transformers", "peft", "bitsandbytes", "accelerate", "datasets"):
        try:
            out[package] = meta.version(package)
        except Exception:
            out[package] = "not installed"
    return out
