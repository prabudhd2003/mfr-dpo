# Joint training reference

This document explains how to build and run the **joint-training reference** on the `project` branch. None of this
code exists yet. Follow the steps below exactly.

## What it is and why we need it

Every method in the main grid learns Helpful, Safe, and Quality **one after another**. Joint training removes the
sequence: one fresh adapter sees all 6,000 frozen training pairs (2,000 per behavior), shuffled together once, in a
single pass with the same QLoRA and DPO settings. Because nothing is learned before or after anything else, there is
nothing to forget.

Joint training is a **reference**, not a competing method. A continual learner cannot see old and new data together,
so joint training is not a fair opponent. Its job is to answer one question: *how much of the damage caused by
sequential training does each replay method repair?* We report this as **share of the gap closed**:

```text
share of gap closed = (final average of method − final average of No replay)
                      / (final average of Joint − final average of No replay)
```

Each order–seed cell is paired with the joint run that has the same seed, because joint training has no order.
A value of 0% means no better than No replay, and 100% means as good as training on everything at once. Continual-learning
work commonly reports this kind of joint or multitask reference. Ibrahim et al. (2024) use re-training on all data as
the target that replay-based continual pre-training tries to match.

## Exact definition

| Setting | Joint training | Why it matches the sequential grid |
|---|---|---|
| Training data | `helpful_train`, `safe_train`, `quality_train` from `data/v2` (6,000 pairs) | Same frozen files and manifest |
| Shuffle | One global shuffle with `random_state = stage_seed(seed, 1)` | Same seed rule as Stage 1 |
| Pairs per step | 18 (`new_per_step`) | No replay uses 18 new pairs per step |
| Optimizer steps | `ceil(6000 / 18) = 334` | No replay uses `3 × ceil(2000 / 18) = 336` |
| Learning rate | `1e-4`, 5% linear warm-up, then linear decay over all 334 steps | Same value and schedule shape |
| DPO β, LoRA r/α/dropout, max tokens, micro-batch | From `configs/experiment_protocol.json` | Identical |
| Training function | `mfr_dpo.train_stage_replay(..., method="none", buffer=None)` | The exact code path No replay uses for each stage |
| Reference model | Frozen base through the existing reference cache | Identical |
| Evaluation | The three 200-pair validation sets with `mfr_dpo.score_pairs` | Identical scorer and `results.csv` format |
| Seeds | 0, 1, 2 | Same seeds as the grid |

There is one difference that cannot be removed. Sequential runs restart the learning-rate schedule at each stage, and
joint training has one schedule. Mention this in the report.

## Files to add or change

### 1. `scripts/run_joint.py` (new)

Create this file exactly as written below.

```python
#!/usr/bin/env python3
"""Train the order-independent joint-training reference for one seed."""

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import mfr_cache
import mfr_data
import mfr_dpo
from mfr_utils import (file_sha256, load_protocol, mark_run_complete, run_info,
                       save_json_atomic, seed_everything, stage_seed)

DATASETS = ("helpful", "safe", "quality")


def joint_train_frame(splits):
    """Every frozen training pair in one frame, before the seeded shuffle."""
    joint = pd.concat([splits[name]["train"] for name in DATASETS], ignore_index=True)
    if joint["id"].duplicated().any():
        raise ValueError("training pair IDs repeat across datasets")
    return joint


def score_validation(model, tokenizer, splits, stage, trained_on, meta, folder, protocol, cache):
    folder.mkdir(parents=True, exist_ok=True)
    rows = []
    for name in DATASETS:
        val = splits[name]["val"]
        scores = mfr_dpo.score_pairs(
            model, tokenizer, val, beta=protocol["beta"], max_tokens=protocol["max_tokens"],
            reference_cache=cache, desc=f"{trained_on}: {name} validation",
        )
        scores.insert(0, "id", val["id"].values)
        scores.to_csv(folder / f"margins_{name}_val.csv", index=False)
        rows.append({**meta, "stage": stage, "trained_on": trained_on, "eval_set": name,
                     **mfr_dpo.summarize(scores)})
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True, help="Artifact root; writes OUTPUT_DIR/joint_runs")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--reference-cache", required=True)
    args = parser.parse_args()

    protocol = load_protocol(ROOT / "configs" / "experiment_protocol.json")
    if args.seed not in protocol["seeds"]:
        raise ValueError(f"seed {args.seed} is not in the protocol seeds {protocol['seeds']}")
    data_dir = ROOT / protocol["data_dir"]
    manifest_hash = file_sha256(data_dir / "manifest.json")
    splits = mfr_data.load_splits(data_dir)
    cache = mfr_cache.load_reference_cache(args.reference_cache, {
        "protocol_version": protocol["protocol_version"], "data_version": protocol["data_version"],
        "model_revision": protocol["model_revision"], "data_manifest_sha256": manifest_hash,
    }).set_index("id", drop=False)

    run_name = f"{protocol['data_version']}_joint_s{args.seed}"
    run_dir = Path(args.output_dir).expanduser().resolve() / "joint_runs" / run_name
    if (run_dir / "COMPLETE.json").exists():
        raise RuntimeError(f"joint run is already complete: {run_dir}")
    if run_dir.exists() and any(run_dir.iterdir()):
        raise RuntimeError(f"partial joint run found; rename it to {run_dir}_failed and resubmit")
    info = run_info(ROOT)
    if info.get("git_dirty"):
        raise RuntimeError("refusing to start a scientific run from an uncommitted Git checkout")

    train = joint_train_frame(splits)
    missing = set(train["id"]) - set(cache.index)
    if missing:
        raise ValueError(f"{len(missing)} training pairs are missing from the reference cache")

    adapter_dir = run_dir / "joint"
    adapter_dir.mkdir(parents=True)
    settings = {
        "run_name": run_name, "method": "joint", "order_id": 0, "order": list(DATASETS),
        "seed": args.seed, "protocol_version": protocol["protocol_version"],
        "data_version": protocol["data_version"], "data_manifest_sha256": manifest_hash,
        "model_name": protocol["model_name"], "model_revision": protocol["model_revision"],
        "lr": protocol["learning_rate"], "beta": protocol["beta"],
        "new_per_step": protocol["new_per_step"], "old_per_step": 0,
        "max_tokens": protocol["max_tokens"], "lora_r": protocol["lora_r"],
        "lora_alpha": protocol["lora_alpha"], "lora_dropout": protocol["lora_dropout"],
        "epochs": protocol["epochs"], "train_pairs": len(train),
        "shuffle_seed": stage_seed(args.seed, 1),
        "joint_script_sha256": file_sha256(Path(__file__)), **info,
    }
    save_json_atomic(settings, run_dir / "settings.json")

    seed_everything(args.seed)
    model, tokenizer = mfr_dpo.load_model(
        protocol["model_name"], protocol["lora_r"], revision=protocol["model_revision"],
        lora_alpha=protocol["lora_alpha"], lora_dropout=protocol["lora_dropout"],
    )
    settings["reference_cache_canary"] = mfr_cache.verify_reference_cache(
        model, tokenizer, splits["helpful"]["val"], cache,
        beta=protocol["beta"], max_tokens=protocol["max_tokens"],
    )
    save_json_atomic(settings, run_dir / "settings.json")

    meta = {"run_name": run_name, "order_id": 0, "method": "joint", "seed": args.seed,
            "protocol_version": protocol["protocol_version"], "data_version": protocol["data_version"]}
    results = score_validation(model, tokenizer, splits, 0, "base", meta,
                               run_dir / "stage0_base", protocol, cache)
    history, _ = mfr_dpo.train_stage_replay(
        model, tokenizer, train, buffer=None, method="none", beta=protocol["beta"],
        lr=protocol["learning_rate"], new_per_step=protocol["new_per_step"], old_per_step=0,
        refreshes=protocol["refreshes"], micro_batch=protocol["micro_batch"],
        max_tokens=protocol["max_tokens"], seed=stage_seed(args.seed, 1),
        desc="joint training", progress_path=adapter_dir / "history.csv", reference_cache=cache,
    )
    model.save_pretrained(adapter_dir)
    history.to_csv(adapter_dir / "history.csv", index=False)
    results = pd.concat([results, score_validation(model, tokenizer, splits, 1, "joint", meta,
                                                   adapter_dir, protocol, cache)], ignore_index=True)
    results.to_csv(run_dir / "results.csv", index=False)
    mark_run_complete(run_dir, [run_dir / "settings.json", run_dir / "results.csv",
                                adapter_dir / "history.csv", adapter_dir / "adapter_config.json"])
    print(results.to_string(index=False))
    print(f"Complete: {run_dir}")


if __name__ == "__main__":
    main()
```

Joint runs are written to `artifacts/joint_runs/`, **not** `artifacts/runs/`. Notebook 07 reads every folder in
`runs/` as an order–seed–method cell, and a joint run has no order.

The script does not change `SCIENTIFIC_CODE_PATHS`. It records its own hash as `joint_script_sha256`, together with
the normal `scientific_code_sha256`.

### 2. `scripts/carc_job.sh`: add a `joint` action

Add this case before the `*)` case:

```bash
  joint)
    SEED="$1"
    "$PYTHON_BIN" -u scripts/run_joint.py \
      --output-dir "$OUTPUT_DIR" \
      --seed "$SEED" \
      --reference-cache "$CACHE_PATH"
    ;;
```

### 3. `scripts/submit_carc.py`: add a `joint` subcommand

In `parse_args()`, after the `group` subparser:

```python
    joint = subparsers.add_parser("joint", help="Train the joint-training reference for one seed")
    add_common(joint, "02:00:00")
    joint.add_argument("--seed", type=int, required=True)
```

In `main()`, replace the `if args.action == "group": ... else: ...` block with:

```python
    if args.action == "group":
        methods = validate_group(args, settings)
        job_name = f"mfr-o{args.order}-s{args.seed}"
        worker_args = ["group", str(ROOT), str(output_dir), args.conda_env, gpu_type,
                       str(args.order), str(args.seed), ",".join(methods)]
    elif args.action == "joint":
        if args.seed not in settings.get("seeds", []):
            raise ValueError(f"unknown seed {args.seed}; configured seeds: {settings.get('seeds', [])}")
        job_name = f"mfr-joint-s{args.seed}"
        worker_args = ["joint", str(ROOT), str(output_dir), args.conda_env, gpu_type, str(args.seed)]
    else:
        job_name = "mfr-cache"
        worker_args = ["cache", str(ROOT), str(output_dir), args.conda_env, gpu_type]
```

### 4. `scripts/final_test.py`: accept joint runs

Replace the two lines that pick the adapter:

```python
    final_dataset = settings["order"][-1]
    adapter = run_dir / f"stage3_{final_dataset}"
```

with:

```python
    if settings.get("method") == "joint":
        adapter = run_dir / "joint"
    else:
        adapter = run_dir / f"stage3_{settings['order'][-1]}"
```

### 5. `tests/test_joint.py` (new, CPU only)

Load `scripts/run_joint.py` with `importlib.util.spec_from_file_location` and test `joint_train_frame` on a small fake
`splits` dictionary:

- the result has every row from all three datasets (for example 3 × 4 = 12 rows) and no duplicate IDs;
- duplicated IDs across datasets raise `ValueError`;
- `df.sample(frac=1, random_state=stage_seed(s, 1))` gives the same order for the same seed and a different order
  for a different seed.

Add one test in `tests/test_carc.py` showing that `submit_carc.py joint --seed 0 --dry-run` builds a command whose
worker arguments start with `joint` and end with `0`.

### 6. Analysis: `src/mfr_analysis.py` and Notebook 07

Add a loader that reads `artifacts/joint_runs/*/results.csv` for folders that contain `COMPLETE.json`. Use only rows with
`stage == 1`. For each seed, compute the joint final average across Helpful, Safe, and Quality. Then add:

- a `Joint training (reference)` row to the final-score tables, drawn as a dashed horizontal line in final-average plots
  and never colored like a method;
- a `Share of gap closed` column next to the final three-behavior average, computed per cell against the joint run
  with the same seed and then averaged;
- per-behavior joint accuracy next to each behavior's accuracy immediately after it was learned. This shows whether
  learning everything together costs anything on its own.

Joint training has no "old behavior" and no "final task", so leave those columns blank for it.

## Checks before submitting

On CARC, in the course-project folder, after the every-session setup in [`PROJECT_FOLDER.md`](PROJECT_FOLDER.md):

```bash
python -m pytest -q
git status --short          # must print nothing
python scripts/submit_carc.py joint \
  --output-dir "$MFR_OUTPUT_DIR" --account xiangren_1987 --gpu l40s --seed 0 --dry-run
```

## Run it

```bash
for seed in 0 1; do
  python scripts/submit_carc.py joint \
    --output-dir "$MFR_OUTPUT_DIR" \
    --account xiangren_1987 \
    --gpu l40s \
    --time 02:00:00 \
    --seed "$seed"
done
```

Seed 2 is submitted the same way (`--seed 2`) once seed 2 has been added to the protocol (see
[`NEXT_STEPS.md`](NEXT_STEPS.md), Step 1). Each job takes about the same time as one full No-replay run, roughly
25 minutes of training plus scoring.

Monitor it like any other job:

```bash
squeue --me
tail -f "$MFR_OUTPUT_DIR/logs/mfr-joint-s0_JOBID.out"
find "$MFR_OUTPUT_DIR/joint_runs" -name COMPLETE.json
```

A finished run contains:

```text
joint_runs/v2_joint_s0/
├── settings.json
├── results.csv              # stage 0 = base model, stage 1 = after joint training
├── COMPLETE.json
├── stage0_base/margins_{helpful,safe,quality}_val.csv
└── joint/                   # adapter, history.csv, margins_{helpful,safe,quality}_val.csv
```

If a job fails, rename the partial folder (for example `mv v2_joint_s0 v2_joint_s0_failed`) and submit again. A
joint run is short, so it restarts from the beginning instead of resuming.

## How to read the result

- **Joint is clearly above every sequential method.** This is the expected case. Report the share of the gap
  each replay method closes.
- **A replay method matches or beats Joint on some behavior.** This is possible. Joint training mixes three
  objectives in every batch and gets a single learning-rate schedule. Report it as it is; do not call joint training an
  upper bound in the report. Call it a reference.
- **Joint is below No replay on the final average.** That points to a bug, such as the wrong split, a cache mismatch,
  or the adapter not being saved. Check `settings.json`, the canary, and `history.csv` before using the number.

## Reference

Ibrahim, A., Thérien, B., Gupta, K., Richter, M. L., Anthony, Q., Lesort, T., Belilovsky, E., and Rish, I. 2024.
*Simple and Scalable Strategies to Continually Pre-train Large Language Models.* arXiv:2403.08763.
https://arxiv.org/abs/2403.08763
