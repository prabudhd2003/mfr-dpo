# Running MFR-DPO on USC CARC

This is the complete terminal workflow for the project. Training uses unattended Slurm jobs with one NVIDIA L40S
GPU. Google Drive and Colab are not part of this workflow.

The examples use:

```text
Repository: /project2/xiangren_1987/grp26-mfr-dpo
CARC account: xiangren_1987
Branch: carc
Artifacts: /project2/xiangren_1987/grp26-mfr-dpo/artifacts
GPU: NVIDIA L40S
```

Do not include the shell prompt, such as `[user@discovery]$` or `bash-4.4$`, when copying a command.

## 1. Get the code

For a first clone into an empty project directory:

```bash
cd /project2/xiangren_1987
git clone --branch carc https://github.com/prabudhd2003/mfr-dpo.git grp26-mfr-dpo
cd /project2/xiangren_1987/grp26-mfr-dpo
```

For an existing clone:

```bash
cd /project2/xiangren_1987/grp26-mfr-dpo
git switch carc
git status --short
git pull --ff-only origin carc
```

If `git pull` says there are local notebook changes and those changes are only executed outputs that can be discarded:

```bash
git restore -- notebooks/07_compare_runs.ipynb
git pull --ff-only origin carc
```

Do not restore a file if it contains code changes that must be kept. Commit or copy those changes first.

## 2. Create the environment once

Environment creation and tests need a CPU allocation, not a GPU:

```bash
cd /project2/xiangren_1987/grp26-mfr-dpo
salloc --partition=main --ntasks=1 --cpus-per-task=4 --mem=16G --time=02:00:00
```

After the allocation starts, run:

```bash
module purge
module load conda

export MFR_CONDA_ROOT=/project2/xiangren_1987/grp26-mfr-dpo/.conda
export CONDA_PKGS_DIRS="$MFR_CONDA_ROOT/pkgs"
mkdir -p "$MFR_CONDA_ROOT/envs" "$CONDA_PKGS_DIRS"

mamba create --prefix "$MFR_CONDA_ROOT/envs/mfr-dpo" python=3.11 pip -y
eval "$(conda shell.bash hook)"
conda activate "$MFR_CONDA_ROOT/envs/mfr-dpo"

mamba install --prefix "$MFR_CONDA_ROOT/envs/mfr-dpo" \
  pytorch pytorch-cuda=11.8 "mkl<2024.1" "intel-openmp<2024.1" \
  -c pytorch -c nvidia -c defaults -y

python -m pip install -r requirements.txt
python -m pip install ipykernel
python -m pytest -q
exit
```

The MKL and Intel OpenMP limits prevent the `iJIT_NotifyEvent` PyTorch import error seen with the 2025 packages.

## 3. Start each CARC terminal session

Run this setup whenever opening a new CARC terminal:

```bash
cd /project2/xiangren_1987/grp26-mfr-dpo
module purge
module load conda
eval "$(conda shell.bash hook)"
conda activate /project2/xiangren_1987/grp26-mfr-dpo/.conda/envs/mfr-dpo

export MFR_OUTPUT_DIR=/project2/xiangren_1987/grp26-mfr-dpo/artifacts
mkdir -p "$MFR_OUTPUT_DIR"
```

`MFR_OUTPUT_DIR` is only a terminal variable containing the artifact path. It does not create a second copy of the
results. Jobs receive the expanded absolute path and continue after the terminal or laptop closes.

Optional checks:

```bash
which python
python -c "import torch; print(torch.__version__, torch.version.cuda)"
python -m pytest -q
```

## 4. Build or verify the reference cache

The reference cache is shared by every method. Check whether it already exists:

```bash
ls -lh \
  "$MFR_OUTPUT_DIR/cache/reference_v2.csv" \
  "$MFR_OUTPUT_DIR/cache/reference_v2.manifest.json"
```

If both files exist and match the active protocol, do not rebuild them. Otherwise submit one cache job:

```bash
python scripts/submit_carc.py cache \
  --output-dir "$MFR_OUTPUT_DIR" \
  --account xiangren_1987 \
  --gpu l40s \
  --time 02:00:00
```

The cache job returns immediately and runs in the background. Never submit several cache jobs at the same time.

## 5. Submit experiment jobs

One group job runs the requested methods sequentially for one order and seed. The existing orders are:

| Order | Sequence |
|---:|---|
| 1 | Helpful → Safe → Quality |
| 2 | Safe → Helpful → Quality |
| 3 | Quality → Helpful → Safe |
| 4 | Quality → Safe → Helpful |

The original seven-method grid—No Replay, Random 10%, original MFR, Random 14.3%, Lowest Margin, FMCR, and CPMR—is complete
for all four orders and both seeds: 56 runs in total. Do not submit those completed combinations again. Their
checkpoints and results are under `$MFR_OUTPUT_DIR/runs`.

CPMR performs reversible virtual training and averaged 48.75 minutes per run, compared with 28.59 minutes for
Lowest Margin. It should remain available as a completed baseline, but it does not need to be resubmitted for seeds
0 and 1.

### Completed: DAPR, DAPR-C, MIR-DPO, and COPR-adapted

These four methods are implemented under the exact command names `dapr`, `dapr_c`, `mir_dpo`, and
`copr_adapted`. Their eight development cells are complete. The command below is retained only for reproduction or
recovery of missing artifacts; do not resubmit completed run folders.

```bash
for order in 1 2 3 4; do
  for seed in 0 1; do
    python scripts/submit_carc.py group \
      --output-dir "$MFR_OUTPUT_DIR" \
      --account xiangren_1987 \
      --gpu l40s \
      --time 04:00:00 \
      --order "$order" \
      --seed "$seed" \
      --methods dapr,dapr_c,mir_dpo,copr_adapted
  done
done
```

This submits eight Slurm jobs. Each job runs its four methods sequentially on one L40S. It reuses the matching
completed No-Replay Stage-1 checkpoint and does not rerun the original seven completed methods. If the queue or four-hour
limit becomes a problem, submit one method at a time by changing `--methods`, for example:

```bash
python scripts/submit_carc.py group \
  --output-dir "$MFR_OUTPUT_DIR" \
  --account xiangren_1987 \
  --gpu l40s \
  --time 04:00:00 \
  --order 1 \
  --seed 0 \
  --methods mir_dpo
```

Do not submit the same order–seed–method combination twice. A failed group job is safe to resubmit: completed runs
are skipped and an incomplete run resumes from its first unfinished stage. MIR-DPO may be the slowest of these four
because every refresh scores the buffer before and after a reversible virtual update.

### Completed: DAPR α = 0.01 and DAPR-Gated

These development variants are configured as `dapr_weak` and `dapr_gated`. DAPR-Weak changes only the directional
anchor strength from 0.1 to 0.01. DAPR-Gated keeps strength 0.1 and performs an eval-mode, no-gradient live margin
check on the two replay pairs before every optimizer step. The gate activates only when `current_margin <
peak_margin`. Its per-occurrence decisions and margins are written to `replay_log.csv`; per-step gate counts and
rates are written to `history.csv`.

The development decision rule is fixed before launch: a cell passes when the variant retains earlier behaviors
better than Random 10% and its final-task accuracy is no more than 2 points below Random 10%. Compare the number of
passing cells first, then the final three-behavior average, with retention and per-behavior acquisition/final scores
reported alongside it. If one variant is selected from seeds 0–1, freeze it before confirming on seeds 2–4.

Both variants are complete in all eight development cells. The command below is retained for reproduction. Before
using it, verify that `git status --short` prints nothing and do not resubmit completed combinations.

```bash
for order in 1 2 3 4; do
  for seed in 0 1; do
    python scripts/submit_carc.py group \
      --output-dir "$MFR_OUTPUT_DIR" \
      --account xiangren_1987 \
      --gpu l40s \
      --time 04:00:00 \
      --order "$order" \
      --seed "$seed" \
      --methods dapr_weak,dapr_gated
  done
done
```

This submits eight jobs. Each job runs the two variants sequentially and reuses the matching completed No-Replay
Stage-1 checkpoint. Do not edit or save tracked repository files while jobs are queued or running. After all runs
finish, rerun notebook 07 from the beginning; its DAPR-Gated section reports gate activation by cell, stage, replay
source, and interval.

### Completed: LoRA-EWC coefficient sweep

LoRA-EWC is a no-replay regularization baseline. After each behavior, it estimates a diagonal empirical Fisher from
per-pair DPO-loss gradients on 500 frozen, method-independent training pairs and only the trainable LoRA parameters. Later stages add the standard
multi-anchor EWC penalty. The three method names differ only in the frozen coefficient:

| Command name | EWC coefficient | Replay examples |
|---|---:|---:|
| `ewc_0_1` | 0.1 | 0 |
| `ewc_1` | 1 | 0 |
| `ewc_10` | 10 | 0 |

All three coefficients are complete on the eight development cells. Their results, ordered from most forgetting to
least forgetting, are:

| Method | Average retention change | Final current-task score | Final three-behavior average |
|---|---:|---:|---:|
| LoRA-EWC (λ = 1) | -7.97 | 74.75 | 70.77 |
| No Replay | -7.94 | 74.81 | 70.77 |
| LoRA-EWC (λ = 0.1) | -7.84 | 74.50 | 70.73 |
| **LoRA-EWC (λ = 10)** | **-7.78** | 74.69 | **70.88** |

The frozen development rule selects λ = 10 as the representative EWC setting, but it remains close to No Replay
and prevents only about 2% of measured forgetting. Use the existing completed No-Replay run as the Stage-1
checkpoint only when reproducing missing EWC artifacts. The command below is retained for that purpose:

```bash
cd /project2/xiangren_1987/grp26-mfr-dpo

module purge
module load conda
eval "$(conda shell.bash hook)"
conda activate /project2/xiangren_1987/grp26-mfr-dpo/.conda/envs/mfr-dpo

export MFR_OUTPUT_DIR=/project2/xiangren_1987/grp26-mfr-dpo/artifacts

git status --short
python -m pytest -q

for order in 1 2 3 4; do
  for seed in 0 1; do
    python scripts/submit_carc.py group \
      --output-dir "$MFR_OUTPUT_DIR" \
      --account xiangren_1987 \
      --gpu l40s \
      --time 04:00:00 \
      --order "$order" \
      --seed "$seed" \
      --methods ewc_0_1,ewc_1,ewc_10
  done
done
```

This creates eight Slurm jobs and must not be run again while the completed artifacts are present. Each job runs the
three coefficients sequentially on one L40S. The runner saves the
Fisher diagonals and learned LoRA reference weights in `ewc_states.pt`, records the coefficient and Fisher settings
in `settings.json`, writes the EWC loss to each stage's `history.csv`, and supports resuming at Stage 3. Do not call
any coefficient the winner until all 24 runs are complete and notebook 07 has compared the same eight cells.

Check completion with:

```bash
for order in 1 2 3 4; do
  for seed in 0 1; do
    for method in ewc_0_1 ewc_1 ewc_10; do
      marker="$MFR_OUTPUT_DIR/runs/v2_o${order}_${method}_s${seed}/COMPLETE.json"
      if [[ -f "$marker" ]]; then
        echo "Order $order Seed $seed $method COMPLETE"
      else
        echo "Order $order Seed $seed $method MISSING"
      fi
    done
  done
done
```

Notebook 07 applies the frozen development rule: first require the final-task score to stay within 2 points of
Random 10%, then prefer the highest final three-behavior average. It selects λ = 10 for unseen-seed confirmation.
Before publication, verify from `history.csv` that the weighted EWC penalty was not negligible relative to DPO loss.

### Completed: offline joint-training reference

Joint training is deliberately separate from the sequential methods. Each run globally shuffles all 6,000 frozen
training pairs (2,000 Helpful, 2,000 Safe, and 2,000 Quality), trains a fresh adapter for one pass, and evaluates all
three validation sets. It has no task order and receives no forgetting score. Outputs go under
`$MFR_OUTPUT_DIR/joint_runs/`, so notebook 07 cannot accidentally treat them as continual runs.

Seeds 0–4 are complete. Their final validation summary is:

| Behavior | Mean accuracy | Seed standard deviation |
|---|---:|---:|
| Helpful | 66.70 | 1.96 |
| Safe | 80.70 | 1.52 |
| Quality | 72.20 | 0.91 |
| **Three-behavior average** | **73.20** | **0.99** |

The commands below are retained for reproduction or recovery of missing joint artifacts. Do not submit them again
while the five completed folders are present.

```bash
cd /project2/xiangren_1987/grp26-mfr-dpo

module purge
module load conda
eval "$(conda shell.bash hook)"
conda activate /project2/xiangren_1987/grp26-mfr-dpo/.conda/envs/mfr-dpo

export MFR_OUTPUT_DIR=/project2/xiangren_1987/grp26-mfr-dpo/artifacts

git status --short
python -m pytest -q

for seed in 0 1 2 3 4; do
  python scripts/submit_carc.py joint \
    --output-dir "$MFR_OUTPUT_DIR" \
    --account xiangren_1987 \
    --gpu l40s \
    --time 02:00:00 \
    --seed "$seed"
done
```

This submits five independent L40S jobs. Check existing or reproduced runs with:

```bash
squeue --me --format="%.18i %.20j %.2t %.10M %.30R"

for seed in 0 1 2 3 4; do
  marker="$MFR_OUTPUT_DIR/joint_runs/v2_joint_s${seed}/COMPLETE.json"
  if [[ -f "$marker" ]]; then
    echo "Joint seed $seed COMPLETE"
  else
    echo "Joint seed $seed MISSING"
  fi
done
```

Notebook 08 reports final Helpful, Safe, Quality, and three-behavior average scores. Do not report retention for
joint training because no behavior is learned in a separate earlier stage.

After Balanced MFR is implemented and configured as `mfr_balanced`, run only that new method across all eight cells:

```bash
for order in 1 2 3 4; do
  for seed in 0 1; do
    python scripts/submit_carc.py group \
      --output-dir "$MFR_OUTPUT_DIR" \
      --account xiangren_1987 \
      --gpu l40s \
      --time 04:00:00 \
      --order "$order" \
      --seed "$seed" \
      --methods mfr_balanced
  done
done
```

After At-Risk MFR is implemented and configured as `mfr_at_risk`, run only that method:

```bash
for order in 1 2 3 4; do
  for seed in 0 1; do
    python scripts/submit_carc.py group \
      --output-dir "$MFR_OUTPUT_DIR" \
      --account xiangren_1987 \
      --gpu l40s \
      --time 04:00:00 \
      --order "$order" \
      --seed "$seed" \
      --methods mfr_at_risk
  done
done
```

The existing completed `none` run supplies the matching Stage-1 checkpoint. Do not rerun the original seven completed methods
just because a new replay method is added. The runner now checks a separate Stage-1 compatibility fingerprint, so a
replay-only extension can reuse the completed checkpoint while a real Stage-1 training change is still rejected.

To submit both new methods together for a single cell:

```bash
python scripts/submit_carc.py group \
  --output-dir "$MFR_OUTPUT_DIR" \
  --account xiangren_1987 \
  --gpu l40s \
  --time 04:00:00 \
  --order 1 \
  --seed 0 \
  --methods mfr_balanced,mfr_at_risk
```

Do not submit the same order–seed–method combination twice at the same time.

## 6. Monitor, inspect, cancel, and resume

List queued and running jobs:

```bash
squeue --me --format="%.18i %.20j %.2t %.10M %.30R"
```

List the newest logs:

```bash
ls -lt "$MFR_OUTPUT_DIR/logs" | head
```

Follow one log, using the exact job name and ID printed by `squeue`:

```bash
tail -f "$MFR_OUTPUT_DIR/logs/mfr-o1-s0_JOBID.out"
```

Press `Ctrl+C` to stop following the log; this does not stop the job.

When a job disappears from `squeue`, check its final state:

```bash
sacct -j JOBID --format=JobID,JobName,State,Elapsed,ExitCode
```

Show its full log:

```bash
cat "$MFR_OUTPUT_DIR/logs/mfr-o1-s0_JOBID.out"
```

Cancel an incorrect or duplicate job:

```bash
scancel JOBID
```

If a job fails or reaches its time limit, make sure it is no longer running and then submit the same command again.
The group runner skips runs with `COMPLETE.json` and resumes a partial run after its last fully saved stage. If final
results exist without `COMPLETE.json`, it stops and asks for inspection rather than guessing.

Useful result checks:

```bash
find "$MFR_OUTPUT_DIR/runs" -name COMPLETE.json | sort
find "$MFR_OUTPUT_DIR/runs" -maxdepth 1 -type d | sort
du -sh "$MFR_OUTPUT_DIR"
```

## 7. Run the notebooks on CARC

Register the kernel once:

```bash
module purge
module load conda
eval "$(conda shell.bash hook)"
conda activate /project2/xiangren_1987/grp26-mfr-dpo/.conda/envs/mfr-dpo
python -m ipykernel install --user --name mfr-dpo --display-name "Python (mfr-dpo)"
```

Before opening Jupyter or code-server from a terminal, set:

```bash
export MFR_OUTPUT_DIR=/project2/xiangren_1987/grp26-mfr-dpo/artifacts
```

Use `notebooks/01_carc_status.ipynb` to check completion and `notebooks/07_compare_runs.ipynb` to analyze all methods,
orders, and seeds. Select the `Python (mfr-dpo)` kernel and run Notebook 07 from the top after the new methods finish.

Do not run the locked-test script while replay methods are still being designed or selected.

## 8. Requirements for CARC-compatible experiment code

Any new training method must follow these rules:

- Read scientific settings from `configs/experiment_protocol.json`; do not hide settings in a notebook or shell file.
- Provide a command-line method name that `scripts/run_experiment.py`, `scripts/run_experiment_group.py`, and
  `scripts/submit_carc.py` can validate.
- Write every large artifact beneath the passed `--output-dir`; never write to Google Drive or a local laptop path.
- Use the existing reference cache and frozen `data/v2` files.
- Keep runs unattended, unbuffered, resumable, and visible through progress logs.
- Record the Git commit, code fingerprint, GPU, package versions, settings, timings, selections, and completion marker.
- Refuse dirty Git checkouts and incompatible resumes.
- Preserve the exact replay budget unless the method is explicitly a budget ablation.
- Add CPU tests before submitting GPU jobs.
- Use only the L40S request shown above for this experiment grid so hardware is consistent.

Before any scientific submission:

```bash
git status --short
python -m pytest -q
git rev-parse --short HEAD
```

`git status --short` must be empty because the experiment runner refuses an uncommitted scientific checkout.
