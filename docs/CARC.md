# Running MFR-DPO on USC CARC

This is the terminal workflow for the course project (branch `project`, folder `/project2/xiangren_1987/group26`).
Training uses unattended Slurm jobs with one NVIDIA L40S GPU. For what this branch is and the folder layout, read
[PROJECT_FOLDER.md](PROJECT_FOLDER.md). For what to run and when, read [NEXT_STEPS.md](NEXT_STEPS.md).

The examples use:

```text
Repository:  /project2/xiangren_1987/group26
Branch:      project
Artifacts:   /project2/xiangren_1987/group26/artifacts
CARC account: xiangren_1987
GPU:         NVIDIA L40S
```

Do not include the shell prompt, such as `[user@discovery]$` or `bash-4.4$`, when copying a command.

## 1. Get the code

The clone already exists. To update it:

```bash
cd /project2/xiangren_1987/group26
git switch project
git status --short
git pull --ff-only origin project
```

If `git pull` complains about local notebook changes that are only executed outputs, discard them and pull again:

```bash
git restore -- notebooks/07_compare_runs.ipynb
git pull --ff-only origin project
```

Do not restore a file that contains code changes you need. Commit or copy those changes first.

## 2. Environments

The training environment already exists at `group26/.conda/envs/mfr-dpo`, a symlink to a shared environment. Do not
recreate it, and do not install packages into it.

The generation environment `group26/.conda/envs/mfr-gen` is created once in [NEXT_STEPS.md](NEXT_STEPS.md),
Step 10.2. It is used only for vLLM generation, IFEval, and the judges.

Register the training environment as a Jupyter kernel once:

```bash
python -m ipykernel install --user --name mfr-dpo --display-name "Python (mfr-dpo)"
```

## 3. Start each CARC terminal session

```bash
cd /project2/xiangren_1987/group26
module purge
module load conda
eval "$(conda shell.bash hook)"
conda activate /project2/xiangren_1987/group26/.conda/envs/mfr-dpo
export MFR_OUTPUT_DIR=/project2/xiangren_1987/group26/artifacts
export HF_HOME=$MFR_OUTPUT_DIR/huggingface
```

`MFR_OUTPUT_DIR` is only a terminal variable holding the artifact path. Jobs receive the expanded absolute path and
keep running after the terminal or laptop closes.

Checks:

```bash
which python                       # inside group26/.conda/envs/mfr-dpo
python -c "import torch; print(torch.__version__, torch.version.cuda)"
python -m pytest -q
```

## 4. Reference cache and model weights

The reference cache is shared by every method and already exists:

```bash
ls -lh "$MFR_OUTPUT_DIR/cache/reference_v2.csv" "$MFR_OUTPUT_DIR/cache/reference_v2.manifest.json"
```

Do not rebuild it. Jobs read model weights from `$MFR_OUTPUT_DIR/huggingface`. Download the pinned model there once
from a login node (NEXT_STEPS Step 0):

```bash
python -c "from huggingface_hub import snapshot_download; snapshot_download('Qwen/Qwen2.5-1.5B-Instruct', revision='989aa7980e4cf806f80c7fef2b1adb7bc71aa306')"
```

## 5. Submit jobs

Always check a new command with `--dry-run` first. Every submission returns immediately; the job runs in the
background.

| Order | Sequence |
|---:|---|
| 1 | Helpful → Safe → Quality |
| 2 | Safe → Helpful → Quality |
| 3 | Quality → Helpful → Safe |
| 4 | Quality → Safe → Helpful |

**New methods on seeds 0–1 (NEXT_STEPS Step 3).** The completed `none` run of each cell supplies Stage 1. Do not
rerun the original five methods.

```bash
for order in 1 2 3 4; do
  for seed in 0 1; do
    python scripts/submit_carc.py group \
      --output-dir "$MFR_OUTPUT_DIR" --account xiangren_1987 --gpu l40s --time 03:00:00 \
      --order "$order" --seed "$seed" --methods mfr_balanced,mfr_at_risk,mfr_stochastic
  done
done
```

**Joint-training reference** ([JOINT_TRAINING.md](JOINT_TRAINING.md)):

```bash
for seed in 0 1; do
  python scripts/submit_carc.py joint \
    --output-dir "$MFR_OUTPUT_DIR" --account xiangren_1987 --gpu l40s --time 02:00:00 --seed "$seed"
done
```

**Seed-2 confirmation (NEXT_STEPS Step 8, only after Step 7):**

```bash
for order in 1 2 3 4; do
  python scripts/submit_carc.py group \
    --output-dir "$MFR_OUTPUT_DIR" --account xiangren_1987 --gpu l40s --time 06:00:00 \
    --order "$order" --seed 2 \
    --methods none,random,random_high,lowest_margin,mfr,mfr_balanced,mfr_at_risk,mfr_stochastic
done
python scripts/submit_carc.py joint \
  --output-dir "$MFR_OUTPUT_DIR" --account xiangren_1987 --gpu l40s --time 02:00:00 --seed 2
```

**Locked test (NEXT_STEPS Step 9, once):**

```bash
python scripts/submit_carc.py test \
  --output-dir "$MFR_OUTPUT_DIR" --account xiangren_1987 --gpu l40s --time 12:00:00
```

**Generation evaluation (NEXT_STEPS Step 10)** uses the `generate`, `ifeval`, and `judge` actions with
`--conda-env /project2/xiangren_1987/group26/.conda/envs/mfr-gen`.

Do not submit the same order–seed–method combination twice at the same time.

## 6. Monitor, inspect, cancel, and resume

```bash
squeue --me --format="%.18i %.20j %.2t %.10M %.30R"     # queued and running jobs
ls -lt "$MFR_OUTPUT_DIR/logs" | head                     # newest logs
tail -f "$MFR_OUTPUT_DIR/logs/mfr-o1-s0_JOBID.out"       # follow one log; Ctrl+C stops following, not the job
sacct -j JOBID --format=JobID,JobName,State,Elapsed,ExitCode
scancel JOBID                                            # cancel a wrong or duplicate job
```

If a group job fails or reaches its time limit, make sure it is no longer running, then submit the same command again.
The group runner skips runs with `COMPLETE.json` and resumes a partial run after its last fully saved stage. If final
results exist without `COMPLETE.json`, it stops and asks for inspection rather than guessing. A failed joint run is
renamed and restarted, as described in JOINT_TRAINING.md.

Result checks:

```bash
find "$MFR_OUTPUT_DIR/runs" -name COMPLETE.json | wc -l
find "$MFR_OUTPUT_DIR/joint_runs" -name COMPLETE.json | wc -l
du -sh "$MFR_OUTPUT_DIR"
```

## 7. Run the notebooks on CARC

Start Jupyter or code-server from a terminal where `MFR_OUTPUT_DIR` is set as in section 3. Select the
`Python (mfr-dpo)` kernel.

- `notebooks/01_carc_status.ipynb` checks completion.
- `notebooks/07_compare_runs.ipynb` analyzes all methods, orders, and seeds. It reads
  `/project2/xiangren_1987/group26/artifacts`. Run it from the top.

## 8. Requirements for CARC-compatible experiment code

Any new training method must follow these rules:

- Read scientific settings from `configs/experiment_protocol.json`; do not hide settings in a notebook or shell file.
- Provide a command-line method name that `scripts/run_experiment.py`, `scripts/run_experiment_group.py`, and
  `scripts/submit_carc.py` can validate.
- Write every large artifact beneath the passed `--output-dir`.
- Use the existing reference cache and frozen `data/v2` files.
- Keep runs unattended, unbuffered, resumable, and visible through progress logs.
- Record the Git commit, code fingerprint, GPU, package versions, settings, timings, selections, and completion marker.
- Refuse dirty Git checkouts and incompatible resumes.
- Preserve the exact replay budget unless the method is explicitly a budget ablation.
- Add CPU tests before submitting GPU jobs.
- Use only the L40S request shown above, so hardware is consistent.

Before any scientific submission:

```bash
git status --short      # must be empty
python -m pytest -q
git rev-parse --short HEAD
```
