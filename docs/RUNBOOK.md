# MFR-DPO CARC Runbook

The active experiments run as unattended Slurm jobs on USC CARC. Google Drive and Colab are not used by this workflow.

## 1. One-time CARC environment

Update the `carc` branch in the shared CARC repository, then request a short interactive CPU session. A GPU is not needed to create the environment or run the automated tests:

```bash
cd /project2/xiangren_1987/grp26-mfr-dpo
git switch carc
git pull origin carc
salloc --partition=main --ntasks=1 --cpus-per-task=4 --mem=16G --time=02:00:00
```

Create the environment inside that allocation:

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

Use the ignored `artifacts` folder inside the shared CARC project directory for every job:

```bash
export MFR_OUTPUT_DIR=/project2/xiangren_1987/grp26-mfr-dpo/artifacts
mkdir -p "$MFR_OUTPUT_DIR"
```

The repository contains the frozen `data/v2` files. Large caches, checkpoints, logs, and results are written under `MFR_OUTPUT_DIR`, not Git.

## 2. Build the CARC reference cache once

From the repository on a CARC login node:

```bash
python scripts/submit_carc.py cache \
  --output-dir "$MFR_OUTPUT_DIR" \
  --account yzhao010_1531 \
  --gpu l40s
```

The command submits the work to Slurm and returns immediately. It is safe to close the laptop after submission.

Monitor it with:

```bash
squeue --me
ls -lt "$MFR_OUTPUT_DIR/logs"
```

Wait until `cache/reference_v2.csv` and `cache/reference_v2.manifest.json` exist before submitting experiments. The cache job never overwrites an existing complete cache.

## 3. Submit the main experiment grid

One command runs all five methods for one order and seed:

```bash
python scripts/submit_carc.py group \
  --output-dir "$MFR_OUTPUT_DIR" \
  --account yzhao010_1531 \
  --gpu l40s \
  --order 1 \
  --seed 0
```

The default methods are:

1. `none`
2. `random`
3. `mfr`
4. `random_high`
5. `lowest_margin`

Submit all four main groups:

```bash
for order in 1 2; do
  for seed in 0 1; do
    python scripts/submit_carc.py group \
      --output-dir "$MFR_OUTPUT_DIR" \
      --account yzhao010_1531 \
      --gpu l40s \
      --order "$order" \
      --seed "$seed"
  done
done
```

The main orders are:

| Order | Sequence |
|---:|---|
| 1 | Helpful → Safe → Quality |
| 2 | Safe → Helpful → Quality |

Each Slurm job runs its five methods sequentially on one L40S. `none` trains Stage 1 first; the four replay methods reuse that exact Stage‑1 checkpoint.

## 4. Quality-retention extension

Orders 3 and 4 put Quality first, allowing its later forgetting to be measured:

| Order | Sequence |
|---:|---|
| 3 | Quality → Helpful → Safe |
| 4 | Quality → Safe → Helpful |

Run the core comparison first:

```bash
for order in 3 4; do
  for seed in 0 1; do
    python scripts/submit_carc.py group \
      --output-dir "$MFR_OUTPUT_DIR" \
      --account yzhao010_1531 \
      --gpu l40s \
      --order "$order" \
      --seed "$seed" \
      --methods none,random,mfr
  done
done
```

The secondary methods can be added later by resubmitting with all five methods. Completed runs are skipped.

## 5. Resume and monitor

Check the queue and logs:

```bash
squeue --me
ls -lt "$MFR_OUTPUT_DIR/logs"
```

If a job stops, first confirm it is no longer running, then submit the same group command again. The group runner:

- skips every run containing `COMPLETE.json`;
- resumes a partial run after its last fully saved stage;
- refuses to guess when final results exist without a completion marker.

Do not submit the same order–seed group twice at the same time.

## 6. Add a future method or order

To add an order, add its sequence to `orders` in `configs/experiment_protocol.json`. The runner reads order choices and stage count from that file.

To add a method such as `mfr_balanced`:

1. implement its selection rule in `src/mfr_replay.py`;
2. add it to `REFRESH_METHODS` there if it needs live margin rescoring;
3. add it to the protocol's `methods` or `secondary_methods` list;
4. add correctness and budget tests;
5. submit it with `--methods none,mfr_balanced` or include it in the default list.

Create a new protocol version before running a new scientific method. Do not combine different protocol versions in one final comparison.

## 7. Analyze the completed runs

Set the same artifact path before starting Jupyter on CARC:

```bash
export MFR_OUTPUT_DIR=/project2/xiangren_1987/grp26-mfr-dpo/artifacts
```

Use:

- `notebooks/01_carc_status.ipynb` to verify data, cache, and run completion;
- `notebooks/07_compare_runs.ipynb` for validation tables and figures.

The previous Colab notebooks and outputs are preserved under `notebooks/colab/` and are not part of the CARC experiment grid.

Do not run locked-test evaluation until the CARC validation analysis and checkpoint-selection decision are frozen.
