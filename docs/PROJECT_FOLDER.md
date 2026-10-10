# Course-project folder (`group26`, branch `project`)

This branch is the **CSCI 544 course project**: the code at commit `1ced1f4` (No Replay, Random 10%,
Random 14.3%, Lowest Margin, MFR) plus At-Risk and Balanced MFR once they are added.

A more complex, extended version of the code lives in a different folder and branch. Do not mix them:

| | Course project | Extended version |
|---|---|---|
| CARC folder | `/project2/xiangren_1987/group26` | `/project2/xiangren_1987/grp26-mfr-dpo` |
| Git branch | `project` | `complete` |
| Artifacts | `/project2/xiangren_1987/group26/artifacts` | `/project2/xiangren_1987/grp26-mfr-dpo/artifacts` |
| Laptop folder | `~/Desktop/CSCI544/project/group26` | `~/Desktop/CSCI544/code/mfr-dpo` |

Both folders use the same conda environment, which lives in the extended folder. Training runs happen
in the extended folder; the finished runs the project needs are copied here.

## Every session

```bash
cd /project2/xiangren_1987/group26
module purge
module load conda
eval "$(conda shell.bash hook)"
conda activate /project2/xiangren_1987/grp26-mfr-dpo/.conda/envs/mfr-dpo
export MFR_OUTPUT_DIR=/project2/xiangren_1987/group26/artifacts
```

Then open `notebooks/07_compare_runs.ipynb` in Jupyter and Run All.

## What is in `artifacts/`

- `cache/`: copied from the extended folder.
- `runs/`: 40 finished runs (`none`, `random`, `random_high`, `lowest_margin`, `mfr` × orders 1–4 ×
  seeds 0–1), copied from the extended folder. 6.2 GB.

`artifacts/` and `.conda/` are ignored by git; they exist only on CARC.

## How this folder was built (for reference; do not rerun)

```bash
cd /project2/xiangren_1987
git clone -b project https://github.com/prabudhd2003/mfr-dpo.git group26
cd group26
mkdir -p .conda/envs
ln -s /project2/xiangren_1987/grp26-mfr-dpo/.conda/envs/mfr-dpo .conda/envs/mfr-dpo

OLD=/project2/xiangren_1987/grp26-mfr-dpo/artifacts
NEW=/project2/xiangren_1987/group26/artifacts
mkdir -p $NEW/runs
rsync -a --exclude runs --exclude joint_runs --exclude logs --exclude huggingface $OLD/ $NEW/
for m in none random random_high lowest_margin mfr; do
  rsync -a $OLD/runs/v2_o[1-4]_${m}_s[01] $NEW/runs/
done
```

## Adding new runs (At-Risk and Balanced MFR)

Train them in the extended folder, then copy them here with the same loop:

```bash
OLD=/project2/xiangren_1987/grp26-mfr-dpo/artifacts
NEW=/project2/xiangren_1987/group26/artifacts
for m in mfr_at_risk mfr_balanced; do
  rsync -a $OLD/runs/v2_o[1-4]_${m}_s[01] $NEW/runs/
done
ls $NEW/runs | wc -l     # 56 once both are copied
```

## Updating the code on CARC

```bash
cd /project2/xiangren_1987/group26
git pull
```

Never run git commands in `grp26-mfr-dpo` from here; that folder follows the `complete` branch.
