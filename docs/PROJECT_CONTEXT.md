# Course project (`project` branch, `group26`): full context

Written 2026-10-10. Hand-off for the CSCI 544 **course project** (branch `project`, CARC folder `group26`).

## Why the project branch exists

The code kept growing on the `complete` branch into a much larger, more complex version (many more methods,
orders, seeds, a second model and a larger evaluation). The course project does not need all of that, so it is
frozen at commit **`1ced1f4` ("updated docs", 2026-10-08)**, which has the core study only. The project adds
just two methods on top: **At-Risk MFR** and **Balanced MFR**.

## What `1ced1f4` contains

- Model: `Qwen/Qwen2.5-1.5B-Instruct`, QLoRA (r 16, α 32), DPO β 0.1; behaviors Helpful (HelpSteer2),
  Safe (PKU-SafeRLHF), Quality (UltraFeedback) learned in sequence.
- Protocol 2.1, data v2, orders 1–4, seeds 0–1 (8 development cells).
- Methods: `none`, `random` (Random 10%), `random_high` (Random 14.3%), `lowest_margin`, `mfr`.
- Design docs `docs/AT_RISK_MFR.md` (method name `mfr_at_risk`) and `docs/BALANCED_MFR.md` (`mfr_balanced`).
  **Neither method is implemented anywhere yet**: no code in any branch, and no runs have been trained.
- Analysis: `notebooks/07_compare_runs.ipynb` (method list and colors in `src/mfr_analysis.py`,
  `METHOD_ORDER`).

## Branches on GitHub (`prabudhd2003/mfr-dpo`)

| Branch | Purpose | State |
|---|---|---|
| `project` | course project | created from `1ced1f4`; adds `docs/PROJECT_FOLDER.md` and notes in README / CARC.md |
| `complete` | extended version | the more complex version of the code |
| `carc` | older branch | untouched |
| `main` | original | untouched |

`carc` was **not** rewound. The `project` branch was made instead, so no history was rewritten.

## Folders

| | Course project | Extended version |
|---|---|---|
| Laptop | `~/Desktop/CSCI544/project/group26` (branch `project`) | `~/Desktop/CSCI544/code/mfr-dpo` (branch `complete`) |
| CARC | `/project2/xiangren_1987/group26` (branch `project`) | `/project2/xiangren_1987/grp26-mfr-dpo` (branch `complete`) |
| Artifacts | `/project2/xiangren_1987/group26/artifacts` | `/project2/xiangren_1987/grp26-mfr-dpo/artifacts` |
| Conda env | symlink `group26/.conda/envs/mfr-dpo` → the extended env | `/project2/xiangren_1987/grp26-mfr-dpo/.conda/envs/mfr-dpo` |

The extended folder must **not be renamed**. The conda env inside it has hard-coded paths, and saved runs record
absolute paths.

## CARC `group26/artifacts` contents

- `cache/`: copied from the extended artifacts.
- `runs/`: 40 finished runs (`none`, `random`, `random_high`, `lowest_margin`, `mfr` × orders 1–4 × seeds 0–1),
  copied with `rsync`. 6.2 GB total. They were produced by code at or before `1ced1f4`, so the project
  notebook reads them unchanged.
- `artifacts/` and `.conda/` are git-ignored and exist only on CARC.

## Every session on CARC

```bash
cd /project2/xiangren_1987/group26
module purge
module load conda
eval "$(conda shell.bash hook)"
conda activate /project2/xiangren_1987/grp26-mfr-dpo/.conda/envs/mfr-dpo
export MFR_OUTPUT_DIR=/project2/xiangren_1987/group26/artifacts
```

Then open `notebooks/07_compare_runs.ipynb` and Run All.

## How it was built (done; do not rerun)

```bash
# laptop
cd ~/Desktop/CSCI544/project
git clone https://github.com/prabudhd2003/mfr-dpo.git group26
cd group26
git switch -c project 1ced1f46f691080e3b68910d970355266fc79721
git push -u origin project

# CARC
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

## What is left for the course project

1. **Implement At-Risk MFR and Balanced MFR** from their design docs, on the `project` branch. The `complete`
   branch needs the same implementation if they are trained from the extended folder, and the two
   implementations must be identical.
2. **Train them** on the 8 development cells: 2 methods × 8 = 16 runs, about 8 GPU-h. Plan: train in the
   extended folder, where the env and cache live, then copy:
   ```bash
   OLD=/project2/xiangren_1987/grp26-mfr-dpo/artifacts
   NEW=/project2/xiangren_1987/group26/artifacts
   for m in mfr_at_risk mfr_balanced; do
     rsync -a $OLD/runs/v2_o[1-4]_${m}_s[01] $NEW/runs/
   done
   ls $NEW/runs | wc -l   # 56
   ```
   One alternative is training directly in `group26` with `--output-dir $NEW`. That keeps the code
   and results in one place, but the job would use the symlinked env.
3. **Add both methods to the analysis**: `METHOD_ORDER` and colors in `src/mfr_analysis.py`, and notebook 07.
4. Write the course report from notebook 07.

## Rules

- Never run git commands in `grp26-mfr-dpo` while doing project work.
