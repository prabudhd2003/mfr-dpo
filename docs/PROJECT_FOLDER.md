# Course-project folder: what this branch is and how to use it

This is the single setup document for the CSCI 544 Group 26 course project. It replaces the former
`PROJECT_CONTEXT.md` and `PROJECT_FOLDER.md`.

## What this branch is

- Branch **`project`** holds the course project. It starts from commit **`1ced1f4`** ("updated docs", 2026-10-08),
  which contains the core study:
  - data v2 and protocol 2.1;
  - `Qwen/Qwen2.5-1.5B-Instruct` with QLoRA (r 16, α 32) and DPO β 0.1;
  - the behaviors Helpful (HelpSteer2), Safe (PKU-SafeRLHF), and Quality (UltraFeedback), learned in sequence;
  - orders 1–4 and seeds 0–1;
  - five methods: `none`, `random`, `random_high`, `lowest_margin`, `mfr`;
  - Notebook 07 for analysis.
- The course project adds only what is listed in [NEXT_STEPS.md](NEXT_STEPS.md):
  - three methods: Balanced MFR, At-Risk MFR, and Stochastic MFR;
  - the joint-training reference;
  - seed 2;
  - the locked test;
  - generation and human evaluation.
- **All course-project work happens in this branch and this folder: code, training, results, and analysis.** The
  other branches in the repository (`complete`, `carc`, `main`) are not used for the course project.

## Folders

| What | Where |
|---|---|
| Laptop clone | `~/Desktop/CSCI544/project/group26` (branch `project`) |
| CARC clone | `/project2/xiangren_1987/group26` (branch `project`) |
| CARC artifacts | `/project2/xiangren_1987/group26/artifacts` (git-ignored) |
| Training environment | `/project2/xiangren_1987/group26/.conda/envs/mfr-dpo` (git-ignored symlink to a shared environment) |
| Generation environment | `/project2/xiangren_1987/group26/.conda/envs/mfr-gen` (created in NEXT_STEPS Step 10.2) |

The training environment is shared through the symlink. **Never install or upgrade packages in it.** Anything new
goes into `mfr-gen`.

## Every CARC session

```bash
cd /project2/xiangren_1987/group26
module purge
module load conda
eval "$(conda shell.bash hook)"
conda activate /project2/xiangren_1987/group26/.conda/envs/mfr-dpo
export MFR_OUTPUT_DIR=/project2/xiangren_1987/group26/artifacts
export HF_HOME=$MFR_OUTPUT_DIR/huggingface
```

`which python` should then print a path inside `group26/.conda/envs/mfr-dpo`. Submission, monitoring, and notebook
commands are in [CARC.md](CARC.md).

## What is in `artifacts/`

| Folder | Contents |
|---|---|
| `cache/` | `reference_v2.csv` and its manifest: frozen base-model log-probabilities for every pair |
| `runs/` | 40 completed runs today (5 methods × orders 1–4 × seeds 0–1, 6.2 GB). 64 after NEXT_STEPS Step 3 and 96 after Step 8 |
| `joint_runs/` | Joint-training reference runs, 3 after Step 8 ([JOINT_TRAINING.md](JOINT_TRAINING.md)) |
| `huggingface/` | Model weights used by Slurm jobs (`HF_HOME` inside jobs) |
| `logs/` | Slurm job logs |

The reference cache and the 40 completed runs were copied into this folder with `rsync` from the artifact directory
where they were first trained. They were produced by code identical to commit `1ced1f4` for these five methods, so
this branch reads them unchanged. They do not need to be retrained.

## Updating the code

1. Edit on the laptop in `~/Desktop/CSCI544/project/group26`, run `python -m pytest -q`, then commit and push to
   `project`.
2. On CARC:
   ```bash
   cd /project2/xiangren_1987/group26
   git pull --ff-only origin project
   git status --short      # must print nothing before any job is submitted
   ```

The experiment runner refuses to start from a checkout with uncommitted changes.

## Rules

- Submit every course-project job from `/project2/xiangren_1987/group26` with `--output-dir "$MFR_OUTPUT_DIR"`.
- Do not train with one branch and analyze with another. Code and results for the course project both live here.
- Never run the locked test before the method selection is frozen (NEXT_STEPS Step 7).
- Never install packages into the shared training environment.

## How this folder was built (history; do not rerun)

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
ln -s <shared training environment> .conda/envs/mfr-dpo
# cache/ and the 40 runs were then copied into artifacts/ with rsync
```
