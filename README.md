# MFR-DPO

**What Should an LLM Rehearse? Budgeted Replay for Continual Preference Tuning**

This project studies catastrophic forgetting during sequential preference tuning. One QLoRA adapter on
`Qwen/Qwen2.5-1.5B-Instruct` learns Helpfulness, Safety, and general response Quality one after another using Direct
Preference Optimization (DPO).

Learning a new behavior can weaken a behavior learned earlier. Most-Forgotten Replay (MFR) tries to reduce this
forgetting by replaying old preference pairs whose scores have fallen the most since they were learned. The main
question is whether MFR retains earlier behaviors better than random replay at the same replay budget, without
substantially reducing new learning.

## Current status

The data, training pipeline, five replay conditions, logging, reference cache, automated checks, and validation
analysis are implemented. The earlier Colab study is preserved under `notebooks/colab/`. Because the final study is
moving to USC CARC, its experiment grid will be rerun there on one consistent A100 type. Colab measurements remain
useful as preliminary evidence but will not be mixed into the final CARC tables.

### Preliminary Colab evidence

In the corrected no-replay run:

- Helpful accuracy fell from 70.0% after Helpful training to 63.5% at the end: **-6.5 points**.
- Safe accuracy fell from 83.5% after Safe training to 74.0% at the end: **-9.5 points**.
- Mean forgetting across the two earlier behaviors was **8 points**.

### First MFR comparison

| Method | Mean retention change | Final Quality | Final average across all three |
|---|---:|---:|---:|
| No replay | -8.00 | 74.5 | 70.67 |
| Random 10% | -5.75 | 74.0 | 71.50 |
| Random 14.3% | -5.25 | 75.0 | 72.17 |
| Lowest margin | **-4.50** | 74.0 | 72.17 |
| MFR 10% | -5.00 | **76.5** | **72.83** |

In this first group, MFR retained 0.75 points better than equal-budget Random 10% and finished 2.5 points higher on
Quality. It passed the predeclared success rule. MFR also slightly exceeded Random 14.3% while using 444 rather than
666 replay examples. Lowest margin had the best pure retention, while MFR had the best observed balance between
retention and final-task learning.

These results are preliminary. Only one comparison group is complete, and every current pair-bootstrap confidence
interval crosses or touches zero. No conclusive MFR claim should be made yet.

## Data and experiment

| Behavior | Source | Train | Validation | Locked test |
|---|---|---:|---:|---:|
| Helpful | NVIDIA HelpSteer2 | 2,000 | 200 | 300 |
| Safe | PKU-SafeRLHF | 2,000 | 200 | 300 |
| Quality | UltraFeedback Binarized | 2,000 | 200 | 300 |

The main experiment uses two orders and two seeds:

- **Order 1:** Helpful -> Safe -> Quality
- **Order 2:** Safe -> Helpful -> Quality
- **Seeds:** 0 and 1

Two additional Quality-first orders are configured for a retention extension:

- **Order 3:** Quality -> Helpful -> Safe
- **Order 4:** Quality -> Safe -> Helpful

The core methods are no replay, Random 10%, and MFR 10%. Random 14.3% tests whether more random replay is enough,
and Lowest margin is a simpler targeted-replay baseline.

The frozen settings, data processing, metrics, method definitions, pilot history, and limitations are explained once
in [`docs/PROJECT_GUIDE.md`](docs/PROJECT_GUIDE.md).

## What remains

1. Build the reference cache once on CARC.
2. Rerun all five methods for Orders 1 and 2, Seeds 0 and 1, on one A100 type.
3. Run the core methods for the Quality-first extension if the compute budget permits.
4. Rerun Notebook 07 across the complete CARC grid and add the required `accuracy_sum` secondary analysis.
5. Run example-level error and forgetting analysis.
6. Freeze the analysis and model-selection decision.
7. Run the locked test once, followed by generation, automatic evaluation, and blinded human review.
8. Complete the report and presentation.

Do not inspect the locked test outcomes before the validation analysis and model-selection decision are frozen.

## Repository structure

```text
mfr-dpo/
├── configs/       # Frozen experiment settings
├── data/v2/       # Active versioned data and manifest
├── docs/          # One project guide and one runbook
├── notebooks/     # Active CARC notebooks; historical Colab notebooks are under notebooks/colab
├── pilot/         # Archived proof-of-concept code; not used for final results
├── scripts/       # Command-line entry points used by notebooks
├── src/           # Reusable data, training, replay, analysis, and evaluation code
└── tests/         # Automated correctness and safeguard checks
```

The active notebooks are:

| Notebook | Purpose |
|---|---|
| `01_carc_status.ipynb` | Verify the frozen data, CARC cache, and run completion. |
| `07_compare_runs.ipynb` | Compare validation results, uncertainty, replay, and cost. |

The previous numbered Colab workflow, including the unchanged Notebook 06 and all saved outputs, is preserved under
`notebooks/colab/`.

## Running the project

Training is designed for USC CARC Slurm with one NVIDIA A100 GPU. Large caches, checkpoints, logs, generations, and
run folders are stored in a user-selected CARC project directory, not in Git.

Clone the [GitHub repository](https://github.com/prabudhd2003/mfr-dpo), switch to the `carc` branch, then follow
[`docs/RUNBOOK.md`](docs/RUNBOOK.md). All scientific settings come from `configs/experiment_protocol.json`.

For local checks:

```bash
python3 -m pip install -r requirements.txt
python3 -m pytest -q
```
