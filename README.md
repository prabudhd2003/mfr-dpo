# MFR-DPO

**What Should an LLM Rehearse? Budgeted Replay for Continual Preference Tuning**

## Project summary

This project studies catastrophic forgetting during sequential preference tuning. One QLoRA adapter on
`Qwen/Qwen2.5-1.5B-Instruct` learns three behaviors—Helpfulness, Safety, and general response Quality—one after
another with Direct Preference Optimization (DPO).

DPO learns from a prompt, a preferred response, and a rejected response. After the model learns one behavior,
training it on a later behavior can weaken what it learned earlier. This is catastrophic forgetting.

Most-Forgotten Replay (MFR) keeps a small memory of old preference pairs. It records how strongly each pair was
learned and later measures how much its reference-relative preference margin has fallen. The pairs with the largest
drops are replayed during later training. The main question is whether this targeted replay protects earlier
behaviors better than random replay using the same 10% replay budget, without harming the behavior currently being
learned.

## What has been completed

- A pilot study confirmed that sequential DPO causes measurable and concentrated forgetting.
- Version 2 of the data was cleaned, deduplicated across all datasets and splits, length-filtered, hashed, and frozen.
- The CARC training pipeline, reference cache, resume logic, Stage-1 reuse, progress logs, replay logs, and automated
  tests are implemented.
- All five current methods were run for four task orders and two seeds on NVIDIA L40S GPUs: **40 completed runs**.
- Notebook 07 reports the complete validation grid, paired uncertainty, per-dataset forgetting, final scores, and
  runtime.

The five completed methods are:

| Method | Meaning |
|---|---|
| No replay | Train only on the current behavior. |
| Random 10% | Fill 10% of each later batch with random old pairs. |
| MFR 10% | Use the same budget for pairs with the largest historical margin drop. |
| Random 14.3% | Use more random replay to test whether budget alone explains the result. |
| Lowest margin | Replay pairs with the lowest current reference-relative margin. |

Forecasted Margin-Crossing Replay (FMCR) is implemented and awaiting evaluation. It forecasts which old preference
pairs are likely to cross a failure boundary before the next buffer refresh and replays them before that happens.
Balanced MFR and At-Risk MFR are planned separately and are **not implemented or evaluated yet**.

## Data

| Behavior | Dataset | Train | Validation | Locked test | Example type |
|---|---|---:|---:|---:|---|
| Helpful | NVIDIA HelpSteer2 | 2,000 | 200 | 300 | Travel planning, explanations, and instruction following. |
| Safe | PKU-SafeRLHF | 2,000 | 200 | 300 | Unsafe or illegal requests where the safer response is preferred. |
| Quality | UltraFeedback Binarized | 2,000 | 200 | 300 | Writing, reasoning, and general question-answering preferences. |

The locked test split has not been used for method development. `data/v2/manifest.json` records the pinned source and
model revisions, row counts, and file hashes.

## Experiment design

Every method uses the same data, model revision, QLoRA settings, DPO settings, task order, seed, and Stage-1 starting
checkpoint. The current grid uses:

| Order | Training sequence |
|---:|---|
| 1 | Helpful → Safe → Quality |
| 2 | Safe → Helpful → Quality |
| 3 | Quality → Helpful → Safe |
| 4 | Quality → Safe → Helpful |

Seeds 0 and 1 are used for each order. Orders 3 and 4 make Quality an earlier behavior so its forgetting can also be
measured.

The primary `accuracy` is the percentage of preference pairs with a positive length-normalized DPO advantage over
the frozen base model. It measures whether training moved the model in the preferred direction; it is not the same
as real-world factual accuracy. For an earlier behavior:

```text
retention change = final validation accuracy - accuracy immediately after learning that behavior
```

A value near zero is better. A negative value means the model forgot some of that behavior.

## Current validation results

The following table averages all eight order–seed cells. “Old behavior change” is retention change, so less negative
is better.

| Method | Average old behavior change | Final current-task accuracy | Final average across all three | Runtime per run |
|---|---:|---:|---:|---:|
| No replay | -7.94 | 74.81 | 70.77 | 22.48 min |
| Random 10% | -5.50 | 73.62 | 71.71 | 24.28 min |
| MFR 10% | -4.88 | 73.75 | 71.98 | 28.28 min |
| Random 14.3% | -5.47 | 73.88 | 71.73 | 24.91 min |
| Lowest margin | **-4.47** | **74.31** | **72.56** | 28.59 min |

What these results support:

- **Forgetting is real.** With no replay, earlier behaviors lost 7.94 accuracy points on average. Safety was the
  hardest behavior to retain: it lost 13.25 points with no replay.
- **Replay helps.** Every replay method improved retention and the final three-behavior average over no replay.
- **MFR improves on equal-budget random replay.** MFR retained 0.62 points more on average. The paired 95% bootstrap
  interval was `[0.22, 1.09]`, while its final-task difference was small and uncertain.
- **MFR also improves on higher-budget random replay.** It retained 0.59 points more and had a 0.25-point higher
  final average while using one-third fewer replay examples. MFR is slower because it must rescore the buffer.
- **Lowest margin is currently the strongest method.** It has the best average retention, final-task score, and
  final three-behavior score. Its final average is 0.58 points above MFR, with a paired interval that excludes zero.
- **The original MFR idea is promising but is not the final winner.** Historical margin decline contains useful
  information, but the current results show that present difficulty is a very strong replay signal.

These are validation results from two seeds, not final test results. They support a controlled project conclusion,
but not a broad claim that MFR is universally better. FMCR will test whether forecasting future failures is more
useful than reacting to current difficulty. Balanced MFR and At-Risk MFR will separately test behavior allocation and
actual current preference failure.

## Repository structure

```text
mfr-dpo/
├── configs/       # Frozen model, data, training, order, seed, and method settings
├── data/v2/       # Frozen preference-pair files and manifest
├── docs/          # CARC instructions, two method specifications, and final next steps
├── notebooks/     # CARC status and validation analysis notebooks
├── pilot/         # Historical proof of concept; not used in final tables
├── scripts/       # Cache, training, Slurm submission, and evaluation entry points
├── src/           # Data, DPO, replay, analysis, and evaluation implementation
└── tests/         # CPU tests for correctness, budgets, resuming, and safeguards
```

Start with:

- [`docs/CARC.md`](docs/CARC.md) for exact CARC setup, submission, monitoring, and analysis commands.
- [`docs/BALANCED_MFR.md`](docs/BALANCED_MFR.md) for the Balanced MFR definition and implementation checklist.
- [`docs/AT_RISK_MFR.md`](docs/AT_RISK_MFR.md) for the At-Risk MFR definition and implementation checklist.
- [`docs/NEXT_STEPS.md`](docs/NEXT_STEPS.md) for the work remaining after those two methods run.

The active notebooks are:

| Notebook | Purpose |
|---|---|
| `01_carc_status.ipynb` | Check the frozen data, reference cache, and completed CARC runs. |
| `07_compare_runs.ipynb` | Produce the complete validation tables, figures, uncertainty, and conclusions. |

Training runs as unattended Slurm jobs on USC CARC using one NVIDIA L40S GPU. Large caches, checkpoints, logs, and
results are stored under the ignored CARC `artifacts/` directory rather than Git.

**Do not inspect or run the locked test until the validation analysis and method-selection rule are frozen.**
