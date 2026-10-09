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
- All six current methods were run for four task orders and two seeds on NVIDIA L40S GPUs: **48 completed runs**.
- Notebook 07 reports the complete validation grid, paired uncertainty, per-dataset forgetting, final scores, and
  runtime.

The six completed methods are:

| Method | Meaning |
|---|---|
| No replay | Train only on the current behavior. |
| Random 10% | Fill 10% of each later batch with random old pairs. |
| MFR 10% | Use the same budget for pairs with the largest historical margin drop. |
| Random 14.3% | Use more random replay to test whether budget alone explains the result. |
| Lowest margin | Replay pairs with the lowest current reference-relative margin. |
| FMCR 10% | Forecast which pairs may cross a preference-failure boundary before the next refresh. |

Forecasted Margin-Crossing Replay (FMCR) is a completed ablation, not the main winning method. Balanced MFR and
At-Risk MFR are planned separately and are **not implemented or evaluated yet**.

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
| FMCR 10% | -5.44 | **74.75** | 72.21 | 29.26 min |

What these results support:

- **Forgetting is real.** With no replay, earlier behaviors lost 7.94 accuracy points on average. Safety was the
  hardest behavior to retain: it lost 13.25 points with no replay.
- **Replay helps.** Every replay method improved retention and the final three-behavior average over no replay.
- **MFR improves on equal-budget random replay.** MFR retained 0.62 points more on average. The paired 95% bootstrap
  interval was `[0.22, 1.09]`, while its final-task difference was small and uncertain.
- **MFR also improves on higher-budget random replay.** It retained 0.59 points more and had a 0.25-point higher
  final average while using one-third fewer replay examples. MFR is slower because it must rescore the buffer.
- **Lowest margin is currently the strongest overall method.** It has the best average retention and final
  three-behavior score. Its final average is 0.58 points above MFR, with a paired interval that excludes zero. FMCR
  has a slightly higher final current-task score, but retains earlier behaviors less effectively.
- **The original MFR idea is promising but is not the final winner.** Historical margin decline contains useful
  information, but the current results show that present difficulty is a very strong replay signal.
- **FMCR is a useful forecasting ablation with a mixed result.** It did not improve retention over Random 10%, but
  it improved final-task accuracy by 1.12 points and the final three-behavior average by 0.50 points. Compared with
  Lowest Margin, FMCR retained 0.97 points less and finished 0.35 points lower overall. Forecasting favored new-task
  learning more than retention and did not replace the simpler current-difficulty signal.

These are validation results from two seeds, not final test results. They support a controlled project conclusion,
but not a broad claim that MFR is universally better. Balanced MFR and At-Risk MFR will separately test behavior
allocation and actual current preference failure.

## How the results fit together

The project is not simply a contest to make MFR win. The report tells a sequence of controlled findings:

1. Sequential DPO causes real, non-uniform forgetting.
2. A fixed amount of replay substantially reduces that forgetting.
3. Historical margin decline is useful: MFR beats equal-budget Random 10% on retention and also beats higher-budget
   random replay while using fewer examples.
4. Present difficulty is even stronger in this setup: Lowest Margin currently gives the best retention and final
   balance.
5. Forecasting is not automatically better. FMCR improves current-task learning and the final average relative to
   random replay, but its retention is similar to random and worse than Lowest Margin.
6. Balanced MFR and At-Risk MFR are controlled ablations that will test whether allocation and actual policy failure
   explain the remaining gap.

This supports a broader contribution: a controlled study of **what an LLM should rehearse during continual
preference tuning**, including positive and negative results, rather than an unsupported claim that one heuristic is
always best.

## What the datasets and task orders reveal

All three datasets use the same basic preference format: a prompt, a chosen answer, and a rejected answer. They also
use the same frozen split sizes and preprocessing. Helpfulness and general Quality overlap because both reward useful,
well-written, instruction-following answers. Safety is more distinct: many examples reward refusing or safely
redirecting harmful requests. This makes the three behaviors related, but not interchangeable.

The measured forgetting reflects that difference:

| Earlier behavior | No replay | Random 10% | MFR 10% | Lowest margin | FMCR 10% |
|---|---:|---:|---:|---:|---:|
| Helpful (6 cells) | -6.25 | -4.50 | -4.83 | -4.42 | **-4.17** |
| Safe (6 cells) | -13.25 | -8.75 | -7.33 | **-6.67** | -8.58 |
| Quality (4 cells) | -2.50 | -2.12 | **-1.25** | **-1.25** | -2.62 |

Safety is the most fragile behavior and Quality is the most stable. A plausible explanation is that later helpfulness
or quality tuning rewards broadly useful responses and can weaken refusal behavior, while helpfulness and quality
share more response characteristics. This is an interpretation of the observed pattern, not yet a causal claim; the
generation and error analyses must inspect actual responses.

Average retention also changes by order:

| Order | Sequence | No replay | MFR 10% | Lowest margin | FMCR 10% |
|---:|---|---:|---:|---:|---:|
| 1 | Helpful → Safe → Quality | -9.75 | -5.38 | **-5.12** | -5.75 |
| 2 | Safe → Helpful → Quality | -13.00 | -9.25 | **-8.50** | -8.75 |
| 3 | Quality → Helpful → Safe | -4.38 | -1.50 | **-1.25** | -3.12 |
| 4 | Quality → Safe → Helpful | -4.62 | -3.38 | **-3.00** | -4.12 |

Order 2 is hardest because Safety is learned first, is the most fragile behavior, and must survive two later stages.
Orders 3 and 4 look easier partly because stable Quality is placed earlier while either Safety or Helpfulness is last
and therefore cannot be measured as forgotten. The order averages therefore mix true sequence effects with which
datasets are exposed to later training. The per-dataset table is the safer basis for behavior-level conclusions.

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
