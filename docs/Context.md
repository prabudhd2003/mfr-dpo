# MFR-DPO Team Handoff

Last updated: September 27, 2026

This document gives teammates and coding assistants enough context to understand the project, inspect the current
work, and continue it without repeating completed experiments or opening the locked test data too early.

## Project resources

- GitHub repository: <https://github.com/prabudhd2003/mfr-dpo>
- Models, reference cache, run folders, and other large artifacts:
  <https://drive.google.com/drive/folders/1Mk7RtTdu0m_f7wnuvzGNzgRSTr8T1bqb?usp=drive_link>
- Frozen protocol: `configs/experiment_protocol.json`
- Detailed run instructions: `docs/RUNBOOK.md`
- Data documentation: `docs/DATA_CARD.md`
- Main validation analysis: `notebooks/07_compare_runs.ipynb`

The local repository and GitHub `main` were synchronized at commit
`d65de533f1acc2c7ff1dab9d8fcde19ded89b00c` when this handoff was written. Large model and experiment artifacts are
stored in Drive and are intentionally not committed to Git.

## Project in one paragraph

The project studies catastrophic forgetting during sequential preference tuning. One QLoRA adapter on
`Qwen/Qwen2.5-1.5B-Instruct` learns helpfulness, safety, and general response quality one after another using Direct
Preference Optimization (DPO). Learning a new preference can weaken preferences learned earlier. Our proposed method,
Most-Forgotten Replay (MFR), uses a small replay budget for old preference pairs whose relative preference margin has
fallen the most since the behavior was learned. The primary question is whether MFR retains old behaviors better than
equal-budget random replay without reducing new-behavior learning by more than two accuracy points.

## What the metrics mean

Each example contains a prompt, a preferred response, and a rejected response. The primary validation metric,
`accuracy`, is the percentage of pairs where training moved the policy toward the preferred answer relative to the
frozen base model, using length-normalized response log probabilities.

This is preference-pair accuracy. It is not the percentage of generated answers that are objectively helpful, safe,
or correct. Generated-response evaluation and blinded human review are required before making behavioral claims.

For a previously learned behavior:

```text
retention change = final accuracy - accuracy immediately after learning that behavior
```

Zero means no forgetting. A negative value means performance fell. For example, `-4` is better retention than `-9`.

The required secondary metric is `accuracy_sum`, which uses summed rather than length-normalized response log
probabilities. The final report must include both metrics.

## Data

Three pinned preference datasets are used:

| Behavior | Source | Pair selection |
|---|---|---|
| Helpful | NVIDIA HelpSteer2 | Keep pairs with absolute preference strength of at least 2. |
| Safe | PKU-SafeRLHF | Keep pairs where exactly one response is marked safe and choose that response. |
| Quality | UltraFeedback Binarized | Keep pairs with a chosen-versus-rejected score gap of at least 1. |

For each behavior, v2 contains:

| Split | Pairs | Use |
|---|---:|---|
| Train | 2,000 | Sequential training and replay candidates |
| Validation | 200 | Development and method comparison |
| Test | 300 | Locked final evaluation only |

The v2 processing pipeline:

1. Loads each dataset at a pinned revision.
2. Removes missing, empty, identical, and duplicate response pairs.
3. Normalizes prompts using Unicode NFKC normalization, case folding, and whitespace normalization.
4. Removes normalized prompt overlap across all behaviors and splits.
5. Counts tokens using the pinned model and removes sequences longer than 1,024 tokens.
6. Creates deterministic train, validation, and test splits with seed 0.
7. Preserves source-dataset, source-split, and source-row provenance.
8. Writes stable IDs, row counts, and SHA-256 hashes to `data/v2/manifest.json`.

Every experiment verifies the data manifest before training. Do not modify the v2 JSONL files.

## Frozen experiment design

The authoritative settings are in `configs/experiment_protocol.json`.

| Setting | Value |
|---|---|
| Model | `Qwen/Qwen2.5-1.5B-Instruct`, pinned revision |
| Hardware | NVIDIA A100 through Google Colab |
| Training | Sequential DPO with one QLoRA adapter |
| Epochs | 1 per behavior |
| DPO beta | 0.1 |
| Learning rate | `1e-4` |
| Seeds | 0 and 1 |
| Replay buffer | 500 pairs |
| MFR refreshes | 5 per later stage |
| Maximum one-behavior replay share | 75% when another old behavior is available |

Two orders are used:

| Order | Stage 1 | Stage 2 | Stage 3 |
|---|---|---|---|
| Order 1 | Helpful | Safe | Quality |
| Order 2 | Safe | Helpful | Quality |

Quality is last in both orders. Therefore, the experiment cannot measure quality as a retention target. This must be
reported as a limitation.

## Methods

| Method | Purpose |
|---|---|
| `none` | No-replay baseline that measures unprotected forgetting. |
| `random` | Equal-budget baseline: 18 new plus 2 uniformly sampled old pairs. |
| `mfr` | Proposed method: 18 new plus 2 old pairs selected by margin loss. |
| `random_high` | Budget control: 18 new plus 3 uniformly sampled old pairs. |
| `lowest_margin` | Secondary targeted baseline using the lowest current margins. |

There is no replay during Stage 1. Replay runs reuse the compatible Stage-1 artifact from the matching no-replay run.
The runner validates the data, settings, model revision, and scientific-code fingerprint before allowing reuse.

## What has been completed

- Original pilot experiments, preserved under `pilot/`
- Corrected and frozen v2 data
- Data manifest and leakage checks
- Frozen protocol v2
- Reference-model probability cache and live numerical cache canary
- Reproducible training, resume, and Stage-1 reuse pipeline
- No replay, random, higher-budget random, lowest-margin, and MFR implementations
- Per-stage validation and per-pair margin storage
- Replay-allocation and runtime logging
- Progress bars and completed-run markers
- Automated tests for data, replay budgets, cache compatibility, DPO math, analysis, evaluation, and review preparation
- A rewritten step-by-step Notebook 07 for validation analysis and visualizations
- One complete five-method comparison cell: Order 1, Seed 0

## Current validation results

The completed cell is Order 1, Seed 0: Helpful -> Safe -> Quality.

| Method | Helpful retention | Safe retention | Mean retention | Final Quality | Final all-behavior average |
|---|---:|---:|---:|---:|---:|
| No replay | -6.5 | -9.5 | -8.00 | 74.5 | 70.67 |
| Random 10% | -6.0 | -5.5 | -5.75 | 74.0 | 71.50 |
| Random 14.3% | -5.5 | -5.0 | -5.25 | 75.0 | 72.17 |
| Lowest margin | -6.5 | **-2.5** | **-4.50** | 74.0 | 72.17 |
| MFR 10% | -6.0 | -4.0 | -5.00 | **76.5** | **72.83** |

Safe interpretation of this cell:

- Forgetting exists: without replay, the Safe score falls by 9.5 points after Quality training.
- Every replay method improves mean retention over no replay on the primary metric.
- MFR beats equal-budget Random 10% by 0.75 mean-retention points.
- MFR finishes 2.5 points higher than Random 10% on the final Quality task.
- MFR passes the predeclared success rule in this one cell.
- MFR slightly exceeds Random 14.3% while using 444 rather than 666 replay examples, suggesting possible replay-data
  efficiency.
- Lowest margin has the best pure retention, mainly from stronger Safe retention.
- MFR has the best observed final average across all three behaviors and currently offers the best
  retention-versus-new-learning balance.
- MFR is not faster. Its replay-candidate scoring adds approximately 4.14 minutes relative to Random 10% and makes it
  approximately 2.5 minutes slower than Random 14.3% in this cell.
- All current pair-bootstrap confidence intervals cross or touch zero. These results are promising but not
  statistically conclusive.

MFR replayed 148 Safe and 74 Helpful slots during Stage 3, while Random 10% replayed 118 Safe and 104 Helpful slots.
Without a forced-equal-allocation control, we cannot fully separate the benefit of choosing particular pairs from the
benefit of assigning more replay to the more vulnerable behavior.

## What is not complete

Only one of four order-and-seed cells is complete. The following 15 runs remain if all five methods are retained:

- Order 1, Seed 1: five methods
- Order 2, Seed 0: five methods
- Order 2, Seed 1: five methods

The core grid contains only `none`, `random`, and `mfr`. Nine of those 12 core runs remain. The additional
`random_high` and `lowest_margin` runs are secondary controls.

The following notebooks have not yet been run for the final study:

- `08_error_analysis.ipynb`
- `09_final_test_eval.ipynb`
- `10_generation_eval.ipynb`
- `11_blind_review.ipynb`

The current simplified Notebook 07 focuses on the primary `accuracy` metric. Before final reporting, add a compact
analysis of the required secondary `accuracy_sum` metric. The result files already contain it, so training does not
need to be repeated.

## Required next steps, in order

### 1. Finish the validation grid

For each remaining order-and-seed cell, run `none` first and then reuse its Stage-1 artifact for the replay methods.
The exact run names and resume instructions are in `docs/RUNBOOK.md`.

Run all five methods for:

1. Order 1, Seed 1
2. Order 2, Seed 0
3. Order 2, Seed 1

### 2. Complete validation analysis

1. Rerun `07_compare_runs.ipynb` after the grid is complete.
2. Restore the required `accuracy_sum` secondary analysis.
3. Compare all four matched cells and calculate aggregate effects.
4. Report pair-bootstrap intervals and every individual order-and-seed cell.
5. Run `08_error_analysis.ipynb` for example-level forgetting and failure analysis.
6. Report replay allocation, replay concentration, runtime, and total examples.

### 3. Freeze the final decision

Before opening test outcomes:

1. Freeze the analysis code and metrics.
2. Record whether the validation evidence supports the MFR hypothesis.
3. Record the selected completed runs/checkpoints for final evaluation.

### 4. Run the locked test once

Run `09_final_test_eval.ipynb` only after the previous decisions are frozen. Never use test results to change the
method, settings, or selected checkpoint.

### 5. Evaluate generated behavior

Run `10_generation_eval.ipynb` with fixed greedy decoding for matched No replay, Random 10%, and MFR checkpoints.
Then run versioned automatic evaluators for instruction-following and safety and preserve per-prompt outputs.

### 6. Run blinded human review

Use `11_blind_review.ipynb` to sample 100 prompts and create a blinded three-system review sheet. Keep the private
method key hidden until ratings are complete. Report preferences, rubric scores, representative failures, and
inter-rater agreement when two reviewers are available.

### 7. Write the final report

Include the method, data, complete grid, primary and secondary metrics, confidence intervals, locked-test outcomes,
generation evaluation, human evaluation, runtime, replay allocation, limitations, and representative examples.

## How future results should be interpreted

- If MFR consistently beats Random 10% across both orders and both seeds while meeting the two-point learning
  tolerance, the results support the hypothesis that most-forgotten selection uses a fixed replay budget more
  effectively than uniform random selection.
- If the aggregate confidence interval is also positive, the evidence becomes substantially stronger.
- If MFR beats Random 10% but Lowest margin retains better, the result is a retention/plasticity trade-off rather
  than proof that MFR is the universal best replay method.
- If Random 14.3% matches or beats MFR, replay quantity may matter more than targeting, although MFR may remain more
  data-efficient.
- If the sign changes across orders, the method is order-dependent.
- If the sign changes across seeds, the advantage is unstable.
- If preference metrics improve but generated and human-rated behavior does not, claims must remain limited to
  preservation of dataset preference margins.
- If validation, locked test, generation, and blinded review agree, the project can make its strongest claim: MFR
  reduces continual preference forgetting under a fixed replay budget and the improvement transfers to generated
  behavior.

## Guardrails for teammates and coding assistants

- Do not change scientific settings outside `configs/experiment_protocol.json`.
- Do not modify `data/v2/` or its manifest.
- Do not mix `pilot/` results with protocol-v2 results.
- Do not include incomplete runs without `COMPLETE.json`.
- Do not compare unmatched orders, seeds, budgets, or model revisions as if they were paired.
- Do not run Notebook 09 or inspect locked test outcomes before the validation analysis and model-selection decision
  are frozen.
- Do not claim that preference accuracy is real-world safety or helpfulness.
- Preserve per-prompt outputs, settings, code fingerprints, runtimes, and evaluator versions.
- Before changing code, check whether completed Drive runs depend on it and whether the change would invalidate a
  comparison or resume.

## Suggested reading order for a new teammate

1. `README.md`
2. `docs/TEAM_HANDOFF.md`
3. `docs/EXPERIMENT_PROTOCOL.md`
4. `docs/DATA_CARD.md`
5. `docs/RUNBOOK.md`
6. `notebooks/07_compare_runs.ipynb`
7. `notebooks/06_run_experiment.ipynb`

After reading these files, inspect the shared Drive run folders and confirm which configurations have `COMPLETE.json`
before scheduling or starting additional experiments.
