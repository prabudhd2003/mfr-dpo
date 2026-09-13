# Experiment protocol v2.0

This is the frozen protocol for final `mfr-dpo` runs. The older notebooks and Drive folders are preliminary.

## Question

At the same replay budget, does Most-Forgotten Replay (MFR) retain earlier preference behavior better than
uniform random replay while keeping new-behavior learning within two accuracy points?

## Core comparison

- Model: `Qwen/Qwen2.5-1.5B-Instruct`, pinned by the revision in `configs/experiment_protocol.json`.
- Data: version 2 splits in `data/v2/`, 2,000 train / 200 validation / 300 test pairs per behavior.
- Orders: helpful -> safe -> quality and safe -> helpful -> quality.
- Core methods: no replay, 10% random replay, and 10% MFR. `random_high` is a secondary
  retention/plasticity control using 18 new plus 3 uniformly sampled old pairs (14.3% replay).
  Lowest-current-margin replay remains optional and secondary.
- Seeds: 0 and 1. Stage 1 is shared only after its complete configuration and file hashes are validated.
- Training: one epoch, 18 new pairs plus 2 old pairs per full optimizer step, beta 0.1, learning rate 1e-4.
  A final partial batch gets proportionally fewer old pairs, making the aggregate replay share as close to 10%
  as integer examples permit.
- Replay: 500 stored pairs, five selection intervals per later stage, at most 75% from one old behavior when
  enough other behavior data exists.

The runner refuses dirty Git checkouts, preserves original Git provenance across resumes, verifies every JSONL
against the data manifest, and checks a 32-pair cached-reference sample against live computation before training.
Checkpoint compatibility uses a SHA-256 fingerprint of the executable training/scoring code rather than the
whole Git commit, so documentation or notebook-only commits do not invalidate a scientifically identical stage 1.

## Metrics

The primary pair score is `accuracy`: the percentage of held-out pairs whose length-normalized DPO advantage
relative to the frozen base is positive. It measures movement toward the chosen response; it is not the percentage
of generated responses that are objectively safe or helpful. `accuracy_sum`, the standard summed DPO score, is a
required secondary metric.

For an old behavior, retention change is final accuracy minus accuracy immediately after learning it. For a newly
trained behavior, new-stage gain is post-stage accuracy minus pre-stage accuracy. MFR is not considered a success
if its new-stage gain is more than two points below random replay.

All primary comparisons are paired by order, seed, validation pair, data version, model revision, and update
budget. Report every order-seed cell and paired bootstrap confidence intervals. Do not average margin-retention
ratios across behaviors.

The `random_high` comparison is not part of the equal-budget primary claim. It tests whether 10% MFR can match or
beat a more conservative random baseline that receives 3 old pairs per full step. Report its extra examples and
runtime explicitly.

## Test lock and generation

Use validation data for development and error analysis. Do not run `09_final_test_eval.ipynb` until the run set,
metrics, and analysis code are frozen. Generated behavior is checked using the base, the checkpoint immediately
after a behavior is learned, and the final checkpoint. IFEval is secondary; safety and blinded response review are
required because pair scores alone are not behavioral accuracy.
