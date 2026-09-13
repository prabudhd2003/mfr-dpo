# MFR-DPO v2 runbook

This is the operational checklist. Check the protocol and data manifest into Git before starting the grid, and record the commit used by every run.

First commit and push the v2 code in this repository; the Colab notebooks pull from GitHub. Local uncommitted
changes are not visible to Colab.

## 1. One-time setup

Run notebooks in this order:

1. `01_load_data.ipynb` — creates `data/v2` from pinned dataset revisions. Run it locally when practical; if
   you run it in Colab, copy the generated folder back into this repository before the runtime is deleted.
2. `02_data_review.ipynb` — must show 2,000/200/300 rows per behavior and no validation failure.
3. Commit and push `data/v2`, its manifest, the protocol, source code, and notebooks.
4. `05_build_reference_cache.ipynb` — creates `Drive/.../cache/reference_v2.csv` and its manifest.

The cache is an optimization, not a different algorithm. If it is missing, notebook 06 computes frozen-base probabilities live and remains correct but slower.

## 2. Core experiment grid

Run all 12 configurations below. Start with `none` for each order/seed because its stage-1 artifact is reused by the two replay methods.

| Order | Seed | Method | Run name | Stage-1 source |
|---:|---:|---|---|---|
| 1 | 0 | none | `v2_o1_none_s0` | `None` |
| 1 | 0 | random | `v2_o1_random_s0` | `.../runs/v2_o1_none_s0` |
| 1 | 0 | mfr | `v2_o1_mfr_s0` | `.../runs/v2_o1_none_s0` |
| 1 | 1 | none | `v2_o1_none_s1` | `None` |
| 1 | 1 | random | `v2_o1_random_s1` | `.../runs/v2_o1_none_s1` |
| 1 | 1 | mfr | `v2_o1_mfr_s1` | `.../runs/v2_o1_none_s1` |
| 2 | 0 | none | `v2_o2_none_s0` | `None` |
| 2 | 0 | random | `v2_o2_random_s0` | `.../runs/v2_o2_none_s0` |
| 2 | 0 | mfr | `v2_o2_mfr_s0` | `.../runs/v2_o2_none_s0` |
| 2 | 1 | none | `v2_o2_none_s1` | `None` |
| 2 | 1 | random | `v2_o2_random_s1` | `.../runs/v2_o2_none_s1` |
| 2 | 1 | mfr | `v2_o2_mfr_s1` | `.../runs/v2_o2_none_s1` |

Optionally add four `lowest_margin` runs after the core grid. They use the same matching stage-1 sources.

For each run, change only `ORDER_ID`, `METHOD`, `SEED`, `START_STAGE`, and `STAGE1_FROM` in notebook 06. All scientific settings come from the protocol file.

## 3. Resume safely

If stage 1 finished and stage 2 did not, set `START_STAGE = 2`. If stage 2 finished and stage 3 did not, set `START_STAGE = 3`. Use the same order, method, seed, Drive folder, Git commit, data, and cache. The runner loads the preceding adapter, buffer, and result rows.

A complete run has `COMPLETE.json`. Do not include partial runs in final tables.

The runner rejects an uncommitted checkout, a different commit/settings combination on resume, a completed
run that would be overwritten, altered data files, and a reference cache that fails its live numerical canary.

## 4. Validation analysis and decision

Run notebooks 07 and 08 after the core grid. Report both normalized and summed metrics.

The main question is whether MFR improves retention relative to random replay across matched order/seed cells while keeping current-stage validation accuracy within the predeclared 2-point tolerance. Use per-pair paired bootstrap intervals; do not treat 600 pooled validation pairs as independent of order and seed.

Write the model-selection decision and chosen runs into the report notes before opening test outcomes.

## 5. Final evaluation

1. Run notebook 09 once on the selected completed run(s).
2. Run notebook 10 with fixed greedy decoding for matched no-replay, random, and MFR checkpoints.
3. Apply versioned automatic evaluators appropriate to each behavior and save per-prompt outputs.
4. Run notebook 11 to sample 100 prompts and create a blinded three-system review sheet.
5. Give reviewers only the blind sheet. Keep `PRIVATE_unblinding_key.csv` hidden until ratings are complete.
6. Report preference outcomes, rubric means, inter-rater agreement when two reviewers are available, failures, and representative examples.

Test results are for final reporting, never for another hyperparameter or method-selection round.
