# Extension protocol (post-freeze, written before any extension run)

Written 2026-11-02, after look 1 of the main study (`docs/FREEZE.md`) and before any run below.
Everything here is fixed in advance. Results are reported as a labeled extension, separate from the
main confirmation study.

## Why
Two questions remain after the main study:
1. **Is DAPR just Dark Experience Replay (DER/DER++)?** DER++ also replays old examples while holding
   their stored outputs in place, but in **both** directions. DAPR only blocks the harmful direction:
   the chosen answer becoming less likely, or the rejected answer becoming more likely.
2. **Does DAPR hold at a larger scale and over a longer task sequence?**

## Part A. Symmetric anchors on Qwen2.5-1.5B (main protocol, `configs/experiment_protocol.json`)

### New methods
All methods use the same 10% replay budget, 500-pair buffer, refresh schedule, data and Stage-1
checkpoints as the main study.

| Code name | Replay selection | Anchor | Strength λ |
|---|---|---|---|
| `der_pp` | random (identical to Random 10%) | symmetric (DER++-style) | 0.1 |
| `der_pp_weak` | random | symmetric | 0.01 |
| `dapr_sym` | lowest margin (identical to DAPR) | symmetric | 0.1 |
| `dapr_sym_weak` | lowest margin | symmetric | 0.01 |
| `dapr_lambda_0p003` | lowest margin | directional (DAPR) | 0.003 |
| `dapr_lambda_0p03` | lowest margin | directional (DAPR) | 0.03 |

The symmetric anchor is DAPR's anchor without the one-sided step: the Huber penalty applies to any
change of a replayed pair's token log-probabilities from their stored peak values, in either
direction. The anchor is otherwise computed exactly as in DAPR (same peak capture, Huber δ = 1, mean
over the replayed pairs of a micro-batch).

### A1. Development selection (validation, 8 development cells)
- Cells: orders 1–4 × seeds 0–1.
- Runs: `der_pp`, `der_pp_weak`, `dapr_sym`, `dapr_sym_weak`.
- Rule: the same rule that chose DAPR's strength (`mfr_stats.select_variant`). Among the two
  strengths of a method, keep those whose mean final-task score is at most 2 points below Random 10%,
  then pick the higher mean final three-behavior average. One DER++ variant and one Symmetric-DAPR
  variant are chosen this way, so both baselines get the same tuning budget DAPR got.

### A2. Confirmation (locked test, 30 cells)
- The two chosen variants run on the remaining 22 cells (orders 5–6 × seeds 0–1, orders 1–6 × seeds
  2–4), giving 30 cells each.
- Locked test at every stage, as in the main study.

### Predeclared tests (seeds 0–4, one look, Holm at α = 0.05; `mfr_stats.extension_analysis`)
| Test | Comparison | Metric |
|---|---|---|
| E1 | DAPR (`dapr_weak`) − chosen DER++ variant | final three-behavior average |
| E2 | DAPR (`dapr_weak`) − chosen Symmetric-DAPR variant | retention |

Same estimator as P1–P3: a one-sample t-test on per-seed mean differences, with a seed-block
bootstrap as a sensitivity check. There is no second look.

### A3. Anchor-strength sensitivity (descriptive, validation, 8 development cells)
`dapr_lambda_0p003` and `dapr_lambda_0p03` run on the development cells. Together with the existing
λ = 0.1 (`dapr`) and λ = 0.01 (`dapr_weak`) runs, they give a four-point sensitivity curve. No test,
no selection: DAPR's strength stays 0.01.

### A4. Replay budget sensitivity (descriptive, validation, 8 development cells)
Does DAPR's advantage over Random 10% depend on the replay budget? Random and DAPR (`dapr_weak`) each
run at four extra budgets on the development cells:

| Code names | Change from the main setting |
|---|---|
| `random_old1`, `dapr_weak_old1` | 1 replayed pair per step (18 new + 1, about 5% replay) |
| `random_old4`, `dapr_weak_old4` | 4 replayed pairs per step (18 new + 4, about 18% replay) |
| `random_buf250`, `dapr_weak_buf250` | 250-pair buffer |
| `random_buf1000`, `dapr_weak_buf1000` | 1,000-pair buffer |

- **Other settings:** unchanged. The DAPR variants keep λ = 0.01.
- **Stage 1:** the buffer is filled only after Stage 1, so Stage 1 does not depend on the buffer size.
  The buffer-size variants reuse the `none` Stage-1 checkpoint. They then refill the Stage-1 buffer at
  their own size, from that checkpoint, with the usual seeded rule. The 250-pair candidates are the
  first 250 of the 500-pair ones.
- **Reporting:** DAPR − Random at each budget (final average, retention, final task), described per
  cell. No test, no selection.

## Part B. Qwen2.5-7B-Instruct, five tasks (`configs/qwen7b_5task_protocol.json`)

### Data (`data/v3`, built by `scripts/prepare_data_v3.py`)
- **Original three tasks:** the nine v2 files are copied byte-identical (same hashes), so the first
  three tasks are exactly the main study's data. The Qwen2.5 models share one tokenizer, and the
  script checks that every v2 pair still fits in 1,024 tokens.
- **summarize:** Reddit TL;DR summary comparisons (Stiennon et al., 2020), from the
  `CarperAI/openai_summarize_comparisons` copy. The official `openai/summarize_from_feedback` loader
  is a dataset script that current `datasets` versions cannot run. All published splits are pooled.
  Each prompt gets a fixed instruction ("Write a short TL;DR summary of the following Reddit post.")
  in place of the trailing "TL;DR:" cue.
- **code:** `coseal/CodeUltraFeedback_binarized` (LLM-judge ratings); pairs with a rating gap of at
  least 1.
- **Filtering:**
  - One pair per prompt.
  - No prompt shared with v2 or with the other new task.
  - Every pair at most 1,024 tokens.
  - Seed 0, 2,000 / 200 / 300 pairs.
  - Dataset and model revisions are pinned in the protocol and the manifest.

### Design
- **Orders (fixed):**

  | Order | Sequence | First three stages = |
  |---|---|---|
  | 1 | helpful → safe → quality → summarize → code | main-study order 1 |
  | 2 | quality → safe → helpful → code → summarize | main-study order 4 |
  | 3 | code → summarize → safe → helpful → quality | new tasks first |

- **Methods:** none, random, lowest margin, DAPR (`dapr_weak`), and the DER++ variant chosen in A1.
- **Seeds:** 0–2 (9 cells per method). With too little time, seeds 0–1 (6 cells); decide before the
  grid starts, not after seeing results.
- **Settings:** all as in the main study (QLoRA r16/α32, β 0.1, lr 1e-4, 18 + 2 per step, 500-pair
  buffer split evenly over learned tasks, 5 refreshes, cap 0.75).
- **Pilot first:** order 1, seed 0, `none` + `lowest_margin`, to measure time and memory. The pilot's
  runs count as grid runs.

### Reporting (descriptive, like the Llama replication)
- **Locked test at every stage.**
- **Stage 5:** retention over the first four tasks, final-task accuracy, and the final five-task
  average.
- **Stage-3 prefix of orders 1–2** (`mfr_stats.cell_metrics(..., upto_stage=3)`): a 3-task Qwen-7B
  result on exactly the main study's tasks, for the 1.5B → 3B → 7B comparison.
- **Per-cell direction** (DAPR vs each baseline) with a sign test.

## Not changed
- **Main-study runs and results.** The code changes add methods and support for more than three
  tasks. Every completed run's settings, Stage-1 fingerprint and results are unchanged.
- **New runs on Qwen-1.5B** borrow Stage 1 from the existing `none` runs, as before.
