# Confirmation freeze (draft — commit and tag `freeze-v1` BEFORE any seed 2–4 run)

Frozen on: ____ (commit: ____). Nothing below changes after the tag, except the EWC λ
(`freeze-v2`, chosen on the 8 development cells with the frozen rule in `docs/CARC.md`).

## Methods
- **Final anchored method:** `dapr_weak` (`anchor_strength = 0.01`). Selected on Oct 9 by the frozen
  rule: passes the 2-point final-task rule (6/8 cells) and has the highest final three-behavior
  average (73.44). No further strengths are added for selection.
- **Qwen confirmation grid** (orders 1–6 × seeds 0–4): `none`, `random`, `random_high`, `mfr`,
  `lowest_margin`, `mir_dpo`, `cpmr`, `dapr_weak`, selected `ewc_*`. Joint reference: seeds 0–4.
- **Development ablations only (8 cells):** `fmcr`, `dapr`, `dapr_gated`, `dapr_c`, `copr_adapted`
  ("COPR-inspired adaptation"), unselected `ewc_*`. At-Risk / Balanced MFR join the grid only if
  they finish before this tag and clearly beat MFR on the development cells.
- **Llama-3.2-3B transfer** (orders 1–6 × seeds 0–2): `none`, `random`, `lowest_margin`, `mfr`,
  `dapr_weak`; joint seeds 0–2. Same hyperparameters as Qwen, no retuning. Chat-template date
  pinned to "26 Jul 2024".

## Hypotheses
H1 sequential DPO forgets, Safety most. H2 targeted selection > random; current difficulty ≥
history; lookahead adds little for its cost. H3 the anchored replay objective moves the
retention–plasticity frontier more than any selection rule. H4 margin forgetting corresponds to
behavioral forgetting. H5 H1–H3 replicate on Llama-3.2-3B.

## Metrics (locked test, `scripts/final_test.py`, every stage checkpoint)
Primary: preference accuracy (normalized margin > 0). Per run: retention (mean change of earlier
behaviors), final-task accuracy, final three-behavior average. Also `accuracy_sum`, margins,
worst-behavior retention.

## Statistics (`src/mfr_stats.py`)
- Unit: order–seed cell; d = target − baseline. Primary test: one-sample t-test on per-seed means
  (each averaged over orders), df = seeds − 1. Sensitivity: seed-block bootstrap, per-order means.
- Primary family (Holm): **P1** dapr_weak − random (final average); **P2** dapr_weak −
  lowest_margin (final average); **P3** mfr − random (retention, seeds 0–4).
- Non-inferiority: one-sided 95% lower bound of dapr_weak − random on final-task > −2.
- **Two-look rule (O'Brien–Fleming):** look 1 after seeds 0–4. Stop if P1 and P2 both have
  p < 0.005 (Holm at 0.05). Otherwise run seeds 5–9 for `none`, `random`, `lowest_margin`,
  `dapr_weak` (120 runs) and test P1, P2 on 10 seeds; Holm at 0.048. No futility stopping; no
  look-1 result changes methods, metrics or analysis. Both looks are reported.
- Everything else is exploratory. Llama: report direction and per-cell sign agreement.

## Generation evaluation (`configs/evaluation_protocol.json` v1.1)
Greedy, max 512 new tokens, batch 32, same prompts for every method.
- Safety: locked PKU prompts, XSTest (250 safe + 200 unsafe), HarmBench standard (200,
  DirectRequest) — graded by WildGuard (harm and refusal).
- IFEval (lm-eval, prompt-level strict).
- Helpful/Quality: Prometheus-2-7B pairwise, no reference, both A/B orders, inconsistent = tie.
  Used only if `scripts/validate_judge.py` agreement with human labels (validation pairs) ≥ 70%;
  otherwise a pinned API judge, validated the same way.
- Behavioral forgetting: post-stage vs final checkpoint (Safety via WildGuard; Helpful/Quality via
  the judge, 150 prompts).
- No human evaluation (stated in Limitations).
