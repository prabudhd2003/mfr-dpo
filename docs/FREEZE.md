# Confirmation freeze (`freeze-v1`, tagged before any seed 2–4 run)

Frozen on: 2026-10-13, at the commit tagged `freeze-v1`. Nothing below changes after the tag.

## Methods
- **Final anchored method:** `dapr_weak` (`anchor_strength = 0.01`). Selected on Oct 9 by the frozen
  rule: passes the 2-point final-task rule (6/8 cells) and has the highest final three-behavior
  average (73.44). No further strengths are added for selection.
- **Qwen confirmation grid** (orders 1–6 × seeds 0–4): `none`, `random`, `random_high`, `mfr`,
  `lowest_margin`, `mir_dpo`, `cpmr`, `dapr_weak`. Joint reference: seeds 0–4.
- **Development ablations only (8 cells):** `fmcr`, `dapr`, `dapr_gated`, `dapr_c`, `copr_adapted`
  ("COPR-inspired adaptation"), LoRA-EWC λ ∈ {100, 1,000, 10,000} (dropped from confirmation:
  no setting beats no replay on the final average; see `docs/EWC_DECISION.md`). At-Risk / Balanced
  MFR were evaluated on the
  development cells and stay development ablations: neither beats MFR (Balanced MFR: retention
  −4.91 vs −4.88, final average 71.94 vs 71.98; At-Risk MFR: retention −5.69, final average 72.00,
  with 100% of its replay slots in the policy-failure tier, so the zero threshold does not select).
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

## Generation evaluation (`configs/evaluation_protocol.json` v1.3)
Greedy, max 512 new tokens, same prompts for every method, vLLM (dequantized NF4 base + LoRA;
parity with the Hugging Face path checked by `scripts/check_vllm_parity.py`: vLLM vs HF 64.0% shared
64-token prefix on 21 validation prompts, HF batch 8 vs HF batch 1 65.7%; passed, since greedy
decoding diverges as much inside HF itself).
- Safety: locked PKU prompts, XSTest (250 safe + 200 unsafe), HarmBench standard (200,
  DirectRequest) — graded by WildGuard (harm and refusal).
- IFEval (lm-eval, prompt-level strict).
- Helpful/Quality: reward-model judge `Skywork/Skywork-Reward-V2-Llama-3.1-8B` (pinned revision): each
  response is scored on its own and the higher reward wins, so there is no A/B position effect.
  Validated on the 400 validation pairs: 84.75% agreement [81.25, 88.25] (Helpful 84.5, Quality 85.0)
  against the 70% bar.
  Prometheus-2-7B was tried first and rejected (56% agreement; `rejected_judges` in the protocol).
- Second, independent judge for the same comparisons: `gemini-3.8-flash` on Vertex AI (run from
  `gemini_judge/`), both A/B orders, inconsistent = tie, same rubric. Chosen by the predeclared rule
  from three candidates on the 400 validation pairs: 3.8 Flash 80.75% [77.0, 84.5], 3.1 Pro preview
  79.5%, 2.5 Pro 77.0% (all ≥ 70%). Both judges are reported; the paper states where they disagree.
- Behavioral forgetting: post-stage vs final checkpoint (Safety via WildGuard; Helpful/Quality via
  the reward judge, 150 prompts).
- No human evaluation (stated in Limitations).
