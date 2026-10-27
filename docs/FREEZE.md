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
- Behavioral forgetting: post-stage vs final checkpoint (Safety via WildGuard; Helpful/Quality via
  the reward judge, 150 prompts).
- No human evaluation (stated in Limitations).

## Amendments after the tag (each made before any result it could affect)
1. **2026-10-14, Llama chat-template date pin.** `pin_chat_template_date` rejected the
   Llama-3.2 template because it also contains the test `strftime_now is defined`. The fix replaces
   every `strftime_now(...)` call as before and allows that test (both branches then use the fixed
   date). It changes nothing for Qwen, whose template has no date, and was made before any Llama run.
2. **2026-10-14, second judge dropped.** The Gemini second judge was removed for cost before any
   confirmation generation existed. Helpful/Quality comparisons use the validated reward-model judge
   only.
3. **2026-10-14, Llama length audit.** `validate_model_data.py` stopped on any frozen pair longer
   than 1,024 tokens under the Llama tokenizer. It now applies the plan's rule: stop only if more
   than 2% of a split is over the limit; otherwise the few long pairs are truncated at 1,024 tokens
   (as `make_batch` already does, identically for every method) and listed in
   `artifacts_llama32_3b/cache/tokenizer_validation.json`. The first audit found 8 such pairs in
   total. Made before any Llama training run.
4. **2026-10-27, buffer pairs with no response token.** Under the Llama tokenizer one frozen
   training pair (`quality-train-0873`, an Oriya-script prompt) has a 1,058-token prompt, so
   truncation at 1,024 tokens leaves no response token and its length-normalized margin is 0/0.
   Only the seed-2 replay buffer samples it, and only in orders 3-6 (where Quality enters the
   buffer), so those 4 Llama `none_s2` runs stopped with "peak margins missing". `update_buffer`
   now leaves out of the buffer any candidate whose prompt alone fills `max_tokens` and logs its id.
   Training on such pairs is unchanged. No completed run (Qwen or Llama) had such a pair in its
   buffer, so all completed runs are reproduced identically; only the 20 Llama seed-2 runs of
   orders 3-6 use the new code (the 4 partial `none_s2` runs were moved aside, not resumed).
   Made before any of those runs' results existed.
