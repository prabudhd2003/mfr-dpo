# LoRA-EWC: dropped from the confirmation grid (decided 2026-10-10)

LoRA-EWC (diagonal empirical Fisher over the trainable LoRA parameters, one anchor per learned
behavior, rule `multi_anchor_diagonal_empirical_fisher_lora_only_v1`) was swept twice on all 8
development cells (orders 1–4 × seeds 0–1, validation set): first at λ = 0.1, 1 and 10, then at
λ = 100, 1,000 and 10,000. Together the two sweeps cover five orders of magnitude. It is
**not** carried into the confirmation grid, the locked test, the generation evaluation or the Llama
transfer. The development results are reported in the paper's appendix.

## Results (8 development cells each)

| Method | Old-behavior change | Final task | Final 3-behavior avg | Median weighted EWC / DPO loss (stages 2–3) |
|---|---:|---:|---:|---:|
| No replay | −7.94 | 74.81 | 70.77 | — |
| LoRA-EWC λ=0.1 (sweep 1) | −7.84 | 74.50 | 70.73 | ≈ 0.001%–0.11% across sweep 1 |
| LoRA-EWC λ=1 (sweep 1) | −7.97 | 74.75 | 70.77 | (same audit) |
| LoRA-EWC λ=10 (sweep 1) | −7.78 | 74.69 | 70.88 | (same audit) |
| LoRA-EWC λ=100 | −7.88 | 74.81 | 70.85 | 0.0039 / 0.497 ≈ 1% |
| LoRA-EWC λ=1,000 | −7.59 | 74.25 | 70.77 | 0.032 / 0.497 ≈ 6% |
| LoRA-EWC λ=10,000 | −7.25 | 73.06 | 70.31 | 0.139 / 0.498 ≈ 28% |
| Random replay 10% (reference) | −5.50 | 73.62 | 71.71 | — |

Safety forgetting, the largest effect in the study, barely moves: −13.25 (no replay) vs −12.75
(λ=10,000).

## Why it was dropped

1. **The penalty was active, not too small.** At λ=10,000 the weighted EWC term is about 28% of the
   DPO loss, and the raw EWC loss shrinks as λ grows (4e-5 → 1e-5), so the parameters are being
   held. Under-scaling is ruled out.
2. **It trades plasticity for almost no retention.** λ=10,000 recovers 0.69 points of forgetting
   (under 9% of what no replay loses) and costs 1.75 points of new-task accuracy. The final average
   falls monotonically with λ (70.85 → 70.77 → 70.31). A larger λ moves further along the same
   curve, so it cannot produce a better setting.
3. **The predeclared selection rule picks λ=100**, which is indistinguishable from no replay
   (−7.88 vs −7.94). Confirming it on 30 more cells would only re-measure no replay.

### Sweep 1 (λ = 0.1, 1, 10)

All three settings behaved like no replay: retention changed by −0.03 to +0.16 points and the final
average by −0.04 to +0.10. The rule picked λ = 10. A loss-scale audit then showed that the weighted
EWC penalty was only about 0.001%–0.11% of the DPO loss, so it could not have done anything. That
is why sweep 2 moved up by two orders of magnitude. Sweep 1 is recorded in
`git show 0a14ca2:docs/NEXT_STEPS.md` (§3) and `git show 0a14ca2:README.md`.

### Across both sweeps

Over λ = 0.1 → 10,000 (penalty from about 0.001% to 28% of the DPO loss), retention moves from
−7.97 to −7.25 against −7.94 for no replay. No coefficient improves the final average by more than
0.11 points, and the strongest one lowers it by 0.46.

## Paper wording (appendix + one sentence in the main text)

> We also evaluated LoRA-EWC (diagonal Fisher over the LoRA parameters, one anchor per behavior) at
> λ ∈ {10⁻¹, 1, 10, 10², 10³, 10⁴} on the development cells. Although the penalty reached 28% of the
> DPO loss at λ = 10⁴, it recovered at most 0.7 points of forgetting (vs 2.4 for 10% random replay) while
> reducing new-task accuracy by 1.8 points, and no coefficient improved the final three-behavior
> average over no replay. We therefore did not carry it into the confirmation grid.

Raw numbers: sweep 2 in `notebooks/07a_results.ipynb` §6, §7 and §11, with histories under
`runs/v2_o[1-4]_ewc_*_s[01]/stage*/history.csv`; sweep 1 in commit `0a14ca2` (`docs/NEXT_STEPS.md`,
`README.md`, `docs/CARC.md`).
