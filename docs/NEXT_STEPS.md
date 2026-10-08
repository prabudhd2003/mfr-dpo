# Next Steps After Balanced MFR and At-Risk MFR

This document begins after both methods are implemented, tested, and run for all four orders and both seeds.

## 1. Verify the run grid

There should be eight completed runs for each new method:

```text
4 orders × 2 seeds = 8 runs per method
```

Use `notebooks/01_carc_status.ipynb` and confirm that every expected run contains `COMPLETE.json`, `settings.json`,
`results.csv`, stage histories, replay logs, margins, buffers, and checkpoints. Confirm that all compared methods use
the same data manifest, model revision, common hyperparameters, order, seed, and replay budget.

Do not move to the locked test if any cell is missing or incompatible.

## 2. Extend the validation analysis

Run Notebook 07 once over the full grid. Add Balanced MFR and At-Risk MFR to:

- average retention change;
- final current-task accuracy;
- final average across all three behaviors;
- per-dataset forgetting for Helpful, Safe, and Quality;
- every order–seed cell;
- both `accuracy` and the required `accuracy_sum` metric;
- absolute policy accuracy and margin diagnostics;
- paired bootstrap intervals;
- runtime and scoring overhead;
- replay allocation, unique selected pairs, and repeat concentration.

The key comparisons are:

1. Balanced MFR vs original MFR — does equal behavior allocation help?
2. At-Risk MFR vs Balanced MFR — does actual current failure add useful information?
3. At-Risk MFR vs lowest margin — does historical decline add value beyond present difficulty?
4. Best MFR variant vs Random 10% — does targeted replay beat an equal-budget random baseline?
5. Best MFR variant vs Random 14.3% — is targeting more data-efficient than extra random replay?

## 3. Perform mechanism analysis

For each targeted method, report:

- overlap between the pairs selected by MFR, Balanced MFR, At-Risk MFR, and lowest margin;
- selected pairs' peak relative margin, current relative margin, historical drop, and current policy margin;
- replay slots and unique pairs per old behavior;
- how often each pair is repeated;
- for At-Risk MFR, the share of slots used by actual failures;
- whether replayed failed pairs recover after the next interval;
- whether one dataset or a small set of pairs dominates selection.

This analysis explains *why* a method wins or loses and is necessary for a meaningful report.

## 4. Freeze the development decision

Write the selection rule before looking at the locked test. A reasonable rule is:

1. A candidate must improve average retention over Random 10%.
2. Its final current-task score may not be more than two points below Random 10%.
3. Among candidates that pass, choose the highest final three-behavior average.
4. Use paired results across all order–seed cells and report uncertainty, not only the mean.
5. If methods are practically tied, prefer the simpler or faster method.

Record the chosen method, checkpoint rule, metrics, and exact Git commit. After this point, do not tune the method on
locked-test outcomes.

## 5. Decide whether to implement the forecasted method

If Balanced MFR or At-Risk MFR still cannot match lowest margin, the next research method is **Forecasted
Margin-Crossing Replay (FMCR)**: replay pairs predicted to cross into failure at the next refresh rather than waiting
until they have already failed.

FMCR would track both the reference-relative margin and absolute policy margin over time, estimate their recent
slopes, forecast the next value, and prioritize:

1. pairs already in absolute preference failure;
2. pairs predicted to enter absolute failure;
3. pairs predicted to lose their relative DPO advantage;
4. largest historical drops as fallback;

with balanced quotas across old behaviors.

FMCR is a separate method-design decision, not part of the current Balanced/At-Risk implementation. Implement it
only if the mechanism analysis supports the need and there is enough time to evaluate it properly.

## 6. Confirm on unseen seeds

Seeds 0 and 1 have been used for development. After selecting the final method, add unseen seeds such as 2, 3, and 4
to the protocol and run a confirmation grid. At minimum include:

- no replay;
- Random 10%;
- original MFR;
- lowest margin;
- the selected final method.

Run all four orders. Do not select a different method after seeing the confirmation results. More independent seeds
are more valuable now than increasing the current 2,000 training pairs.

## 7. Run the locked test once

After the method and statistical plan are frozen:

1. evaluate the chosen checkpoints on all three locked test datasets;
2. report retention and final performance using both normalized and summed metrics;
3. compare the predeclared methods with paired uncertainty;
4. preserve the untouched test outputs and exact code commit;
5. do not change the method because of the test result.

The locked test is for final confirmation, not method development.

## 8. Evaluate generated responses

Preference-pair accuracy does not prove that generated answers are helpful, safe, or high quality. Generate responses
from the final checkpoints using the frozen deterministic generation settings, then evaluate:

- instruction following;
- safety and refusal behavior on harmful prompts;
- general response quality;
- retention across behaviors learned earlier in the sequence;
- obvious failure cases and qualitative examples.

Use the same prompt sets and generation settings for every compared method.

## 9. Conduct blinded human evaluation

Create a small, balanced prompt sample across all three behaviors. Hide method names, randomize response order, and
ask reviewers to judge the relevant criteria. Record the rubric, reviewer agreement, ties, and uncertainty. Human
evaluation should support—not replace—the automatic and preference-pair metrics.

## 10. Complete the report and presentation

The final report should clearly separate:

- the pilot evidence from the final CARC study;
- validation development from locked-test confirmation;
- historical forgetting, present difficulty, behavior allocation, and replay budget;
- statistical evidence from practical effect size;
- pair-level preference metrics from generated-response quality;
- positive results from limitations or negative findings.

A defensible final contribution is a controlled study of which preference pairs should be replayed during continual
DPO. If an MFR variant wins, explain which signal made it work. If lowest margin remains strongest, the honest and
useful finding is that current difficulty is more effective than historical decline under this setup.

## Completion checklist

- [ ] Balanced MFR implemented, tested, and run in all eight cells
- [ ] At-Risk MFR implemented, tested, and run in all eight cells
- [ ] Full validation analysis and mechanism analysis complete
- [ ] Final method and statistical plan frozen
- [ ] Unseen-seed confirmation complete
- [ ] Locked test run once
- [ ] Generation and automatic evaluation complete
- [ ] Blinded human evaluation complete
- [ ] Error analysis complete
- [ ] Final figures, report, and presentation complete
