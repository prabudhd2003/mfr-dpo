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

## 5. Completed ablation: Forecasted Margin-Crossing Replay

**Forecasted Margin-Crossing Replay (FMCR)** was evaluated in all four orders and both seeds. It replays pairs
predicted to cross into failure at the next refresh rather than waiting until they have already failed.

FMCR tracks both the reference-relative margin and absolute policy margin, uses an exponential moving average of
their per-refresh changes, forecasts one refresh ahead, and prioritizes:

```text
velocity_t = 0.5 × velocity_(t-1) + 0.5 × (margin_t - margin_(t-1))
forecast_t = margin_t + velocity_t
```

The same calculation is applied separately to both margins. At the start of each new task the velocities reset to
zero, so a trend from the previous task is never projected into a different task.

1. pairs already in absolute preference failure;
2. pairs predicted to enter absolute failure;
3. pairs that already lost their relative DPO advantage;
4. pairs predicted to lose their relative DPO advantage;
5. largest historical drops as fallback;

with balanced quotas across old behaviors. The frozen settings were a velocity decay of `0.5` and a one-refresh
forecast horizon.

### FMCR result

| Method | Average retention change | Final current-task score | Final three-behavior average | Runtime |
|---|---:|---:|---:|---:|
| Random 10% | -5.50 | 73.62 | 71.71 | 24.28 min |
| Original MFR | -4.88 | 73.75 | 71.98 | 28.28 min |
| Lowest Margin | **-4.47** | 74.31 | **72.56** | 28.59 min |
| FMCR | -5.44 | **74.75** | 72.21 | 29.26 min |

FMCR did not beat Lowest Margin. Against Lowest Margin it retained 0.97 points less, with a paired 95% interval of
`[-1.56, -0.47]`, and its final three-behavior average was 0.35 points lower, with an interval of `[-0.67, -0.06]`.
Its 0.44-point final-task advantage was uncertain.

FMCR also did not improve retention over Random 10%: the difference was only 0.06 points and the interval included
zero. It did, however, improve final-task accuracy by 1.12 points and the final average by 0.50 points; both intervals
excluded zero. The correct interpretation is that this forecasting rule favored plasticity and new-task learning,
not that it solved retention better.

FMCR remains a useful ablation. It shows that anticipating a short-horizon boundary crossing is not automatically
more effective than reacting to current difficulty. The report should include this result, but original MFR remains
the main historical-forgetting method and Lowest Margin remains the baseline to beat.

Each completed FMCR stage saved `fmcr_refresh_*.csv` snapshots for the full buffer. Its replay log records the
selected pair's tier, current state, forecast, next observed state when available, forecast correctness, and
recovery. These files remain available for mechanism analysis. A selected pair that avoids a crossing after replay
does not by itself prove replay caused the recovery; method-level baseline comparisons supply the causal evidence.

### Research positioning

FMCR should be described as a preference-specific extension of prioritized replay, not as the first method to
anticipate forgetting in any continual-learning setting. [Maximally Interfered Retrieval](https://proceedings.neurips.cc/paper/2019/hash/15825aee15eb335cc13f9b559f166ee8-Abstract.html)
predicts which memories would be harmed by a virtual incoming update. [COPR](https://aclanthology.org/2025.findings-acl.281/)
studies continual preference learning through policy regularization. FMCR differs by forecasting the observed time
trajectory of two preference-pair margins and replaying before a meaningful preference boundary is crossed, without
a virtual gradient update. No exact prior method was found in the literature search, but the paper should still use
the careful wording **“to our knowledge”** and include a complete related-work review.

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
- [x] FMCR run and analyzed in all eight cells
- [ ] Full validation analysis and mechanism analysis complete
- [ ] Final method and statistical plan frozen
- [ ] Unseen-seed confirmation complete
- [ ] Locked test run once
- [ ] Generation and automatic evaluation complete
- [ ] Blinded human evaluation complete
- [ ] Error analysis complete
- [ ] Final figures, report, and presentation complete
