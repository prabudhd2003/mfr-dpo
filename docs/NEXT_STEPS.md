# Method Development and Final Steps

## 1. Completed method: Counterfactual Projected-Margin Replay

CPMR is implemented as a preference-specific, interval-level interference method. At each replay refresh it:

1. scores every old buffer pair's current reference-relative DPO margin;
2. snapshots all trainable adapter parameters, AdamW state, learning-rate scheduler state, and CPU/GPU random state;
3. virtually trains on the exact upcoming shuffled new-task batches with no replay;
4. scores the old buffer again to obtain each projected margin;
5. restores the snapshot exactly, so the virtual pass cannot alter real training;
6. defines `worst_case_margin = min(current_margin, projected_margin)`;
7. ranks lower worst-case margins first, with larger predicted drop and historical drop as tie-breakers;
8. applies the same 10% budget and 0.75 per-behavior cap as the existing targeted methods.

Using the worse of the present and projected states is deliberate. It preserves Lowest Margin's protection of pairs
that are already weak while allowing an upcoming update to reveal additional endangered pairs. There is no fitted
mixing weight or failure threshold.

The closest conceptual predecessor is [Maximally Interfered Retrieval (MIR)](https://proceedings.neurips.cc/paper/2019/hash/15825aee15eb335cc13f9b559f166ee8-Abstract.html),
which retrieves memories whose loss is harmed by a foreseen update. CPMR should not be described as the first use of
a virtual update in continual learning. Its proposed contribution is preference-specific: an exact multi-step DPO
lookahead over one refresh interval, reference-relative preference margins rather than classification loss, and a
worst-present-or-projected rule that contains Lowest Margin as the no-interference case. [COPR](https://aclanthology.org/2025.findings-acl.281/)
is the closest published continual-preference context, but it uses policy regularization rather than this retrieval
rule. Any novelty claim must say **“to our knowledge”** and be checked again before submission.

CPMR has now been run for all four orders and both seeds. The completed validation result is:

| Method | Average retention change | Final current-task score | Final three-behavior average | Runtime |
|---|---:|---:|---:|---:|
| Random 10% | -5.50 | 73.62 | 71.71 | 24.28 min |
| Original MFR | -4.88 | 73.75 | 71.98 | 28.28 min |
| Lowest Margin | -4.47 | 74.31 | 72.56 | 28.59 min |
| CPMR | **-4.28** | **74.56** | **72.85** | 48.75 min |

CPMR has the best aggregate retention and final-average point estimates. Against equal-budget Random 10%, it
improved retention by 1.22 points with a paired 95% interval of `[0.50, 1.94]`, final-task accuracy by 0.94 points
with an interval of `[0.31, 1.56]`, and final average by 1.15 points with an interval of `[0.71, 1.56]`. These are
the strongest completed comparisons because every interval excludes zero.

Against Lowest Margin, CPMR improved retention by 0.19 points, final-task accuracy by 0.25 points, and final average
by 0.29 points. The respective intervals `[-0.56, 0.94]`, `[-0.12, 0.56]`, and `[-0.08, 0.69]` include zero.
Therefore CPMR beats Lowest Margin descriptively, but the current two-seed grid does not show a reliable difference.

Mechanistically, CPMR and Lowest Margin have a mean distinct-pair Jaccard overlap of 0.55 and chance-adjusted overlap
of 0.67. CPMR and Random 10% overlap by only 0.04. CPMR therefore often agrees with the current-difficulty baseline
while making a clearly targeted, non-random modification. The main drawback is computation: its reversible
lookahead raises mean runtime by about 20 minutes relative to Lowest Margin.

## 2. Finish Balanced MFR and At-Risk MFR

Next, implement, test, and run both methods for all four orders and both seeds.

## 3. Verify the run grid

There should be eight completed runs for each new method:

```text
4 orders × 2 seeds = 8 runs per method
```

Use `notebooks/01_carc_status.ipynb` and confirm that every expected run contains `COMPLETE.json`, `settings.json`,
`results.csv`, stage histories, replay logs, margins, buffers, and checkpoints. Confirm that all compared methods use
the same data manifest, model revision, common hyperparameters, order, seed, and replay budget.

Do not move to the locked test if any cell is missing or incompatible.

## 4. Extend the validation analysis

Notebook 07 already includes CPMR throughout the full grid. After Balanced MFR and At-Risk MFR are complete, add
them to:

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

Notebook 07 now reads CPMR's projected margins, predicted drops, observed next-refresh margins, selection overhead,
and overlap with Lowest Margin. Its `cpmr_refresh_*.csv` files preserve the full candidate ranking inputs at every
refresh.

The key comparisons are:

1. Balanced MFR vs original MFR — does equal behavior allocation help?
2. At-Risk MFR vs Balanced MFR — does actual current failure add useful information?
3. At-Risk MFR vs lowest margin — does historical decline add value beyond present difficulty?
4. Best MFR variant vs Random 10% — does targeted replay beat an equal-budget random baseline?
5. Best MFR variant vs Random 14.3% — is targeting more data-efficient than extra random replay?

## 5. Perform mechanism analysis

Notebook 07 now reports selection overlap, repeat concentration, behavior allocation, interval-to-interval
stability, and CPMR projection diagnostics. Extend the same analysis to each remaining targeted method and report:

- overlap between the pairs selected by MFR, Balanced MFR, At-Risk MFR, CPMR, and lowest margin;
- selected pairs' peak relative margin, current relative margin, historical drop, and current policy margin;
- replay slots and unique pairs per old behavior;
- how often each pair is repeated;
- for At-Risk MFR, the share of slots used by actual failures;
- whether replayed failed pairs recover after the next interval;
- whether one dataset or a small set of pairs dominates selection.

This analysis explains *why* a method wins or loses and is necessary for a meaningful report.

## 6. Freeze the development decision

Write the selection rule before looking at the locked test. A reasonable rule is:

1. A candidate must improve average retention over Random 10%.
2. Its final current-task score may not be more than two points below Random 10%.
3. Among candidates that pass, choose the highest final three-behavior average.
4. Use paired results across all order–seed cells and report uncertainty, not only the mean.
5. If methods are practically tied, prefer the simpler or faster method.

Record the chosen method, checkpoint rule, metrics, and exact Git commit. After this point, do not tune the method on
locked-test outcomes.

## 7. Completed ablation: Forecasted Margin-Crossing Replay

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

FMCR remains a useful ablation. It shows that extrapolating a short-horizon margin trajectory is not automatically
more effective than reacting to current difficulty. The report should include this result. CPMR then supplies the
positive forecasting result: simulating the actual upcoming DPO updates works substantially better than
extrapolating past score movement.

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

## 8. Confirm on unseen seeds

Seeds 0 and 1 have been used for development. After the remaining variants are evaluated and the final method is
selected, add unseen seeds such as 2, 3, and 4 to the protocol and run a confirmation grid. CPMR is the current
provisional winner. At minimum include:

- no replay;
- Random 10%;
- original MFR;
- lowest margin;
- the selected final method.

Run all four orders. Do not select a different method after seeing the confirmation results. More independent seeds
are more valuable now than increasing the current 2,000 training pairs.

## 9. Run the locked test once

After the method and statistical plan are frozen:

1. evaluate the chosen checkpoints on all three locked test datasets;
2. report retention and final performance using both normalized and summed metrics;
3. compare the predeclared methods with paired uncertainty;
4. preserve the untouched test outputs and exact code commit;
5. do not change the method because of the test result.

The locked test is for final confirmation, not method development.

## 10. Evaluate generated responses

Preference-pair accuracy does not prove that generated answers are helpful, safe, or high quality. Generate responses
from the final checkpoints using the frozen deterministic generation settings, then evaluate:

- instruction following;
- safety and refusal behavior on harmful prompts;
- general response quality;
- retention across behaviors learned earlier in the sequence;
- obvious failure cases and qualitative examples.

Use the same prompt sets and generation settings for every compared method.

## 11. Conduct blinded human evaluation

Create a small, balanced prompt sample across all three behaviors. Hide method names, randomize response order, and
ask reviewers to judge the relevant criteria. Record the rubric, reviewer agreement, ties, and uncertainty. Human
evaluation should support—not replace—the automatic and preference-pair metrics.

## 12. Complete the report and presentation

The final report should clearly separate:

- the pilot evidence from the final CARC study;
- validation development from locked-test confirmation;
- historical forgetting, present difficulty, behavior allocation, and replay budget;
- statistical evidence from practical effect size;
- pair-level preference metrics from generated-response quality;
- positive results from limitations or negative findings.

A defensible final contribution is a controlled study of which preference pairs should be replayed during continual
DPO. The current evidence supports a progression from historical forgetting, to present difficulty, to
counterfactual interference from the actual upcoming updates. CPMR is the strongest aggregate method so far, but
its advantage over Lowest Margin must be confirmed on unseen seeds and the locked test before it is presented as a
reliable win.

## Completion checklist

- [ ] Balanced MFR implemented, tested, and run in all eight cells
- [ ] At-Risk MFR implemented, tested, and run in all eight cells
- [x] CPMR implemented and tested
- [x] CPMR run and analyzed in all eight cells
- [x] FMCR run and analyzed in all eight cells
- [x] Current seven-method validation and CPMR mechanism analysis complete
- [ ] Balanced MFR and At-Risk MFR added to the full analysis
- [ ] Final method and statistical plan frozen
- [ ] Unseen-seed confirmation complete
- [ ] Locked test run once
- [ ] Generation and automatic evaluation complete
- [ ] Blinded human evaluation complete
- [ ] Error analysis complete
- [ ] Final figures, report, and presentation complete
