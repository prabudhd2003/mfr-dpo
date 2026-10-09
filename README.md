# MFR-DPO

**What Should an LLM Rehearse? Budgeted Replay for Continual Preference Tuning**

## Project summary

This project studies catastrophic forgetting during sequential preference tuning. One QLoRA adapter on
`Qwen/Qwen2.5-1.5B-Instruct` learns three behaviors—Helpfulness, Safety, and general response Quality—one after
another with Direct Preference Optimization (DPO).

DPO learns from a prompt, a preferred response, and a rejected response. After the model learns one behavior,
training it on a later behavior can weaken what it learned earlier. This is catastrophic forgetting.

Most-Forgotten Replay (MFR) keeps a small memory of old preference pairs. It records how strongly each pair was
learned and later measures how much its reference-relative preference margin has fallen. The pairs with the largest
drops are replayed during later training. The main question is whether this targeted replay protects earlier
behaviors better than random replay using the same 10% replay budget, without harming the behavior currently being
learned.

## What has been completed

- A pilot study confirmed that sequential DPO causes measurable and concentrated forgetting.
- Version 2 of the data was cleaned, deduplicated across all datasets and splits, length-filtered, hashed, and frozen.
- The CARC training pipeline, reference cache, resume logic, Stage-1 reuse, progress logs, replay logs, and automated
  tests are implemented.
- All seven methods were run for four task orders and two seeds on NVIDIA L40S GPUs: **56 completed runs**.
- Notebook 07 reports the complete validation grid, paired uncertainty, per-dataset forgetting, replay-selection
  overlap, concentration, final scores, and runtime.

The seven completed methods are:

| Method | Meaning |
|---|---|
| No replay | Train only on the current behavior. |
| Random 10% | Fill 10% of each later batch with random old pairs. |
| MFR 10% | Use the same budget for pairs with the largest historical margin drop. |
| Random 14.3% | Use more random replay to test whether budget alone explains the result. |
| Lowest margin | Replay pairs with the lowest current reference-relative margin. |
| FMCR 10% | Forecast which pairs may cross a preference-failure boundary before the next refresh. |
| CPMR 10% | Virtually train on the upcoming new-task interval and replay pairs with the lowest present-or-projected margin. |

Forecasted Margin-Crossing Replay (FMCR) is a completed forecasting ablation. **Counterfactual Projected-Margin
Replay (CPMR)** is the strongest method by aggregate point estimate. CPMR briefly simulates the exact upcoming
new-task interval without replay, measures each old pair before and after that reversible lookahead, and replays the
pairs with the lowest worst-case margin across the present and projected states. It uses the same 10% replay budget
and 0.75 per-behavior cap as Lowest Margin. The virtual pass restores the adapter, optimizer, learning-rate schedule,
and random state before real training continues.

Balanced MFR and At-Risk MFR are planned controlled ablations and are **not implemented or evaluated yet**.

Six additional controlled methods are implemented as secondary comparisons or development ablations:

| Method | Fixed-budget test |
|---|---|
| DAPR-Strong 10% (`dapr`) | Uses Lowest Margin retrieval, then one-sided token anchors with strength 0.1 to stop a replayed chosen response from weakening or a rejected response from strengthening beyond its peak-time state. |
| DAPR-Weak 10% (`dapr_weak`) | Changes only DAPR's anchor strength from 0.1 to 0.01. |
| DAPR-Gated 10% (`dapr_gated`) | Keeps strength 0.1 but applies the anchor only when an eval-mode, per-step live margin is below the pair's stored peak margin. |
| DAPR-C 10% | Centers DAPR on the pair's common likelihood shift, so it constrains preference direction rather than penalizing both responses merely becoming more or less likely. |
| MIR-DPO 10% | Performs one reversible incoming-task update and replays old pairs whose DPO loss is predicted to increase most. This is the preference-learning adaptation of MIR. |
| COPR-adapted 10% | Uses Lowest Margin retrieval and constrains the replayed pair's two-response policy distribution toward its peak-time distribution. It is a controlled COPR-inspired adaptation, not a reproduction of full COPR. |

All six use exactly the same 10% replay budget, buffer, refresh schedule, behavior cap, data, and Stage-1 checkpoint
as the established methods. Every DAPR variant, DAPR-C, and COPR-adapted saves auditable peak anchors in
`preference_anchors.npz`; MIR-DPO saves each virtual-loss snapshot in `mir_dpo_refresh_*.csv`. DAPR-Weak and
DAPR-Gated should be described as **implemented, not evaluated** until their eight-cell grids are complete.

## Data

| Behavior | Dataset | Train | Validation | Locked test | Example type |
|---|---|---:|---:|---:|---|
| Helpful | NVIDIA HelpSteer2 | 2,000 | 200 | 300 | Travel planning, explanations, and instruction following. |
| Safe | PKU-SafeRLHF | 2,000 | 200 | 300 | Unsafe or illegal requests where the safer response is preferred. |
| Quality | UltraFeedback Binarized | 2,000 | 200 | 300 | Writing, reasoning, and general question-answering preferences. |

The locked test split has not been used for method development. `data/v2/manifest.json` records the pinned source and
model revisions, row counts, and file hashes.

## Experiment design

Every method uses the same data, model revision, QLoRA settings, DPO settings, task order, seed, and Stage-1 starting
checkpoint. The current grid uses:

| Order | Training sequence |
|---:|---|
| 1 | Helpful → Safe → Quality |
| 2 | Safe → Helpful → Quality |
| 3 | Quality → Helpful → Safe |
| 4 | Quality → Safe → Helpful |

Seeds 0 and 1 are used for each order. Orders 3 and 4 make Quality an earlier behavior so its forgetting can also be
measured.

The primary `accuracy` is the percentage of preference pairs with a positive length-normalized DPO advantage over
the frozen base model. It measures whether training moved the model in the preferred direction; it is not the same
as real-world factual accuracy. For an earlier behavior:

```text
retention change = final validation accuracy - accuracy immediately after learning that behavior
```

A value near zero is better. A negative value means the model forgot some of that behavior.

## Current validation results

The following table averages all eight order–seed cells. “Average forgetting” is the change in an earlier behavior
between the point when it was learned and the end of training. Less negative is better. All changes in the table are
**percentage-point changes**, not relative percentages.

| Method | Average forgetting | Forgetting prevented vs no replay | Final average accuracy | Change vs no replay |
|---|---:|---:|---:|---:|
| No replay | -7.94 | — | 70.77 | — |
| Random 10% | -5.50 | about **31%** | 71.71 | +2.44 retention, +0.94 final |
| MFR 10% | -4.88 | about **39%** | 71.98 | +3.06 retention, +1.21 final |
| Random 14.3% | -5.47 | about **31%** | 71.73 | +2.47 retention, +0.96 final |
| Lowest margin | -4.47 | about **44%** | 72.56 | +3.47 retention, +1.79 final |
| FMCR 10% | -5.44 | about **31%** | 72.21 | +2.50 retention, +1.44 final |
| CPMR 10% | **-4.28** | about **46%** | **72.85** | **+3.66 retention, +2.08 final** |

“Forgetting prevented” expresses the improvement relative to the 7.94-point forgetting observed without replay.
For example, MFR recovers 3.06 of those 7.94 points, so it prevents approximately `3.06 / 7.94 = 39%` of the
measured forgetting. It does **not** mean that MFR raises total model accuracy by 39%.

In simple terms:

- Random 10% prevents approximately **31% of the forgetting**.
- MFR 10% prevents approximately **39% of the forgetting**.
- Random 14.3% also prevents approximately **31% of the forgetting** despite using more replay examples.
- Lowest Margin prevents approximately **44% of the forgetting**.
- FMCR prevents approximately **31% of the forgetting** and favors final-task learning more than retention.
- CPMR prevents approximately **46% of the forgetting**, the strongest aggregate point estimate.

For additional context, final current-task accuracy was 74.81 for No Replay, 73.62 for Random 10%, 73.75 for MFR,
73.88 for Random 14.3%, 74.31 for Lowest Margin, 74.75 for FMCR, and 74.56 for CPMR. CPMR therefore recovered
the most retention while remaining only 0.25 points below No Replay on the final task. Average runtime was 22.48
minutes for No Replay, 28.59 for Lowest Margin, and 48.75 for CPMR. CPMR's reversible lookahead makes it the most
computationally expensive method.

### Is this size of improvement normal?

Yes. In continual-learning research, a replay method may produce a modest change in final average accuracy while
producing a clearer reduction in forgetting. The closest comparisons below do **not** use our exact combination of
datasets, model, or protocol, so their raw scores should not be treated as direct benchmarks for this project. They
show how replay improvements are normally interpreted.

- [Maximally Interfered Retrieval (MIR), NeurIPS 2019](https://proceedings.neurips.cc/paper/2019/file/15825aee15eb335cc13f9b559f166ee8-Paper.pdf)
  compared targeted selection with ordinary experience replay. On MiniImageNet, targeted replay increased final
  accuracy from 24.7 to 25.2—only 0.5 points—but reduced forgetting from 23.5 to 18.0, a 5.5-point improvement. The
  paper therefore described accuracy as only slightly better while forgetting improved substantially.
- [Leitner-Guided Memory Replay, NAACL 2024](https://aclanthology.org/2024.naacl-long.432/) found that selecting
  informative replay examples consistently reduced forgetting while maintaining accuracy across tasks, task orders,
  and languages. This supports evaluating retention separately from final accuracy.
- [SEEKR, EMNLP 2024](https://aclanthology.org/2024.emnlp-main.190/) achieved comparable or better continual-learning
  performance with one-tenth of the replay data used by comparison methods and reduced replay to 1%. This shows that
  replay efficiency can itself be a meaningful result even when the absolute accuracy difference is small.
- [COPR, Findings of ACL 2025](https://aclanthology.org/2025.findings-acl.281.pdf) studies continual human-preference
  learning more directly. It uses 5% historical data for experience-replay baselines and reports average performance,
  backward transfer, and forgetting as separate quantities rather than judging a method from final preference score
  alone.

Our results follow the same general pattern: replay changes final average accuracy by roughly 1–2 points, but it
prevents 31–46% of the forgetting measured without replay. MFR's 0.62-point retention advantage over equal-budget
Random 10% is modest, but its paired 95% bootstrap interval, `[0.22, 1.09]`, excludes zero. This makes the retention
result credible within the current experiment grid, although generation and human evaluation are still needed to
show whether the difference is noticeable in model responses.

What these results support:

- **Forgetting is real.** With no replay, earlier behaviors lost 7.94 accuracy points on average. Safety was the
  hardest behavior to retain: it lost 13.25 points with no replay.
- **Replay helps.** Every replay method improved retention and the final three-behavior average over no replay.
- **MFR improves on equal-budget random replay.** MFR retained 0.62 points more on average. The paired 95% bootstrap
  interval was `[0.22, 1.09]`, while its final-task difference was small and uncertain.
- **MFR also improves on higher-budget random replay.** It retained 0.59 points more and had a 0.25-point higher
  final average while using one-third fewer replay examples. MFR is slower because it must rescore the buffer.
- **Lowest Margin is a strong baseline, but CPMR has the best aggregate point estimates.** CPMR retained 0.19 points
  more, scored 0.25 points higher on the final task, and finished 0.29 points higher overall. These three paired
  intervals include zero, so the current two-seed grid does not establish that CPMR reliably beats Lowest Margin.
- **The original MFR idea is promising but is not the final winner.** Historical margin decline contains useful
  information, but the current results show that present difficulty is a very strong replay signal.
- **FMCR is a useful forecasting ablation with a mixed result.** It did not improve retention over Random 10%, but
  it improved final-task accuracy by 1.12 points and the final three-behavior average by 0.50 points. Compared with
  Lowest Margin, FMCR retained 0.97 points less and finished 0.35 points lower overall. Forecasting favored new-task
  learning more than retention and did not replace the simpler current-difficulty signal.
- **CPMR is the strongest completed forecasting method.** Against equal-budget Random 10%, it improved retention by
  1.22 points, final-task accuracy by 0.94 points, and the final three-behavior average by 1.15 points; all three
  paired 95% intervals exclude zero. It also beat Random 14.3% on retention and final average while using one-third
  fewer replay examples.
- **CPMR is related to, but not identical to, Lowest Margin.** Their mean distinct-pair Jaccard overlap is 0.55 and
  chance-adjusted overlap is 0.67. CPMR's overlap with Random 10% is only 0.04, showing that its selections are
  targeted rather than effectively random.

These are validation results from two seeds, not final test results. They support a controlled project conclusion,
but not a broad claim that CPMR is universally better. Balanced MFR and At-Risk MFR will separately test behavior
allocation and actual current preference failure. Unseen seeds must confirm the CPMR result before the locked test.

## How the results fit together

The project is not simply a contest to make MFR win. The report tells a sequence of controlled findings:

1. Sequential DPO causes real, non-uniform forgetting.
2. A fixed amount of replay substantially reduces that forgetting.
3. Historical margin decline is useful: MFR beats equal-budget Random 10% on retention and also beats higher-budget
   random replay while using fewer examples.
4. Present difficulty is even stronger than historical decline in this setup: Lowest Margin beats original MFR.
5. Forecasting is not automatically better. FMCR improves current-task learning and the final average relative to
   random replay, but its retention is similar to random and worse than Lowest Margin.
6. A forecast grounded in the actual upcoming updates works better: CPMR combines current difficulty with a
   reversible interval-level simulation and achieves the best aggregate retention and final average. It clearly
   beats random replay, while its small advantage over Lowest Margin remains uncertain.
7. Balanced MFR and At-Risk MFR are remaining controlled ablations that will test whether allocation and actual
   policy failure explain the remaining gap.

This supports a broader contribution: a controlled study of **what an LLM should rehearse during continual
preference tuning**, including positive and negative results, rather than an unsupported claim that one heuristic is
always best.

## What the datasets and task orders reveal

All three datasets use the same basic preference format: a prompt, a chosen answer, and a rejected answer. They also
use the same frozen split sizes and preprocessing. Helpfulness and general Quality overlap because both reward useful,
well-written, instruction-following answers. Safety is more distinct: many examples reward refusing or safely
redirecting harmful requests. This makes the three behaviors related, but not interchangeable.

The measured forgetting reflects that difference:

| Earlier behavior | No replay | Random 10% | MFR 10% | Lowest margin | FMCR 10% | CPMR 10% |
|---|---:|---:|---:|---:|---:|---:|
| Helpful (6 cells) | -6.25 | -4.50 | -4.83 | -4.42 | -4.17 | **-3.92** |
| Safe (6 cells) | -13.25 | -8.75 | -7.33 | **-6.67** | -8.58 | -7.00 |
| Quality (4 cells) | -2.50 | -2.12 | -1.25 | -1.25 | -2.62 | **-0.75** |

Safety is the most fragile behavior and Quality is the most stable. A plausible explanation is that later helpfulness
or quality tuning rewards broadly useful responses and can weaken refusal behavior, while helpfulness and quality
share more response characteristics. This is an interpretation of the observed pattern, not yet a causal claim; the
generation and error analyses must inspect actual responses.

Average retention also changes by order:

| Order | Sequence | No replay | MFR 10% | Lowest margin | FMCR 10% | CPMR 10% |
|---:|---|---:|---:|---:|---:|---:|
| 1 | Helpful → Safe → Quality | -9.75 | -5.38 | **-5.12** | -5.75 | -6.25 |
| 2 | Safe → Helpful → Quality | -13.00 | -9.25 | -8.50 | -8.75 | **-7.25** |
| 3 | Quality → Helpful → Safe | -4.38 | -1.50 | -1.25 | -3.12 | **-1.00** |
| 4 | Quality → Safe → Helpful | -4.62 | -3.38 | -3.00 | -4.12 | **-2.62** |

Order 2 is hardest because Safety is learned first, is the most fragile behavior, and must survive two later stages.
Orders 3 and 4 look easier partly because stable Quality is placed earlier while either Safety or Helpfulness is last
and therefore cannot be measured as forgotten. The order averages therefore mix true sequence effects with which
datasets are exposed to later training. The per-dataset table is the safer basis for behavior-level conclusions.

## Repository structure

```text
mfr-dpo/
├── configs/       # Frozen model, data, training, order, seed, and method settings
├── data/v2/       # Frozen preference-pair files and manifest
├── docs/          # CARC instructions, two method specifications, and final next steps
├── notebooks/     # CARC status and validation analysis notebooks
├── pilot/         # Historical proof of concept; not used in final tables
├── scripts/       # Cache, training, Slurm submission, and evaluation entry points
├── src/           # Data, DPO, replay, analysis, and evaluation implementation
└── tests/         # CPU tests for correctness, budgets, resuming, and safeguards
```

Start with:

- [`docs/CARC.md`](docs/CARC.md) for exact CARC setup, submission, monitoring, and analysis commands.
- [`docs/BALANCED_MFR.md`](docs/BALANCED_MFR.md) for the Balanced MFR definition and implementation checklist.
- [`docs/AT_RISK_MFR.md`](docs/AT_RISK_MFR.md) for the At-Risk MFR definition and implementation checklist.
- [`docs/NEXT_STEPS.md`](docs/NEXT_STEPS.md) for CPMR's completed result and the remaining evaluation plan.

The active notebooks are:

| Notebook | Purpose |
|---|---|
| `01_carc_status.ipynb` | Check the frozen data, reference cache, and completed CARC runs. |
| `07_compare_runs.ipynb` | Produce the complete validation tables, figures, uncertainty, and conclusions. |

Training runs as unattended Slurm jobs on USC CARC using one NVIDIA L40S GPU. Large caches, checkpoints, logs, and
results are stored under the ignored CARC `artifacts/` directory rather than Git.

**Do not inspect or run the locked test until the validation analysis and method-selection rule are frozen.**
