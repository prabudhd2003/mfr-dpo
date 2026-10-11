# MFR-DPO

**What Should an LLM Rehearse? Budgeted Replay for Continual Preference Tuning**

> **Course-project branch.** This is the `project` branch of the CSCI 544 Group 26 project. Folder setup and
> every-session commands: [docs/PROJECT_FOLDER.md](docs/PROJECT_FOLDER.md). Everything still to do, in order:
> [docs/NEXT_STEPS.md](docs/NEXT_STEPS.md).

## Project summary

This project studies catastrophic forgetting during sequential preference tuning. One QLoRA adapter [2] on
`Qwen/Qwen2.5-1.5B-Instruct` learns three behaviors one after another with Direct Preference Optimization (DPO) [1]:
Helpfulness, Safety, and general response Quality.

DPO learns from a prompt, a preferred response, and a rejected response. After the model learns one behavior,
training it on a later behavior can weaken what it learned earlier. This is catastrophic forgetting.

Most-Forgotten Replay (MFR) keeps a small memory of old preference pairs. It records how strongly each pair was
learned and later measures how much its reference-relative preference margin has fallen. The pairs with the largest
drops are replayed during later training. The main question is whether this targeted replay protects earlier
behaviors better than random replay with the same 10% replay budget, without harming the behavior currently being
learned.

## Status

| Done | Next |
|---|---|
| Data v2 cleaned, deduplicated across datasets and splits, length-filtered, hashed, and frozen | Implement Balanced, At-Risk, and Stochastic MFR, plus the joint-training reference |
| CARC pipeline: reference cache, resume logic, Stage-1 reuse, replay logs, CPU tests | Train them on orders 1–4 × seeds 0–1 |
| **40 validation runs**: 5 methods × 4 orders × 2 seeds on NVIDIA L40S | Freeze the method choice, then confirm on unseen seed 2 |
| Notebook 07: full grid, paired uncertainty, per-behavior forgetting, runtime | Run the locked test once, then evaluate generated responses, run human evaluation, and write the report |

## Methods

| Method | Status | What it does | Question it answers |
|---|---|---|---|
| No replay | done | Train only on the current behavior. | How much is forgotten without protection? |
| Random 10% | done | 18 new + 2 random old pairs per step. | Does untargeted replay help? This is the main baseline. |
| Random 14.3% | done | 18 new + 3 random old pairs per step. | Is any gain just more replay? |
| Lowest margin | done | Replay pairs with the lowest *current* reference-relative margin. | Is present difficulty a better signal than past decline? |
| MFR 10% | done | Replay pairs with the largest *historical* margin drop. | The proposed method. |
| Balanced MFR | to build | MFR ranking with equal replay slots per old behavior. | Does MFR's gain come from which pairs it picks or from how it splits slots across behaviors? |
| At-Risk MFR | to build | Balanced MFR, but pairs the policy currently gets wrong come first. | Does actual current failure add information? |
| Stochastic MFR | to build | Sample pairs in proportion to `softmax(z-scored drop / τ)` instead of always taking the top ones. | Does greedy top-k repeat the same pairs too often? |
| Joint training | to build | One adapter, all 6,000 pairs shuffled together, one pass. | How much of the forgetting gap does each method close? This is a reference, not a competitor. |

Specifications: [Balanced MFR](docs/BALANCED_MFR.md), [At-Risk MFR](docs/AT_RISK_MFR.md),
[Stochastic MFR](docs/NEXT_STEPS.md#stochastic-mfr-specification), [Joint training](docs/JOINT_TRAINING.md).

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
checkpoint, so every comparison is **paired**.

| Order | Training sequence |
|---:|---|
| 1 | Helpful → Safe → Quality |
| 2 | Safe → Helpful → Quality |
| 3 | Quality → Helpful → Safe |
| 4 | Quality → Safe → Helpful |

Seeds 0 and 1 are used for each order, which gives eight order–seed cells per method. Orders 3 and 4 make Quality an
earlier behavior, so its forgetting can also be measured.

### How to read the numbers

- **Accuracy** is the percentage of preference pairs with a positive length-normalized DPO advantage over the frozen
  base model. It measures whether training moved the model in the preferred direction. It is not factual accuracy.
- **Old-behavior change** = final validation accuracy − accuracy immediately after that behavior was learned,
  averaged over the two earlier behaviors. Zero means nothing was forgotten. More negative is worse.
- **Forgetting** is the size of the old-behavior change. "X% less forgetting than B" means
  `(forgetting of B − forgetting of X) / forgetting of B`.
- **Resolution.** Each validation set has 200 pairs, so **one pair = 0.5 accuracy points**. Old-behavior change
  averages two behaviors, so it moves in steps of 0.25 points.
- **Tables are sorted from worst method to best** on the metric named in the table caption.

## Validation results (40 runs, 8 cells per method)

### Headline

- **Forgetting is real.** With no replay, earlier behaviors lose **7.94 points** on average, about 16 of 200 pairs.
  Safety is hit hardest: it loses **13.25 points**.
- **Any replay helps.** Random 10% removes **30.7%** of that forgetting (7.94 → 5.50) and raises the final
  three-behavior average from 70.77 to 71.71.
- **MFR beats random replay at the same budget.** MFR has **11.3% less forgetting than Random 10%** (4.88 vs 5.50
  points; paired 95% interval for the difference [0.22, 1.09] points). It retains more in **7 of 8 cells**, with the
  same 444 replay examples per run.
- **MFR's gain is not explained by more replay.** MFR has **10.8% less forgetting than Random 14.3%**, which replays
  **50% more** pairs (666 vs 444). Extra random replay barely helps: Random 14.3% forgets only 0.5% less than Random 10%.
- **Lowest margin is the strongest method so far.** It has **18.7% less forgetting than Random 10%** and **8.4% less
  than MFR**. It also has the best final three-behavior average (72.56, +0.58 over MFR, interval [0.19, 1.04]) and the
  best final-task score of any replay method. It beats Random 10% on the final average in **8 of 8 cells**.
- **Targeting costs time.** MFR and Lowest margin take **16–18% longer** per run than Random 10%, because they
  re-score the replay buffer (about 4.3 minutes per run).

### Table 1. Overall: sorted worst to best by forgetting

| Method | Old-behavior change | Less forgetting than No replay | Less forgetting than Random 10% | Final average (3 behaviors) | Final current task | Replay examples per run | Runtime per run |
|---|---:|---:|---:|---:|---:|---:|---:|
| No replay | −7.94 | — | 44.4% *more* | 70.77 | 74.81 | 0 | 22.48 min |
| Random 10% | −5.50 | 30.7% | — | 71.71 | 73.62 | 444 | 24.28 min |
| Random 14.3% | −5.47 | 31.1% | 0.5% | 71.73 | 73.88 | 666 | 24.91 min |
| MFR 10% | −4.88 | 38.5% | 11.3% | 71.98 | 73.75 | 444 | 28.28 min |
| **Lowest margin** | **−4.47** | **43.7%** | **18.7%** | **72.56** | 74.31 | 444 | 28.59 min |

The ranking by final three-behavior average is the same as the ranking by forgetting. **No replay has the highest
final-task score**, because nothing old competes with the new behavior in its batches. Every replay method gives up
0.5–1.2 points on the newest behavior in exchange for keeping the old ones.

### Table 2. Forgetting by behavior: each block sorted worst to best

Helpful and Safe are earlier behaviors in six cells each. Quality is earlier only in orders 3–4 (four cells).

| Behavior | Method | Change | Less forgetting than No replay | Less forgetting than Random 10% |
|---|---|---:|---:|---:|
| Helpful | No replay | −6.25 | — | 38.9% *more* |
| Helpful | Random 14.3% | −5.00 | 20.0% | 11.1% *more* |
| Helpful | MFR 10% | −4.83 | 22.7% | 7.3% *more* |
| Helpful | Random 10% | −4.50 | 28.0% | — |
| Helpful | **Lowest margin** | **−4.42** | **29.3%** | **1.8%** |
| Safe | No replay | −13.25 | — | 51.4% *more* |
| Safe | Random 10% | −8.75 | 34.0% | — |
| Safe | Random 14.3% | −8.08 | 39.0% | 7.7% |
| Safe | MFR 10% | −7.33 | 44.7% | 16.2% |
| Safe | **Lowest margin** | **−6.67** | **49.7%** | **23.8%** |
| Quality | No replay | −2.50 | — | 17.9% *more* |
| Quality | Random 14.3% | −2.25 | 10.0% | 6.1% *more* |
| Quality | Random 10% | −2.12 | 15.2% | — |
| Quality | **MFR 10%** (tie) | **−1.25** | **50.0%** | **41.0%** |
| Quality | **Lowest margin** (tie) | **−1.25** | **50.0%** | **41.0%** |

MFR's advantage comes from **Safe** and **Quality**. On **Helpful**, MFR forgets 7.3% *more* than Random 10%.
In the order 1, seed 0 cell, MFR gave 148 of its 222 Stage-3 replay slots (67%) to Safe pairs, while Random 10% gave
118 (53%). This is why Balanced MFR, which forces equal slots per behavior, is the next ablation. Quality percentages
are large but the absolute changes are small: 2.5 points is 5 pairs.

### Table 3. Paired comparisons with MFR: sorted from weakest to strongest baseline

A positive difference favors MFR. Intervals are 95% paired bootstrap intervals over the eight order–seed cells
(20,000 resamples) [15, 16].

| MFR compared with | Retention difference | 95% interval | MFR retains more in | Final-average difference | 95% interval |
|---|---:|---:|---:|---:|---:|
| No replay | +3.06 | [2.06, 4.03] | 8 of 8 cells | +1.21 | [0.46, 1.98] |
| Random 10% | +0.62 | [0.22, 1.09] | 7 of 8 cells | +0.27 | [−0.06, 0.62] |
| Random 14.3% | +0.59 | [0.16, 1.06] | 6 of 8 cells (1 tie) | +0.25 | [0.04, 0.44] |
| Lowest margin | −0.41 | [−1.12, 0.34] | 2 of 8 cells | −0.58 | [−1.04, −0.19] |

### What these results support

1. Sequential DPO causes measurable forgetting, and replay reduces it by roughly a third to almost half.
2. Using past margin decline to choose replay pairs (MFR) beats choosing randomly at the same budget, and beats a
   random budget that is 50% larger. The retention interval excludes zero, but the effect is small: 0.62 points is
   about **one validation pair per behavior**.
3. Present difficulty (Lowest margin) is a stronger signal than past decline in this setup. The original MFR idea is
   useful but is **not** the best method so far.
4. These are validation results from two seeds. Seed 2 and the locked test, both planned in
   [NEXT_STEPS.md](docs/NEXT_STEPS.md), will confirm or overturn them.

## Are such small differences normal? Context from the literature

Yes. The gaps between replay methods are small, but this is typical for this kind of research, and we expected it.

**Replay itself is the big win, and even a tiny memory is enough.** Chaudhry et al. [3] found that simply training
on the current task together with a very small memory of past examples "significantly outperforms specifically
designed" continual-learning methods. Rolnick et al. [4] reached a similar conclusion for experience replay in
reinforcement learning. For language models, Scialom et al. [10] kept continual instruction tuning stable with a
rehearsal memory holding only 1% of past data. Our results agree: 10% random replay alone removes 30.7% of forgetting.

**Random replay is a famously strong baseline, and smarter selection usually wins by little.**
- Isele and Cosgun [8] compared four ways of choosing what to store. Matching the overall training distribution,
  which is close to a random sample, was "consistently the best approach on all domains tested". Selection by surprise
  or reward still forgot.
- GDumb [5] showed that a deliberately simple method, a greedy class-balanced memory plus retraining on it, matched or
  beat many published continual-learning methods. The title states the point: it "questions our progress".
- Buzzega et al. [6] showed that a few simple fixes make plain experience replay much stronger, which raises the bar
  for any selection method.
- Maximally Interfered Retrieval [7], the closest idea to MFR, picks the memory examples a coming update would hurt
  most. It reports consistent gains over random retrieval.

So a targeted method that beats Random 10% by about 11% in relative terms is in line with prior work.

**There is not much room left to win.** Random 10% already brings forgetting down to 5.50 points. A better
selection rule can only compete for that remaining amount. On that scale, MFR removes 11% and Lowest margin 19% of
what is left after random replay. The joint-training reference ([docs/JOINT_TRAINING.md](docs/JOINT_TRAINING.md))
will show how high the ceiling really is. Ibrahim et al. [12] use re-training on all data as the target for
replay-based continual pre-training in the same way.

**Small evaluation sets limit what can be detected.** Card et al. [13] showed that underpowered experiments are common
in NLP: small test sets cannot reliably detect small differences. Bouthillier et al. [14] showed that variation from
random seeds and data sampling must be measured with repeated runs before two methods can be compared. Our safeguards
follow from this:
- every comparison is paired, with the same data, seed, order, Stage-1 checkpoint, and validation pairs;
- we report paired bootstrap intervals [15, 16] and the number of cells won, not only means;
- we add an unseen seed and a larger locked test set (300 pairs per behavior) before making final claims.

**Some examples are simply more forgettable, which is MFR's motivation.** Toneva et al. [9] showed that individual
training examples differ greatly in how often they are forgotten, and some are forgotten repeatedly. Bhandari et al.
[17] report uneven margin changes across preference pairs in sequential DPO. Luo et al. [11] document catastrophic
forgetting in 1B–7B language models during continual fine-tuning. MFR tests whether that pair-level signal can direct a
fixed replay budget.

**What this means for the report.** Present small but consistent, paired, well-measured differences honestly. Give
relative percentages next to absolute points and pair counts. If Lowest margin stays best, that is a useful
finding: for this setup, *current* difficulty is a better replay signal than *past* decline.

## Repository structure

```text
mfr-dpo/
├── configs/       # Frozen model, data, training, order, seed, and method settings
├── data/v2/       # Frozen preference-pair files and manifest
├── docs/          # Setup, CARC workflow, method specifications, and the remaining plan
├── notebooks/     # CARC status and validation analysis notebooks
├── pilot/         # Historical proof of concept; not used in final tables
├── scripts/       # Cache, training, Slurm submission, and evaluation entry points
├── src/           # Data, DPO, replay, analysis, and evaluation implementation
└── tests/         # CPU tests for correctness, budgets, resuming, and safeguards
```

| Document | Purpose |
|---|---|
| [docs/PROJECT_FOLDER.md](docs/PROJECT_FOLDER.md) | What this branch is, folders, every-session CARC commands |
| [docs/CARC.md](docs/CARC.md) | Submitting, monitoring, resuming, and analyzing jobs on CARC |
| [docs/NEXT_STEPS.md](docs/NEXT_STEPS.md) | Every remaining step until the final report, with dates |
| [docs/BALANCED_MFR.md](docs/BALANCED_MFR.md) | Balanced MFR definition and implementation checklist |
| [docs/AT_RISK_MFR.md](docs/AT_RISK_MFR.md) | At-Risk MFR definition and implementation checklist |
| [docs/JOINT_TRAINING.md](docs/JOINT_TRAINING.md) | Joint-training reference: code to add and how to run it |

| Notebook | Purpose |
|---|---|
| `01_carc_status.ipynb` | Check the frozen data, reference cache, and completed CARC runs. |
| `07_compare_runs.ipynb` | Produce the complete validation tables, figures, uncertainty, and conclusions. |

Training runs as unattended Slurm jobs on USC CARC using one NVIDIA L40S GPU. Large caches, checkpoints, logs, and
results are stored in the git-ignored `artifacts/` directory on CARC.

**Do not run the locked test until the method-selection rule is frozen (NEXT_STEPS Step 7).**

## References

[1] R. Rafailov, A. Sharma, E. Mitchell, S. Ermon, C. D. Manning, and C. Finn. "Direct Preference Optimization: Your
Language Model Is Secretly a Reward Model." NeurIPS 2023. https://arxiv.org/abs/2305.18290

[2] T. Dettmers, A. Pagnoni, A. Holtzman, and L. Zettlemoyer. "QLoRA: Efficient Finetuning of Quantized LLMs."
NeurIPS 2023. https://arxiv.org/abs/2305.14314

[3] A. Chaudhry, M. Rohrbach, M. Elhoseiny, T. Ajanthan, P. K. Dokania, P. H. S. Torr, and M. Ranzato. "On Tiny
Episodic Memories in Continual Learning." arXiv:1902.10486, 2019. https://arxiv.org/abs/1902.10486

[4] D. Rolnick, A. Ahuja, J. Schwarz, T. P. Lillicrap, and G. Wayne. "Experience Replay for Continual Learning."
NeurIPS 2019. https://arxiv.org/abs/1811.11682

[5] A. Prabhu, P. H. S. Torr, and P. K. Dokania. "GDumb: A Simple Approach that Questions Our Progress in Continual
Learning." ECCV 2020. https://www.ecva.net/papers/eccv_2020/papers_ECCV/html/3587_ECCV_2020_paper.php

[6] P. Buzzega, M. Boschini, A. Porrello, and S. Calderara. "Rethinking Experience Replay: a Bag of Tricks for
Continual Learning." ICPR 2020. https://arxiv.org/abs/2010.05595

[7] R. Aljundi, L. Caccia, E. Belilovsky, M. Caccia, M. Lin, L. Charlin, and T. Tuytelaars. "Online Continual
Learning with Maximal Interfered Retrieval." NeurIPS 2019. https://arxiv.org/abs/1908.04742

[8] D. Isele and A. Cosgun. "Selective Experience Replay for Lifelong Learning." AAAI 2018.
https://arxiv.org/abs/1802.10269

[9] M. Toneva, A. Sordoni, R. Tachet des Combes, A. Trischler, Y. Bengio, and G. J. Gordon. "An Empirical Study of
Example Forgetting during Deep Neural Network Learning." ICLR 2019. https://arxiv.org/abs/1812.05159

[10] T. Scialom, T. Chakrabarty, and S. Muresan. "Fine-tuned Language Models are Continual Learners." EMNLP 2022.
https://aclanthology.org/2022.emnlp-main.410

[11] Y. Luo, Z. Yang, F. Meng, Y. Li, J. Zhou, and Y. Zhang. "An Empirical Study of Catastrophic Forgetting in Large
Language Models During Continual Fine-tuning." arXiv:2308.08747, 2023. https://arxiv.org/abs/2308.08747

[12] A. Ibrahim, B. Thérien, K. Gupta, M. L. Richter, Q. Anthony, T. Lesort, E. Belilovsky, and I. Rish. "Simple and
Scalable Strategies to Continually Pre-train Large Language Models." arXiv:2403.08763, 2024.
https://arxiv.org/abs/2403.08763

[13] D. Card, P. Henderson, U. Khandelwal, R. Jia, K. Mahowald, and D. Jurafsky. "With Little Power Comes Great
Responsibility." EMNLP 2020. https://arxiv.org/abs/2010.06595

[14] X. Bouthillier, P. Delaunay, M. Bronzi, et al. "Accounting for Variance in Machine Learning Benchmarks."
MLSys 2021. https://arxiv.org/abs/2103.03098

[15] R. Dror, G. Baumer, S. Shlomov, and R. Reichart. "The Hitchhiker's Guide to Testing Statistical Significance in
Natural Language Processing." ACL 2018. https://aclanthology.org/P18-1128

[16] B. Efron and R. J. Tibshirani. *An Introduction to the Bootstrap.* Chapman & Hall/CRC, 1993.

[17] P. Bhandari et al. "Beyond Uniform Forgetting: A Study of Sequential Direct Preference Optimization Across
Preference Settings." arXiv:2606.19744, 2026.
