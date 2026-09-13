# mfr-dpo

CSCI 544, Group 26: **What Should an LLM Rehearse? Budgeted Replay for Continual Preference Tuning**

We fine-tune Qwen2.5-1.5B-Instruct with DPO in three stages (e.g. safe, then helpful, then quality). When a new
stage starts, the model can forget what it learned earlier. We test whether replaying the old examples
the model has forgotten most (Most-Forgotten Replay, MFR) works better than replaying random ones.

## Data

| Stage | Dataset | What we keep |
|---|---|---|
| helpful | [HelpSteer2](https://huggingface.co/datasets/nvidia/HelpSteer2) (preference) | clear preferences, single-turn |
| safe | [PKU-SafeRLHF](https://huggingface.co/datasets/PKU-Alignment/PKU-SafeRLHF) | one safe + one unsafe response |
| quality | [UltraFeedback](https://huggingface.co/datasets/HuggingFaceH4/ultrafeedback_binarized) | score gap ≥ 1 |

All three are loaded as tables with the same columns: `prompt`, `chosen`, `rejected`.

### Splits and the token limit

In the proposal we planned for sequences of 512–768 tokens. At 768 tokens, HelpSteer2 runs short
(434 of its pairs are too long), which leaves only **1,828 train / 200 val / 300 test** pairs per dataset.

So we raised the limit to **1024 tokens**, which gives the full **2,000 train / 200 val / 300 test**
for every dataset (only 154 HelpSteer2 pairs are still too long). The pilot showed this is affordable
(14–35 minutes per stage on an L4), so we're keeping it. If we ever need to go back: set `max_tokens=768`
in the split cell of `01_load_data.ipynb` and re-run it (gives 1,828 / 200 / 300).

Either way, every dataset gets the same split sizes, each prompt appears only once, and no prompt is
shared between splits or between datasets.

## Results so far

### One stage (`03_train_one_stage`)

Trained on helpful only (lr 5e-5): 48.7 min on an L4, 12.6 GB peak.
Val accuracy afterwards: helpful 58%, safe 44%, quality 71%. Training on helpful alone already pushes the
model toward the *unsafe* answer on safety pairs, a first hint of the conflict between the two.

### Pilot: safe → helpful → quality, no replay (`04_pilot`)

Val accuracy = % of the 200 val pairs where the model prefers the chosen answer (length-normalized margin > 0).
Bold = the stage that trained on that dataset.

| val set | after 1. safe | after 2. helpful | after 3. quality |
|---|---|---|---|
| safe | **87.5** | 70.5 | 69.5 |
| helpful | 52.5 | **61.5** | 61.5 |
| quality | 57.5 | 73.0 | **77.0** |

- **Forgetting is real.** Safe drops 18 points and keeps only 31% of its margin, almost all of it during the helpful stage.
- **Helpful is not forgotten** on our main score (its margin even grows to 134%). On the standard DPO score
  (summed, not length-normalized) helpful does drop 69.5 → 60.5 in the quality stage, so we report both scores.
- **Forgetting is widespread but uneven.** 81% of safe pairs lose some margin, and the worst 10% of pairs carry
  37% of the total loss. That unevenness is what MFR tries to exploit.
- **Cost.** Stages took 14.2 / 34.7 / 34.7 min on an L4 (peak 11 GB): about 1.5 h per full run, ~5 Colab compute units.
- Settings: lr 1e-4, beta 0.1, 1 epoch per stage, 16 pairs per step, LoRA r=16, seed 0. One order and one seed so far.

## Plan

**Step 1: Replay code.** One memory buffer and four ways to fill the replay slots:
- *none*, *random*, *lowest current margin*, and *MFR* (largest drop since the pair was learned).
- Buffer: 500 pairs from the earlier stages' train splits (split evenly between them), each with its margin
  right after its own stage.
- Every batch: 18 new pairs + 2 old pairs (10% old). Every method sees all 2,000 new pairs in the same number of
  steps; *none* simply leaves the 2 old slots empty. MFR and lowest-margin re-score the buffer 5 times per stage.
- Speed-up: the base model never changes, so its scores for every pair are computed once and reused.
- Tests that every method gets exactly the same budget.

**Step 2: First comparison.** Random vs. MFR on safe → helpful → quality, seed 0 (~2 h each), plus no replay with the
new 18-pair batches. This is our go / no-go before the full study.

**Step 3: Main experiments.** 2 orders × 4 methods × 2 seeds = 16 runs, ~1.5–2 h each on an L4 (~30 GPU-hours).
- order 1: helpful → safe → quality
- order 2: safe → helpful → quality
- Each run saves to `runs/<order>_<method>_s<seed>/` in the team Drive.

**Step 4: Milestone 2 table (Week 10).** For each method: retention of earlier stages and learning of the current
stage, with paired differences between methods and bootstrap confidence intervals.

**Step 5: Final evaluation.** First and only use of the test split, IFEval on generated answers, a blinded review of
100–150 generated answers, and error analysis (which pairs get forgotten, length effects).

**Step 6: Report.**

**Settings fixed for all main runs:** Qwen2.5-1.5B-Instruct, 4-bit QLoRA r=16, one adapter across all stages,
DPO beta 0.1, lr 1e-4, 1 epoch per stage, 1024 tokens, 2,000 / 200 / 300 pairs per dataset.
**Still to decide before step 3:** how much worse MFR may be on the *current* stage and still count as a win
(e.g. at most 2 points below random replay).

## How to run

The splits are already in `data/`. Training notebooks need a GPU: in Colab, Runtime → Change runtime type → **L4**.
Run only one notebook on the GPU at a time (restart the previous one first), and keep your Mac awake during long
runs (`caffeinate -dims` in Terminal).

**Colab:** File → Open notebook → GitHub → `prabudhd2003/mfr-dpo` → pick a notebook, then Run all.

**Locally:** `pip install -r requirements.txt`, then open the notebook (data notebooks only, no GPU).

## Shared Drive

[Team Drive folder](https://drive.google.com/drive/folders/1Mk7RtTdu0m_f7wnuvzGNzgRSTr8T1bqb?usp=sharing): for anything too big for GitHub, like trained adapters and results.
Runs are saved under `runs/<run name>/`.

To use it in Colab, first add it to your own Drive: open the folder, right-click it, then Organize → Add shortcut → My Drive. Then in a notebook:

```python
from google.colab import drive
drive.mount("/content/drive")
# the folder is now at /content/drive/MyDrive/<folder name>
```

## What's where

- `src/mfr_data.py`: loads the three datasets and makes the splits
- `src/mfr_dpo.py`: loads the model (4-bit QLoRA), trains with DPO, measures margins
- `data/`: our train / val / test splits (see "Splits and the token limit" above)
- `notebooks/01_load_data.ipynb`: downloads the datasets and makes `data/`. **Already done, don't re-run** unless we change a data rule (filters, token limit, split sizes)
- `notebooks/02_data_review.ipynb`: counts, sanity checks, lengths and examples from `data/`
- `notebooks/03_train_one_stage.ipynb`: trains on helpful and checks it learned (GPU)
- `notebooks/04_pilot.ipynb`: three stages in a row with no replay, measures forgetting (GPU, ~1.5 h, saves to Drive after each stage, can resume after a disconnect)

## Progress

- [x] Load the three datasets
- [x] Train / val / test splits (2,000 / 200 / 300 per dataset, in `data/`)
- [x] DPO training, one stage (`03`)
- [x] Pilot: forgetting is real (safe 87.5% → 69.5%) (`04`)
- [ ] Step 1: replay code (none / random / lowest margin / MFR)
- [ ] Step 2: first comparison, random vs. MFR
- [ ] Step 3: 16 main runs
- [ ] Step 4: Milestone 2 table
- [ ] Step 5: test set, IFEval, blinded review, error analysis
- [ ] Step 6: report
