# mfr-dpo

CSCI 544, Group 26: **What Should an LLM Rehearse? Budgeted Replay for Continual Preference Tuning**

We fine-tune Qwen2.5-1.5B-Instruct with DPO in three stages (helpful, then safe, then quality). When a new
stage starts, the model can forget what it learned earlier. We test whether replaying the old examples
the model has forgotten most (Most-Forgotten Replay) works better than replaying random ones.

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
for every dataset (only 154 HelpSteer2 pairs are still too long). Longer sequences make training slower (our guess is 20–30% more time per stage; the pilot will tell us the real number). If it's too slow, we go back to 768 tokens and 1,828 / 200 / 300: set
`max_tokens=768` in the split cell of `01_load_data.ipynb` and re-run it.

Either way, every dataset gets the same split sizes, each prompt appears only once, and no prompt is
shared between splits or between datasets.

## How to run

The splits are already in `data/`, so we now only need `notebooks/02_data_review.ipynb`.

**Colab:** File → Open notebook → GitHub → `prabudhd2003/mfr-dpo` → pick a notebook, then Run all. No GPU needed.

**Locally:** `pip install -r requirements.txt`, then open the notebook.

## Shared Drive

[Team Drive folder](https://drive.google.com/drive/folders/1Mk7RtTdu0m_f7wnuvzGNzgRSTr8T1bqb?usp=sharing): for anything too big for GitHub, like trained models, checkpoints and results.

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
- `notebooks/03_train_one_stage.ipynb`: trains on helpful and checks it learned (needs a GPU: L4 or A100)
- `notebooks/04_pilot.ipynb`: safe → helpful → quality with no replay, measures forgetting (GPU, ~2.5 h, saves to Drive after each stage)

## Progress

- [x] Load the three datasets
- [x] Train / val / test splits (2,000 / 200 / 300 per dataset, in `data/`)
- [x] DPO training, one stage (`03`): 48.7 min on an L4, 12.6 GB; val accuracy helpful 58%, safe 44%, quality 71%
- [ ] Pilot: three stages in a row, is there forgetting? (`04`)
- [ ] Replay methods: random, lowest margin, MFR
- [ ] Experiments + results
