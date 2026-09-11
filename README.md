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

## How to run

**Colab:** File → Open notebook → GitHub → `prabudhd2003/mfr-dpo` → `notebooks/01_load_data.ipynb`, then Run all. No GPU needed.

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

- `src/data.py`: loads the three datasets
- `notebooks/`: the notebooks we run

## Progress

- [x] Load the three datasets
- [ ] Train / test splits
- [ ] DPO training (one stage, then three in a row)
- [ ] Replay methods: random, lowest margin, MFR
- [ ] Experiments + results
