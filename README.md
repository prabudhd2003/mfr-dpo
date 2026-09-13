# MFR-DPO

**What Should an LLM Rehearse? Budgeted Replay for Continual Preference Tuning**

This project studies catastrophic forgetting during sequential preference tuning. A single QLoRA adapter on Qwen2.5-1.5B-Instruct learns three behaviors in sequence: helpfulness, safety, and general response quality. At later stages, 10% of each replay-enabled batch is old data. Most-Forgotten Replay (MFR) chooses old pairs whose preference margin has fallen furthest from the value measured immediately after they were learned.

## Status

Notebooks 03 and 04 and the existing Drive runs are useful preliminary evidence: sequential DPO learns each stage, safety can fall sharply after helpfulness training, and forgetting is uneven across examples. They are **not final experimental runs** because the earlier pipeline used unversioned data, an unseeded pilot initialization, and 16-new/2-old batches (11.1% replay).

Protocol v2 fixes those issues. Its authoritative settings live in [`configs/experiment_protocol.json`](configs/experiment_protocol.json). Do not change individual notebook constants during the main study.

## Correct experiment

- Data: 2,000 train / 200 validation / 300 locked test pairs for each behavior.
- Global normalized-prompt de-duplication across datasets and splits.
- Two orders: helpful → safe → quality; safe → helpful → quality.
- Core methods: no replay, 10% random replay, and 10% MFR. A 14.3% `random_high` budget control and lowest-margin replay are secondary.
- Seeds: 0 and 1.
- Replay-enabled batch: 18 new + 2 old = exactly 10% replay.
- Buffer: 500 pairs, rebalanced across learned behaviors.
- Primary metric: length-normalized relative preference accuracy.
- Secondary metric: summed DPO relative preference accuracy.
- Also reported: absolute policy accuracy, new-stage gain, retention loss, costs, per-pair uncertainty, generated behavior, and blinded human review.

See [`docs/EXPERIMENT_PROTOCOL.md`](docs/EXPERIMENT_PROTOCOL.md), [`docs/DATA_CARD.md`](docs/DATA_CARD.md), and [`docs/RUNBOOK.md`](docs/RUNBOOK.md).

## Run in this order

Use Colab with an L4/A100 GPU for model work. Add the shared Drive folder as `MyDrive/CSCI544/mfr-dpo`.
Commit and push these code changes before opening Colab, because the notebooks pull their code from GitHub.

1. Run `01_load_data.ipynb` once; review with `02_data_review.ipynb`; commit `data/v2`.
2. Run `05_build_reference_cache.ipynb` once.
3. In `06_run_experiment.ipynb`, run the two v2 no-replay baselines for both seeds.
4. Run matched random, MFR, and higher-budget random runs, reusing the compatible v2 no-replay stage 1.
5. Run `07_compare_runs.ipynb`; then `08_error_analysis.ipynb`.
6. Freeze the final model-selection decision. Only then run `09_final_test_eval.ipynb` once.
7. Run `10_generation_eval.ipynb`, automatic behavior evaluators, and `11_blind_review.ipynb`.

The exact 12-run core grid, four-run higher-budget random addition, and resume instructions are in the runbook.

## Repository map

- `src/mfr_data.py`: pinned data loading, normalized de-duplication, validation, manifests.
- `src/mfr_dpo.py`: QLoRA model loading, DPO, replay training, relative and absolute metrics.
- `src/mfr_replay.py`: deterministic buffer and replay selection rules.
- `src/mfr_cache.py`: frozen-reference cache with compatibility checks.
- `src/mfr_analysis.py`: retention, new learning, costs, paired bootstrap, diagnostics.
- `src/mfr_eval.py`: locked test evaluation and deterministic generation.
- `src/mfr_review.py`: blinded review sheets and private method key.
- `scripts/`: reproducible command-line entry points used by notebooks.
- `notebooks/03_train_one_stage.ipynb`, `04_pilot.ipynb`: historical preliminary work; do not use for final numbers.
- `tests/`: CPU bookkeeping, leakage, cache, metric, and review tests.

## Local checks

Use Python 3.10 or newer:

```bash
python -m pip install -r requirements.txt
PYTHONPATH=src:tests python -m pytest -q
```

Training requires a supported NVIDIA GPU and is intended for Colab. Large adapters, caches, generations, and results remain in Drive rather than Git.
