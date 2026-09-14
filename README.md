# MFR-DPO

**What Should an LLM Rehearse? Budgeted Replay for Continual Preference Tuning**

MFR-DPO studies catastrophic forgetting during continual preference tuning. A single QLoRA adapter on
`Qwen/Qwen2.5-1.5B-Instruct` learns three preference behaviors one after another: helpfulness, safety, and general
response quality. The central question is whether a small replay budget is more useful when it is spent on the
examples the model has forgotten most, rather than on uniformly random old examples.

The project uses Direct Preference Optimization (DPO). Every training example contains a prompt, a preferred
response, and a rejected response. As the model learns a new behavior, its preference for previously learned
responses can weaken. This loss is the form of forgetting measured here.

## Research question

At the same replay budget, does **Most-Forgotten Replay (MFR)** retain earlier preference behavior better than
uniform random replay while preserving the ability to learn the current behavior?

MFR stores each replay candidate's preference margin immediately after its behavior is learned. During later
training, it periodically scores the candidates again and prioritizes pairs with the largest drop from that stored
margin. The v2 protocol uses a 500-pair replay buffer, five selection intervals per later stage, and a 75% cap on the
share of replay that may come from one old behavior when another is available.

## Experimental design

The authoritative settings are frozen in [`configs/experiment_protocol.json`](configs/experiment_protocol.json).
Scientific settings should not be changed inside individual notebooks during the main experiment grid.

| Component | Protocol v2 setting |
|---|---|
| Base model | `Qwen/Qwen2.5-1.5B-Instruct`, pinned revision |
| Training | One QLoRA adapter, sequential DPO, one epoch per behavior |
| Orders | helpful → safe → quality; safe → helpful → quality |
| Seeds | 0 and 1 |
| Core methods | `none`, `random`, `mfr` |
| Secondary methods | `random_high`, `lowest_margin` |
| Standard replay batch | 18 new + 2 old pairs, exactly 10% replay |
| Higher-budget batch | 18 new + 3 random old pairs, 14.3% replay |
| Optimizer settings | DPO beta 0.1, learning rate `1e-4`, one epoch |
| Context limit | 1,024 tokens |
| Training hardware | Google Colab with an NVIDIA A100 GPU |

The methods answer different questions:

- **No replay (`none`)** measures how much forgetting occurs without protection.
- **Random replay (`random`)** is the equal-budget baseline. It uniformly samples two old pairs per full step.
- **Most-Forgotten Replay (`mfr`)** uses the same number of old pairs but prioritizes the largest margin drops.
- **Higher-budget random (`random_high`)** samples three old pairs per step to test whether MFR can compete with
  random replay that receives more old data.
- **Lowest current margin (`lowest_margin`)** prioritizes pairs the current model scores poorly, without considering
  how far each pair has fallen from its own post-learning peak.

There is no replay in Stage 1, so compatible methods share the matching completed no-replay Stage-1 checkpoint.
The runner verifies its scientific-code fingerprint, data hashes, model revision, order, seed, and training settings
before allowing this reuse.

## Data

The project builds three preference streams from pinned Hugging Face dataset revisions.

| Behavior | Source | How the preferred response is selected |
|---|---|---|
| Helpful | [NVIDIA HelpSteer2](https://huggingface.co/datasets/nvidia/HelpSteer2) | Keep pairs with absolute preference strength of at least 2 and select the preferred response indicated by the label. |
| Safe | [PKU-SafeRLHF](https://huggingface.co/datasets/PKU-Alignment/PKU-SafeRLHF) | Keep pairs where exactly one response is labelled safe and choose that response. |
| Quality | [UltraFeedback Binarized](https://huggingface.co/datasets/HuggingFaceH4/ultrafeedback_binarized) | Keep pairs where the chosen response's score exceeds the rejected response's score by at least 1. |

Each behavior has:

| Split | Pairs per behavior | Purpose |
|---|---:|---|
| Train | 2,000 | Adapter training and replay candidates |
| Validation | 200 | Development, method comparison, and error analysis |
| Test | 300 | Locked final evaluation only |

### Data-processing pipeline

[`notebooks/01_load_data.ipynb`](notebooks/01_load_data.ipynb) and [`src/mfr_data.py`](src/mfr_data.py) perform the
following steps:

1. Download each source at its pinned revision and retain source-dataset, source-split, and source-row provenance.
2. Remove missing or empty text, identical chosen/rejected responses, and duplicate preference triples.
3. Normalize prompts using Unicode NFKC normalization, case folding, whitespace trimming, and whitespace collapse.
4. Remove normalized prompt overlap globally across all three behaviors and all splits to prevent leakage.
5. Apply the pinned model's chat template, count tokens, and remove any pair whose longer prompt-response sequence
   exceeds 1,024 tokens.
6. Shuffle deterministically with split seed 0 and create prompt-disjoint train, validation, and test splits.
7. Assign stable IDs and save nine JSONL files in `data/v2/`.
8. Validate required columns, sizes, unique IDs, prompt separation, and sequence lengths.
9. Record every file's row count and SHA-256 hash in [`data/v2/manifest.json`](data/v2/manifest.json).

The runner verifies the manifest before every experiment. The test files remain closed until the experiment grid,
analysis, and model-selection decision have been frozen. More detail is available in
[`docs/DATA_CARD.md`](docs/DATA_CARD.md).

## Metrics

The primary metric, `accuracy`, is the percentage of held-out preference pairs whose **length-normalized DPO
advantage relative to the frozen base model is positive**. In simple terms, it asks whether training moved the model
toward the preferred response relative to where the base model started.

This is not the percentage of generated answers that are objectively correct, safe, or helpful. Generated-response
evaluation and human review are separate final steps.

The required secondary metric, `accuracy_sum`, uses summed response log probabilities rather than length-normalized
log probabilities. The project also records absolute policy accuracy, mean margins, new-stage learning, retention,
replay allocation, runtime, and paired-bootstrap uncertainty.

For an old behavior:

`retention change = final accuracy − accuracy immediately after that behavior was learned`

A value closer to zero means less forgetting. For example, −4 points is better retention than −9 points.

The frozen primary success rule asks whether MFR has better mean retention than equal-budget random replay while
its final new-stage score is no more than 2 points worse.

## Current results

The first complete matched comparison cell is finished: **Order 1, Seed 0**
(`helpful → safe → quality`). These are validation results, not final test results.

| Method | Helpful retention change | Safe retention change | Mean retention change | Final quality accuracy | Final mean across all behaviors |
|---|---:|---:|---:|---:|---:|
| No replay | −6.5 | −9.5 | −8.00 | 74.5 | 70.67 |
| Random | −6.0 | −5.5 | −5.75 | 74.0 | 71.50 |
| Random high | −5.5 | −5.0 | −5.25 | 75.0 | 72.17 |
| Lowest margin | −6.5 | **−2.5** | **−4.50** | 74.0 | 72.17 |
| MFR | −6.0 | −4.0 | −5.00 | **76.5** | **72.83** |

What this first cell shows:

- Forgetting is real: without replay, safe accuracy falls by 9.5 points after the final quality stage.
- Every replay method improves mean retention over no replay on the primary metric.
- MFR beats equal-budget random by 0.75 retention points and finishes 2.5 points higher on quality, so it satisfies
  the predeclared success rule in this cell.
- `lowest_margin` has the best pure retention, while MFR has the best final average across all three behaviors. The
  current evidence therefore supports a promising retention–plasticity balance for MFR, not a claim that MFR always
  minimizes forgetting.
- MFR slightly outperforms higher-budget random while using 444 rather than 666 replayed examples across Stages 2
  and 3. This is encouraging for replay-data efficiency.
- MFR takes 37.72 minutes versus 33.58 minutes for equal-budget random on this run. Its approximately 4.14-minute
  overhead comes almost entirely from scoring replay candidates.
- All current paired-bootstrap intervals cross or touch zero. The observed differences are therefore encouraging
  but not yet statistically conclusive.

Only one of four order/seed cells is complete. The results must be replicated across the remaining seed and task
order before making a final claim. Notebook 07 contains the current tables, plots, replay audit, uncertainty analysis,
and cost comparison.

## Project status and next steps

Completed:

- Frozen v2 data, manifest, and experiment protocol
- Reference-model probability cache with a live numerical canary
- Reproducible training, replay, resume, and Stage-1 reuse pipeline
- Five completed Order-1/Seed-0 runs: no replay, random, random high, lowest margin, and MFR
- Validation summaries, retention plots, replay-allocation audit, bootstrap intervals, and runtime accounting

Remaining:

1. Complete all five methods for Order 1/Seed 1, Order 2/Seed 0, and Order 2/Seed 1.
2. Re-run Notebook 07 on the complete grid and use Notebook 08 for example-level forgetting analysis.
3. Freeze the analysis and final model-selection decision.
4. Run Notebook 09 once on the locked test sets.
5. Generate matched model responses with Notebook 10 and run automatic behavior evaluations.
6. Build and score the blinded human review with Notebook 11.
7. Report aggregate results, uncertainty, compute cost, failure cases, and limitations.

Important current limitations are the single completed order/seed cell, one small base model, quality always appearing
as the final behavior, and the lack of a forced-equal-behavior-allocation control. Because MFR replayed more safe than
helpful pairs during Stage 3, the current experiment cannot fully separate better individual-pair targeting from
better allocation across old behaviors.

## Repository structure

```text
mfr-dpo/
├── configs/
│   └── experiment_protocol.json   # Frozen scientific configuration
├── data/
│   └── v2/                        # Active versioned train/validation/test JSONL files and manifest
├── docs/                          # Protocol, data card, runbook, rubric, and pilot notes
├── notebooks/                     # Active v2 workflow, numbered in execution order
├── pilot/                         # Archived original project code, data, notebooks, and tests; not used for v2 results
├── scripts/                       # Reproducible command-line entry points used by the notebooks
├── src/                           # Data, DPO, replay, cache, analysis, evaluation, and review modules
├── tests/                         # Automated checks for the active v2 implementation
├── requirements.txt               # Main environment
├── requirements-eval.txt          # Additional final-evaluation dependencies
└── pytest.ini                     # Test-discovery configuration
```

### Notebooks

| Notebook | Purpose |
|---|---|
| `01_load_data.ipynb` | Build and validate the frozen v2 splits. |
| `02_data_review.ipynb` | Inspect counts, overlap checks, lengths, and representative examples. |
| `03_train_one_stage.ipynb` | Optional v2 single-stage training sanity check; not part of the final grid. |
| `04_pilot.ipynb` | Optional v2 forgetting demonstration; not used for final claims. |
| `05_build_reference_cache.ipynb` | Build the frozen-base log-probability cache once. |
| `06_run_experiment.ipynb` | Run or resume one protocol-v2 experiment. |
| `07_compare_runs.ipynb` | Compare retention, plasticity, uncertainty, replay behavior, and cost. |
| `08_error_analysis.ipynb` | Inspect which validation examples were forgotten and why. |
| `09_final_test_eval.ipynb` | Perform the one-time locked preference test evaluation. |
| `10_generation_eval.ipynb` | Generate responses and prepare automatic behavioral evaluation. |
| `11_blind_review.ipynb` | Create and score blinded human-review sheets. |

### Source modules and scripts

| Path | Purpose |
|---|---|
| `src/mfr_data.py` | Download, clean, split, validate, hash, and load preference data. |
| `src/mfr_dpo.py` | Load QLoRA models, train DPO, score pairs, and compute preference metrics. |
| `src/mfr_replay.py` | Maintain the replay buffer and implement all selection strategies. |
| `src/mfr_cache.py` | Build and verify the frozen-reference cache. |
| `src/mfr_analysis.py` | Calculate retention, learning, cost, diagnostics, and paired bootstraps. |
| `src/mfr_eval.py` | Guard the locked test, score final checkpoints, and generate responses. |
| `src/mfr_review.py` | Create blinded review materials and summarize ratings. |
| `src/mfr_utils.py` | Load protocol settings, seed runs, record provenance, and validate resumes. |
| `scripts/prepare_data.py` | Command-line data builder used by Notebook 01. |
| `scripts/build_reference_cache.py` | Command-line cache builder used by Notebook 05. |
| `scripts/run_experiment.py` | Main experiment runner used by Notebook 06. |
| `scripts/final_test.py` | Locked final-test runner used by Notebook 09. |
| `scripts/run_ifeval.py` | Optional IFEval runner for generated instruction-following behavior. |

## Running the project

Training and model evaluation are designed for Google Colab with an **NVIDIA A100 GPU**. The repository is cloned
into `/content/mfr-dpo`, while large artifacts are written to the mounted shared Drive folder
`/content/drive/MyDrive/CSCI544/mfr-dpo`. Models, reference caches, run outputs, and generations are intentionally not
committed to Git.

Shared artifact folder: [Google Drive](https://drive.google.com/drive/folders/1Mk7RtTdu0m_f7wnuvzGNzgRSTr8T1bqb?usp=drive_link)

Before using Colab, commit and push the active code and frozen data because the notebooks update from
[the GitHub repository](https://github.com/prabudhd2003/mfr-dpo).

Run the active workflow in numerical notebook order, using the detailed checklist in
[`docs/RUNBOOK.md`](docs/RUNBOOK.md). Do not use anything in `pilot/` for protocol-v2 results, and do not run Notebook
09 until the validation grid and model-selection decision are frozen.

For local installation and automated checks:

```bash
python3 -m pip install -r requirements.txt
python3 -m pytest -q
```

The test suite is lightweight and checks replay budgets, deterministic selection, data leakage, manifest and cache
compatibility, metric calculations, analysis logic, locked-test safeguards, and blinded-review bookkeeping.

## Reproducibility safeguards

- Dataset and model revisions are pinned.
- Every active data file is covered by the checked-in manifest and a SHA-256 hash.
- Runs record their settings, scientific-code fingerprint, Git commit, timings, replay decisions, and checkpoints.
- Dirty working trees, incompatible resumes, invalid Stage-1 sources, changed data, and overwritten completed runs are
  rejected.
- A live 32-pair numerical canary verifies the reference cache before training.
- Large artifacts remain in Drive, while complete runs are marked with `COMPLETE.json`.
- Pilot results are isolated in `pilot/` and must never be mixed with v2 results.
