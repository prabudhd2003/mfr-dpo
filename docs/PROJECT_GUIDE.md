# MFR-DPO Project Guide

This document contains the scientific details of the project: the problem, data, method, experiment, metrics, and
limitations. Current results are kept in the repository README, and execution instructions are kept in `RUNBOOK.md`.

## Problem

Direct Preference Optimization trains a model from pairs containing a prompt, a preferred response, and a rejected
response. In this project, one adapter learns three kinds of preferences in sequence: Helpfulness, Safety, and general
Quality.

Sequential training can cause catastrophic forgetting: after the model learns a later behavior, its preference for
responses from an earlier behavior can weaken. Replay reduces this problem by mixing a small number of old examples
into later training stages.

MFR asks which old examples should be replayed. Immediately after a behavior is learned, the method stores each replay
candidate's preference margin. During later stages, it measures the margin again and prioritizes the pairs with the
largest drop.

## What the pilot established

The original Safe -> Helpful -> Quality pilot was a proof of concept. Safety validation accuracy fell from 87.5%
immediately after Safety training to 69.5% after all three stages, an 18-point drop. Margin analysis also showed that
forgetting was concentrated: some pairs lost much more than others. These findings established the problem and
motivated targeted replay.

The pilot is not used for final conclusions. It used older splits, an initialization that was not fully reproducible,
16 new plus 2 old pairs rather than the final 10% replay budget, and less complete provenance and evaluation. Its code
is preserved under `pilot/`; final tables must use only runs beginning with `v2_`.

## Data

| Behavior | Source | Selection rule |
|---|---|---|
| Helpful | NVIDIA HelpSteer2 | Keep pairs with absolute preference strength of at least 2. |
| Safe | PKU-SafeRLHF | Keep pairs where exactly one response is marked safe and choose the safe response. |
| Quality | UltraFeedback Binarized | Keep pairs whose chosen score exceeds the rejected score by at least 1. |

Representative tasks include travel budgeting and instruction following for Helpful, unsafe or illegal requests for
Safe, and writing feedback or general question answering for Quality.

Each behavior contains 2,000 training, 200 validation, and 300 locked test pairs. `data/v2/manifest.json` records the
pinned dataset and model revisions, row counts, and SHA-256 hashes.

The processing pipeline:

1. Removes missing, empty, identical, and duplicate response pairs.
2. Normalizes prompts with Unicode NFKC normalization, case folding, and whitespace normalization.
3. Removes normalized prompt overlap across all behaviors and splits.
4. Applies the pinned tokenizer and removes sequences longer than 1,024 tokens.
5. Shuffles with seed 0, assigns stable IDs, and preserves source provenance.
6. Validates sizes, IDs, prompt separation, lengths, and file hashes.

## Frozen protocol

The authoritative settings are in `configs/experiment_protocol.json`.

| Setting | Value |
|---|---|
| Model | `Qwen/Qwen2.5-1.5B-Instruct`, pinned revision |
| Training | One QLoRA adapter, sequential DPO, one epoch per behavior |
| Hardware | One fixed NVIDIA A100 type on USC CARC |
| Main orders | Helpful -> Safe -> Quality; Safe -> Helpful -> Quality |
| Quality-retention extension | Quality -> Helpful -> Safe; Quality -> Safe -> Helpful |
| Seeds | 0 and 1 |
| DPO beta | 0.1 |
| Learning rate | `1e-4` |
| Context limit | 1,024 tokens |
| Replay buffer | 500 pairs |
| MFR refreshes | 5 per later stage |
| Maximum dataset share | 75% when another old behavior is available |

There is no replay in Stage 1, so later methods may reuse the compatible Stage-1 checkpoint from the matching
no-replay run.

## Methods

| Method | Training batch and purpose |
|---|---|
| `none` | No old pairs; measures unprotected forgetting. |
| `random` | 18 new + 2 uniformly sampled old pairs; equal-budget baseline. |
| `mfr` | 18 new + 2 old pairs selected by the largest margin loss; proposed method. |
| `random_high` | 18 new + 3 random old pairs; tests whether more replay is sufficient. |
| `lowest_margin` | 18 new + 2 pairs with the lowest current margins; simpler targeted baseline. |

The primary claim compares `mfr` with `random`. `random_high` and `lowest_margin` are secondary controls.

## Metrics and success rule

The primary metric, `accuracy`, is the percentage of validation pairs with a positive length-normalized DPO advantage
relative to the frozen base model. It measures movement toward the preferred response; it is not real-world behavioral
accuracy.

The required secondary metric, `accuracy_sum`, uses summed response log probabilities. Final reporting must include
both.

For an old behavior:

```text
retention change = final accuracy - accuracy immediately after learning it
```

For a newly trained behavior:

```text
new-stage gain = accuracy after its stage - accuracy before its stage
```

MFR passes a matched order-and-seed cell when it retains old behaviors better than equal-budget random replay and its
final new-task score is no more than two points lower. Comparisons are paired by order, seed, validation pair, data,
model revision, and update budget. Report individual cells and paired-bootstrap confidence intervals.

## Reproducibility safeguards

- Dataset and model revisions are pinned.
- Every active data file has a saved hash.
- Runs record settings, Git commit, scientific-code fingerprint, timings, replay decisions, and checkpoints.
- Dirty checkouts, incompatible resumes, changed data, invalid Stage-1 sources, and overwritten completed runs are
  rejected.
- A live 32-pair numerical canary verifies the reference cache before training.
- Complete runs contain `COMPLETE.json`.

## Limitations

- The study uses one 1.5B model and two seeds.
- Quality retention is not measured in the main two orders; the optional Quality-first extension measures it.
- Preference accuracy does not establish generated helpfulness, safety, or quality.
- Dataset labels may be noisy, and source responses may contain factual errors.
- MFR can allocate more replay to one old behavior. Without a forced-equal-allocation control, pair selection cannot be
  fully separated from cross-behavior allocation.
- Candidate scoring adds runtime even when MFR uses fewer replay examples.

These limitations are addressed through matched baselines, confidence intervals, locked-test evaluation, generated
responses, automatic behavior evaluation, blinded human review, and explicit reporting.
