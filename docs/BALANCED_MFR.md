# Balanced MFR

## Purpose

Original MFR ranks every old pair together by:

```text
forgetting score = peak reference-relative margin - current reference-relative margin
```

It then replays the largest drops. At Stage 3, the buffer contains two earlier behaviors. If most of the largest drops
come from one behavior, that behavior can receive most of the replay slots. The current 75% cap limits this imbalance,
but it does not force equal allocation.

Balanced MFR answers a specific question:

> Does MFR work because it finds better pairs within each old behavior, or because it gives more replay to one
> behavior than another?

Balanced MFR uses the same examples, scoring schedule, training steps, and 10% replay budget as MFR. It changes only
how the replay slots are allocated.

## Exact method definition

Use the method name:

```text
mfr_balanced
```

At each replay-plan refresh:

1. Find the old behaviors currently represented in the buffer.
2. Divide the interval's replay slots as evenly as possible across those behaviors.
3. Within each behavior, rank pairs by `peak_margin - current_margin`, largest first.
4. Break exact ties with the existing seeded random generator.
5. Take the highest-ranked pairs from each behavior until its quota is full.
6. If the slot count is not divisible by the number of behaviors, assign the remainder deterministically and rotate
   the remainder recipient across refresh intervals so one behavior is not always favored.
7. If a behavior has fewer unique pairs than its quota, cycle only after all of its available pairs have been used.

With one old behavior, Balanced MFR must produce the same selection as ordinary MFR. With two old behaviors and an
even number of slots, each behavior must receive exactly half.

The method must retain the existing batch budget:

```text
18 new pairs + 2 replay pairs per full optimizer step
```

It is not a larger-budget method. The existing `max_share_per_dataset` cap is unnecessary after equal quotas are
applied, but the public planning function may keep the argument for compatibility.

## How it differs from existing methods

| Method | Pair ranking | Allocation across old behaviors |
|---|---|---|
| MFR | Largest historical margin drop | Global ranking, with a 75% maximum share |
| Lowest margin | Lowest current relative margin | Global ranking, with a 75% maximum share |
| Balanced MFR | Largest historical margin drop | Equal quota for every old behavior |

This is an ablation, not a claim that equal allocation itself is a new idea. Its scientific value is that it separates
MFR's pair-selection signal from its behavior-allocation effect.

## Required code changes

### `src/mfr_replay.py`

- Add `mfr_balanced` to `METHODS` and `REFRESH_METHODS`.
- Add a balanced quota helper that receives ranked rows, `n_slots`, the seeded RNG or interval index, and returns an
  exact-length plan.
- For each dataset, sort by descending `peak_margin - current_margin` with seeded tie-breaking.
- Interleave or otherwise combine the per-dataset selections without changing their quotas.
- Preserve the existing output columns: `id`, `dataset`, `selection_score`, `selection_rank`, and `method`.
- Add an allocation field such as `dataset_quota` if useful for auditability. If a new replay-log column is added,
  update the empty replay-log schema in `src/mfr_dpo.py`.
- Do not change the behavior of `none`, `random`, `random_high`, `lowest_margin`, or `mfr`.

### `configs/experiment_protocol.json`

- Add `mfr_balanced` to `secondary_methods`.
- Do not add an `old_per_step_overrides` entry; it must use the standard two replay pairs.
- Keep the model, data, optimizer, DPO beta, LoRA settings, buffer size, and refresh count unchanged.

### `tests/test_replay.py`

Add tests proving that:

- the method fills exactly the requested number of slots;
- two old behaviors receive equal counts when the slot count is even;
- counts differ by at most one when the slot count is odd;
- the remainder assignment is deterministic and is not permanently biased to one behavior;
- the largest drops within each behavior are selected;
- with one old behavior, the result matches MFR;
- selection is deterministic for the same seed and changes only where expected for another seed;
- the standard 10% replay budget is unchanged;
- `needs_refresh("mfr_balanced")` is true.

### `tests/test_carc.py` and `tests/test_protocol.py`

- Confirm that the configured method is accepted by the group runner.
- Confirm that `--methods mfr_balanced` can run without requesting the already completed methods.
- Add or update protocol-validation expectations for the new method.

### `src/mfr_analysis.py` and `notebooks/07_compare_runs.ipynb`

- Add the display label `Balanced MFR 10%` and a distinct, consistent color.
- Include the method in the full-grid summary, per-dataset forgetting, heatmaps, paired comparisons, and runtime table.
- Compare it directly with original MFR, equal-budget random, and lowest margin.

## Stage-1 compatibility

The Stage-1-specific compatibility fingerprint is already implemented and tested. It allows a replay-only extension
to borrow the completed no-replay Stage-1 checkpoint while keeping full resume validation strict. Do not replace or
weaken it.

When adding Balanced MFR, verify that:

1. Keep the full `scientific_code_sha256` for complete-run provenance and resume validation.
2. Keep using the separate Stage-1 fingerprint already stored by new runs.
3. Preserve the explicit legacy compatibility path for the completed v2 protocol-2.1 artifacts.
4. Confirm that replay-only changes allow Stage-1 reuse while data, model, seed, order, optimizer, DPO, LoRA, or
   Stage-1 training changes are rejected.

## Checks before GPU submission

```bash
python -m pytest -q
git status --short
git diff --check
```

Commit and push the code, then pull the same commit on CARC. The CARC checkout must be clean.

## CARC command

After setting up the terminal as described in `CARC.md`, run only Balanced MFR across the existing eight cells:

```bash
for order in 1 2 3 4; do
  for seed in 0 1; do
    python scripts/submit_carc.py group \
      --output-dir "$MFR_OUTPUT_DIR" \
      --account xiangren_1987 \
      --gpu l40s \
      --time 04:00:00 \
      --order "$order" \
      --seed "$seed" \
      --methods mfr_balanced
  done
done
```

The six completed methods do not need to be rerun if the common data, model, training, scoring, and replay budgets are
unchanged and Stage-1 compatibility is verified.

## How to interpret the result

- **Balanced MFR beats original MFR:** the original method was hurt by unequal behavior allocation.
- **Balanced MFR matches original MFR:** the 75% cap was already sufficient; allocation was not the main issue.
- **Balanced MFR improves retention but lowers final-task learning:** forced allocation makes replay more conservative.
- **Balanced MFR still loses to lowest margin:** current difficulty remains more useful than historical drop even after
  allocation is controlled.

Report the complete eight-cell result and paired intervals, not only the cells where Balanced MFR wins.
