# At-Risk MFR

At-Risk MFR is the risk-aware MFR variant. Use the method name:

```text
mfr_at_risk
```

## Purpose

Original MFR asks: **Which pairs deteriorated most from their earlier peak?**

Lowest-margin replay asks: **Which pairs are weakest right now relative to the base model?**

The completed validation grid shows that lowest margin is currently stronger overall than original MFR. This suggests
that present difficulty matters. At-Risk MFR combines present failure with historical deterioration instead of using
either signal alone.

There are two different margins in the current scorer:

- `margin`: the length-normalized DPO advantage of the tuned policy relative to the frozen reference model;
- `policy_margin`: the tuned policy's absolute chosen-versus-rejected log-probability difference.

These must not be confused. `margin <= 0` only means the adapter is not improving that pair over the reference. It
does **not** necessarily mean the current policy prefers the rejected response. Actual preference failure is:

```text
policy_margin <= 0
```

## Exact method definition

At every buffer refresh:

1. Score both `margin` and `policy_margin` for every buffered pair.
2. Divide replay slots equally across the old behaviors, using the same quota rule as Balanced MFR.
3. Within each behavior, split pairs into two tiers:
   - **Tier 1 — failed:** `current_policy_margin <= 0`;
   - **Tier 2 — not failed:** `current_policy_margin > 0`.
4. Within each tier, rank by historical relative-margin drop:

   ```text
   peak_margin - current_margin
   ```

   Largest drop comes first; use the existing seeded random tie-breaker.
5. Fill each behavior's quota from Tier 1 first, then use Tier 2 as ordinary MFR fallback.
6. Preserve the standard 10% replay budget: 18 new plus 2 replay pairs per full step.

This definition deliberately has no tuned risk threshold. It uses zero because zero has a clear meaning: the policy
currently does not prefer the chosen response. A positive “near failure” threshold may be studied later, but it must
not be selected by looking at locked-test results.

At-Risk MFR includes balanced behavior quotas. This creates a clean sequence of ablations:

| Method | Historical drop | Actual current failure | Equal behavior allocation |
|---|---:|---:|---:|
| Original MFR | Yes | No | No |
| Balanced MFR | Yes | No | Yes |
| At-Risk MFR | Yes | Yes | Yes |
| Lowest margin | No | No; uses current relative margin | No |

The comparison of Balanced MFR with At-Risk MFR isolates the value of the actual-failure signal.

## Required code changes

### `src/mfr_replay.py`

- Add `mfr_at_risk` to `METHODS` and `REFRESH_METHODS`.
- Reuse the existing `current_policy_margin` and trajectory fields added for FMCR. Storing `peak_policy_margin` is
  optional for analysis; selection does not require it.
- Reuse the existing aligned relative and absolute policy-margin score updates rather than creating parallel state.
- Add a selector that applies equal behavior quotas, prioritizes `current_policy_margin <= 0`, and ranks by historical
  relative-margin drop within each tier.
- Preserve deterministic tie-breaking and exact slot counts.
- Add replay-log fields that make the mechanism auditable, for example `risk_tier`, `current_policy_margin`,
  `relative_drop`, and `dataset_quota`.
- Keep loading compatible with old buffer CSV files only through an explicit validation or migration path. Never fill
  a missing current policy margin with an arbitrary number.
- Do not change existing method behavior.

### `src/mfr_dpo.py`

The FMCR path already passes both score types to the replay buffer:

```text
scores["margin"]
scores["policy_margin"]
```

Extend that tested mechanism to `mfr_at_risk` and update the empty replay-log schema if new audit columns are added.

The existing Stage-1 `buffer.csv` files do not contain `current_policy_margin`. At-Risk MFR must therefore perform a
live buffer scoring pass at the start of Stage 2, before its first replay plan is created. This scoring pass computes
both current margin types and allows the existing Stage-1 checkpoint and candidate IDs to be reused safely. Do not
invent, zero-fill, or infer the missing policy margin. Later refreshes follow the normal refresh schedule.

### `scripts/run_experiment.py`

`update_buffer()` already stores aligned relative and policy margins for new candidates and has an FMCR path for
updating the full forecast state. Extend the appropriate branch to `mfr_at_risk`. The saved `buffer.csv` must retain
the required fields so interrupted jobs resume correctly.

### `configs/experiment_protocol.json`

- Add `mfr_at_risk` to `secondary_methods`.
- Do not add an `old_per_step_overrides` entry.
- Keep every common model, data, training, buffer, and scoring setting unchanged.

### Tests

Add tests proving that:

- a pair with `policy_margin <= 0` is chosen before a nonfailed pair in the same behavior;
- among failed pairs, the largest historical relative-margin drop is chosen first;
- if too few failed pairs exist, remaining slots use the ordinary MFR ranking;
- equal behavior quotas are preserved even if all failures come from one behavior;
- `margin <= 0` alone does not place a pair in the failure tier;
- exact ties are deterministic;
- the plan fills exactly the replay budget without duplicate use until cycling is necessary;
- saving and loading preserves both margin types;
- an old Stage-1 buffer without policy margins is scored before the first At-Risk replay selection;
- no At-Risk replay plan can be created while required current policy margins are missing;
- `needs_refresh("mfr_at_risk")` is true;
- the six completed methods are unchanged.

Update `tests/test_budget.py`, `tests/test_replay.py`, `tests/test_carc.py`, and `tests/test_protocol.py` as needed.

### Analysis and Notebook 07

- Add the display label `At-Risk MFR 10%` and a distinct color.
- Include it in all full-grid tables, plots, paired intervals, and runtime comparisons.
- Report how many selected replay slots came from the failure tier.
- Report the number of unique selected pairs and allocation by old behavior.
- Compare directly with Balanced MFR, original MFR, lowest margin, and equal-budget random.

## Stage-1 compatibility

The Stage-1-specific compatibility fingerprint described in `BALANCED_MFR.md` is already implemented and tested.
Use it to borrow existing Stage-1 checkpoints, and keep full resume validation strict.

Because this method adds buffer columns, do not resume a partial original-MFR run as At-Risk MFR. It must start at
Stage 2 from the matching completed no-replay Stage-1 source and create its own run directory.

## Checks before GPU submission

```bash
python -m pytest -q
git status --short
git diff --check
```

Commit and push the code, pull that exact commit on CARC, and confirm the checkout is clean.

## CARC command

After setting up the terminal as described in `CARC.md`, run only the new method:

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
      --methods mfr_at_risk
  done
done
```

The six completed methods do not need to be rerun if their common protocol is unchanged and compatibility checks
pass.

## How to interpret the result

- **At-Risk beats Balanced MFR:** actual current preference failure adds value beyond historical decline and equal
  allocation.
- **At-Risk matches Balanced MFR:** the binary failure signal adds little; allocation was the more important change.
- **At-Risk beats lowest margin:** historical trajectory contributes information beyond present relative difficulty.
- **At-Risk loses to lowest margin:** the simpler current-state signal remains sufficient or more stable.
- **At-Risk helps retention but harms current-task learning:** it may be too conservative; report the trade-off rather
  than calling it an unconditional win.

The strongest result requires improvement across the complete grid and paired uncertainty, not only a higher overall
mean.
