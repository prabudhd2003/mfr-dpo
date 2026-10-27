# PID-inspired replay: branch setup and CARC gain search

Use a new branch named `feat/pid-replay-gains` in a separate Git worktree. A worktree is another folder for the same repository, with its own checked-out branch. This lets the shared `group26` folder keep its current branch while these jobs use the new code.

This package implements the proposed selector; it has not been trained or benchmarked on a GPU. It was prepared against the uploaded `mfr-dpo-project.zip`, using the separately attached `CARC.md` for the CARC paths, account, environment, and L40S request. That guide describes methods and Slurm actions absent from the ZIP, so the CARC checkout may be newer. Apply the patch with Git's compatibility check. The `changed_files/` folder contains reference copies for reviewing or porting individual changes, rather than copies to overwrite newer files wholesale.

## 1. What the experiment changes

Each stored pair has a fixed learned-margin target, `a`, and its latest margin, `m`. The new selector ranks old pairs by:

```text
e = a - m
I = previous_I + e * dt
D = (e - previous_e) / dt
priority = e + Ki * I + Kd * D
```

`Kp` is fixed at 1 because multiplying every gain by the same positive number leaves the ranking unchanged. `dt` is measured in nominal replay windows: 1 at an ordinary refresh, and 20/23 for the last window of a 2,000-pair stage. Integration uses the signed error at the end of the elapsed window. This initial experiment uses neither integral clipping nor derivative smoothing; those choices stay fixed throughout the gain search.

The selector continues to use DPO, 18 new + 2 replay pairs per full update, the 500-pair memory, the existing interval planner and dataset cap, and the existing scoring passes. There are five plans and four in-stage rescoring passes. For 2,000 new pairs, the actual replay counts are 46, 46, 46, 46, and 38: 222 presentations total. At a topic boundary, surviving older pairs retain their controller history, newly added pairs start at zero, and the last interval is incorporated once before saving `buffer.csv`.

These priorities are a heuristic. They do not predict the effect of an update on every other pair, and classical PID stability guarantees do not follow.

## 2. Files to change

The patch modifies six files and adds five. Apply it as one change so the method registration, saved state, and provenance checks agree.

| File | Action | Purpose |
|---|---|---|
| `configs/experiment_protocol.json` | Modify | Register 16 named profiles and their gains. |
| `src/mfr_replay.py` | Modify | Register the selectors, preserve controller state, and rank by PID priority. |
| `src/mfr_dpo.py` | Modify | Advance state at refreshes and log score components. |
| `scripts/run_experiment.py` | Modify | Record gains, save boundary state, and publish stage results after the buffer is saved. |
| `src/mfr_utils.py` | Modify | Fingerprint PID code/configuration and reject resumes with changed gains. |
| `tests/test_carc.py` | Modify | Replace a stale hard-coded default method list with the configured list. |
| `src/mfr_pid.py` | Add | CPU-only controller math and profile loading. |
| `scripts/pid_results.py` | Add | Paired validation summaries and suggestions for the integral search. |
| `tests/test_pid.py` | Add | Controller, regression, budget, resume, and analysis tests. |
| `tests/fixtures/replay_snapshot_plans.json` | Add | Golden plans captured from the original uploaded replay code. |
| `docs/PID_GAIN_SEARCH.md` | Add | This runbook. |

In the uploaded snapshot, `run_experiment_group.py` and `submit_carc.py` already validate methods from the protocol, so they accept these new names without edits. `carc_job.sh` already forwards those names. Preserve any newer CARC actions or replay variants when porting the patch. The frozen data and reference cache are reused.

The existing `NEXT_STEPS.md` describes a previously frozen method list. Treat this gain search as a separate pilot extending that plan, and record it as such in the report. The old experiment results remain in their original artifact folder.

## 3. Create the branch on CARC

Connect to CARC as usual. These commands fetch the committed `project` branch and create a separate checkout without switching the shared folder's branch:

```bash
cd /project2/xiangren_1987/group26
git status --short
git fetch origin project
git worktree add -b feat/pid-replay-gains /project2/xiangren_1987/group26-pid-gains origin/project
cd /project2/xiangren_1987/group26-pid-gains
git branch --show-current
```

The last command should print `feat/pid-replay-gains`. This worktree contains committed files from `origin/project`; uncommitted work in the shared checkout is not copied into it.

## 4. Upload and apply the package

Download `pid-replay-delivery.zip` and transfer it using your usual CARC file-transfer method to `/project2/xiangren_1987/pid-replay-delivery.zip`. Unpack it outside the checkout:

```bash
unzip /project2/xiangren_1987/pid-replay-delivery.zip -d /project2/xiangren_1987/pid-replay-package
export PID_PACKAGE_DIR=/project2/xiangren_1987/pid-replay-package
cd /project2/xiangren_1987/group26-pid-gains
git apply --check "$PID_PACKAGE_DIR/pid-replay.patch" && git apply "$PID_PACKAGE_DIR/pid-replay.patch"
git diff --stat
git status --short
```

If the compatibility check fails, Git leaves the checkout unchanged. Compare the files in the table with the reference copies under `changed_files/` and port the indicated changes into the newer code. A failed check is a version mismatch to resolve before running jobs. Keep existing methods and their settings when adding the `pid_replay_profiles` object and profile names. Run the full CPU suite after any port.

## 5. Activate the existing training environment and verify

Use the absolute shared environment path because the new worktree has no local `.conda` symlink:

```bash
module purge
module load conda
eval "$(conda shell.bash hook)"
conda activate /project2/xiangren_1987/group26/.conda/envs/mfr-dpo
which python
python -m pytest -q
git diff --check
```

No new Python package is required. The shared training environment stays as provided in `CARC.md`.

Local verification passed 16 new `unittest` cases and all 52 existing test functions using a small standard-library runner. `pytest` was unavailable in the preparation workspace, so run the full `pytest` command above on CARC before submission. The new tests can also be run independently:

```bash
PYTHONPATH=src:tests python -m unittest discover -s tests -p test_pid.py -v
```

The tests check that P-only selection matches MFR, the original controls match the uploaded snapshot's golden plans, replay stays at 222 presentations, four refreshes occur, and controller history survives saving, loading, and topic changes. They do not establish training performance.

Commit the complete implementation before running jobs:

```bash
git add configs/experiment_protocol.json src/mfr_pid.py src/mfr_replay.py src/mfr_dpo.py src/mfr_utils.py scripts/run_experiment.py scripts/pid_results.py tests/test_pid.py tests/test_carc.py tests/fixtures/replay_snapshot_plans.json docs/PID_GAIN_SEARCH.md
git commit -m "Add PID-inspired replay gain search"
git status --short
git rev-parse HEAD
```

`git status --short` must be empty. Record the commit. Keep this worktree on that commit while jobs are queued or running: the existing worker executes files from the worktree when the job starts. Optional backup after review: `git push -u origin feat/pid-replay-gains`.

## 6. Isolate results and share the existing cache and weights

Set these variables in every new terminal session used for this experiment:

```bash
export MFR_OUTPUT_DIR=/project2/xiangren_1987/group26/artifacts
export PID_OUTPUT_DIR=/project2/xiangren_1987/group26/artifacts/pid_gain_search_v1
export HF_HOME="$MFR_OUTPUT_DIR/huggingface"
```

Create a separate artifact root and link its cache and model-weight paths to the existing assets. This script verifies existing links and refuses to overwrite a different destination:

```bash
python - <<'PY'
import os
from pathlib import Path
shared = Path(os.environ['MFR_OUTPUT_DIR'])
pilot = Path(os.environ['PID_OUTPUT_DIR'])
(pilot / 'cache').mkdir(parents=True, exist_ok=True)
for relative in ['huggingface', 'cache/reference_v2.csv', 'cache/reference_v2.manifest.json']:
    source = shared / relative
    destination = pilot / relative
    if not source.exists():
        raise FileNotFoundError(source)
    if destination.is_symlink() or destination.exists():
        if destination.resolve() != source.resolve():
            raise RuntimeError(f'Existing path has a different target: {destination}')
    else:
        destination.symlink_to(source, target_is_directory=source.is_dir())
    print(f'{destination} -> {source}')
PY
```

The uploaded runner requires the same scientific code fingerprint for borrowed Stage 1 checkpoints. Adding this method changes that fingerprint. This runbook therefore trains a fresh `none` run once per order/seed in the pilot folder; the other methods in that cell reuse its Stage 1. It keeps the existing compatibility checks intact and provides controls from the same code. This is additional pilot compute; existing main-study runs are not overwritten. If the newer CARC code has a separate, verified Stage 1 compatibility mechanism, retain that mechanism when porting.

Always pass `--methods` explicitly: the protocol also lists future methods that are absent from the uploaded source, and the 16 profiles are candidates to screen selectively.

## 7. Run one smoke cell first

This cell runs fresh controls and one PD candidate (`Kp=1, Ki=0, Kd=0.25`). The group runner executes `none` first and then shares Stage 1:

```bash
python scripts/submit_carc.py group \
  --output-dir "$PID_OUTPUT_DIR" --account xiangren_1987 --gpu l40s --time 06:00:00 \
  --conda-env /project2/xiangren_1987/group26/.conda/envs/mfr-dpo \
  --order 1 --seed 0 --methods none,random,lowest_margin,mfr,pid_i000_d025 --dry-run
```

Inspect the printed command for the `group26-pid-gains` checkout, pilot output path, absolute training environment, and one L40S. Then submit the same command without `--dry-run`:

```bash
python scripts/submit_carc.py group \
  --output-dir "$PID_OUTPUT_DIR" --account xiangren_1987 --gpu l40s --time 06:00:00 \
  --conda-env /project2/xiangren_1987/group26/.conda/envs/mfr-dpo \
  --order 1 --seed 0 --methods none,random,lowest_margin,mfr,pid_i000_d025
```

Six hours is a starting time limit for the multi-method group, not a runtime estimate. Check the observed runtime before choosing limits for larger groups.

Monitor with:

```bash
squeue --me --format="%.18i %.20j %.2t %.10M %.30R"
ls -lt "$PID_OUTPUT_DIR/logs"
```

Use the exact log filename from that listing with `tail -f`. After completion, inspect the PID run's settings, replay counts, and trace:

```bash
python - <<'PY'
import json
import os
from pathlib import Path
import pandas as pd
run = Path(os.environ['PID_OUTPUT_DIR']) / 'runs/v2_o1_pid_i000_d025_s0'
assert (run / 'COMPLETE.json').exists(), f'Not complete: {run}'
settings = json.loads((run / 'settings.json').read_text())
print('Gains:', settings['pid_profile'])
for stage in ['stage2_safe', 'stage3_quality']:
    folder = run / stage
    history = pd.read_csv(folder / 'history.csv')
    replay = pd.read_csv(folder / 'replay_log.csv')
    trace = pd.read_csv(folder / 'pid_trace.csv')
    assert history['n_new'].sum() == 2000
    assert history['n_replay'].sum() == len(replay) == 222
    assert replay.groupby('interval').size().tolist() == [46, 46, 46, 46, 38]
    assert len(trace) == 2500
    print(stage, 'replay counts:', replay.groupby('interval').size().to_dict())
    print(trace.groupby('interval')[['pid_p_term', 'pid_i_term', 'pid_d_term']].mean())
PY
```

## 8. Screen the derivative gain

After the smoke cell finishes, use orders 1 and 3 with seed 0. The four settings are MFR (`Ki=Kd=0`) and PD with `Kd` 0.25, 0.5, or 1.0. Random and lowest-margin are paired controls. The completed smoke runs will be skipped.

Dry-run both commands first:

```bash
export PID_PD_METHODS=none,random,lowest_margin,mfr,pid_i000_d025,pid_i000_d050,pid_i000_d100
for order in 1 3; do
  python scripts/submit_carc.py group \
    --output-dir "$PID_OUTPUT_DIR" --account xiangren_1987 --gpu l40s --time 06:00:00 \
    --conda-env /project2/xiangren_1987/group26/.conda/envs/mfr-dpo \
    --order "$order" --seed 0 --methods "$PID_PD_METHODS" --dry-run
done
```

After checking them, repeat that loop with `--dry-run` removed. Do not overlap jobs that contain the same order, seed, and method. This screen has 14 completed run folders across two cells, including the smoke runs, with Stage 1 shared within each cell.

When all finish:

```bash
python scripts/pid_results.py --output-dir "$PID_OUTPUT_DIR" \
  --orders 1,3 --seeds 0 --methods "$PID_PD_METHODS" --label pd_screen
```

The script writes `pid_analysis/pd_screen_cells.csv` and `pd_screen_summary.csv`. It refuses incomplete cells or mixed scientific code fingerprints. It reports old-behavior retention change, final three-behavior average, final current-topic score, runtime, and paired differences from Random and lowest margin. Larger retention change is better: -3 is better than -5.

## 9. Add the integral term to the best two P/PD settings

Keep `Kp=1`. Screen `Ki` 0.02, 0.1, and 0.3 for the best two derivative settings, including plain MFR as a possible `Kd=0` setting. The helper chooses by retention among settings whose mean final current-topic score is within 2 points of Random. Review the PD summary first, then generate the six profile names:

```bash
export PID_I_METHODS="$(python scripts/pid_results.py --output-dir "$PID_OUTPUT_DIR" \
  --orders 1,3 --seeds 0 --methods "$PID_PD_METHODS" --suggest-integral)"
printf '%s\n' "$PID_I_METHODS"
```

An example profile is `pid_i010_d050`: `Ki=0.10`, `Kd=0.50`. The code suffixes are labels; the authoritative numeric values are in `pid_replay_profiles` and each run's `settings.json`. If fewer than two derivative settings pass the new-learning guard, the helper raises an error; review the screen before expanding it.

Submit the six named profiles on orders 1 and 3, seed 0. As before, first use `--dry-run`, inspect the commands, and repeat without it:

```bash
for order in 1 3; do
  python scripts/submit_carc.py group \
    --output-dir "$PID_OUTPUT_DIR" --account xiangren_1987 --gpu l40s --time 06:00:00 \
    --conda-env /project2/xiangren_1987/group26/.conda/envs/mfr-dpo \
    --order "$order" --seed 0 --methods "${PID_I_METHODS:?Generate the integral profile list first}" --dry-run
done
```

This adds up to 12 run folders. The profiles are already committed, so no code or configuration edit is needed between trials. Summarize all screened settings together:

```bash
export PID_SCREEN_METHODS="$PID_PD_METHODS,$PID_I_METHODS"
python scripts/pid_results.py --output-dir "$PID_OUTPUT_DIR" \
  --orders 1,3 --seeds 0 --methods "$PID_SCREEN_METHODS" --label full_screen
```

## 10. Validate finalists on every development cell

Choose one or two promising PID profiles from the complete screen. Prefer retention gains that persist across both orders and keep new-topic learning within the Random tolerance. Inspect the component traces and selection logs: identical selected sets mean different gains may be equivalent on these runs. A large integral contribution or alternating selected sets is a diagnostic to investigate, not proof of instability.

Set `PID_FINALISTS` to the actual selected profile name, or two comma-separated names. For example, the following assignment is illustrative, not a recommended winner:

```bash
export PID_FINALISTS=pid_i010_d050
```

Dry-run the full development grid, then repeat without `--dry-run`:

```bash
for order in 1 2 3 4; do
  for seed in 0 1; do
    python scripts/submit_carc.py group \
      --output-dir "$PID_OUTPUT_DIR" --account xiangren_1987 --gpu l40s --time 06:00:00 \
      --conda-env /project2/xiangren_1987/group26/.conda/envs/mfr-dpo \
      --order "$order" --seed "$seed" --methods "none,random,lowest_margin,mfr,$PID_FINALISTS" --dry-run
  done
done
```

Completed runs are skipped. Summarize the eight paired cells:

```bash
python scripts/pid_results.py --output-dir "$PID_OUTPUT_DIR" \
  --orders 1,2,3,4 --seeds 0,1 \
  --methods "none,random,lowest_margin,mfr,$PID_FINALISTS" --label finalists
```

For this gain search, compare mean earlier-behavior retention change, enforce the 2-point mean final current-topic tolerance versus Random, and report final average and runtime. Lowest margin is the stronger observed baseline to beat. Use paired order/seed results and the existing paired-margin analysis for uncertainty; the summary table alone does not establish significance.

The main study's `NEXT_STEPS.md` also has a final method-selection rule: pass the retention/new-learning gates, then rank the eligible methods by final average and use runtime for a near tie. Keep that separate from the exploratory gain screen. If PID joins the main study, document its candidate status and the selection rule before confirmation.

## 11. Freeze, confirm, and evaluate

Freeze one shared gain profile after development. Record its exact gains, commit, selection rule, and screened configurations. Seed 2 remains confirmation data; it is not used to choose or revise gains. The uploaded JSON lists only seeds 0 and 1, so the confirmation step must explicitly add seed 2 on a later committed revision. That changes the fingerprint; finish development jobs and resumes first, then create fresh compatible seed-2 sources and run the frozen profile with the controls across all four orders.

Use the locked test once after the method is frozen, following the main project's evaluation plan. This package uses only the existing training process and validation scores during tuning; it does not submit the test action.

## 12. Failure and resume behavior

If a job fails or times out, first check `squeue` and `sacct` to confirm it is no longer running, then resubmit the same command with the same committed code. Keep completed and partial run folders. The group runner skips `COMPLETE.json` runs and resumes partial runs from the last completed stage; it restarts an unfinished stage rather than continuing an individual window. The preceding stage's `buffer.csv` restores controller history.

Do not change gains under an existing profile name and resume into its run folder. The config fingerprint and saved gains reject that combination. Any later, genuinely different experiment needs its own committed configuration and output namespace. If final results exist without a completion marker, follow the existing runner's request to inspect them rather than manufacturing the marker.

The first action is to create the worktree and apply the patch. The first GPU action is the single smoke cell in Step 7.
