# Frozen PD test evaluation on CARC

## What this evaluates

This adds a dedicated launcher for the selected `pid_i000_d100` profile:
`kp=1`, `ki=0`, `kd=1`. It evaluates the already-trained final adapters on the
frozen `data/v2` test split. Defaults are orders 1, 2, 3, 4 and seeds 0, 1:
eight final checkpoints, with 300 Helpful, 300 Safe, and 300 Quality pairs each.

This follows Step 9 of `docs/NEXT_STEPS.md`: test reports final preference
accuracy per behavior and the final three-behavior average. Retention remains
a validation-only result. It is preference-pair scoring, not generation/judging.

The launcher creates four Slurm jobs, one per order, with the two seeds evaluated
sequentially inside each job. Each job requests one L40S, 8 CPUs, 64 GB RAM and
3 hours. Its worker activates the existing shared environment and uses the
existing pilot Hugging Face cache. It calculates frozen-base reference scores
directly, because the training/validation reference cache need not contain test IDs.

## Add the files to your existing PD branch

Extract the package into the root of the local checkout that has your PD code.
The three additions are:

- `scripts/pd_test.py`
- `tests/test_pd_test.py`
- `docs/PD_TEST_CARC.md`

No existing file needs replacing. Adding these files preserves the training and
scoring fingerprint used by the completed runs. Keep the protocol, model revision,
training files, saved adapters, and data splits at their existing versions.

On your LOCAL computer, inside that repository:

```bash
git branch --show-current
git add scripts/pd_test.py tests/test_pd_test.py docs/PD_TEST_CARC.md
git commit -m "Add frozen PD test evaluation"
git push
```

Use your existing PD feature branch. There is no need to create another branch
or apply the original PID patch again. `git push` assumes the existing branch
already has its upstream, as it did for the gain search.

## Reconnect and initialize CARC

Open CARC OnDemand > Clusters > Discovery Cluster Shell Access. Enter one command
per line. All commands below are for the CARC terminal:

```bash
shared=/project2/xiangren_1987/group26
cd "${shared}-pid-gains"
git pull --ff-only
module purge
module load conda
eval "$(conda shell.bash hook)"
conda activate "$shared/.conda/envs/mfr-dpo"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
out="$shared/artifacts/pid_gain_search_v1"
p=scripts/pd_test.py
```

The existing shared Conda environment is sufficient; this package adds no dependency.

## CPU verification and preflight

```bash
python -m unittest discover -s tests -p test_pd_test.py -q
python "$p" check --output-dir "$out"
```

The first command runs 17 CPU-only checks. The second verifies all eight completed
PD runs, saved scoring settings, model revision, final adapter hashes, the frozen
test files and their 300-pair denominators, and the scientific-code fingerprint.
It should print `READY: 8 completed pid_i000_d100 runs` and eight `TO SCORE` entries
(or `DONE` entries if a test from this launcher is already complete).

Preflight performs no model inference. It does not train or submit a job.

If it finds earlier `final_test` scores created by another evaluator, it stops
instead of overwriting or rescoring them. Preserve those files and inspect the
specific message with the team. A code-fingerprint mismatch similarly means the
current checkout no longer matches the code that produced the selected adapters.

## Preview and submit

```bash
python "$p" submit --output-dir "$out" --dry-run
```

The preview should show four `sbatch` commands, with order arguments 1 through 4,
`--gpus-per-task=l40s:1`, the correct pilot artifact directory, and the shared
`group26/.conda/envs/mfr-dpo` environment. Preview does not freeze the plan, score
the test set, or schedule jobs.

Schedule the four jobs once:

```bash
python "$p" submit --output-dir "$out"
squeue --me
```

This freezes `pd_test/PLAN.json` before submission. The plan records the selected
profile, run list, evaluation batch size, source settings, data, evaluator, and
adapter hashes. Test-time changes to those inputs are rejected. It evaluates no
other gain candidates and starts no training.

For a different allocation or environment prefix, pass `--account` or
`--conda-env`. `--time` changes the scheduling limit, not the scientific plan.

## Monitor

The submission prints each job ID and exact log filename. To list recent PD logs:

```bash
ls -lt "$out/logs"/pdtest-*.out
```

Check an actual job ID, replacing `JOBID`:

```bash
sacct -j JOBID -o JobID,State,Elapsed,ExitCode
```

The job can keep running after the terminal disconnects. On reconnect, repeat
the environment/path setup; the jobs receive absolute paths.

After an interrupted or timed-out job is no longer active, use the same submission
command. Completed orders are skipped, active jobs with the same plan are skipped,
and completed behavior test sets within a partial run are reused. An interrupted
behavior test set is recomputed from the same frozen checkpoint. Never delete
scores or the frozen plan to restart a scientific comparison.

## Summarize after all four jobs finish

```bash
python "$p" summary --output-dir "$out"
```

The summary requires all eight planned runs to be complete and prints `cells: 8`.
It reports exact percentages from the 300-pair counts rather than rounding each
behavior's accuracy to one decimal before averaging.

Share these files for analysis:

- `$out/pd_test/pd_test_summary.csv`: overall mean and per-behavior mean accuracies.
- `$out/pd_test/pd_test_cells.csv`: eight matched order/seed results.
- `$out/pd_test/pd_test_by_topic.csv`: 24 per-run, per-behavior results.

Each selected run also gets the standard `final_test/test_results.csv` and
`final_test/{helpful,safe,quality}_test_scores.csv`, plus evaluation provenance
and a `PD_TEST_COMPLETE.json` marker. Training/validation results and adapters
are preserved. Pair-level scores retain the original test IDs.

For comparisons with MFR or Lowest Margin, use their held-out test scores on the
same data split and matching orders/seeds. Validation accuracy and test accuracy
are different evaluations. The launcher deliberately evaluates the selected PD
profile only, rather than sweeping every PID candidate on the test split.

## If seed 2 has already been trained

The default scope matches the eight completed runs discussed so far. If the team
also has completed PD seed-2 runs, specify `--seeds 0,1,2` on `check`, on the
submission preview, and on the first actual submission. Decide that scope before
creating the frozen test plan. Evaluation does not require adding a seed to the
training protocol or launching another training run.

## Local verification

The package was checked with 17 standard-library CPU tests, synthetic-model
end-to-end scoring and resume checks, mocked Slurm submissions for all four
orders, Python compilation, shell syntax checking of the generated worker, and
comparison of the training/scoring fingerprint to the original PID delivery.
It was not run on a real GPU or submitted to CARC from the authoring environment.
