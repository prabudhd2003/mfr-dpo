# CARC runbook: from today to the paper tables

Run these in order. Each block starts from a fresh CARC login shell. `squeue -u $USER` shows
running jobs; wait for a block's jobs to finish before starting a block marked **WAIT**.

## 0. Every session starts with this

```bash
cd /project2/xiangren_1987/grp26-mfr-dpo
module purge
module load conda
eval "$(conda shell.bash hook)"
conda activate /project2/xiangren_1987/grp26-mfr-dpo/.conda/envs/mfr-dpo
export MFR_OUTPUT_DIR=/project2/xiangren_1987/grp26-mfr-dpo/artifacts
export LLAMA_OUT=/project2/xiangren_1987/grp26-mfr-dpo/artifacts_llama32_3b
export ACCT=xiangren_1987
```

## 1. Wait for the running jobs (EWC, At-Risk MFR, Balanced MFR)

Do **not** pull the new code until they finish: the code hash changes and an interrupted run
would refuse to resume.

```bash
squeue -u $USER
```

## 2. Pull the new code and check it (CPU, login node)

```bash
git pull
python -m pytest -q
lm-eval --help | head -5
```

## 3. Choose the EWC coefficient, then freeze

Open `notebooks/07a_results.ipynb` (section 11), apply the frozen rule, then on your laptop:
fill the date, commit and EWC choice into `docs/FREEZE.md`, commit, and tag `freeze-v1`; push the
tag. Then on CARC:

```bash
git pull --tags
git describe --tags        # must show freeze-v1
export EWC=ewc_1000        # <- the coefficient you chose
```

If At-Risk or Balanced MFR qualify (see FREEZE.md), append them to `METHODS_B` in step 5.

## 4. Qwen confirmation, part A: `none` for every new cell (Stage 1 for everyone else)

```bash
submit_none() {
  python scripts/submit_carc.py group --output-dir "$MFR_OUTPUT_DIR" --account $ACCT \
    --gpu l40s --time 02:00:00 --order "$1" --seed "$2" --methods none
}
for order in 5 6; do for seed in 0 1; do submit_none $order $seed; done; done
for order in 1 2 3 4 5 6; do for seed in 2 3 4; do submit_none $order $seed; done; done
```

## 5. Qwen confirmation, part B: all other methods (**WAIT** for step 4)

```bash
METHODS_A=random,random_high,mfr,lowest_margin
METHODS_B=mir_dpo,cpmr,dapr_weak,$EWC
submit_methods() {
  for methods in $METHODS_A $METHODS_B; do
    python scripts/submit_carc.py group --output-dir "$MFR_OUTPUT_DIR" --account $ACCT \
      --gpu l40s --time 04:00:00 --order "$1" --seed "$2" --methods "$methods"
  done
}
for order in 5 6; do for seed in 0 1; do submit_methods $order $seed; done; done
for order in 1 2 3 4 5 6; do for seed in 2 3 4; do submit_methods $order $seed; done; done
```

## 6. Llama-3.2-3B: data check, reference cache, pilot (can run in parallel with step 5)

Needs Hugging Face access to `meta-llama/Llama-3.2-3B-Instruct` and `HF_TOKEN` set.

```bash
python scripts/submit_second_model_carc.py cache --output-dir "$LLAMA_OUT" --account $ACCT
# WAIT for the cache job, then the pilot (check runtime and memory in its log):
python scripts/submit_second_model_carc.py group --output-dir "$LLAMA_OUT" --account $ACCT \
  --time 06:00:00 --order 1 --seed 0 --methods none,lowest_margin
```

## 7. Llama grid (**WAIT** for a good pilot)

```bash
for order in 1 2 3 4 5 6; do for seed in 0 1 2; do
  python scripts/submit_second_model_carc.py group --output-dir "$LLAMA_OUT" --account $ACCT \
    --time 08:00:00 --order $order --seed $seed --methods none
done; done
# WAIT for the none runs, then:
for order in 1 2 3 4 5 6; do for seed in 0 1 2; do
  for methods in random,lowest_margin mfr,dapr_weak; do
    python scripts/submit_second_model_carc.py group --output-dir "$LLAMA_OUT" --account $ACCT \
      --time 08:00:00 --order $order --seed $seed --methods $methods
  done
done; done
for seed in 0 1 2; do
  python scripts/submit_second_model_carc.py joint --output-dir "$LLAMA_OUT" --account $ACCT --seed $seed
done
```
(Order 1 seed 0 `none` and `lowest_margin` already exist from the pilot; resubmitting skips them.)

## 8. Look 1: locked preference test on all 30 Qwen cells (**WAIT** for step 5)

```bash
QWEN_METHODS="random random_high mfr lowest_margin mir_dpo cpmr dapr_weak $EWC"
locked() {
  python scripts/submit_eval_carc.py locked-test --output-dir "$MFR_OUTPUT_DIR" --account $ACCT \
    --run-dir "$1" --confirm-final-evaluation
}
for run in "$MFR_OUTPUT_DIR"/runs/v2_o*_none_s[0-4]; do locked "$run"; done
# WAIT for the none jobs (their Stage-1 scores are copied by the others), then:
for m in $QWEN_METHODS; do for run in "$MFR_OUTPUT_DIR"/runs/v2_o*_${m}_s[0-4]; do locked "$run"; done; done
for seed in 0 1 2 3 4; do locked "$MFR_OUTPUT_DIR/joint_runs/v2_joint_s$seed"; done
```

Then the predeclared analysis (CPU):

```bash
python - <<'PY'
import os, sys; sys.path.insert(0, "src")
import mfr_analysis, mfr_stats
cells = mfr_stats.cell_metrics(mfr_analysis.load_locked_test(os.environ["MFR_OUTPUT_DIR"]))
table, decision = mfr_stats.primary_analysis(cells, look=1)
print(table.to_string(index=False)); print("DECISION:", decision)
PY
```

## 9. Only if DECISION is `extend`: seeds 5–9, then look 2

```bash
for order in 1 2 3 4 5 6; do for seed in 5 6 7 8 9; do submit_none $order $seed; done; done
# WAIT, then:
for order in 1 2 3 4 5 6; do for seed in 5 6 7 8 9; do
  python scripts/submit_carc.py group --output-dir "$MFR_OUTPUT_DIR" --account $ACCT \
    --gpu l40s --time 04:00:00 --order $order --seed $seed --methods random,lowest_margin,dapr_weak
done; done
# WAIT, then locked test for the new runs (none first, then the rest):
for run in "$MFR_OUTPUT_DIR"/runs/v2_o*_none_s[5-9]; do locked "$run"; done
# WAIT
for m in random lowest_margin dapr_weak; do for run in "$MFR_OUTPUT_DIR"/runs/v2_o*_${m}_s[5-9]; do locked "$run"; done; done
```
Rerun the step-8 Python block with `look=2`. That result is final.

## 10. Generation evaluation (Qwen)

```bash
E="python scripts/submit_eval_carc.py"
OUT="--output-dir $MFR_OUTPUT_DIR --account $ACCT"
$E validate-judge $OUT                       # WAIT: must print accepted=True
$E generate $OUT --base-model --confirm-final-evaluation
for m in none $QWEN_METHODS; do for run in "$MFR_OUTPUT_DIR"/runs/v2_o*_${m}_s[0-4]; do
  $E generate $OUT --run-dir "$run" --confirm-final-evaluation
done; done
for run in "$MFR_OUTPUT_DIR"/runs/v2_o[1-4]_dapr_s[01]; do        # DAPR-Strong, development cells
  $E generate $OUT --run-dir "$run" --confirm-final-evaluation
done
for seed in 0 1 2 3 4; do
  $E generate $OUT --run-dir "$MFR_OUTPUT_DIR/joint_runs/v2_joint_s$seed" --confirm-final-evaluation
done
# Behavioral forgetting checkpoints: Safety (orders where Safe is not last), Helpful/Quality (seeds 2-4)
for o in 1 2 4 6; do for m in none $QWEN_METHODS; do for run in "$MFR_OUTPUT_DIR"/runs/v2_o${o}_${m}_s[0-4]; do
  $E generate $OUT --run-dir "$run" --checkpoint post_stage --behavior safe --confirm-final-evaluation
done; done; done
for m in lowest_margin dapr_weak random mfr; do for run in "$MFR_OUTPUT_DIR"/runs/v2_o*_${m}_s[2-4]; do
  order=$(basename "$run" | sed 's/v2_o\([0-9]\).*/\1/')
  case $order in 1|5) behaviors=helpful;; 3|4) behaviors=quality;; *) behaviors="";; esac
  for b in $behaviors; do
    $E generate $OUT --run-dir "$run" --checkpoint post_stage --behavior $b --max-prompts 150 --confirm-final-evaluation
  done
done; done
```

**WAIT** for generation, then grading:

```bash
for run in "$MFR_OUTPUT_DIR"/runs/*/ "$MFR_OUTPUT_DIR"/joint_runs/*/ "$MFR_OUTPUT_DIR"/base_model/*/; do
  [ -f "$run/generation/final_eval/safe_test.jsonl" ] || continue
  $E safety $OUT --run-dir "$run"
  $E ifeval $OUT --run-dir "$run" --confirm-final-evaluation
  [ -f "$run/generation/post_stage_safe/safe_test.jsonl" ] && \
    $E safety $OUT --run-dir "$run" --checkpoint post_stage --behavior safe
  ls "$run"/generation/post_stage_helpful "$run"/generation/post_stage_quality >/dev/null 2>&1 && \
    $E judge $OUT --within-run "$run"
done
for o in 1 2 3 4 5 6; do for s in 2 3 4; do
  R="$MFR_OUTPUT_DIR/runs/v2_o${o}"
  $E judge $OUT --candidate-run ${R}_dapr_weak_s$s --baseline-run ${R}_lowest_margin_s$s
  $E judge $OUT --candidate-run ${R}_mfr_s$s --baseline-run ${R}_random_s$s
done; done
```
(`--within-run` judges only the behaviors that have post-stage files; a missing one is an error to
re-run with `--behaviors helpful` or `--behaviors quality`.)

## 11. Llama locked test and generation (**WAIT** for step 7)

```bash
LP="--protocol configs/second_model_protocol.json"
LOUT="--output-dir $LLAMA_OUT --account $ACCT"
for run in "$LLAMA_OUT"/runs/v2_o*_none_s*; do $E locked-test $LOUT $LP --run-dir "$run" --confirm-final-evaluation; done
# WAIT
for run in "$LLAMA_OUT"/runs/v2_o*_s* "$LLAMA_OUT"/joint_runs/*; do
  case $run in *_none_*) continue;; esac
  $E locked-test $LOUT $LP --run-dir "$run" --confirm-final-evaluation
done
for run in "$LLAMA_OUT"/runs/v2_o*_s* "$LLAMA_OUT"/joint_runs/*; do
  $E generate $LOUT $LP --run-dir "$run" --confirm-final-evaluation
done
$E generate $LOUT $LP --base-model --confirm-final-evaluation
# WAIT, then:
for run in "$LLAMA_OUT"/runs/*/ "$LLAMA_OUT"/joint_runs/*/ "$LLAMA_OUT"/base_model/*/; do
  $E safety $LOUT --run-dir "$run"; $E ifeval $LOUT --run-dir "$run" --confirm-final-evaluation
done
```

## 12. Gradient conflict (any time after step 5)

```bash
for o in 1 2 3 4 5 6; do for s in 0 1; do for stage in 2 3; do
  $E gradient-conflict $OUT --run-dir "$MFR_OUTPUT_DIR/runs/v2_o${o}_none_s$s" --stage $stage
done; done; done
```

## 13. Analysis

Open `notebooks/07a_results.ipynb`, `07b_mechanisms.ipynb`, `08_joint_baseline.ipynb`,
`09_generation_evaluation.ipynb` and `11_second_model.ipynb` in Jupyter on CARC (with
`MFR_OUTPUT_DIR` exported) and Run All.
