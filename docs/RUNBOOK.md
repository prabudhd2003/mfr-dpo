# CARC runbook: from today to the paper tables

Run the steps in order. **WAIT** means: let the previous block's jobs finish (`squeue -u $USER`).
GPU-h are L40S estimates. Training numbers come from measured runtimes; evaluation numbers are
rougher, so check the first job of each kind with `sacct -j JOBID --format=JobName,Elapsed,State`.

Total: about **300 GPU-h** (about 370 if step 9 runs).

## 0. Every login

```bash
cd /project2/xiangren_1987/grp26-mfr-dpo
module purge
module load conda
eval "$(conda shell.bash hook)"
conda activate /project2/xiangren_1987/grp26-mfr-dpo/.conda/envs/mfr-dpo
export MFR_OUTPUT_DIR=/project2/xiangren_1987/grp26-mfr-dpo/artifacts
export LLAMA_OUT=/project2/xiangren_1987/grp26-mfr-dpo/artifacts_llama32_3b
export ACCT=xiangren_1987

# helpers (from step 4 on; set EWC to your step-3 choice)
export EWC=ewc_1000
QWEN_METHODS="random random_high mfr lowest_margin mir_dpo cpmr dapr_weak $EWC"
R=$MFR_OUTPUT_DIR/runs
LR=$LLAMA_OUT/runs
E="python scripts/submit_eval_carc.py"
OUT="--output-dir $MFR_OUTPUT_DIR --account $ACCT"
LP="--protocol configs/second_model_protocol.json"
LOUT="--output-dir $LLAMA_OUT --account $ACCT"
submit_none() {
  python scripts/submit_carc.py group --output-dir "$MFR_OUTPUT_DIR" --account $ACCT \
    --gpu l40s --time 02:00:00 --order "$1" --seed "$2" --methods none
}
locked() { $E locked-test $OUT --run-dir "$1" --confirm-final-evaluation; }
```

Run globs use `v2_o[1-6]_<method>_s<seeds>`, never `v2_o*`: `*` would also match
`balanced_mfr` when you ask for `mfr`.

## 1. Wait for the running jobs (0 GPU-h)

EWC, At-Risk MFR and Balanced MFR must finish before you switch branches (the code hash changes).

```bash
squeue -u $USER
```

## 2. Switch to the new code and check it (CPU)

```bash
git fetch origin
git checkout complete
python -m pytest -q
```

## 2b. vLLM environment, once (CPU, login node, ~20 min)

vLLM 0.30 needs an NVIDIA driver >= 580 (CUDA 13). Check a GPU node first:

```bash
srun --account=$ACCT --partition=gpu --gpus-per-task=l40s:1 --time=00:05:00 \
  nvidia-smi --query-gpu=name,driver_version --format=csv
```

```bash
conda create -y -p /project2/xiangren_1987/grp26-mfr-dpo/.conda/envs/mfr-eval python=3.12
conda activate /project2/xiangren_1987/grp26-mfr-dpo/.conda/envs/mfr-eval
pip install -r requirements-vllm.txt
conda activate /project2/xiangren_1987/grp26-mfr-dpo/.conda/envs/mfr-dpo
```

If the driver is older than 580, tell me before installing; we pin an older vLLM instead.

## 2c. Smoke test on one finished run (1 job, ~0.5 GPU-h)

Builds the vLLM base, checks HF-vs-vLLM parity, and runs a few prompts through generation,
WildGuard, Prometheus and IFEval. It writes only to smoke folders.

```bash
sbatch --account=$ACCT scripts/carc_vllm_smoke.sh $R/v2_o1_dapr_weak_s0 $MFR_OUTPUT_DIR
# WAIT, then:
tail -40 vllm_smoke_*.out
```

It must end with `SMOKE TEST PASSED`, and step 2 of its log must say `PARITY PASSED`. Send me
the log either way.

## 2d. Balanced MFR and At-Risk MFR on the 8 development cells (8 jobs, ~8 GPU-h)

Development ablations (docs/BALANCED_MFR.md, docs/AT_RISK_MFR.md). They borrow Stage 1 from the
existing `none` runs. Can run alongside 2c.

```bash
for order in 1 2 3 4; do for seed in 0 1; do
  python scripts/submit_carc.py group --output-dir "$MFR_OUTPUT_DIR" --account $ACCT \
    --gpu l40s --time 03:00:00 --order $order --seed $seed --methods mfr_balanced,mfr_at_risk
done; done
```

When they finish, look at 07a (ranking, uncertainty) and the last section of 07b (slot allocation and
failure-tier share). They qualify for the confirmation grid under the FREEZE.md rule; if so, add them
to `METHODS_B` in step 5.

## 3. Choose the EWC coefficient, then freeze (0 GPU-h)

Open `notebooks/07a_results.ipynb` (section 11) and apply the frozen rule. On your laptop, fill
`docs/FREEZE.md`, commit, `git tag freeze-v1`, then `git push origin complete --tags`. On CARC:

```bash
git pull --tags
git describe --tags        # must show freeze-v1
export EWC=ewc_1000        # <- your choice; re-paste the helpers
```

If At-Risk or Balanced MFR qualify, append them to `METHODS_B` in step 5 and use `--time 05:00:00`.

## 4. Qwen `none` for the 22 new cells (22 jobs, ~8 GPU-h)

```bash
for order in 5 6; do for seed in 0 1; do submit_none $order $seed; done; done
for order in 1 2 3 4 5 6; do for seed in 2 3 4; do submit_none $order $seed; done; done
```

## 5. Qwen, all other methods (44 jobs, ~92 GPU-h; WAIT for step 4)

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

## 6. Llama cache and pilot (~3 GPU-h; can run during step 5; needs HF_TOKEN)

```bash
python scripts/submit_second_model_carc.py cache --output-dir "$LLAMA_OUT" --account $ACCT
# WAIT
python scripts/submit_second_model_carc.py group --output-dir "$LLAMA_OUT" --account $ACCT \
  --time 06:00:00 --order 1 --seed 0 --methods none,lowest_margin
```

Send me the pilot's elapsed time and peak memory.

## 7. Llama grid (57 jobs, ~82 GPU-h; WAIT for a good pilot)

```bash
for order in 1 2 3 4 5 6; do for seed in 0 1 2; do
  python scripts/submit_second_model_carc.py group --output-dir "$LLAMA_OUT" --account $ACCT \
    --time 08:00:00 --order $order --seed $seed --methods none
done; done
# WAIT
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

The pilot's runs are skipped automatically. The Qwen joint runs (seeds 0–4) are already done.

## 8. Look 1: locked test (275 jobs, ~35 GPU-h; WAIT for step 5)

```bash
for run in $R/v2_o[1-6]_none_s[0-4]; do locked $run; done
# WAIT (the other methods copy Stage-1 scores from none)
for m in $QWEN_METHODS; do for run in $R/v2_o[1-6]_${m}_s[0-4]; do locked $run; done; done
for run in $MFR_OUTPUT_DIR/joint_runs/v2_joint_s[0-4]; do locked $run; done
# optional appendix (~5 GPU-h): development ablations on their 8 cells
for m in fmcr dapr dapr_gated dapr_c copr_adapted; do
  for run in $R/v2_o[1-4]_${m}_s[01]; do locked $run; done
done
```

If sbatch reports a job limit, submit one method at a time. Then the analysis (CPU):

```bash
python - <<'PY'
import os, sys; sys.path.insert(0, "src")
import mfr_analysis, mfr_stats
cells = mfr_stats.cell_metrics(mfr_analysis.load_locked_test(os.environ["MFR_OUTPUT_DIR"]))
table, decision = mfr_stats.primary_analysis(cells, look=1)
print(table.to_string(index=False)); print("DECISION:", decision)
PY
```

`stop`: skip step 9. `extend`: run step 9.

## 9. Only if DECISION is `extend` (120 runs + 120 locked tests, ~68 GPU-h)

```bash
for order in 1 2 3 4 5 6; do for seed in 5 6 7 8 9; do submit_none $order $seed; done; done
# WAIT
for order in 1 2 3 4 5 6; do for seed in 5 6 7 8 9; do
  python scripts/submit_carc.py group --output-dir "$MFR_OUTPUT_DIR" --account $ACCT \
    --gpu l40s --time 04:00:00 --order $order --seed $seed --methods random,lowest_margin,dapr_weak
done; done
# WAIT
for run in $R/v2_o[1-6]_none_s[5-9]; do locked $run; done
# WAIT
for m in random lowest_margin dapr_weak; do for run in $R/v2_o[1-6]_${m}_s[5-9]; do locked $run; done; done
```

Rerun the step-8 Python block with `look=2`. That result is final.

## 10. Qwen generation evaluation with vLLM (~45 jobs, ~40 GPU-h)

One job takes a whole method (all its runs); vLLM loads once and swaps LoRA adapters.

```bash
$E vllm-base $OUT                           # skipped if the smoke test built it
$E validate-judge $OUT                      # WAIT: must print accepted=True

# 10a. final checkpoints (12 jobs, ~14 GPU-h)
$E generate $OUT --base-model --confirm-final-evaluation
for m in none $QWEN_METHODS; do
  $E generate $OUT --run-dir $R/v2_o[1-6]_${m}_s[0-4] --confirm-final-evaluation
done
$E generate $OUT --run-dir $R/v2_o[1-4]_dapr_s[01] --confirm-final-evaluation      # DAPR-Strong (dev)
$E generate $OUT --run-dir $MFR_OUTPUT_DIR/joint_runs/v2_joint_s[0-4] --confirm-final-evaluation

# 10b. behavioral forgetting checkpoints (17 jobs, ~4 GPU-h)
for m in none $QWEN_METHODS; do
  $E generate $OUT --run-dir $R/v2_o[1246]_${m}_s[0-4] --checkpoint post_stage --behavior safe \
    --confirm-final-evaluation
done
for m in random mfr lowest_margin dapr_weak; do
  $E generate $OUT --run-dir $R/v2_o[15]_${m}_s[2-4] --checkpoint post_stage --behavior helpful \
    --max-prompts 150 --confirm-final-evaluation
  $E generate $OUT --run-dir $R/v2_o[34]_${m}_s[2-4] --checkpoint post_stage --behavior quality \
    --max-prompts 150 --confirm-final-evaluation
done
```

**WAIT** for 10a and 10b, then grade:

```bash
# 10c. WildGuard + IFEval on final checkpoints (22 jobs, ~20 GPU-h)
EXTRA="$R/v2_o[1-4]_dapr_s[01] $MFR_OUTPUT_DIR/joint_runs/v2_joint_s[0-4] $MFR_OUTPUT_DIR/base_model/*"
for m in none $QWEN_METHODS; do
  $E safety $OUT --run-dir $R/v2_o[1-6]_${m}_s[0-4]
  $E ifeval $OUT --run-dir $R/v2_o[1-6]_${m}_s[0-4] --confirm-final-evaluation
done
$E safety $OUT --run-dir $EXTRA
$E ifeval $OUT --run-dir $EXTRA --confirm-final-evaluation

# 10d. WildGuard on post-Safe checkpoints (9 jobs, ~1 GPU-h)
for m in none $QWEN_METHODS; do
  $E safety $OUT --run-dir $R/v2_o[1246]_${m}_s[0-4] --checkpoint post_stage --behavior safe
done

# 10e. Prometheus (10 jobs, ~3 GPU-h)
for m in random mfr lowest_margin dapr_weak; do
  $E judge $OUT --within-run $R/v2_o[15]_${m}_s[2-4] --behaviors helpful
  $E judge $OUT --within-run $R/v2_o[34]_${m}_s[2-4] --behaviors quality
done
$E judge $OUT --candidate-run $R/v2_o[1-6]_dapr_weak_s[2-4] --baseline-run $R/v2_o[1-6]_lowest_margin_s[2-4]
$E judge $OUT --candidate-run $R/v2_o[1-6]_mfr_s[2-4] --baseline-run $R/v2_o[1-6]_random_s[2-4]
```

Every job skips runs that are already done, so resubmitting the same line after a failure is safe.

## 11. Llama evaluation (~35 GPU-h; WAIT for step 7)

```bash
# 11a. locked test (93 jobs, ~19 GPU-h)
for run in $LR/v2_o[1-6]_none_s[0-2]; do $E locked-test $LOUT $LP --run-dir $run --confirm-final-evaluation; done
# WAIT
for m in random lowest_margin mfr dapr_weak; do for run in $LR/v2_o[1-6]_${m}_s[0-2]; do
  $E locked-test $LOUT $LP --run-dir $run --confirm-final-evaluation
done; done
for run in $LLAMA_OUT/joint_runs/*; do $E locked-test $LOUT $LP --run-dir $run --confirm-final-evaluation; done

# 11b. generation (7 jobs, ~8 GPU-h)
$E vllm-base $LOUT $LP
# WAIT
$E generate $LOUT $LP --base-model --confirm-final-evaluation
for m in none random lowest_margin mfr dapr_weak; do
  $E generate $LOUT $LP --run-dir $LR/v2_o[1-6]_${m}_s[0-2] --confirm-final-evaluation
done
$E generate $LOUT $LP --run-dir $LLAMA_OUT/joint_runs/* --confirm-final-evaluation
# WAIT

# 11c. WildGuard + IFEval (14 jobs, ~8 GPU-h)
for m in none random lowest_margin mfr dapr_weak; do
  $E safety $LOUT --run-dir $LR/v2_o[1-6]_${m}_s[0-2]
  $E ifeval $LOUT --run-dir $LR/v2_o[1-6]_${m}_s[0-2] --confirm-final-evaluation
done
$E safety $LOUT --run-dir $LLAMA_OUT/joint_runs/* $LLAMA_OUT/base_model/*
$E ifeval $LOUT --run-dir $LLAMA_OUT/joint_runs/* $LLAMA_OUT/base_model/* --confirm-final-evaluation
```

## 12. Gradient conflict (24 jobs, ~4 GPU-h; any time after step 5)

```bash
for o in 1 2 3 4 5 6; do for s in 0 1; do for stage in 2 3; do
  $E gradient-conflict $OUT --run-dir $R/v2_o${o}_none_s$s --stage $stage
done; done; done
```

## 13. Analysis (CPU)

With `MFR_OUTPUT_DIR` exported, open `notebooks/07a_results.ipynb`, `07b_mechanisms.ipynb`,
`08_joint_baseline.ipynb`, `09_generation_evaluation.ipynb` and `11_second_model.ipynb` in
Jupyter on CARC and Run All.
