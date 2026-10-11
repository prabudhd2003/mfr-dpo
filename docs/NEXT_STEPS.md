# Next steps: from Oct 11, 2026 to the final report

This is the complete plan for the rest of the CSCI 544 course project, on the `project` branch only. Every decision
below is final. Do the steps in order. Do not add methods, orders, or evaluations that are not listed here.

## Calendar

| Date | Deliverable or milestone | Step |
|---|---|---|
| Sun Oct 11 – Mon Oct 12 | Folder fixes, protocol frozen | 0, 1 |
| Mon Oct 12 – Fri Oct 16 | Three new methods + joint training implemented and tested | 2 |
| Fri Oct 16 | Commit, pull on CARC, submit seeds 0–1 jobs | 3 |
| **Tue Oct 20 / Thu Oct 22** | **Midterm presentation** (2 slides, 2 min + 1 min Q&A, 5% of grade) | 4 |
| **Tue Oct 27** | **Midterm report** (4 pages, single space, 10 pt, 15% of grade) | 5 |
| Mon Oct 26 – Fri Oct 30 | Full validation and mechanism analysis on seeds 0–1 | 6 |
| **Fri Oct 30** | **Method selection frozen** | 7 |
| Fri Oct 30 – Fri Nov 6 | Seed-2 confirmation runs | 8 |
| Sat Nov 7 – Mon Nov 9 | Locked test, run once | 9 |
| Mon Nov 9 – Fri Nov 13 | Generation evaluation (vLLM, reward model, Llama Guard, IFEval) | 10 |
| Thu Nov 12 – Mon Nov 16 | Human evaluation and error analysis | 11, 12 |
| **Wed Nov 18** | **Final presentation ready.** Slots are Nov 20, Nov 24, Dec 1, and Dec 3 (6 min, 5 slides, 10% of grade) | 13 |
| Fri Dec 11 | Full report draft | 14 |
| **Tue Dec 15, 6:30 pm** | **Final report** (8 pages + up to 15 with appendix, 20% of grade). **Late days cannot be used.** | 14 |

Coursework collisions: HW1 is due Oct 13, and HW2 runs from Oct 15 to Nov 10. That is why the implementation work below
is split across all six people.

## Frozen study design

| Item | Decision |
|---|---|
| Model and training | Unchanged: Qwen2.5-1.5B-Instruct, QLoRA r 16 / α 32, DPO β 0.1, LR 1e-4, 18 new pairs per step, buffer 500, 5 refreshes |
| Orders | 1–4 only. Orders 5–6 are not added; the report states as a limitation that Quality is never the middle behavior. |
| Seeds | 0 and 1 for development, **2 for confirmation**. No more seeds. |
| Methods (8) | `none`, `random`, `random_high`, `lowest_margin`, `mfr`, `mfr_balanced`, `mfr_at_risk`, `mfr_stochastic` |
| Reference | Joint training, seeds 0–2 ([JOINT_TRAINING.md](JOINT_TRAINING.md)) |
| Primary metric | Old-behavior change (retention) on validation, plus "% less forgetting than Random 10%" |
| Secondary metrics | Final three-behavior average, final current-task score, `accuracy_sum`, share of the No-replay→Joint gap closed, runtime |
| Primary comparisons | (1) selected method vs Random 10%, (2) MFR vs Random 10% |
| Ablation comparisons (descriptive) | Balanced vs MFR (allocation), At-Risk vs Balanced (failure signal), Stochastic vs MFR (greedy vs sampled), At-Risk vs Lowest margin (past + present vs present) |
| Statistics | Paired bootstrap over order–seed cells, 20,000 resamples, 95% percentile interval, plus cells won / tied / lost |
| Locked test | Run once on every final checkpoint after Step 7. Never used for any decision. |

## Step 0: Folder fixes (Oct 11–12)

1. **Notebook 07 path.** Cell 4 used to set `MFR_OUTPUT_DIR` to another folder's artifacts. It must read
   `/project2/xiangren_1987/group26/artifacts`. This is fixed in this commit; confirm after `git pull`.
2. **Model weights for jobs in this folder.** Slurm jobs set `HF_HOME=$MFR_OUTPUT_DIR/huggingface`, and that folder
   is empty in `group26`. Download the pinned model once from a login node:
   ```bash
   export HF_HOME=$MFR_OUTPUT_DIR/huggingface
   python -c "from huggingface_hub import snapshot_download; snapshot_download('Qwen/Qwen2.5-1.5B-Instruct', revision='989aa7980e4cf806f80c7fef2b1adb7bc71aa306')"
   ```
3. **Environment.** Run the every-session commands in [PROJECT_FOLDER.md](PROJECT_FOLDER.md), then
   `which python` must point inside `group26/.conda/envs/mfr-dpo`, and `python -m pytest -q` must pass.

## Step 1: Freeze the protocol (Oct 12)

Edit `configs/experiment_protocol.json` and commit it **before any new run**:

```json
"secondary_methods": ["random_high", "lowest_margin", "mfr_balanced", "mfr_at_risk", "mfr_stochastic"],
"seeds": [0, 1, 2],
"stochastic_temperature": 1.0,
"development_seeds": [0, 1],
"confirmation_seeds": [2],
"selection_rule": "see docs/NEXT_STEPS.md Step 7",
"selected_method": null
```

Keep every other value unchanged. Update `tests/test_protocol.py` for the new keys.

## Step 2: Implement and test (Oct 12–16)

### Work split (one owner each)

| Owner | Work |
|---|---|
| Member 1 | Stage-1 compatibility fingerprint (shared by all new methods) and Balanced MFR |
| Member 2 | At-Risk MFR |
| Member 3 | Stochastic MFR |
| Member 4 | Joint training, plus the `joint` and `test` CARC actions |
| Member 5 | `src/mfr_analysis.py` and Notebook 07 extensions |
| Member 6 | Generation-evaluation environment and scripts (Step 10), prepared now and run later |

### 2.1 Stage-1 compatibility fingerprint (do this first)

`validate_stage1_source()` in `src/mfr_utils.py` compares the full `scientific_code_sha256`. That hash includes
`src/mfr_replay.py`, so adding any method rejects every existing Stage-1 checkpoint. Implement the fingerprint
exactly as specified in [BALANCED_MFR.md](BALANCED_MFR.md), section "Stage-1 compatibility change required before
running":
- keep the full hash for resume validation;
- add a Stage-1 fingerprint;
- add a tested legacy path for the existing seed-0/1 runs.

### 2.2 Balanced MFR and At-Risk MFR

Follow [BALANCED_MFR.md](BALANCED_MFR.md) and [AT_RISK_MFR.md](AT_RISK_MFR.md) exactly, including all listed tests.

### Stochastic MFR specification

Method name `mfr_stochastic`, display label `Stochastic MFR 10%`.

**Definition.** Stochastic MFR is identical to MFR except for one line: the ranking order. At each refresh, for every
buffered pair:

```text
drop_i = peak_margin_i − current_margin_i
z_i    = (drop_i − mean(drop)) / std(drop)          (z_i = 0 for all pairs if std = 0)
key_i  = z_i / τ + g_i,     g_i ~ Gumbel(0, 1),     τ = stochastic_temperature = 1.0
order  = pairs sorted by key, largest first
```

Sorting by `key` is the Gumbel-top-k trick. It is an exact sample without replacement from `softmax(z / τ)` [Kool et
al. 2019]. Everything after the ranking is MFR's code unchanged: the 75% per-behavior cap (`_apply_cap`), cycling
when slots exceed pairs, 5 refreshes, and 18 new + 2 replay pairs per step.

**Why τ = 1.0.** It is fixed before any result is seen. With z-scores, a pair one standard deviation above the
average drop is e ≈ 2.7 times as likely to be picked first as an average pair. Selection stays clearly targeted but
not greedy. τ → 0 recovers MFR and τ → ∞ recovers Random 10%. τ is never tuned.

**Code changes:**
- `src/mfr_replay.py`:
  - add `mfr_stochastic` to `METHODS` and `REFRESH_METHODS`;
  - add a `temperature=1.0` keyword to `plan_interval` and `plan_interval_details`;
  - add the branch below.

  ```python
  elif method == "mfr_stochastic":
      drop = (rows["peak_margin"] - rows["current_margin"]).to_numpy()
      spread = drop.std()
      z = (drop - drop.mean()) / spread if spread > 0 else np.zeros_like(drop)
      key = z / temperature + rng.gumbel(size=len(rows))
      score = drop
      order = np.argsort(-key, kind="stable")
  ```
- `src/mfr_dpo.py`: add `temperature=1.0` to `train_stage_replay` and pass it to `plan_interval_details`.
- `scripts/run_experiment.py`:
  - pass `temperature=protocol["stochastic_temperature"]`;
  - store `stochastic_temperature` in `settings`;
  - add it to the `validate_resume_settings` keys.
- Analysis: label `Stochastic MFR 10%` and its own color.

**Tests (`tests/test_replay.py`):**
- exact slot count;
- the 75% cap is respected;
- the same seed gives the same plan, and a different seed gives a different plan;
- with τ = 1e-6 and no tied drops, the plan equals MFR's;
- with τ = 1e6, over 2,000 seeded draws, each pair's first-place frequency is within 3 standard errors of uniform;
- `needs_refresh("mfr_stochastic")` is true;
- the five original methods produce byte-identical plans before and after the change.

### 2.3 Joint training

Implement exactly as written in [JOINT_TRAINING.md](JOINT_TRAINING.md).

### 2.4 `test` action for the locked test

Add to `scripts/carc_job.sh`, before `*)`:

```bash
  test)
    for RUN in "$OUTPUT_DIR"/runs/v2_o*_s* "$OUTPUT_DIR"/joint_runs/v2_joint_s*; do
      [[ -f "$RUN/COMPLETE.json" && ! -f "$RUN/final_test/test_results.csv" ]] || continue
      "$PYTHON_BIN" -u scripts/final_test.py --run-dir "$RUN"
    done
    ;;
```

Add a `test` subcommand to `scripts/submit_carc.py` that takes only the common arguments, with a default time of
`12:00:00`, job name `mfr-test`, and worker arguments `["test", ROOT, output_dir, conda_env, gpu_type]`. This needs
the joint-run change to `scripts/final_test.py` from JOINT_TRAINING.md.

### 2.5 Analysis code

In `src/mfr_analysis.py` and Notebook 07:
- `METHOD_ORDER` and colors for all 8 methods;
- the joint reference ([JOINT_TRAINING.md](JOINT_TRAINING.md), section 6);
- a "% less forgetting than Random 10%" column and a cells won / tied / lost column in every summary table;
- every table sorted from worst to best on its main metric;
- a `SEEDS` switch at the top of the notebook: `[0, 1]` for development and `[0, 1, 2]` for confirmation.

### 2.6 Gate before any GPU job

```bash
python -m pytest -q
git diff --check
git status --short      # must be empty after committing
```

The new code must leave all five original methods unchanged. Prove this with the byte-identical-plan tests above.

## Step 3: Train the new methods on seeds 0–1 (Oct 16–19)

On CARC, in `group26`, after `git pull` and the every-session setup:

```bash
for order in 1 2 3 4; do
  for seed in 0 1; do
    python scripts/submit_carc.py group \
      --output-dir "$MFR_OUTPUT_DIR" --account xiangren_1987 --gpu l40s --time 03:00:00 \
      --order "$order" --seed "$seed" --methods mfr_balanced,mfr_at_risk,mfr_stochastic
  done
done
for seed in 0 1; do
  python scripts/submit_carc.py joint \
    --output-dir "$MFR_OUTPUT_DIR" --account xiangren_1987 --gpu l40s --time 02:00:00 --seed "$seed"
done
```

That is 24 method runs plus 2 joint runs, about 12 GPU-hours. Each group reuses the completed `none` run in
`group26/artifacts/runs` as its Stage-1 source. When everything finishes:

```bash
find "$MFR_OUTPUT_DIR/runs" -name COMPLETE.json | wc -l         # 64
find "$MFR_OUTPUT_DIR/joint_runs" -name COMPLETE.json | wc -l   # 2
```

## Step 4: Midterm presentation (Oct 20 / 22)

There are two slides and one presenter. Use the 40 completed runs. Add the new methods only if all 8 cells of a
method are complete by Oct 19.

- **Slide 1: problem, method, and difference from prior work.**
  - Sequential DPO forgets: Safety drops 13.25 points without replay.
  - MFR replays the pairs whose margins fell most.
  - Prior selection methods (MIR, selective replay) target supervised learning or RL, not preference margins in
    sequential DPO.
- **Slide 2: takeaway and what remains.**
  - Table 1 from the README.
  - MFR: 11.3% less forgetting than equal-budget random replay, in 7 of 8 cells.
  - Lowest margin: 18.7% less forgetting; it is the method to beat.
  - Next: three ablations, the joint reference, seed 2, the locked test, and generation evaluation.

Rehearse to 2:00. Prepare answers to three questions: "Why not more seeds?", "Is 0.6 points real?" (paired interval
[0.22, 1.09]), and "Why does lowest margin win?"

## Step 5: Midterm report (due Oct 27)

Four pages, single-spaced, 10 pt. It must answer the syllabus questions.

| Section | Pages | Content |
|---|---:|---|
| Objective, in plain words | 0.4 | What we do, who cares, and what difference it makes |
| Related work | 0.8 | Replay and selection in continual learning (README references [3]–[9]), LLM continual tuning [10–12], sequential DPO [17] |
| Method | 0.8 | MFR, the five baselines, the three ablations, and joint training; the method → question table from the README |
| Experimental plan | 0.6 | Orders, seeds, metrics, hypotheses (one per comparison), statistics, the locked-test rule |
| Preliminary results | 1.0 | README Tables 1–3, with % and absolute points, plus a one-paragraph literature context on small gaps |
| Revised milestones | 0.4 | The calendar at the top of this document |

The TA's feedback asked for proper baselines and ablations. The method → question table is the direct answer; put it
on page 2.

## Step 6: Validation and mechanism analysis on seeds 0–1 (Oct 26–30)

Run Notebook 07 with `SEEDS = [0, 1]`. Produce:

1. Tables 1–3 of the README for all 8 methods plus joint training.
2. Ablation answers, one sentence each with numbers:
   - Balanced vs MFR;
   - At-Risk vs Balanced;
   - Stochastic vs MFR;
   - At-Risk vs Lowest margin.
3. Mechanism analysis from the saved replay logs, buffers, and margins. No GPU is needed.
   - Overlap (Jaccard) between the pairs selected by MFR, Balanced, At-Risk, Stochastic, and Lowest margin at each
     refresh.
   - Unique pairs, uses per pair, and the share of slots taken by the top 10% most-replayed pairs, per method.
   - Slots per old behavior, per method and stage.
   - For At-Risk: the share of slots that went to failed pairs (`policy_margin ≤ 0`).
   - Recovery: for pairs replayed in interval *k*, the change in margin by interval *k + 1*, compared with buffered
     pairs that were not replayed.
   - The validation-set view: which validation pairs flip from correct to wrong under each method (pair-level
     margins in `margins_*_val.csv`).
4. An updated conclusion cell in Notebook 07, rewritten from the new numbers.

## Step 7: Freeze the method selection (Oct 30)

Apply this rule to **seeds 0–1 validation only**. Candidates: `mfr`, `mfr_balanced`, `mfr_at_risk`,
`mfr_stochastic`, `lowest_margin`.

1. The candidate's mean old-behavior change must be better than Random 10%.
2. Its mean final current-task score must be no more than 2.0 points below Random 10%
   (`new_learning_tolerance_points`).
3. Among candidates that pass, choose the highest mean final three-behavior average.
4. If the top two are within 0.25 points on the final average, choose the one with the lower mean runtime.

Write the result into `configs/experiment_protocol.json` as `"selected_method": "<name>"`. Commit it with the
message `Freeze method selection` and record the commit hash in the report. From this point, no method, setting, or
analysis choice changes.

## Step 8: Seed-2 confirmation (Oct 30 – Nov 6)

Run all 8 methods on all 4 orders with seed 2, plus joint training seed 2:

```bash
for order in 1 2 3 4; do
  python scripts/submit_carc.py group \
    --output-dir "$MFR_OUTPUT_DIR" --account xiangren_1987 --gpu l40s --time 06:00:00 \
    --order "$order" --seed 2 \
    --methods none,random,random_high,lowest_margin,mfr,mfr_balanced,mfr_at_risk,mfr_stochastic
done
python scripts/submit_carc.py joint \
  --output-dir "$MFR_OUTPUT_DIR" --account xiangren_1987 --gpu l40s --time 02:00:00 --seed 2
```

That is 32 runs plus 1 joint run, about 15 GPU-hours. The group runner trains `none` first and reuses its Stage 1 for
the other seven methods. Expected totals: 96 completed runs in `runs/` and 3 in `joint_runs/`.

Then run Notebook 07 twice:
- `SEEDS = [2]`: does the seed-2 result agree in direction with seeds 0–1 for both primary comparisons?
- `SEEDS = [0, 1, 2]`: the final validation numbers, over 12 cells per method.

Report both. If seed 2 disagrees with seeds 0–1, report it as it is. Do not reselect the method.

## Step 9: Locked test, run once (Nov 7–9)

```bash
python scripts/submit_carc.py test \
  --output-dir "$MFR_OUTPUT_DIR" --account xiangren_1987 --gpu l40s --time 12:00:00
```

This scores the three 300-pair test sets for all 96 final checkpoints and 3 joint checkpoints, about 6 GPU-hours.
Each run gets `final_test/test_results.csv` and pair-level scores. Test resolution is 300 pairs, so one pair =
0.33 points.

The test can only measure **final** accuracy, because test scores were not recorded after each stage. Report final
accuracy per behavior, the final three-behavior average, and the share of the gap closed. Use the same tables, sorting,
% columns, and paired bootstrap as validation. Retention stays a validation-only metric.

## Step 10: Generation evaluation (Nov 9–13)

Preference-pair accuracy does not show that generated answers are helpful, safe, or good. We generate responses and
judge them automatically.

### 10.1 Decision: use vLLM for generation and IFEval

We use vLLM [Kwon et al. 2023] for all generation in this step, and Hugging Face only for training and pair scoring.

- **Scale.** There are 100 checkpoints: 96 final method checkpoints, 3 joint checkpoints, and the base model. Each
  generates for 900 test prompts with up to 512 new tokens, which is up to about 46 million generated tokens.
  `mfr_eval.generate_responses` decodes in fixed batches of 4 on a 4-bit model and would need tens of GPU-hours.
  vLLM's continuous batching and paged attention load the base model once and swap each LoRA adapter per request.
  This brings the job down to a few GPU-hours.
- **Matching the training setup.** vLLM loads the base model in 4-bit bitsandbytes
  (`quantization="bitsandbytes"`), the same NF4 base the adapters were trained against. It applies each adapter with
  `enable_lora=True, max_lora_rank=16`. This is vLLM's documented LoRA-with-quantization path.
- **Scope.** vLLM is not used for training, buffer scoring, or the locked pair test. Those need exact
  log-probabilities of fixed responses and stay in the existing code.

### 10.2 Separate environment

vLLM pins its own PyTorch, so it gets its own environment. **Never install vLLM into the training environment.**

```bash
cd /project2/xiangren_1987/group26
salloc --partition=gpu --gpus-per-task=l40s:1 --cpus-per-task=8 --mem=64G --time=02:00:00
module purge && module load conda && eval "$(conda shell.bash hook)"
mamba create --prefix "$PWD/.conda/envs/mfr-gen" python=3.11 pip -y
conda activate "$PWD/.conda/envs/mfr-gen"
python -m pip install vllm "lm_eval[vllm,ifeval]" bitsandbytes pandas tqdm
python -m pip freeze | grep -iE '^(vllm|torch|transformers|lm.eval|bitsandbytes)==' > requirements-gen.txt
```

Commit `requirements-gen.txt` so the exact versions are recorded. Accept the Llama Guard 3 licence on Hugging Face
with one team account. Then run `huggingface-cli login` once in this environment, and download
`meta-llama/Llama-Guard-3-8B` and `Skywork/Skywork-Reward-Llama-3.1-8B-v0.2` into `$MFR_OUTPUT_DIR/huggingface`.

### 10.3 Scripts to write (Member 6)

1. **`scripts/run_generation_eval.py`** (gen environment):
   - Collect every completed final adapter: `runs/*/stage3_*`, `joint_runs/*/joint`, and the base model with no
     adapter.
   - Create one `vllm.LLM` with the pinned model revision, `quantization="bitsandbytes"`, `enable_lora=True`,
     `max_lora_rank=16`, `max_model_len=2048`, `seed=0`.
   - Render each of the 900 test prompts with the tokenizer's chat template: one user message,
     `add_generation_prompt=True`. This is exactly what `mfr_dpo.chat_prompt` does.
   - Generate with `SamplingParams(temperature=0.0, max_tokens=512)`. This matches the protocol's `generation`
     block: greedy decoding, 512 new tokens.
   - Pass each adapter as `LoRARequest(run_name, unique_int_id, adapter_path)`.
   - Write `<run>/generation/test_generations.jsonl` with `id`, `dataset`, `prompt`, `response`,
     `n_response_tokens`, `finish_reason`. Use `mfr_eval.save_generations`'s format and manifest, copied into the
     script so the gen environment does not import the training code.
   - Skip runs whose file already exists.
2. **`scripts/run_ifeval.py`**: switch to `--model vllm` with
   `pretrained=…,revision=…,quantization=bitsandbytes,lora_local_path=<adapter>,max_lora_rank=16`. Accept joint
   runs (adapter `joint/`) and the base model.
3. **`scripts/judge_generations.py`** (gen environment, Hugging Face `transformers`, bf16, one judge loaded at a time):
   - **Reward model.** Score every (prompt, response) with `Skywork/Skywork-Reward-Llama-3.1-8B-v0.2` [Liu et al.
     2024] and write `reward_scores.csv`.
   - **Safety.** Classify every response to the 300 Safe (PKU-SafeRLHF) prompts with `meta-llama/Llama-Guard-3-8B`
     [Inan et al. 2023; Llama Team 2024] and write `safety_labels.csv` (`safe` / `unsafe` plus category).
4. Add `generate`, `ifeval`, and `judge` actions to `carc_job.sh` and `submit_carc.py`. They work like `test`, but are
   submitted with `--conda-env /project2/xiangren_1987/group26/.conda/envs/mfr-gen`.

### 10.4 Equivalence gate (must pass before the full generation run)

On checkpoint `v2_o1_mfr_s0` and 30 test prompts (10 per behavior):

1. The rendered prompt strings from vLLM's tokenizer and from `mfr_dpo.chat_prompt` are byte-identical.
2. vLLM outputs **with** the adapter differ from vLLM outputs **without** it on at least 15 of 30 prompts. This
   proves the adapter is applied.
3. The first 20 generated tokens match `mfr_eval.generate_responses` (Hugging Face, training environment) on at
   least 24 of 30 prompts.

If any check fails, fix the cause before generating. The usual causes are a chat-template, EOS-token, or
adapter-path mismatch. Do not generate with a setup that fails the gate. Save the gate outputs and results in
`artifacts/generation_gate/`.

### 10.5 What to report

Every row of these tables is sorted worst to best and paired by cell and prompt.

| Metric | Prompts | Comparison |
|---|---|---|
| Mean reward | All 900, per behavior | All methods, Joint, and the base model |
| Win rate vs Random 10% (reward, tie = ½) | Per behavior | All methods; bootstrap over the 12 cells |
| Unsafe-response rate (Llama Guard 3) | 300 Safe prompts | All methods, Joint, base model |
| IFEval prompt-level strict accuracy [Zhou et al. 2023] | 541 IFEval prompts | All methods, Joint, base model |
| Mean response length (tokens) | All 900 | Reported next to reward, because reward models favor longer answers |

## Step 11: Human evaluation (Nov 12–16)

- **Systems:** the selected method vs Random 10%, final checkpoints from the four seed-2 cells.
- **Items:** 96 prompts. That is 4 cells × 3 behaviors × 8 prompts, drawn from the test generations with
  `mfr_review.make_blind_review(..., n_prompts=8, seed=544)`, called once per cell and behavior.
- **Raters:** all six members in three fixed pairs. Each pair rates 32 items, so every item gets two independent
  ratings.
- **Sheet:** `preference_rank` (1 = better) and the 1–5 score for that behavior: helpfulness for Helpful prompts,
  safety for Safe prompts, quality for Quality prompts. Use the rubric written at the top of the sheet before rating
  starts.
- **Blinding:** no one opens `PRIVATE_blinding_key.csv` until all six sheets are submitted.
- **Report:**
  - the win rate of the selected method, with a bootstrap interval over items;
  - inter-rater agreement (`mfr_review.inter_rater_agreement`);
  - agreement between the human majority and the reward-model preference on the same 96 items.

## Step 12: Error analysis (Nov 12–17)

1. For each earlier behavior, list the validation pairs that **Random 10% forgets but the selected method keeps**,
   and the reverse. Use pairs correct after learning and wrong at the end, from `margins_*_val.csv` on seeds 0–2.
2. Read 15 pairs from each list per behavior and label them with these fixed categories:
   - refusal vs compliance;
   - response length difference;
   - topic;
   - formatting;
   - near-duplicate responses.
3. From the replay logs, describe the 20 most-replayed pairs for MFR and for the selected method.
4. Pick 3 generated examples per behavior for the slides and report: one improvement, one shared failure, one
   regression.

## Step 13: Final presentation (ready by Nov 18)

Six minutes, 5 slides, plus 2 minutes of questions. Two presenters: the person who gave the midterm and one other.

1. Motivation: forgetting in sequential preference tuning, with Safety's 13-point drop.
2. Key insight: replay the pairs whose preference margin has fallen; prior selection work, and why random replay is
   hard to beat.
3. Method and ablations: one diagram, and the method → question table.
4. Results:
   - test and validation forgetting in % and points, with seed-2 confirmation;
   - share of the gap closed;
   - one generation metric.
5. Takeaways, limitations, and what remains before the report.

Rehearse twice with a timer. Slides are frozen two days before the assigned slot.

## Step 14: Final report (draft Dec 11, submit by Dec 14)

Eight pages, single-spaced, 10 pt, plus an appendix up to 15 pages. It must answer every syllabus question.

| Section | Pages |
|---|---:|
| Introduction: goal in plain words, who cares, difference it makes | 0.75 |
| Related work and limits of current practice | 1.0 |
| Method: MFR, baselines, three ablations, joint reference | 1.25 |
| Experimental setup: data, orders, seeds, metrics, hypotheses, statistics, locked-test rule | 1.0 |
| Results: validation (12 cells), locked test, generation, human evaluation | 2.25 |
| Analysis: mechanism, error analysis | 1.0 |
| Limitations, takeaways, next questions | 0.75 |

The appendix contains:
- every per-cell table;
- all hyperparameters and commit hashes;
- the equivalence-gate result;
- the human-evaluation rubric and agreement;
- the **AI-use statement**. The syllabus requires citing AI-generated material and listing the prompts used.

Submit on Dec 14 to leave a buffer. There are no late days for the final report.

## Completion checklist

- [ ] Step 0: notebook path, model download, environment check
- [ ] Step 1: protocol frozen and committed
- [ ] Step 2: Stage-1 fingerprint, Balanced, At-Risk, Stochastic, joint, `test` action, analysis; all tests pass
- [ ] Step 3: 24 runs + 2 joint runs complete
- [ ] Step 4: midterm presentation
- [ ] Step 5: midterm report submitted
- [ ] Step 6: validation and mechanism analysis
- [ ] Step 7: `selected_method` committed
- [ ] Step 8: 32 seed-2 runs + 1 joint run, confirmation analysis
- [ ] Step 9: locked test complete
- [ ] Step 10: gate passed; generation, reward, Llama Guard, IFEval complete
- [ ] Step 11: human evaluation complete
- [ ] Step 12: error analysis complete
- [ ] Step 13: final presentation delivered
- [ ] Step 14: final report submitted

## References for this plan

Method citations [1]–[17] are in the [README](../README.md#references). Additional references for this plan:

- Kool, W., van Hoof, H., and Welling, M. "Stochastic Beams and Where to Find Them: The Gumbel-Top-k Trick for
  Sampling Sequences Without Replacement." ICML 2019. https://arxiv.org/abs/1903.06059
- Kwon, W., Li, Z., Zhuang, S., et al. "Efficient Memory Management for Large Language Model Serving with
  PagedAttention." SOSP 2023. https://arxiv.org/abs/2309.06180
- vLLM documentation, "LoRA With Quantization Inference."
  https://docs.vllm.ai/en/v0.8.5/getting_started/examples/lora_with_quantization_inference.html
- Liu, C. Y., Zeng, L., Liu, J., et al. "Skywork-Reward: Bag of Tricks for Reward Modeling in LLMs."
  arXiv:2410.18451, 2024. https://arxiv.org/abs/2410.18451
- Inan, H., Upasani, K., Chi, J., et al. "Llama Guard: LLM-based Input-Output Safeguard for Human-AI Conversations."
  arXiv:2312.06674, 2023. https://arxiv.org/abs/2312.06674
- Llama Team, AI @ Meta. "The Llama 3 Herd of Models." arXiv:2407.21783, 2024. https://arxiv.org/abs/2407.21783
- Zhou, J., Lu, T., Mishra, S., et al. "Instruction-Following Evaluation for Large Language Models."
  arXiv:2311.07911, 2023. https://arxiv.org/abs/2311.07911
- Gao, L., Tow, J., et al. "The Language Model Evaluation Harness" (EleutherAI `lm-evaluation-harness`).
  https://github.com/EleutherAI/lm-evaluation-harness
