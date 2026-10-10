# Notebooks

The active workflow runs training through CARC terminal jobs, not notebooks.

- `01_carc_status.ipynb` checks the local data, reference cache, and completed CARC runs.
- `07a_results.ipynb` — **results**: ranking of every method, averages over all cells, forgetting by behavior,
  every order–seed cell, paired uncertainty, runtime, and the LoRA-EWC coefficient choice.
- `07b_mechanisms.ipynb` — **why**: one example run followed stage by stage, then replay mechanisms over all
  cells (selection overlap, concentration, whether replayed pairs had regressed, CPMR and DAPR-Gated checks).
- `archive/07_compare_runs_full.ipynb` — the previous single notebook, kept for reference.
- `08_joint_baseline.ipynb` analyzes the completed five-seed offline joint-training reference. Joint training has
  final behavior scores but no forgetting score because it is not sequential.
- `09_generation_evaluation.ipynb` combines WildGuard safety/harm results, XSTest over-refusal, IFEval, and
  position-controlled Prometheus helpfulness/quality comparisons.
- `10_human_evaluation.ipynb` is not part of the plan (no human evaluation); kept unused.
- `11_second_model.ipynb` compares the selected methods within Qwen2.5-1.5B and the pinned
  Llama-3.2-3B-Instruct replication. The two artifact roots remain separate.
- `colab/` preserves the previous Colab notebooks and their outputs. They are historical and should not be used to launch the new experiment grid.

Before opening an active notebook on CARC, set the artifact directory in the terminal that starts Jupyter:

```bash
export MFR_OUTPUT_DIR=/project2/xiangren_1987/grp26-mfr-dpo/artifacts
```

Training and cache construction are submitted with `scripts/submit_carc.py` as described in `docs/CARC.md`.
