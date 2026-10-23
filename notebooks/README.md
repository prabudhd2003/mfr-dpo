# Notebooks

The active workflow runs training through CARC terminal jobs, not notebooks.

- `01_carc_status.ipynb` checks the local data, reference cache, and completed CARC runs.
- `07_compare_runs.ipynb` builds the validation tables and figures from the CARC artifact directory.
- `08_mid_term_results.ipynb` produces every table, figure, and quoted number for the midterm report (all methods, including DAPR-weak, plus joint training). Comment out methods at the top to drop them; nothing is saved until you call `save('F1')`, `save('T1')`, etc., which write to `artifacts/analysis/midterm/`.
- `09_pair_level_analysis.ipynb` looks at individual validation pairs: which pairs each method forgets or recovers, how much methods, seeds, and orders agree (vs chance), the core every method forgets, what replay saves vs No replay, what predicts forgetting, the link to replayed training pairs, and the pairs' text (`examples('core')`, `show_pair('safe-val-0012')`). Same settings and `save(...)` as 08; writes to `artifacts/analysis/pair_level/`.
- `colab/` preserves the previous Colab notebooks and their outputs. They are historical and should not be used to launch the new experiment grid.

Before opening an active notebook on CARC, set the artifact directory in the terminal that starts Jupyter:

```bash
export MFR_OUTPUT_DIR=/project2/xiangren_1987/group26/artifacts
```

Training and cache construction are submitted with `scripts/submit_carc.py` as described in `docs/CARC.md`.
