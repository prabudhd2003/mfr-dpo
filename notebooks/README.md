# Notebooks

The active workflow runs training through CARC terminal jobs, not notebooks.

- `01_carc_status.ipynb` checks the local data, reference cache, and completed CARC runs.
- `07_compare_runs.ipynb` builds the validation tables and figures from the CARC artifact directory.
- `colab/` preserves the previous Colab notebooks and their outputs. They are historical and should not be used to launch the new experiment grid.

Before opening an active notebook on CARC, set the artifact directory in the terminal that starts Jupyter:

```bash
export MFR_OUTPUT_DIR=/project2/xiangren_1987/grp26-mfr-dpo/artifacts
```

Training and cache construction are submitted with `scripts/submit_carc.py` as described in `docs/CARC.md`.
