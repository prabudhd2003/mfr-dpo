"""CARC group-orchestration tests; no GPU or Slurm required."""

import importlib.util
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "run_experiment_group", ROOT / "scripts" / "run_experiment_group.py"
)
GROUP = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(GROUP)


def test_default_group_contains_all_configured_methods():
    protocol = GROUP.load_protocol()
    assert GROUP.requested_methods(protocol, None) == [
        "none", "random", "mfr", "random_high", "lowest_margin", "fmcr", "cpmr",
        "dapr", "dapr_c", "mir_dpo", "copr_adapted",
    ]


def test_group_accepts_a_configured_subset():
    protocol = GROUP.load_protocol()
    assert GROUP.requested_methods(protocol, "none,random,mfr") == ["none", "random", "mfr"]


def test_group_resumes_after_last_saved_stage(tmp_path):
    run = tmp_path / "run"
    run.mkdir()
    pd.DataFrame({"stage": [0, 1]}).to_csv(run / "results.csv", index=False)
    assert GROUP.first_unfinished_stage(run, initial_stage=1, n_stages=3) == 2


def test_group_skips_complete_run(tmp_path):
    run = tmp_path / "run"
    run.mkdir()
    (run / "COMPLETE.json").write_text("{}", encoding="utf-8")
    assert GROUP.first_unfinished_stage(run, initial_stage=1, n_stages=3) is None
