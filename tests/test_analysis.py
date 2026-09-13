"""Metric semantics and paired uncertainty tests."""

from pathlib import Path

import pandas as pd

import mfr_analysis


def tiny_runs():
    rows = []
    for stage, values in [(0, [50, 50, 50]), (1, [80, 50, 50]), (2, [70, 75, 50]), (3, [65, 70, 85])]:
        for dataset, value in zip(["safe", "helpful", "quality"], values):
            rows.append({"run_name": "run", "order_id": 2, "method": "mfr", "seed": 0,
                         "order": ["safe", "helpful", "quality"], "folder": "/missing",
                         "protocol_version": "2.0", "data_version": "v2", "stage1_from": None,
                         "stage": stage, "eval_set": dataset, "accuracy": value,
                         "mean_margin": value / 100})
    return pd.DataFrame(rows)


def test_retention_and_new_learning_are_distinct():
    runs = tiny_runs()
    retained = mfr_analysis.retention_table(runs)
    learned = mfr_analysis.new_learning_table(runs)
    assert retained.set_index("dataset").loc["safe", "change"] == -15
    assert learned.set_index("dataset").loc["helpful", "gain"] == 25


def test_paired_bootstrap_uses_shared_ids():
    left = pd.DataFrame({"id": ["a", "b", "c"], "correct": [1, 1, 0]})
    right = pd.DataFrame({"id": ["a", "b", "c"], "correct": [0, 1, 0]})
    result = mfr_analysis.paired_binary_difference(left, right, n_boot=200, seed=0)
    assert round(result["difference_points"], 2) == 33.33 and result["n_pairs"] == 3


def test_numeric_bootstrap_reports_left_minus_right():
    left = pd.DataFrame({"id": ["a", "b"], "change": [0, -1]})
    right = pd.DataFrame({"id": ["a", "b"], "change": [-1, -1]})
    result = mfr_analysis.paired_numeric_difference(left, right, "change", n_boot=100, seed=0, scale=100)
    assert result["difference"] == 50 and result["n_pairs"] == 2


def test_borrowed_stage_resolves_next_to_local_run(tmp_path):
    runs = tmp_path / "runs"
    current = runs / "v2_o2_mfr_s0"
    source = runs / "v2_o2_none_s0"
    stage = source / "stage1_safe"
    current.mkdir(parents=True)
    stage.mkdir(parents=True)
    recorded_colab_path = "/content/drive/MyDrive/CSCI544/mfr-dpo/runs/v2_o2_none_s0"
    resolved = mfr_analysis.stage_folder(
        str(current), ["safe", "helpful", "quality"], 1, recorded_colab_path
    )
    assert Path(resolved) == stage
