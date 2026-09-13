"""Aggregate metric semantics without loading a real language model."""

import pandas as pd

import fake_deps
fake_deps.install()

import mfr_dpo


def test_summarize_distinguishes_relative_and_absolute_preferences():
    scores = pd.DataFrame({"margin": [1.0, -1.0], "margin_sum": [1.0, 2.0],
                           "policy_margin": [-1.0, -2.0], "policy_margin_sum": [1.0, -1.0]})
    summary = mfr_dpo.summarize(scores)
    assert summary["accuracy"] == 50.0
    assert summary["accuracy_sum"] == 100.0
    assert summary["policy_accuracy"] == 0.0
    assert summary["policy_accuracy_sum"] == 50.0
