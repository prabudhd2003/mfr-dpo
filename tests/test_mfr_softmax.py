"""Focused CPU tests for the probabilistic MFR replay rule."""

import numpy as np
import pandas as pd

from mfr_replay import ReplayBuffer, plan_interval_details


def make_buffer():
    buffer = ReplayBuffer(size=4, seed=7)
    buffer._rows = pd.DataFrame({
        "id": ["a", "b", "c", "d"],
        "dataset": ["old_1", "old_1", "old_2", "old_2"],
        "peak_margin": [3.0, 2.0, 1.0, 0.0],
        "current_margin": [0.0, 0.0, 0.0, 0.0],
    })
    return buffer


def test_softmax_plan_is_seeded_and_auditable():
    left = plan_interval_details(
        make_buffer(), "mfr_softmax", 100, np.random.default_rng(123),
        mfr_softmax_temperature=1.0,
    )
    right = plan_interval_details(
        make_buffer(), "mfr_softmax", 100, np.random.default_rng(123),
        mfr_softmax_temperature=1.0,
    )
    assert left.equals(right)
    assert set((
        "selection_score", "selection_rank", "softmax_probability",
        "selection_probability", "temperature",
    )) <= set(left.columns)
    counts = left["id"].value_counts()
    assert counts.get("a", 0) > counts.get("d", 0)


def test_lower_temperature_concentrates_probability_on_largest_drop():
    warm = plan_interval_details(
        make_buffer(), "mfr_softmax", 1, np.random.default_rng(1),
        mfr_softmax_temperature=2.0,
    )
    cold = plan_interval_details(
        make_buffer(), "mfr_softmax", 1, np.random.default_rng(1),
        mfr_softmax_temperature=0.25,
    )
    assert cold.iloc[0]["softmax_probability"] > warm.iloc[0]["softmax_probability"]


def test_invalid_temperature_is_rejected():
    try:
        plan_interval_details(
            make_buffer(), "mfr_softmax", 1, np.random.default_rng(1),
            mfr_softmax_temperature=0.0,
        )
    except ValueError as error:
        assert "temperature" in str(error)
    else:
        raise AssertionError("zero temperature should fail")
