"""Tests for the replay buffer and the four selection rules. CPU only: pytest -q"""

import numpy as np
import pandas as pd

import mfr_replay
from mfr_replay import ReplayBuffer, plan_interval


def fake_split(dataset, n=600):
    return pd.DataFrame({
        "id": [f"{dataset}-train-{i:04d}" for i in range(n)],
        "prompt": [f"{dataset} prompt {i}" for i in range(n)],
        "chosen": [f"good {i}" for i in range(n)],
        "rejected": [f"bad {i}" for i in range(n)],
        "prompt_tokens": 50, "chosen_tokens": 100, "rejected_tokens": 90,
    })


def margins(rows, values):
    return pd.Series(values, index=rows["id"].values)


def filled_buffer(seed=0, size=100, two_stages=False):
    safe = fake_split("safe")
    buffer = ReplayBuffer(size=size, seed=seed)
    cand = buffer.candidates("safe", safe)
    buffer.add_stage("safe", cand, margins(cand, np.linspace(0.01, 0.10, len(cand))))
    if two_stages:
        helpful = fake_split("helpful")
        cand2 = buffer.candidates("helpful", helpful)
        buffer.add_stage("helpful", cand2, margins(cand2, np.linspace(0.02, 0.05, len(cand2))))
    return buffer


# ------------------------------------------------------------------ the buffer

def test_candidates_depend_on_seed_and_dataset_only():
    safe = fake_split("safe")
    a = ReplayBuffer(seed=0).candidates("safe", safe)["id"].tolist()
    b = ReplayBuffer(seed=0).candidates("safe", safe)["id"].tolist()
    c = ReplayBuffer(seed=1).candidates("safe", safe)["id"].tolist()
    assert a == b            # same seed -> same pairs, whatever the method does later
    assert a != c            # different seed -> different pairs


def test_buffer_size_and_even_split():
    buffer = filled_buffer(size=100)
    assert len(buffer) == 100 and set(buffer.rows()["dataset"]) == {"safe"}
    buffer = filled_buffer(size=100, two_stages=True)
    counts = buffer.rows()["dataset"].value_counts()
    assert len(buffer) == 100 and counts["safe"] == 50 and counts["helpful"] == 50


def test_buffer_remainder_is_not_silently_dropped():
    buffer = filled_buffer(size=101, two_stages=True)
    assert len(buffer) == 101
    assert sorted(buffer.rows()["dataset"].value_counts().tolist()) == [50, 51]


def test_current_starts_at_peak_and_updates():
    buffer = filled_buffer(size=20)
    rows = buffer.rows()
    assert (rows["current_margin"] == rows["peak_margin"]).all()
    buffer.set_current(pd.Series(0.0, index=rows["id"].values))
    assert (buffer.rows()["current_margin"] == 0).all()
    assert np.allclose(buffer.forgetting().values, rows["peak_margin"].values)


def test_get_returns_rows_in_order():
    buffer = filled_buffer(size=20)
    ids = buffer.rows()["id"].tolist()[:3][::-1]
    got = buffer.get(ids)
    assert [row["id"] for row in got] == ids
    assert {"prompt", "chosen", "rejected", "prompt_tokens"} <= set(got[0])


def test_save_and_load(tmp_path="/tmp"):
    import tempfile, os
    buffer = filled_buffer(size=30, two_stages=True)
    with tempfile.TemporaryDirectory() as folder:
        path = os.path.join(folder, "buffer.csv")
        buffer.to_csv(path)
        again = ReplayBuffer.from_csv(path, size=30, seed=0)
    assert len(again) == len(buffer)
    assert again.rows()["id"].tolist() == buffer.rows()["id"].tolist()


# ------------------------------------------------------------------ selection

def test_none_replays_nothing():
    buffer = filled_buffer(size=20)
    assert plan_interval(buffer, "none", 10, np.random.default_rng(0)) == []


def test_every_method_fills_exactly_the_slots():
    buffer = filled_buffer(size=50, two_stages=True)
    for method in ("random", "random_high", "lowest_margin", "mfr"):
        plan = plan_interval(buffer, method, 24, np.random.default_rng(0))
        assert len(plan) == 24
        assert set(plan) <= set(buffer.rows()["id"])


def test_mfr_picks_the_largest_drops():
    buffer = filled_buffer(size=20)
    rows = buffer.rows()
    current = rows["peak_margin"].copy()
    current.iloc[[3, 7, 11]] -= 1.0                      # these three "forgot" the most
    buffer.set_current(pd.Series(current.values, index=rows["id"].values))

    picked = plan_interval(buffer, "mfr", 3, np.random.default_rng(0))
    assert set(picked) == set(rows["id"].iloc[[3, 7, 11]])


def test_lowest_margin_picks_the_lowest_current_margins():
    buffer = filled_buffer(size=20)
    rows = buffer.rows()
    picked = plan_interval(buffer, "lowest_margin", 3, np.random.default_rng(0))
    lowest = rows.nsmallest(3, "current_margin")["id"].tolist()
    assert set(picked) == set(lowest)


def test_mfr_and_lowest_margin_differ():
    """A pair can have a low margin because it was never learned, not because it was forgotten."""
    buffer = filled_buffer(size=20)
    rows = buffer.rows()
    best = rows["peak_margin"].idxmax()                  # the best-learned pair...
    current = rows["peak_margin"].copy()
    current.iloc[best] -= 0.02                           # ...loses a little: the biggest drop
    buffer.set_current(pd.Series(current.values, index=rows["id"].values))

    mfr = plan_interval(buffer, "mfr", 1, np.random.default_rng(0))
    low = plan_interval(buffer, "lowest_margin", 1, np.random.default_rng(0))
    assert mfr == [rows["id"].iloc[best]]                # MFR: forgotten the most
    assert low == [rows["id"].iloc[current.idxmin()]]    # lowest margin: weakest right now
    assert mfr != low


def test_random_is_uniform_and_seeded():
    buffer = filled_buffer(size=50)
    a = plan_interval(buffer, "random", 10, np.random.default_rng(0))
    b = plan_interval(buffer, "random", 10, np.random.default_rng(0))
    c = plan_interval(buffer, "random", 10, np.random.default_rng(1))
    assert a == b and a != c
    assert len(set(a)) == 10                              # no pair twice in one interval


def test_random_high_uses_the_same_seeded_selection_rule():
    buffer = filled_buffer(size=50)
    regular = plan_interval(buffer, "random", 10, np.random.default_rng(0))
    high = plan_interval(buffer, "random_high", 10, np.random.default_rng(0))
    assert high == regular


def test_more_slots_than_pairs_cycles():
    buffer = filled_buffer(size=4)
    plan = plan_interval(buffer, "mfr", 10, np.random.default_rng(0))
    assert len(plan) == 10 and set(plan) == set(buffer.rows()["id"])


def test_dataset_cap():
    buffer = filled_buffer(size=100, two_stages=True)
    rows = buffer.rows()
    current = rows["peak_margin"].copy()
    current[rows["dataset"] == "safe"] -= 1.0             # every safe pair looks most-forgotten
    buffer.set_current(pd.Series(current.values, index=rows["id"].values))

    uncapped = plan_interval(buffer, "mfr", 20, np.random.default_rng(0))
    capped = plan_interval(buffer, "mfr", 20, np.random.default_rng(0), max_share_per_dataset=0.75)
    datasets = buffer.rows().set_index("id")["dataset"]
    assert (datasets[uncapped] == "safe").sum() == 20
    assert (datasets[capped] == "safe").sum() == 15       # 75% of 20


def test_buffer_contents_do_not_depend_on_method():
    """The pairs stored are identical for every method, which is what makes comparisons paired."""
    ids = [filled_buffer(seed=0, size=60, two_stages=True).rows()["id"].tolist() for _ in range(2)]
    assert ids[0] == ids[1]


def test_unknown_method_raises():
    buffer = filled_buffer(size=10)
    try:
        plan_interval(buffer, "best", 5, np.random.default_rng(0))
        raise AssertionError("expected ValueError")
    except ValueError:
        pass
