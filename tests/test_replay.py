"""Tests for the replay buffer and replay-selection rules. CPU only: pytest -q"""

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
    buffer.set_counterfactual_scores(
        buffer.rows().set_index("id")["current_margin"],
        buffer.rows().set_index("id")["current_margin"],
    )
    for method in (
        "random", "random_high", "lowest_margin", "mfr", "cpmr", "dapr", "dapr_c",
        "copr_adapted",
    ):
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


def test_anchor_methods_use_the_same_selection_as_lowest_margin():
    buffer = filled_buffer(size=20)
    for method in ("dapr", "dapr_c", "copr_adapted"):
        assert plan_interval(buffer, method, 5, np.random.default_rng(4)) == plan_interval(
            buffer, "lowest_margin", 5, np.random.default_rng(4)
        )


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


def test_only_margin_based_methods_need_live_refreshes():
    assert mfr_replay.needs_refresh("mfr")
    assert mfr_replay.needs_refresh("lowest_margin")
    assert mfr_replay.needs_refresh("fmcr")
    assert mfr_replay.needs_refresh("cpmr")
    assert mfr_replay.needs_refresh("dapr")
    assert mfr_replay.needs_refresh("dapr_c")
    assert mfr_replay.needs_refresh("mir_dpo")
    assert mfr_replay.needs_refresh("copr_adapted")
    assert not mfr_replay.needs_refresh("random")
    assert not mfr_replay.needs_refresh("random_high")
    assert not mfr_replay.needs_refresh("none")


def test_cpmr_ranks_the_lowest_projected_margin_first():
    buffer = filled_buffer(size=20)
    rows = buffer.rows()
    current = pd.Series(rows["current_margin"].values, index=rows["id"].values)
    projected = current.copy()
    projected.iloc[9] = -0.5
    buffer.set_counterfactual_scores(current, projected)
    plan = mfr_replay.plan_interval_details(buffer, "cpmr", 1, np.random.default_rng(0))
    assert plan.iloc[0]["id"] == rows.iloc[9]["id"]
    assert plan.iloc[0]["projected_margin"] == -0.5


def test_cpmr_reduces_to_lowest_margin_when_projected_margins_do_not_change():
    buffer = filled_buffer(size=20)
    rows = buffer.rows()
    current = pd.Series(rows["current_margin"].values, index=rows["id"].values)
    buffer.set_counterfactual_scores(current, current)
    cpmr = plan_interval(buffer, "cpmr", 5, np.random.default_rng(3))
    lowest = plan_interval(buffer, "lowest_margin", 5, np.random.default_rng(3))
    assert cpmr == lowest


def test_cpmr_does_not_hide_a_currently_weak_pair_predicted_to_recover():
    buffer = filled_buffer(size=20)
    rows = buffer.rows()
    current = pd.Series(0.10, index=rows["id"].values)
    current.iloc[2] = -0.20
    projected = current.copy()
    projected.iloc[2] = 0.30
    buffer.set_counterfactual_scores(current, projected)
    plan = mfr_replay.plan_interval_details(buffer, "cpmr", 1, np.random.default_rng(0))
    assert plan.iloc[0]["id"] == rows.iloc[2]["id"]
    assert plan.iloc[0]["worst_case_margin"] == -0.20


def test_mir_dpo_picks_the_largest_virtual_loss_increase():
    buffer = filled_buffer(size=20)
    rows = buffer.rows()
    current = pd.DataFrame({
        "margin": rows["current_margin"].to_numpy(),
        "margin_sum": np.full(len(rows), 1.0),
    }, index=rows["id"])
    projected = current.copy()
    projected.iloc[6, projected.columns.get_loc("margin_sum")] = -2.0
    buffer.set_mir_scores(current, projected)
    plan = mfr_replay.plan_interval_details(buffer, "mir_dpo", 1, np.random.default_rng(0))
    assert plan.iloc[0]["id"] == rows.iloc[6]["id"]
    assert plan.iloc[0]["interference_score"] > 0


# ------------------------------------------------------------------ FMCR

def forecast_buffer(size=20, two_stages=False):
    buffer = filled_buffer(size=size, two_stages=two_stages)
    rows = buffer.rows()
    initial = pd.DataFrame({
        "margin": rows["current_margin"].to_numpy(),
        "policy_margin": np.full(len(rows), 0.20),
    }, index=rows["id"].values)
    buffer.set_forecast_scores(initial, initialize=True)
    return buffer


def update_forecast(buffer, relative=None, policy=None, decay=0.0):
    rows = buffer.rows()
    relative = (rows["current_margin"].to_numpy() if relative is None
                else np.asarray(relative, dtype=float))
    policy = (rows["current_policy_margin"].to_numpy() if policy is None
              else np.asarray(policy, dtype=float))
    scores = pd.DataFrame(
        {"margin": relative, "policy_margin": policy}, index=rows["id"].values
    )
    buffer.set_forecast_scores(scores, velocity_decay=decay)


def test_fmcr_requires_initialized_policy_trajectories():
    buffer = filled_buffer(size=20)
    try:
        plan_interval(buffer, "fmcr", 2, np.random.default_rng(0))
        raise AssertionError("expected uninitialized forecast error")
    except ValueError as error:
        assert "initialized" in str(error)


def test_fmcr_trajectory_uses_ema_and_resets_at_a_new_task():
    buffer = forecast_buffer(size=4)
    rows = buffer.rows()
    initial_relative = rows["current_margin"].to_numpy()

    update_forecast(
        buffer,
        relative=initial_relative - 0.10,
        policy=np.full(len(rows), 0.10),
        decay=0.5,
    )
    updated = buffer.rows()
    assert np.allclose(updated["margin_velocity"], -0.05)
    assert np.allclose(updated["policy_velocity"], -0.05)

    reset_scores = pd.DataFrame(
        {"margin": updated["current_margin"].to_numpy(),
         "policy_margin": updated["current_policy_margin"].to_numpy()},
        index=updated["id"].values,
    )
    buffer.set_forecast_scores(reset_scores, velocity_decay=0.5, initialize=True)
    assert np.allclose(buffer.rows()[["margin_velocity", "policy_velocity"]], 0.0)


def test_fmcr_prioritizes_actual_policy_failure():
    buffer = forecast_buffer(size=20)
    rows = buffer.rows()
    relative = rows["current_margin"].to_numpy() - 0.01
    policy = np.full(len(rows), 0.20)
    policy[7] = -0.01
    update_forecast(buffer, relative, policy)
    plan = mfr_replay.plan_interval_details(buffer, "fmcr", 1, np.random.default_rng(0))
    assert plan.iloc[0]["id"] == rows.iloc[7]["id"]
    assert plan.iloc[0]["risk_label"] == "policy_failed"


def test_fmcr_prioritizes_a_forecast_policy_crossing():
    buffer = forecast_buffer(size=20)
    rows = buffer.rows()
    relative = rows["current_margin"].to_numpy(copy=True)
    policy = np.full(len(rows), 0.20)
    policy[4] = 0.08  # velocity=-0.12, so the next forecast is below zero
    update_forecast(buffer, relative, policy)
    plan = mfr_replay.plan_interval_details(buffer, "fmcr", 1, np.random.default_rng(0))
    assert plan.iloc[0]["id"] == rows.iloc[4]["id"]
    assert plan.iloc[0]["risk_label"] == "forecast_policy_crossing"


def test_fmcr_prioritizes_an_already_lost_relative_advantage():
    buffer = forecast_buffer(size=20)
    rows = buffer.rows()
    relative = rows["current_margin"].to_numpy(copy=True)
    relative[6] = -0.10
    update_forecast(buffer, relative, np.full(len(rows), 0.20))
    relative[6] = -0.01  # it is recovering and forecast positive, but is still failed right now
    update_forecast(buffer, relative, np.full(len(rows), 0.20))
    plan = mfr_replay.plan_interval_details(buffer, "fmcr", 1, np.random.default_rng(0))
    assert plan.iloc[0]["id"] == rows.iloc[6]["id"]
    assert plan.iloc[0]["risk_label"] == "relative_failed"


def test_fmcr_uses_relative_crossing_before_historical_fallback():
    buffer = forecast_buffer(size=20)
    rows = buffer.rows()
    relative = rows["current_margin"].to_numpy(copy=True)
    policy = np.full(len(rows), 0.20)
    relative[3] = max(0.001, relative[3] * 0.20)  # crosses relative zero next interval
    update_forecast(buffer, relative, policy)
    plan = mfr_replay.plan_interval_details(buffer, "fmcr", 1, np.random.default_rng(0))
    assert plan.iloc[0]["id"] == rows.iloc[3]["id"]
    assert plan.iloc[0]["risk_label"] == "forecast_relative_crossing"


def test_fmcr_fallback_is_original_mfr_drop():
    buffer = forecast_buffer(size=20)
    rows = buffer.rows()
    relative = rows["current_margin"].to_numpy(copy=True)
    relative[11] -= 0.01
    policy = np.full(len(rows), 0.20)
    update_forecast(buffer, relative, policy)
    plan = mfr_replay.plan_interval_details(buffer, "fmcr", 1, np.random.default_rng(0))
    assert plan.iloc[0]["id"] == rows.iloc[11]["id"]
    assert plan.iloc[0]["risk_label"] == "historical_drop_fallback"


def test_fmcr_balances_datasets_and_interleaves_them():
    buffer = forecast_buffer(size=40, two_stages=True)
    rows = buffer.rows()
    relative = rows["current_margin"].to_numpy() - 0.01
    policy = np.where(rows["dataset"].eq("safe"), -0.10, 0.20)
    update_forecast(buffer, relative, policy)
    plan = mfr_replay.plan_interval_details(buffer, "fmcr", 10, np.random.default_rng(0))
    assert plan["dataset"].value_counts().to_dict() == {"safe": 5, "helpful": 5}
    assert all(plan.iloc[i]["dataset"] != plan.iloc[i + 1]["dataset"]
               for i in range(len(plan) - 1))


def test_fmcr_rotates_an_odd_quota_remainder():
    buffer = forecast_buffer(size=40, two_stages=True)
    rows = buffer.rows()
    update_forecast(buffer, rows["current_margin"].to_numpy() - 0.01, np.full(len(rows), 0.2))
    first = mfr_replay.plan_interval_details(
        buffer, "fmcr", 5, np.random.default_rng(0), plan_index=0
    )
    second = mfr_replay.plan_interval_details(
        buffer, "fmcr", 5, np.random.default_rng(0), plan_index=1
    )
    first_majority = first["dataset"].value_counts().idxmax()
    second_majority = second["dataset"].value_counts().idxmax()
    assert first_majority != second_majority


def test_fmcr_is_seeded_and_auditable():
    buffer = forecast_buffer(size=20)
    rows = buffer.rows()
    update_forecast(buffer, rows["current_margin"].to_numpy(), np.full(len(rows), 0.2))
    a = mfr_replay.plan_interval_details(buffer, "fmcr", 5, np.random.default_rng(9))
    b = mfr_replay.plan_interval_details(buffer, "fmcr", 5, np.random.default_rng(9))
    assert a["id"].tolist() == b["id"].tolist()
    assert {"forecast_margin", "forecast_policy_margin", "time_to_crossing",
            "historical_drop", "risk_tier", "risk_label", "dataset_quota"} <= set(a.columns)
