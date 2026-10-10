"""Predeclared confirmatory statistics (src/mfr_stats.py). CPU only."""

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("scipy")

import mfr_stats

ORDERS = {1: ["helpful", "safe", "quality"], 2: ["safe", "helpful", "quality"],
          3: ["quality", "helpful", "safe"], 4: ["quality", "safe", "helpful"],
          5: ["helpful", "quality", "safe"], 6: ["safe", "quality", "helpful"]}


def run_rows(method, order_id, seed, after, final):
    """after[b]: score right after b was learned; final[b]: score at the end."""
    order = ORDERS[order_id]
    rows = []
    for stage, trained in enumerate(order, 1):
        for b in mfr_stats.BEHAVIORS:
            value = final[b] if stage == 3 else (after[b] if b == trained else 50.0)
            rows.append({"run_name": f"o{order_id}_{method}_s{seed}", "order_id": order_id,
                         "method": method, "seed": seed, "stage": stage, "trained_on": trained,
                         "eval_set": b, "accuracy": value})
    return rows


def test_cell_metrics_by_hand():
    rows = run_rows("random", 1, 0, after={"helpful": 70, "safe": 80, "quality": 0},
                    final={"helpful": 66, "safe": 74, "quality": 75})
    cells = mfr_stats.cell_metrics(pd.DataFrame(rows))
    row = cells.iloc[0]
    assert row["retention"] == pytest.approx(((66 - 70) + (74 - 80)) / 2)
    assert row["final_task"] == 75
    assert row["final_average"] == pytest.approx((66 + 74 + 75) / 3)


def synthetic_cells(seeds, effect=1.0, noise=0.5, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for s in seeds:
        seed_shift = rng.normal(0, noise)
        for o in ORDERS:
            for method, bonus in (("random", 0), ("lowest_margin", 0.5), ("mfr", 0.3),
                                  ("dapr_weak", effect)):
                base = 70 + rng.normal(0, 0.2)
                rows.append({"order_id": o, "seed": s, "method": method,
                             "retention": -5 + bonus + seed_shift * (method == "dapr_weak"),
                             "final_task": 74 - 0.5 * (method == "dapr_weak"),
                             "final_average": base + bonus + seed_shift * (method == "dapr_weak")})
    return pd.DataFrame(rows)


def test_seed_mean_ttest_matches_manual_computation():
    cells = synthetic_cells(range(5))
    diffs = mfr_stats.paired_differences(cells, "dapr_weak", "random", "final_average")
    result = mfr_stats.seed_mean_ttest(diffs)
    means = diffs.groupby("seed")["diff"].mean()
    t = means.mean() / (means.std(ddof=1) / np.sqrt(5))
    assert result["df"] == 4 and result["n_cells"] == 30
    assert result["t"] == pytest.approx(t)
    assert result["ci_low"] < result["estimate"] < result["ci_high"]


def test_unbalanced_design_is_rejected():
    cells = synthetic_cells(range(5))
    cells = cells[~((cells["seed"] == 3) & (cells["order_id"] == 6))]
    diffs = mfr_stats.paired_differences(cells, "dapr_weak", "random", "final_average")
    with pytest.raises(ValueError, match="unbalanced"):
        mfr_stats.seed_mean_ttest(diffs)


def test_holm_step_down():
    out = mfr_stats.holm({"P1": 0.01, "P2": 0.04, "P3": 0.03}, 0.05)
    assert out["P1"] == (pytest.approx(0.03), True)
    assert out["P3"] == (pytest.approx(0.06), False)
    assert out["P2"] == (pytest.approx(0.06), False)


def test_two_look_rule():
    strong = {"P1": {"p_value": 0.001}, "P2": {"p_value": 0.004}}
    weak = {"P1": {"p_value": 0.001}, "P2": {"p_value": 0.006}}
    assert mfr_stats.two_look_decision(strong) == "stop"
    assert mfr_stats.two_look_decision(weak) == "extend"


def test_primary_analysis_look1_and_look2():
    cells = synthetic_cells(range(5), effect=1.0, noise=0.05)
    table, decision = mfr_stats.primary_analysis(cells, look=1)
    assert decision in ("stop", "extend") and list(table["test"]) == ["P1", "P2", "P3"]
    with pytest.raises(ValueError, match="no paired cells"):
        mfr_stats.primary_analysis(cells, look=2)
    ten = synthetic_cells(range(10), effect=1.0, noise=0.05)
    ten = ten[(ten["seed"] < 5) | ten["method"].isin(["random", "lowest_margin", "dapr_weak"])]
    table2, decision2 = mfr_stats.primary_analysis(ten, look=2)
    assert decision2 == "final"
    assert set(table2["family_alpha"]) == {mfr_stats.FINAL_ALPHA_AFTER_EXTENSION}
    assert table2.set_index("test").loc["P1", "n_seeds"] == 10
    assert table2.set_index("test").loc["P3", "n_seeds"] == 5
    assert "noninferiority" in table2.attrs
