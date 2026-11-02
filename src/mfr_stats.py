"""Predeclared confirmatory statistics (docs/FREEZE.md §Statistics). CPU only.

Unit of analysis: one order-seed cell. For a comparison, d[o, s] = target - baseline in that
cell. Seeds are crossed with orders (shared LoRA initialization and Stage-1 checkpoints), so the
primary test is a one-sample t-test on the per-seed means of d (each averaged over all orders),
df = n_seeds - 1. In this balanced design that equals the mixed model with order fixed and a
random seed intercept.

Two-look group-sequential rule (O'Brien-Fleming, two equally spaced looks, overall alpha 0.05):
look 1 after seeds 0-4; stop only if P1 and P2 both have p < 0.005; otherwise run seeds 5-9 for
none/random/lowest_margin/dapr_weak and test P1, P2 on all 10 seeds. Holm is applied to
{P1, P2, P3} at alpha 0.05 after a stop at look 1, or 0.048 at look 2. P3 always uses seeds 0-4.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

BEHAVIORS = ("helpful", "safe", "quality")   # the original three; cell_metrics reads each run's own
INTERIM_ALPHA = 0.005
FINAL_ALPHA_AFTER_EXTENSION = 0.048
ALPHA_NO_EXTENSION = 0.05
CONFIRMATION_SEEDS = (0, 1, 2, 3, 4)
EXTENSION_SEEDS = (5, 6, 7, 8, 9)
PRIMARY = {
    "P1": ("dapr_weak", "random", "final_average"),
    "P2": ("dapr_weak", "lowest_margin", "final_average"),
    "P3": ("mfr", "random", "retention"),
}
NONINFERIORITY = ("dapr_weak", "random", "final_task", -2.0)
METRICS = ("retention", "final_task", "final_average")


def cell_metrics(results, score="accuracy", upto_stage=None):
    """Per-run retention, final-task and final average over the run's own behaviors.

    `results` has one row per (run, stage, eval_set) with columns run_name, order_id, method,
    seed, stage, trained_on, eval_set and `score` -- the format of results.csv and of
    locked_test/results_test.csv. Joint runs (trained_on == "joint") get final_average only.

    Works for any number of behaviors (3 in the main study, 5 in the extension). With
    `upto_stage=k`, a sequential run is cut after stage k and scored on its first k behaviors only:
    the stage-3 prefix of a 5-task run is then exactly a 3-task run.
    """
    rows = []
    for run_name, part in results.groupby("run_name", sort=False):
        first = part.iloc[0]
        stages = part.drop_duplicates("stage").sort_values("stage")
        stages = stages[stages["stage"] > 0]
        order = list(stages["trained_on"])
        joint = order == ["joint"]
        if upto_stage is not None and not joint:
            if upto_stage > len(order):
                raise ValueError(f"{run_name}: has {len(order)} stages, cannot cut at {upto_stage}")
            order = order[:upto_stage]
            stages = stages[stages["stage"] <= upto_stage]
        last = int(stages["stage"].max())
        final = part[part["stage"] == last].set_index("eval_set")[score]
        behaviors = sorted(final.index) if joint else sorted(order)
        if not joint and len(set(order)) != len(order):
            raise ValueError(f"{run_name}: stages {order} repeat a behavior")
        if set(behaviors) - set(final.index):
            raise ValueError(f"{run_name}: missing final scores for {set(behaviors) - set(final.index)}")
        row = {"run_name": run_name, "order_id": first["order_id"], "method": first["method"],
               "seed": int(first["seed"]), "final_average": float(final[behaviors].mean())}
        if joint:
            row.update({"retention": np.nan, "final_task": np.nan})
        else:
            changes = []
            for learned_at, behavior in enumerate(order[:-1], start=1):
                right_after = part[(part["stage"] == learned_at) & (part["eval_set"] == behavior)][score]
                changes.append(float(final[behavior]) - float(right_after.iloc[0]))
            row.update({"retention": float(np.mean(changes)) if changes else np.nan,
                        "final_task": float(final[order[-1]])})
        rows.append(row)
    return pd.DataFrame(rows)


def paired_differences(cells, target, baseline, metric, seeds=None):
    """d[order, seed] = target - baseline for cells where both methods exist."""
    if metric not in METRICS:
        raise ValueError(f"unknown metric {metric!r}")
    pivot = cells.pivot_table(index=["order_id", "seed"], columns="method", values=metric)
    for method in (target, baseline):
        if method not in pivot:
            raise ValueError(f"no completed runs for {method!r}")
    diff = (pivot[target] - pivot[baseline]).dropna().rename("diff").reset_index()
    if seeds is not None:
        diff = diff[diff["seed"].isin(seeds)]
    return diff


def check_balanced(diffs):
    """Every seed must have the same set of orders, or the seed means are not comparable."""
    orders = diffs.groupby("seed")["order_id"].apply(lambda x: tuple(sorted(x)))
    if orders.nunique() != 1:
        raise ValueError(f"unbalanced design; orders per seed: {orders.to_dict()}")
    return len(orders.iloc[0]), len(orders)


def seed_mean_ttest(diffs, alpha=0.05):
    """Two-sided one-sample t-test on per-seed mean differences, with a (1 - alpha) CI."""
    from scipy import stats

    n_orders, n_seeds = check_balanced(diffs)
    seed_means = diffs.groupby("seed")["diff"].mean()
    if n_seeds < 2:
        raise ValueError("need at least two seeds")
    mean = float(seed_means.mean())
    se = float(seed_means.std(ddof=1) / np.sqrt(n_seeds))
    df = n_seeds - 1
    t = mean / se if se > 0 else np.copysign(np.inf, mean) if mean else 0.0
    p = float(2 * stats.t.sf(abs(t), df)) if np.isfinite(t) else 0.0
    half = float(stats.t.ppf(1 - alpha / 2, df) * se)
    return {
        "estimate": mean, "cell_mean": float(diffs["diff"].mean()), "se": se, "t": float(t),
        "df": df, "p_value": p, "ci_low": mean - half, "ci_high": mean + half,
        "n_seeds": n_seeds, "n_orders": n_orders, "n_cells": int(len(diffs)),
        "seed_means": {int(k): float(v) for k, v in seed_means.items()},
    }


def noninferiority_lower_bound(diffs, alpha=0.05):
    """One-sided (1 - alpha) lower confidence bound on the seed-mean difference."""
    from scipy import stats

    _, n_seeds = check_balanced(diffs)
    seed_means = diffs.groupby("seed")["diff"].mean()
    se = float(seed_means.std(ddof=1) / np.sqrt(n_seeds))
    return float(seed_means.mean() - stats.t.ppf(1 - alpha, n_seeds - 1) * se)


def seed_block_bootstrap(diffs, n_boot=10000, seed=0, alpha=0.05):
    """Sensitivity CI: resample seeds with replacement, keeping all orders of each seed."""
    check_balanced(diffs)
    by_seed = diffs.groupby("seed")["diff"].mean()
    values = by_seed.to_numpy()
    rng = np.random.default_rng(seed)
    draws = rng.choice(values, size=(n_boot, len(values)), replace=True).mean(axis=1)
    return float(np.quantile(draws, alpha / 2)), float(np.quantile(draws, 1 - alpha / 2))


def holm(p_values, alpha):
    """Holm step-down. p_values: {name: p}. Returns {name: (adjusted_p, reject)}."""
    names = sorted(p_values, key=lambda name: p_values[name])
    m = len(names)
    adjusted, running = {}, 0.0
    for rank, name in enumerate(names):
        running = max(running, min(1.0, (m - rank) * p_values[name]))
        adjusted[name] = running
    return {name: (adjusted[name], adjusted[name] <= alpha) for name in names}


def two_look_decision(look1):
    """'stop' iff P1 and P2 both have p < INTERIM_ALPHA at look 1, else 'extend'."""
    if look1["P1"]["p_value"] < INTERIM_ALPHA and look1["P2"]["p_value"] < INTERIM_ALPHA:
        return "stop"
    return "extend"


def primary_analysis(cells, look):
    """Predeclared P1-P3 analysis at look 1 or look 2. Returns (table, decision)."""
    if look not in (1, 2):
        raise ValueError("look must be 1 or 2")
    rows, tests = [], {}
    for name, (target, baseline, metric) in PRIMARY.items():
        seeds = CONFIRMATION_SEEDS
        if look == 2 and name in ("P1", "P2"):
            seeds = CONFIRMATION_SEEDS + EXTENSION_SEEDS
        diffs = paired_differences(cells, target, baseline, metric, seeds=seeds)
        missing = sorted(set(seeds) - set(diffs["seed"]))
        if missing:
            raise ValueError(f"{name}: seeds {missing} have no paired cells yet")
        result = seed_mean_ttest(diffs)
        result["bootstrap_ci"] = seed_block_bootstrap(diffs)
        tests[name] = result
        rows.append({"test": name, "target": target, "baseline": baseline, "metric": metric,
                     **{k: v for k, v in result.items() if k != "seed_means"}})
    decision = two_look_decision(tests) if look == 1 else "final"
    alpha = (ALPHA_NO_EXTENSION if decision == "stop" else
             FINAL_ALPHA_AFTER_EXTENSION if look == 2 else None)
    table = pd.DataFrame(rows)
    if alpha is not None:
        adjusted = holm({name: tests[name]["p_value"] for name in tests}, alpha)
        table["holm_p"] = table["test"].map(lambda name: adjusted[name][0])
        table["significant"] = table["test"].map(lambda name: adjusted[name][1])
        table["family_alpha"] = alpha
        target, baseline, metric, margin = NONINFERIORITY
        seeds = CONFIRMATION_SEEDS + (EXTENSION_SEEDS if look == 2 else ())
        bound = noninferiority_lower_bound(
            paired_differences(cells, target, baseline, metric, seeds=seeds))
        table.attrs["noninferiority"] = {"lower_bound": bound, "margin": margin,
                                         "passes": bound > margin}
    table.attrs["decision"] = decision
    return table, decision


# ----------------------------------------------------------------------------- extension

def select_variant(dev_cells, candidates, baseline="random", tolerance=2.0):
    """The development selection rule DAPR used, applied to a new method's strength variants.

    Among `candidates` whose mean final-task score on the development cells is at most `tolerance`
    points below `baseline`, return the one with the highest mean final three-behavior average.
    Returns (chosen_method, table).
    """
    means = dev_cells.groupby("method")[["final_task", "final_average"]].mean()
    missing = [method for method in [*candidates, baseline] if method not in means.index]
    if missing:
        raise ValueError(f"no development cells for {missing}")
    table = means.loc[list(candidates)].copy()
    table["final_task_vs_baseline"] = table["final_task"] - means.loc[baseline, "final_task"]
    table["passes_tolerance"] = table["final_task_vs_baseline"] >= -tolerance
    passing = table[table["passes_tolerance"]]
    if passing.empty:
        raise ValueError("no candidate is within the new-learning tolerance")
    return str(passing["final_average"].idxmax()), table


def extension_analysis(cells, der_method, sym_method, alpha=ALPHA_NO_EXTENSION):
    """Predeclared extension tests (docs/EXTENSION.md), confirmation seeds 0-4, Holm at alpha.

    E1: DAPR - DER++-style on final average. E2: DAPR - Symmetric-DAPR on retention.
    `der_method` / `sym_method` are the strength variants chosen by select_variant on the
    development cells before the 30-cell runs.
    """
    tests = {"E1": ("dapr_weak", der_method, "final_average"),
             "E2": ("dapr_weak", sym_method, "retention")}
    rows, p_values = [], {}
    for name, (target, baseline, metric) in tests.items():
        diffs = paired_differences(cells, target, baseline, metric, seeds=CONFIRMATION_SEEDS)
        result = seed_mean_ttest(diffs)
        result["bootstrap_ci"] = seed_block_bootstrap(diffs)
        p_values[name] = result["p_value"]
        rows.append({"test": name, "target": target, "baseline": baseline, "metric": metric,
                     **{k: v for k, v in result.items() if k != "seed_means"}})
    adjusted = holm(p_values, alpha)
    table = pd.DataFrame(rows)
    table["holm_p"] = table["test"].map(lambda name: adjusted[name][0])
    table["significant"] = table["test"].map(lambda name: adjusted[name][1])
    table["family_alpha"] = alpha
    return table
