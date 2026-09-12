"""Collect every run from the team Drive into tables we can compare and plot.

Reads only saved files (no GPU): each run folder has settings.json, results.csv, and one folder per
stage with history.csv, replay_log.csv, margins_*.csv and buffer.csv.

Main entry points:
    runs = load_runs(DRIVE_DIR)          every result row of every run, with run metadata attached
    retention_table(runs)                per run and earlier dataset: score right after vs at the end
    summary_table(runs)                  one row per run: retention, new learning, worst case
    cost_table(runs)                     minutes trained, minutes spent re-scoring, peak steps
    replay_summary(runs)                 which pairs each method actually replayed
"""

import glob
import json
import os

import pandas as pd

METHOD_ORDER = ["none", "random", "lowest_margin", "mfr"]
METHOD_COLORS = {"none": "#6b6a66", "random": "#2a78d6", "lowest_margin": "#1baf7a", "mfr": "#eb6834"}
DATASETS = ["safe", "helpful", "quality"]


# ------------------------------------------------------------------ loading

def read_settings(folder):
    """Run metadata, filled in for older runs (the pilot has no method: it is a no-replay run)."""
    path = os.path.join(folder, "settings.json")
    saved = json.load(open(path)) if os.path.exists(path) else {}
    order = saved.get("order") or DATASETS
    return {
        "run_name": saved.get("run_name", os.path.basename(os.path.normpath(folder))),
        "order_id": saved.get("order_id", 2 if order[0] == "safe" else 1),
        "order": order,
        "method": saved.get("method", "none"),
        "seed": saved.get("seed", 0),
        "folder": os.path.normpath(folder),
    }


def load_runs(drive_dir, quiet=False):
    """Every (run, stage, eval set) row of every finished-or-partial run in drive_dir/runs/."""
    frames = []
    for folder in sorted(glob.glob(os.path.join(drive_dir, "runs", "*"))):
        results = os.path.join(folder, "results.csv")
        if not os.path.isdir(folder) or not os.path.exists(results):
            continue
        settings = read_settings(folder)
        df = pd.read_csv(results)
        for key in ("run_name", "order_id", "method", "seed"):
            if key not in df.columns:
                df[key] = settings[key]
        df["folder"] = settings["folder"]
        df["order"] = [settings["order"]] * len(df)
        frames.append(df)
    if not frames:
        raise FileNotFoundError(f"no runs with a results.csv found in {drive_dir}/runs/")
    runs = pd.concat(frames, ignore_index=True)
    runs["method"] = pd.Categorical(runs["method"], categories=METHOD_ORDER, ordered=True)
    if not quiet:
        done = runs.groupby("run_name")["stage"].max()
        print(f"{len(done)} runs found:")
        for name, stage in done.items():
            print(f"  {name:24s} stages finished: {stage}/3")
    return runs


def _run_rows(runs):
    """Iterate over runs: (metadata dict, that run's rows)."""
    for name, part in runs.groupby("run_name", observed=True):
        meta = {"run_name": name, "order_id": part["order_id"].iloc[0], "method": part["method"].iloc[0],
                "seed": part["seed"].iloc[0], "order": part["order"].iloc[0], "folder": part["folder"].iloc[0]}
        yield meta, part


# ------------------------------------------------------------------ tables

def retention_table(runs, score="accuracy"):
    """For every run and every dataset learned before the last stage: how much survived.

    right_after  the score just after that dataset's own stage
    at_end       the score after the final stage
    change       at_end - right_after (negative = forgotten)
    margin_kept  final mean margin as a % of the mean margin right after (100 = nothing lost)
    """
    rows = []
    for meta, part in _run_rows(runs):
        last = part["stage"].max()
        if last < len(meta["order"]):
            continue                                        # run not finished yet
        for k, dataset in enumerate(meta["order"][:-1], start=1):
            after = part[(part["stage"] == k) & (part["eval_set"] == dataset)]
            end = part[(part["stage"] == last) & (part["eval_set"] == dataset)]
            if not len(after) or not len(end):
                continue
            after, end = after.iloc[0], end.iloc[0]
            rows.append({**{key: meta[key] for key in ("run_name", "order_id", "method", "seed")},
                         "dataset": dataset, "learned_at": k,
                         "right_after": after[score], "at_end": end[score],
                         "change": round(end[score] - after[score], 2),
                         "margin_kept": round(100 * end["mean_margin"] / after["mean_margin"], 1)
                                        if after["mean_margin"] > 0 else None})
    return pd.DataFrame(rows)


def summary_table(runs, score="accuracy"):
    """One row per finished run: how much of the earlier stages survived, and what it learned last."""
    retention = retention_table(runs, score=score)
    rows = []
    for meta, part in _run_rows(runs):
        last = part["stage"].max()
        if last < len(meta["order"]):
            continue
        mine = retention[retention["run_name"] == meta["run_name"]]
        newest = meta["order"][-1]
        new_score = part[(part["stage"] == last) & (part["eval_set"] == newest)][score]
        rows.append({**{key: meta[key] for key in ("run_name", "order_id", "method", "seed")},
                     "mean change (points)": round(mine["change"].mean(), 2) if len(mine) else None,
                     "worst change (points)": round(mine["change"].min(), 2) if len(mine) else None,
                     "mean margin kept (%)": round(mine["margin_kept"].mean(), 1) if len(mine) else None,
                     f"new stage ({newest}) (%)": new_score.iloc[0] if len(new_score) else None})
    out = pd.DataFrame(rows)
    return out.sort_values(["order_id", "seed", "method"]).reset_index(drop=True) if len(out) else out


def cost_table(runs):
    """Training minutes and buffer re-scoring minutes per run (from each stage's history.csv)."""
    rows = []
    for meta, part in _run_rows(runs):
        minutes, scoring, steps = 0.0, 0.0, 0
        for stage in sorted(part["stage"].unique()):
            dataset = meta["order"][int(stage) - 1]
            path = os.path.join(meta["folder"], f"stage{int(stage)}_{dataset}", "history.csv")
            if not os.path.exists(path):
                continue
            history = pd.read_csv(path)
            minutes += history["minutes"].max()
            scoring += history["scoring_minutes"].max() if "scoring_minutes" in history else 0.0
            steps += len(history)
        rows.append({**{key: meta[key] for key in ("run_name", "method", "order_id", "seed")},
                     "steps": steps, "minutes": round(minutes, 1),
                     "of which re-scoring": round(scoring, 1)})
    return pd.DataFrame(rows)


def replay_summary(runs):
    """What each run actually replayed: how many slots, how many different pairs, from which dataset."""
    rows = []
    for meta, part in _run_rows(runs):
        for stage in sorted(part["stage"].unique()):
            dataset = meta["order"][int(stage) - 1]
            path = os.path.join(meta["folder"], f"stage{int(stage)}_{dataset}", "replay_log.csv")
            if not os.path.exists(path):
                continue
            log = pd.read_csv(path)
            if not len(log):
                continue
            rows.append({"run_name": meta["run_name"], "method": meta["method"], "stage": int(stage),
                         "trained_on": dataset, "replayed slots": len(log),
                         "different pairs": log["id"].nunique(),
                         "from": log["dataset"].value_counts().to_dict()})
    return pd.DataFrame(rows)


def stage_curve(runs, order_id, seed, score="accuracy"):
    """Table for the line chart: score of each eval set after each stage, one column per method."""
    part = runs[(runs["order_id"] == order_id) & (runs["seed"] == seed)]
    return part.pivot_table(index=["eval_set", "stage"], columns="method", values=score, observed=True)


def margins_at(folder, order, stage, eval_set, split="val"):
    """The per-pair margins saved after a stage (index = pair id)."""
    path = os.path.join(folder, f"stage{stage}_{order[stage - 1]}", f"margins_{eval_set}_{split}.csv")
    return pd.read_csv(path, index_col=0) if os.path.exists(path) else None


def concentration(folder, order, dataset, score="margin"):
    """How concentrated the forgetting is: share of the total margin loss from the worst x% of pairs."""
    learned_at = order.index(dataset) + 1
    after = margins_at(folder, order, learned_at, dataset)
    end = margins_at(folder, order, len(order), dataset)
    if after is None or end is None:
        return None
    drop = (after[score] - end[score]).clip(lower=0).sort_values(ascending=False)
    if drop.sum() <= 0:
        return None
    shares = {}
    for percent in (10, 20, 50):
        top = drop.head(max(1, len(drop) * percent // 100)).sum()
        shares[f"worst {percent}% of pairs"] = round(100 * top / drop.sum(), 1)
    shares["pairs that lost margin (%)"] = round(100 * (drop > 0).mean(), 1)
    return shares
