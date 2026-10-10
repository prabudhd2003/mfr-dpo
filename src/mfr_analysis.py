"""Load runs, validate comparisons, and compute retention, learning, cost, and uncertainty."""

from __future__ import annotations

import glob
import json
import os
import warnings

import numpy as np
import pandas as pd

METHOD_ORDER = [
    "none", "random", "random_high", "lowest_margin", "mfr", "fmcr", "cpmr",
    "dapr", "dapr_weak", "dapr_gated", "dapr_c", "mir_dpo", "copr_adapted",
    "ewc_100", "ewc_1000", "ewc_10000",
]
METHOD_COLORS = {
    "none": "#6b6a66", "random": "#2a78d6", "random_high": "#7857c5",
    "lowest_margin": "#1baf7a", "mfr": "#eb6834", "fmcr": "#b52fb5",
    "cpmr": "#7a3db8",
    "dapr": "#d68a00", "dapr_weak": "#e6ad38", "dapr_gated": "#8a5a00",
    "dapr_c": "#a45c00", "mir_dpo": "#008c95",
    "copr_adapted": "#b13c63",
    "ewc_100": "#4c78a8", "ewc_1000": "#355f8a", "ewc_10000": "#1f3b57",
}
DATASETS = ["safe", "helpful", "quality"]


def read_settings(folder):
    path = os.path.join(folder, "settings.json")
    if os.path.exists(path):
        with open(path, encoding="utf-8") as handle:
            saved = json.load(handle)
    else:
        saved = {}
    order = saved.get("order") or DATASETS
    return {
        "run_name": saved.get("run_name", os.path.basename(os.path.normpath(folder))),
        "order_id": int(saved.get("order_id", 2 if order[0] == "safe" else 1)),
        "order": order,
        "method": saved.get("method", "none"),
        "seed": int(saved.get("seed", 0)),
        "protocol_version": saved.get("protocol_version", "pre-v2"),
        "data_version": saved.get("data_version", "v1"),
        "model_revision": saved.get("model_revision", "unpinned"),
        "stage1_from": saved.get("stage1_from"),
        "folder": os.path.normpath(folder),
    }


def load_runs(artifact_dir, quiet=False, include_preliminary=True, completed_only=False,
              protocol_version=None):
    frames = []
    for folder in sorted(glob.glob(os.path.join(artifact_dir, "runs", "*"))):
        result_path = os.path.join(folder, "results.csv")
        if not os.path.isfile(result_path):
            continue
        if completed_only and not os.path.exists(os.path.join(folder, "COMPLETE.json")):
            continue
        settings = read_settings(folder)
        if protocol_version is not None and str(settings["protocol_version"]) != str(protocol_version):
            continue
        if not include_preliminary and settings["protocol_version"] == "pre-v2":
            continue
        frame = pd.read_csv(result_path)
        for key in ("run_name", "order_id", "method", "seed", "protocol_version", "data_version"):
            if key not in frame:
                frame[key] = settings[key]
        frame["folder"] = settings["folder"]
        frame["order"] = [settings["order"]] * len(frame)
        frame["stage1_from"] = settings["stage1_from"]
        frames.append(frame)
    if not frames:
        raise FileNotFoundError(f"no runs with results.csv found under {artifact_dir}/runs")
    runs = pd.concat(frames, ignore_index=True)
    found = list(dict.fromkeys(runs["method"].astype(str)))
    method_order = [method for method in METHOD_ORDER if method in found]
    method_order += [method for method in found if method not in method_order]
    runs["method"] = pd.Categorical(runs["method"], method_order, ordered=True)
    if not quiet:
        status = runs.groupby("run_name")["stage"].max()
        print(f"{len(status)} runs found")
        for name, stage in status.items():
            print(f"  {name:28s} stages finished: {stage}/3")
    return runs


def _run_rows(runs):
    for name, part in runs.groupby("run_name", observed=True):
        keys = ("order_id", "method", "seed", "order", "folder", "protocol_version", "data_version", "stage1_from")
        yield {"run_name": name, **{key: part[key].iloc[0] for key in keys}}, part


def validate_compatible_runs(runs, strict=True):
    """Warn or fail when a comparison mixes protocols, data versions, or incomplete method cells."""
    issues = []
    for column in ("protocol_version", "data_version"):
        values = sorted(map(str, runs[column].dropna().unique()))
        if len(values) > 1:
            issues.append(f"mixed {column}: {values}")
    cells = runs[["order_id", "seed", "method", "run_name"]].drop_duplicates()
    for (order_id, seed), group in cells.groupby(["order_id", "seed"], observed=True):
        methods = set(group["method"].astype(str))
        if not {"none", "random", "mfr"} <= methods:
            issues.append(f"order {order_id}, seed {seed} lacks one of none/random/mfr")
    if issues and strict:
        raise ValueError("incompatible run comparison:\n- " + "\n- ".join(issues))
    for issue in issues:
        warnings.warn(issue)
    return not issues


def retention_table(runs, score="accuracy"):
    """Earlier-stage score immediately after learning versus after the final stage."""
    rows = []
    for meta, part in _run_rows(runs):
        last = int(part["stage"].max())
        if last < len(meta["order"]):
            continue
        for learned_at, dataset in enumerate(meta["order"][:-1], start=1):
            after = part[(part["stage"] == learned_at) & (part["eval_set"] == dataset)]
            end = part[(part["stage"] == last) & (part["eval_set"] == dataset)]
            if after.empty or end.empty:
                continue
            after, end = after.iloc[0], end.iloc[0]
            baseline_margin = after.get("mean_margin", np.nan)
            rows.append({
                **{key: meta[key] for key in ("run_name", "order_id", "method", "seed")},
                "dataset": dataset, "learned_at": learned_at,
                "right_after": after[score], "at_end": end[score],
                "change": end[score] - after[score],
                "margin_kept": 100 * end.get("mean_margin", np.nan) / baseline_margin
                if pd.notna(baseline_margin) and baseline_margin != 0 else np.nan,
            })
    return pd.DataFrame(rows)


def new_learning_table(runs, score="accuracy"):
    """How much each stage improved its own validation set from before to after training."""
    rows = []
    for meta, part in _run_rows(runs):
        for stage, dataset in enumerate(meta["order"], start=1):
            after = part[(part["stage"] == stage) & (part["eval_set"] == dataset)]
            before_stage = stage - 1
            before = part[(part["stage"] == before_stage) & (part["eval_set"] == dataset)]
            if after.empty:
                continue
            after_value = after.iloc[0][score]
            before_value = before.iloc[0][score] if not before.empty else np.nan
            rows.append({
                **{key: meta[key] for key in ("run_name", "order_id", "method", "seed")},
                "stage": stage, "dataset": dataset, "before": before_value, "after": after_value,
                "gain": after_value - before_value if pd.notna(before_value) else np.nan,
            })
    return pd.DataFrame(rows)


def summary_table(runs, score="accuracy"):
    retention = retention_table(runs, score)
    learning = new_learning_table(runs, score)
    rows = []
    for meta, part in _run_rows(runs):
        last = int(part["stage"].max())
        if last < len(meta["order"]):
            continue
        retained = retention[retention["run_name"] == meta["run_name"]]
        learned = learning[learning["run_name"] == meta["run_name"]]
        rows.append({
            **{key: meta[key] for key in ("run_name", "order_id", "method", "seed")},
            "mean_retention_change": retained["change"].mean(),
            "worst_retention_change": retained["change"].min(),
            "mean_margin_kept_pct": retained["margin_kept"].mean(),
            "mean_new_stage_gain": learned["gain"].mean(),
            "final_new_stage_score": learned.loc[learned["stage"] == last, "after"].iloc[0]
            if (learned["stage"] == last).any() else np.nan,
        })
    out = pd.DataFrame(rows)
    return out.sort_values(["order_id", "seed", "method"]).reset_index(drop=True) if len(out) else out


def resolve_borrowed_run(folder, recorded_path):
    """Resolve a recorded source path after an artifact directory is moved.

    The recorded path is tried first. If it does not exist, the source run's basename is
    resolved as a sibling of the current run.
    """
    if not recorded_path:
        return None
    recorded = os.path.normpath(recorded_path)
    if os.path.exists(recorded):
        return recorded
    sibling = os.path.join(os.path.dirname(os.path.normpath(folder)), os.path.basename(recorded))
    return sibling if os.path.exists(sibling) else recorded


def stage_folder(folder, order, stage, stage1_from=None):
    """Portable folder for a local stage or a borrowed stage-1 artifact."""
    dataset = order[int(stage) - 1]
    direct = os.path.join(os.path.normpath(folder), f"stage{int(stage)}_{dataset}")
    if os.path.exists(direct):
        return direct
    if int(stage) == 1 and stage1_from:
        source_run = resolve_borrowed_run(folder, stage1_from)
        return os.path.join(source_run, f"stage1_{dataset}")
    return direct


def _stage_folder(meta, stage, dataset=None):
    return stage_folder(meta["folder"], meta["order"], stage, meta.get("stage1_from"))


def cost_table(runs):
    """Actual training/scoring time, including a reused stage-1 artifact once per run."""
    rows = []
    for meta, part in _run_rows(runs):
        minutes = scoring = 0.0
        steps = examples = 0
        for stage in sorted(map(int, part["stage"].unique())):
            dataset = meta["order"][stage - 1]
            path = os.path.join(_stage_folder(meta, stage, dataset), "history.csv")
            if not os.path.exists(path):
                continue
            history = pd.read_csv(path)
            minutes += history["minutes"].max()
            scoring += history["scoring_minutes"].max() if "scoring_minutes" in history else 0.0
            steps += len(history)
            if "n_new" in history:
                replayed = history["n_replay"].sum() if "n_replay" in history else 0
                examples += int(history["n_new"].sum() + replayed)
        rows.append({**{key: meta[key] for key in ("run_name", "method", "order_id", "seed")},
                     "steps": steps, "examples": examples, "minutes": minutes,
                     "scoring_minutes": scoring, "training_minutes": minutes - scoring})
    return pd.DataFrame(rows)


def replay_summary(runs):
    rows = []
    for meta, part in _run_rows(runs):
        for stage in sorted(map(int, part["stage"].unique())):
            dataset = meta["order"][stage - 1]
            path = os.path.join(_stage_folder(meta, stage, dataset), "replay_log.csv")
            if not os.path.exists(path):
                continue
            log = pd.read_csv(path)
            if len(log):
                rows.append({"run_name": meta["run_name"], "method": meta["method"], "stage": stage,
                             "trained_on": dataset, "replayed_slots": len(log),
                             "different_pairs": log["id"].nunique(),
                             "from": log["dataset"].value_counts().to_dict()})
    return pd.DataFrame(rows)


def stage_curve(runs, order_id, seed, score="accuracy"):
    part = runs[(runs["order_id"] == order_id) & (runs["seed"] == seed)]
    return part.pivot_table(index=["eval_set", "stage"], columns="method", values=score, observed=True)


def margins_at(folder, order, stage, eval_set, split="val", stage1_from=None):
    """Load pair margins, including portable resolution of a borrowed stage 1."""
    if int(stage) == 1 and stage1_from is None:
        settings_path = os.path.join(folder, "settings.json")
        if os.path.exists(settings_path):
            with open(settings_path, encoding="utf-8") as handle:
                stage1_from = json.load(handle).get("stage1_from")
    path = os.path.join(
        stage_folder(folder, order, stage, stage1_from), f"margins_{eval_set}_{split}.csv"
    )
    return pd.read_csv(path).set_index("id")


def paired_binary_difference(left, right, column="correct", n_boot=10000, seed=0):
    """Paired percentage-point difference and prompt-bootstrap 95% interval."""
    merged = left[["id", column]].merge(right[["id", column]], on="id", suffixes=("_left", "_right"))
    differences = 100 * (merged[f"{column}_left"].astype(float) - merged[f"{column}_right"].astype(float)).to_numpy()
    if not len(differences):
        raise ValueError("no paired IDs")
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(differences), size=(n_boot, len(differences)))
    samples = differences[indices].mean(axis=1)
    return {"difference_points": differences.mean(), "ci_low": np.quantile(samples, 0.025),
            "ci_high": np.quantile(samples, 0.975), "n_pairs": len(differences)}


def paired_numeric_difference(left, right, column, n_boot=10000, seed=0, scale=1.0):
    """Mean paired left-minus-right difference with a row-bootstrap interval."""
    merged = left[["id", column]].merge(right[["id", column]], on="id", suffixes=("_left", "_right"))
    differences = scale * (
        merged[f"{column}_left"].astype(float) - merged[f"{column}_right"].astype(float)
    ).to_numpy()
    if not len(differences):
        raise ValueError("no paired IDs")
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(differences), size=(n_boot, len(differences)))
    samples = differences[indices].mean(axis=1)
    return {"difference": differences.mean(), "ci_low": np.quantile(samples, 0.025),
            "ci_high": np.quantile(samples, 0.975), "n_pairs": len(differences)}


def paired_retention_comparison(runs, method="mfr", baseline="random", margin="margin",
                                n_boot=10000, seed=0):
    """Pair-level comparison of retention changes for matched order/seed/old-behavior cells."""
    rows = []
    all_meta = [meta for meta, _ in _run_rows(runs)]
    for order_id in sorted({int(meta["order_id"]) for meta in all_meta}):
        seeds = sorted({int(meta["seed"]) for meta in all_meta if int(meta["order_id"]) == order_id})
        for run_seed in seeds:
            cell = [meta for meta in all_meta
                    if int(meta["order_id"]) == order_id and int(meta["seed"]) == run_seed]
            by_method = {str(meta["method"]): meta for meta in cell}
            if method not in by_method or baseline not in by_method:
                continue
            left_meta, right_meta = by_method[method], by_method[baseline]
            order = left_meta["order"]
            for learned_at, dataset in enumerate(order[:-1], start=1):
                frames = []
                for meta in (left_meta, right_meta):
                    learned_path = os.path.join(
                        _stage_folder(meta, learned_at, dataset), f"margins_{dataset}_val.csv"
                    )
                    final_path = os.path.join(
                        _stage_folder(meta, len(order), order[-1]), f"margins_{dataset}_val.csv"
                    )
                    if not os.path.exists(learned_path) or not os.path.exists(final_path):
                        frames = []
                        break
                    learned = pd.read_csv(learned_path)[["id", margin]].rename(columns={margin: "learned"})
                    final = pd.read_csv(final_path)[["id", margin]].rename(columns={margin: "final"})
                    frame = learned.merge(final, on="id")
                    frame["retention_change"] = (
                        (frame["final"] > 0).astype(int) - (frame["learned"] > 0).astype(int)
                    )
                    frames.append(frame[["id", "retention_change"]])
                if len(frames) != 2:
                    continue
                result = paired_numeric_difference(
                    frames[0], frames[1], "retention_change", n_boot=n_boot,
                    seed=seed + order_id * 100 + run_seed * 10 + learned_at, scale=100,
                )
                rows.append({"order_id": order_id, "seed": run_seed, "dataset": dataset,
                             "method": method, "baseline": baseline,
                             "difference_points": result["difference"],
                             "ci_low": result["ci_low"], "ci_high": result["ci_high"],
                             "n_pairs": result["n_pairs"]})
    return pd.DataFrame(rows)


def concentration(values, top_fraction=0.10):
    """Share of total positive loss carried by the worst fraction of examples."""
    losses = np.maximum(np.asarray(values, dtype=float), 0)
    if losses.sum() == 0:
        return 0.0
    n = max(1, int(np.ceil(len(losses) * top_fraction)))
    return np.sort(losses)[-n:].sum() / losses.sum()


# ----------------------------------------------------------------------------- confirmation

def load_locked_test(artifact_dir, subdir="runs"):
    """All locked_test/results_test.csv rows (format accepted by mfr_stats.cell_metrics)."""
    paths = sorted(glob.glob(os.path.join(artifact_dir, subdir, "*", "locked_test", "results_test.csv")))
    if not paths:
        raise FileNotFoundError(f"no locked_test/results_test.csv under {artifact_dir}/{subdir}")
    return pd.concat([pd.read_csv(path) for path in paths], ignore_index=True)


# ----------------------------------------------------------------------------- mechanisms

# How each selection score is computed from a buffer row: higher = selected first.
SELECTION_SCORES = {
    "lowest_margin": lambda rows: -rows["current_margin"],
    "dapr": lambda rows: -rows["current_margin"],
    "dapr_weak": lambda rows: -rows["current_margin"],
    "dapr_gated": lambda rows: -rows["current_margin"],
    "dapr_c": lambda rows: -rows["current_margin"],
    "copr_adapted": lambda rows: -rows["current_margin"],
    "mfr": lambda rows: rows["peak_margin"] - rows["current_margin"],
    "cpmr": lambda rows: -np.minimum(rows["current_margin"], rows["projected_margin"]),
}
SNAPSHOT_PREFIX = {"cpmr": "cpmr_refresh", "fmcr": "fmcr_refresh", "mir_dpo": "mir_dpo_refresh"}


def replay_regression_share(runs):
    """For every replayed slot: had the pair actually regressed below its own peak margin?

    current margin at selection is recovered from the logged selection score (Lowest-Margin
    family: -score; MFR: peak - score). Peak margins come from the buffer saved at the end of
    the previous stage. `never_learned` = peak margin <= 0. One row per (run, stage).
    """
    rows = []
    for meta, part in _run_rows(runs):
        method = str(meta["method"])
        if method not in SELECTION_SCORES or method == "cpmr":
            continue
        for stage in range(2, len(meta["order"]) + 1):
            log_path = os.path.join(_stage_folder(meta, stage), "replay_log.csv")
            buffer_path = os.path.join(_stage_folder(meta, stage - 1), "buffer.csv")
            if not (os.path.exists(log_path) and os.path.exists(buffer_path)):
                continue
            log = pd.read_csv(log_path)
            if log.empty:
                continue
            peak = pd.read_csv(buffer_path).set_index("id")["peak_margin"]
            log["peak_margin"] = log["id"].map(peak)
            if method == "mfr":
                log["current_margin"] = log["peak_margin"] - log["selection_score"]
            else:
                log["current_margin"] = -log["selection_score"]
            regressed = log["current_margin"] < log["peak_margin"]
            rows.append({
                **{key: meta[key] for key in ("run_name", "order_id", "method", "seed")},
                "stage": stage, "trained_on": meta["order"][stage - 1], "slots": len(log),
                "regressed_pct": 100.0 * float(regressed.mean()),
                "never_learned_pct": 100.0 * float((log["peak_margin"] <= 0).mean()),
                "regressed_and_learned_pct": 100.0 * float((regressed & (log["peak_margin"] > 0)).mean()),
            })
    return pd.DataFrame(rows)


def score_validity(runs):
    """Does a method's selection score predict the next interval's margin drop?

    For refresh i and i+1 snapshots of the full buffer, drop = current_i - current_(i+1) on pairs
    NOT replayed during interval i (replayed pairs are changed by the intervention itself).
    Reports Spearman rho(score_i, drop) per (run, stage, interval). Needs per-refresh snapshots:
    buffer_refresh_*.csv (written for refresh methods from this commit on) or the CPMR/FMCR/MIR
    files.
    """
    from scipy.stats import spearmanr

    rows = []
    for meta, part in _run_rows(runs):
        method = str(meta["method"])
        if method not in SELECTION_SCORES:
            continue
        prefix = SNAPSHOT_PREFIX.get(method, "buffer_refresh")
        for stage in range(2, len(meta["order"]) + 1):
            folder = _stage_folder(meta, stage)
            snapshots = sorted(glob.glob(os.path.join(folder, f"{prefix}_*.csv")),
                               key=lambda path: int(path.rsplit("_", 1)[1].split(".")[0]))
            log_path = os.path.join(folder, "replay_log.csv")
            if len(snapshots) < 2 or not os.path.exists(log_path):
                continue
            log = pd.read_csv(log_path)
            for index in range(len(snapshots) - 1):
                now = pd.read_csv(snapshots[index]).set_index("id")
                later = pd.read_csv(snapshots[index + 1]).set_index("id")
                interval = index + 1
                replayed = set(log.loc[log["interval"] == interval, "id"])
                keep = [pair for pair in now.index if pair not in replayed and pair in later.index]
                if len(keep) < 10:
                    continue
                score = SELECTION_SCORES[method](now.loc[keep])
                drop = now.loc[keep, "current_margin"] - later.loc[keep, "current_margin"]
                rho, p = spearmanr(score, drop)
                rows.append({
                    **{key: meta[key] for key in ("run_name", "order_id", "method", "seed")},
                    "stage": stage, "interval": interval, "pairs": len(keep),
                    "spearman_rho": float(rho), "p_value": float(p),
                })
    return pd.DataFrame(rows)
