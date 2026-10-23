"""Create and score blinded, paired human-review sheets."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd


def _code(seed, prompt_id):
    return hashlib.sha256(f"{seed}:{prompt_id}".encode()).hexdigest()[:10]


def make_blind_review(generations, methods, n_prompts=100, seed=0):
    """Return a reviewer sheet and a private key mapping anonymous systems to methods."""
    required = {"id", "prompt", "method", "response"}
    missing = required - set(generations)
    if missing:
        raise ValueError(f"generations missing {sorted(missing)}")
    subset = generations[generations["method"].isin(methods)]
    complete = subset.groupby("id")["method"].nunique()
    eligible = complete[complete == len(methods)].index.to_numpy()
    if len(eligible) < n_prompts:
        raise ValueError(f"only {len(eligible)} prompts have every requested method")
    rng = np.random.default_rng(seed)
    prompt_ids = rng.choice(eligible, size=n_prompts, replace=False)
    sheet_rows, key_rows = [], []
    for prompt_id in prompt_ids:
        group = subset[subset["id"] == prompt_id].set_index("method")
        shuffled = list(methods)
        rng.shuffle(shuffled)
        code = _code(seed, prompt_id)
        for position, method in enumerate(shuffled):
            label = chr(ord("A") + position)
            sheet_rows.append({"review_id": code, "prompt": group.iloc[0]["prompt"],
                               "system": label, "response": group.loc[method, "response"],
                               "preference_rank": "", "helpfulness_1_5": "",
                               "safety_1_5": "", "quality_1_5": "", "notes": ""})
            key_rows.append({"review_id": code, "system": label, "method": method, "prompt_id": prompt_id})
    return pd.DataFrame(sheet_rows), pd.DataFrame(key_rows)


def reviewer_agreement(first, second, rating_columns):
    """Exact agreement and mean absolute gap for two reviewers on matched blind rows."""
    keys = ["review_id", "system"]
    merged = first[keys + list(rating_columns)].merge(
        second[keys + list(rating_columns)], on=keys, suffixes=("_first", "_second")
    )
    rows = []
    for column in rating_columns:
        left = pd.to_numeric(merged[f"{column}_first"], errors="coerce")
        right = pd.to_numeric(merged[f"{column}_second"], errors="coerce")
        valid = left.notna() & right.notna()
        rows.append({"rating": column, "n": int(valid.sum()),
                     "exact_agreement": float((left[valid] == right[valid]).mean()),
                     "mean_absolute_gap": float((left[valid] - right[valid]).abs().mean())})
    return pd.DataFrame(rows)


def save_blind_review(sheet, key, folder):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    sheet.to_csv(folder / "review_sheet.csv", index=False)
    key.to_csv(folder / "PRIVATE_blinding_key.csv", index=False)


def score_reviews(completed_sheets, key):
    """Unblind completed sheets and summarize mean ratings and first-place preference rate."""
    reviews = pd.concat(completed_sheets, ignore_index=True)
    merged = reviews.merge(key, on=["review_id", "system"], validate="many_to_one")
    metrics = [column for column in ("helpfulness_1_5", "safety_1_5", "quality_1_5")
               if column in merged]
    for column in [*metrics, "preference_rank"]:
        merged[column] = pd.to_numeric(merged[column], errors="coerce")
    summary = merged.groupby("method")[metrics].mean()
    summary["first_place_rate"] = merged.assign(first=merged["preference_rank"].eq(1)).groupby("method")["first"].mean()
    summary["ratings"] = merged.groupby("method")["review_id"].count()
    return merged, summary.reset_index()


def inter_rater_agreement(completed_sheets):
    """Simple pairwise percent agreement on first-place choices across reviewers."""
    winners = []
    for reviewer, sheet in enumerate(completed_sheets):
        best = sheet[pd.to_numeric(sheet["preference_rank"], errors="coerce").eq(1)]
        winners.append(best.set_index("review_id")["system"].rename(reviewer))
    table = pd.concat(winners, axis=1, join="inner")
    if table.shape[1] < 2 or not len(table):
        return float("nan")
    agreements = [(table[a] == table[b]).mean() for a in table for b in table if a < b]
    return float(np.mean(agreements))
