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


def make_pairwise_blind_review(generations, candidate_method, baseline_method,
                               prompts_per_behavior=50, seed=544):
    """Create a wide, blinded A/B sheet stratified by behavior and its private key."""
    required = {"id", "behavior", "prompt", "method", "response"}
    if missing := required - set(generations):
        raise ValueError(f"generations missing {sorted(missing)}")
    methods = [candidate_method, baseline_method]
    subset = generations[generations["method"].isin(methods)].copy()
    rng = np.random.default_rng(seed)
    sheet_rows, key_rows = [], []
    for behavior, frame in subset.groupby("behavior", sort=True):
        complete = frame.groupby("id")["method"].nunique()
        eligible = complete[complete == 2].index.to_numpy()
        if len(eligible) < prompts_per_behavior:
            raise ValueError(
                f"only {len(eligible)} complete {behavior} prompts; need {prompts_per_behavior}"
            )
        selected = rng.choice(eligible, size=prompts_per_behavior, replace=False)
        for prompt_id in selected:
            group = frame[frame["id"] == prompt_id].set_index("method")
            a_method, b_method = methods if rng.random() < 0.5 else methods[::-1]
            code = _code(seed, f"{behavior}:{prompt_id}")
            sheet_rows.append({
                "review_id": code,
                "behavior": behavior,
                "prompt": group.iloc[0]["prompt"],
                "response_A": group.loc[a_method, "response"],
                "response_B": group.loc[b_method, "response"],
                "preferred_response": "",
                "confidence_1_5": "",
                "notes": "",
            })
            key_rows.append({
                "review_id": code,
                "behavior": behavior,
                "prompt_id": prompt_id,
                "method_A": a_method,
                "method_B": b_method,
                "candidate_method": candidate_method,
                "baseline_method": baseline_method,
            })
    sheet = pd.DataFrame(sheet_rows).sample(frac=1, random_state=seed).reset_index(drop=True)
    return sheet, pd.DataFrame(key_rows)


def score_pairwise_reviews(completed_sheets, key, seed=544, bootstrap_samples=10000):
    """Unblind A/B/TIE choices and bootstrap prompts, not individual reviewer rows."""
    if len(completed_sheets) < 1:
        raise ValueError("at least one completed review sheet is required")
    rows = []
    for reviewer, sheet in enumerate(completed_sheets, start=1):
        frame = sheet.copy()
        frame["reviewer"] = reviewer
        rows.append(frame)
    reviews = pd.concat(rows, ignore_index=True)
    reviews["preferred_response"] = reviews["preferred_response"].astype(str).str.strip().str.upper()
    invalid = reviews.loc[~reviews["preferred_response"].isin(["A", "B", "TIE"]),
                          ["review_id", "preferred_response"]]
    if len(invalid):
        raise ValueError(
            "every preferred_response must be A, B, or TIE; invalid rows: "
            + invalid.head(5).to_dict("records").__repr__()
        )
    merged = reviews.merge(key, on=["review_id", "behavior"], validate="many_to_one")
    merged["candidate_score"] = np.where(
        merged["preferred_response"].eq("TIE"), 0.5,
        np.where(
            (merged["preferred_response"].eq("A") &
             merged["method_A"].eq(merged["candidate_method"])) |
            (merged["preferred_response"].eq("B") &
             merged["method_B"].eq(merged["candidate_method"])),
            1.0, 0.0,
        ),
    )
    per_prompt = merged.groupby(
        ["review_id", "behavior", "candidate_method", "baseline_method"], as_index=False
    )["candidate_score"].mean()
    rng = np.random.default_rng(seed)
    summaries = []
    for behavior, group in list(per_prompt.groupby("behavior")) + [("overall", per_prompt)]:
        values = group["candidate_score"].to_numpy(float)
        boot = np.empty(bootstrap_samples)
        for index in range(bootstrap_samples):
            boot[index] = rng.choice(values, size=len(values), replace=True).mean()
        summaries.append({
            "behavior": behavior,
            "candidate_method": group["candidate_method"].iloc[0],
            "baseline_method": group["baseline_method"].iloc[0],
            "prompts": int(len(values)),
            "reviewers": int(merged["reviewer"].nunique()),
            "candidate_score_pct": 100.0 * float(values.mean()),
            "ci_low_pct": 100.0 * float(np.quantile(boot, 0.025)),
            "ci_high_pct": 100.0 * float(np.quantile(boot, 0.975)),
        })
    return merged, pd.DataFrame(summaries)


def pairwise_reviewer_agreement(completed_sheets):
    """Exact A/B/TIE agreement for every reviewer pair on matched prompts."""
    choices = []
    for reviewer, sheet in enumerate(completed_sheets, start=1):
        choices.append(sheet.set_index("review_id")["preferred_response"].astype(str)
                       .str.strip().str.upper().rename(reviewer))
    table = pd.concat(choices, axis=1, join="inner")
    rows = []
    for first in table:
        for second in table:
            if first < second:
                rows.append({
                    "reviewer_1": first,
                    "reviewer_2": second,
                    "matched_prompts": int(len(table)),
                    "exact_agreement_pct": 100.0 * float((table[first] == table[second]).mean()),
                })
    return pd.DataFrame(rows)
