#!/usr/bin/env python3
"""Unblind completed human-review sheets and report preference plus agreement."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd

from mfr_review import pairwise_reviewer_agreement, score_pairwise_reviews


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--review-sheets", nargs="+", required=True)
    parser.add_argument("--key", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, default=544)
    args = parser.parse_args()

    sheets = [pd.read_csv(path, keep_default_na=False) for path in args.review_sheets]
    key = pd.read_csv(args.key)
    merged, summary = score_pairwise_reviews(sheets, key, seed=args.seed)
    agreement = pairwise_reviewer_agreement(sheets)
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    merged.to_csv(output_dir / "unblinded_ratings.csv", index=False)
    summary.to_csv(output_dir / "human_summary.csv", index=False)
    agreement.to_csv(output_dir / "reviewer_agreement.csv", index=False)
    print("Human preference summary")
    print(summary.round(2).to_string(index=False))
    print("\nReviewer agreement")
    print(agreement.round(2).to_string(index=False) if len(agreement) else
          "Only one reviewer sheet was supplied; agreement is not measurable.")


if __name__ == "__main__":
    main()
