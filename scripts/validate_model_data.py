#!/usr/bin/env python3
"""Length audit of the frozen pairs under another model's tokenizer before replication.

The data were frozen so every pair fits max_tokens under the Qwen tokenizer. Another tokenizer can
push a few pairs over; training and scoring then truncate them at max_tokens (mfr_dpo.make_batch),
identically for every method. Rule from the plan (§2, Llama port): if more than 2% of any split is
over the limit, stop (raise max_tokens for that model); otherwise record the truncated pairs and go on.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from transformers import AutoTokenizer

import mfr_data
from mfr_utils import load_protocol, pin_chat_template_date, save_json_atomic

MAX_TRUNCATED_SHARE = 0.02


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    protocol = load_protocol(args.protocol)
    tokenizer = pin_chat_template_date(AutoTokenizer.from_pretrained(
        protocol["model_name"], revision=protocol["model_revision"]
    ))
    splits = mfr_data.load_splits(ROOT / protocol["data_dir"])
    report, too_long = [], []
    for behavior, parts in splits.items():
        for split, frame in parts.items():
            counted = mfr_data.add_token_counts(frame, tokenizer)
            longest = counted["prompt_tokens"] + counted[["chosen_tokens", "rejected_tokens"]].max(axis=1)
            bad = counted[longest > protocol["max_tokens"]]
            too_long.extend(bad["id"].tolist())
            report.append({
                "behavior": behavior,
                "split": split,
                "rows": int(len(frame)),
                "maximum_tokens": int(longest.max()),
                "over_limit": int(len(bad)),
                "over_limit_pct": round(100.0 * len(bad) / max(1, len(frame)), 3),
            })
    result = {
        "model_name": protocol["model_name"],
        "model_revision": protocol["model_revision"],
        "max_tokens": protocol["max_tokens"],
        "max_truncated_share_per_split": MAX_TRUNCATED_SHARE,
        "valid": all(row["over_limit"] <= MAX_TRUNCATED_SHARE * row["rows"] for row in report),
        "truncated_pairs": len(too_long),
        "over_limit_ids": too_long,
        "splits": report,
    }
    save_json_atomic(result, args.output)
    print(json.dumps(result, indent=2))
    if not result["valid"]:
        worst = max(report, key=lambda row: row["over_limit_pct"])
        raise ValueError(
            f"{worst['behavior']}/{worst['split']}: {worst['over_limit_pct']}% of pairs exceed "
            f"{protocol['max_tokens']} tokens (limit {100 * MAX_TRUNCATED_SHARE:.0f}%); raise max_tokens "
            "for this model before running it")
    if too_long:
        print(f"Note: {len(too_long)} pairs exceed {protocol['max_tokens']} tokens and will be truncated "
              "(within the 2% per-split rule).", flush=True)


if __name__ == "__main__":
    main()
