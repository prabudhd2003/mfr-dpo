#!/usr/bin/env python3
"""Verify that every frozen pair fits under another model's tokenizer before replication."""

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
            })
    result = {
        "model_name": protocol["model_name"],
        "model_revision": protocol["model_revision"],
        "max_tokens": protocol["max_tokens"],
        "valid": not too_long,
        "first_over_limit_ids": too_long[:20],
        "splits": report,
    }
    save_json_atomic(result, args.output)
    print(json.dumps(result, indent=2))
    if too_long:
        raise ValueError(
            f"{len(too_long)} frozen pairs exceed {protocol['max_tokens']} under the second tokenizer"
        )


if __name__ == "__main__":
    main()
