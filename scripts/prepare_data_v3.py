#!/usr/bin/env python3
"""Build data/v3 for the 5-task extension (docs/EXTENSION.md).

The nine v2 files (helpful, safe, quality) are copied byte-identical, so the first three stages of a
5-task run see exactly the main study's data. Two behaviors are added: summarize (Reddit TL;DR
comparisons) and code (CodeUltraFeedback). They are split prompt-disjoint from everything in v2 and
from each other, one pair per prompt, and length-checked under the protocol's tokenizer.

    # step 1: print the revisions to pin in configs/qwen7b_5task_protocol.json
    python scripts/prepare_data_v3.py --print-revisions
    # step 2: build into a folder outside the repo, then copy it to data/v3 and commit
    python scripts/prepare_data_v3.py --output ~/data_v3_build
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import mfr_data
from mfr_utils import file_sha256, load_protocol, pin_chat_template_date

PLACEHOLDER = "PIN_ME"


def print_revisions(protocol):
    from huggingface_hub import HfApi

    api = HfApi()
    revisions = {"model_revision": api.model_info(protocol["model_name"]).sha}
    for name, source in protocol["data_sources"].items():
        revisions[name] = api.dataset_info(source["repo"]).sha
    print(json.dumps(revisions, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", default=str(ROOT / "configs" / "qwen7b_5task_protocol.json"))
    parser.add_argument("--source", default=str(ROOT / "data" / "v2"), help="frozen v2 data folder")
    parser.add_argument("--output", help="new, empty folder for the v3 files")
    parser.add_argument("--print-revisions", action="store_true")
    args = parser.parse_args()
    protocol = load_protocol(args.protocol)
    if args.print_revisions:
        print_revisions(protocol)
        return
    if not args.output:
        parser.error("--output is required")
    sources = protocol["data_sources"]
    pins = [protocol["model_revision"], *(source["revision"] for source in sources.values())]
    if PLACEHOLDER in pins:
        raise SystemExit("Pin the revisions first: run with --print-revisions and put the values "
                         "into the protocol (model_revision and data_sources.*.revision).")
    source = Path(args.source).expanduser().resolve()
    output = Path(args.output).expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        raise SystemExit(f"{output} is not empty; choose a new folder")

    old = mfr_data.load_splits(source)                       # verifies the v2 hashes
    old_manifest = json.loads((source / "manifest.json").read_text(encoding="utf-8"))
    from transformers import AutoTokenizer
    tokenizer = pin_chat_template_date(AutoTokenizer.from_pretrained(
        protocol["model_name"], revision=protocol["model_revision"]))

    over_limit = []
    for behavior, parts in old.items():
        for split, frame in parts.items():
            counted = mfr_data.add_token_counts(frame, tokenizer)
            longest = counted["prompt_tokens"] + counted[["chosen_tokens", "rejected_tokens"]].max(axis=1)
            over_limit += counted.loc[longest > protocol["max_tokens"], "id"].tolist()
    if over_limit:
        raise ValueError(f"{len(over_limit)} v2 pairs exceed {protocol['max_tokens']} tokens under "
                         f"{protocol['model_name']}: {over_limit[:5]}")
    print(f"All v2 pairs fit in {protocol['max_tokens']} tokens under {protocol['model_name']}.")

    new = {
        "summarize": mfr_data.load_tldr(revision=sources["summarize"]["revision"]),
        "code": mfr_data.load_codeultrafeedback(
            min_rating_gap=float(sources["code"].get("min_rating_gap", 1.0)),
            revision=sources["code"]["revision"]),
    }
    sizes = {"train": protocol["train_pairs"], "val": protocol["val_pairs"], "test": protocol["test_pairs"]}
    new_splits = mfr_data.make_extra_splits(new, old, tokenizer, sizes, protocol["max_tokens"], seed=0)

    output.mkdir(parents=True, exist_ok=True)
    files = {}
    for name, entry in old_manifest["files"].items():
        shutil.copy2(source / name, output / name)
        files[name] = entry
    for behavior, parts in new_splits.items():
        for split, frame in parts.items():
            path = output / f"{behavior}_{split}.jsonl"
            frame.to_json(path, orient="records", lines=True, force_ascii=False)
            files[path.name] = {"rows": len(frame), "sha256": file_sha256(path)}
    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "data_version": protocol["data_version"],
        "protocol_version": protocol["protocol_version"],
        "model_name": protocol["model_name"],
        "model_revision": protocol["model_revision"],
        "max_tokens": protocol["max_tokens"],
        "split_seed": 0,
        "files": files,
        "source_revisions": {
            **old_manifest.get("source_revisions", {}),
            "summarize": sources["summarize"]["revision"],
            "code": sources["code"]["revision"],
        },
        "sources": {"summarize": sources["summarize"]["repo"], "code": sources["code"]["repo"]},
        "code_min_rating_gap": float(sources["code"].get("min_rating_gap", 1.0)),
        "derived_from": {"data_version": old_manifest.get("data_version", "v2"),
                         "manifest_sha256": file_sha256(source / "manifest.json")},
    }
    with open(output / "manifest.json", "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
    mfr_data.load_splits(output)                              # read back and verify everything

    for behavior, parts in new_splits.items():
        row = parts["train"].iloc[0]
        print(f"\n--- {behavior} example ({row['id']}) ---")
        print("PROMPT:", str(row["prompt"])[:600])
        print("CHOSEN:", str(row["chosen"])[:300])
        print("REJECTED:", str(row["rejected"])[:300])
    print(f"\nWrote {len(files)} files to {output}. Copy the folder to data/v3 in the repo and commit it.")


if __name__ == "__main__":
    main()
