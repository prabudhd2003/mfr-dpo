#!/usr/bin/env python3
"""Compute the frozen base model's log probabilities once for all v2 pairs."""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import mfr_cache
import mfr_data
import mfr_dpo
from mfr_utils import file_sha256, load_protocol, seed_everything


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, help="CSV path, preferably in the shared Drive")
    parser.add_argument("--batch-size", type=int, default=4)
    args = parser.parse_args()
    protocol = load_protocol(ROOT / "configs" / "experiment_protocol.json")
    data_dir = ROOT / protocol["data_dir"]
    splits = mfr_data.load_splits(data_dir)
    seed_everything(0)
    model, tokenizer = mfr_dpo.load_model(
        protocol["model_name"], protocol["lora_r"], revision=protocol["model_revision"],
        lora_alpha=protocol["lora_alpha"], lora_dropout=protocol["lora_dropout"],
    )
    frames = []
    for dataset, parts in splits.items():
        for split, frame in parts.items():
            frames.append(mfr_cache.build_reference_cache(
                model, tokenizer, frame, args.batch_size, protocol["max_tokens"], f"{dataset}/{split}"
            ))
    cache = mfr_cache.merge_reference_caches(frames)
    manifest_path = data_dir / "manifest.json"
    metadata = {
        "protocol_version": protocol["protocol_version"], "data_version": protocol["data_version"],
        "model_name": protocol["model_name"], "model_revision": protocol["model_revision"],
        "data_manifest_sha256": file_sha256(manifest_path), "max_tokens": protocol["max_tokens"],
    }
    saved = mfr_cache.save_reference_cache(cache, args.output, metadata)
    print(json.dumps(saved, indent=2))


if __name__ == "__main__":
    main()
