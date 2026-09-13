#!/usr/bin/env python3
"""Build validated, versioned preference splits from pinned source revisions."""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from transformers import AutoTokenizer

import mfr_data
from mfr_utils import load_protocol


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    protocol = load_protocol(ROOT / "configs" / "experiment_protocol.json")
    output = Path(args.output or ROOT / protocol["data_dir"])
    tokenizer = AutoTokenizer.from_pretrained(protocol["model_name"], revision=protocol["model_revision"])
    datasets = mfr_data.load_all()
    sizes = {"train": protocol["train_pairs"], "val": protocol["val_pairs"], "test": protocol["test_pairs"]}
    splits = mfr_data.make_splits(datasets, tokenizer, sizes, protocol["max_tokens"], seed=0)
    manifest = mfr_data.save_splits(
        splits, output,
        {"protocol_version": protocol["protocol_version"], "data_version": protocol["data_version"],
         "model_name": protocol["model_name"], "model_revision": protocol["model_revision"],
         "max_tokens": protocol["max_tokens"], "split_seed": 0},
    )
    print(json.dumps(manifest, indent=2))
    print(f"Validated data written to {output}")


if __name__ == "__main__":
    main()
