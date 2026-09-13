"""Load, clean, split, validate, and version the three preference datasets."""

from __future__ import annotations

import hashlib
import json
import os
import re
import unicodedata
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


def _load_dataset(*args, **kwargs):
    """Import lazily so split-validation tests do not require the download stack."""
    from datasets import load_dataset
    return load_dataset(*args, **kwargs)


SOURCE_REVISIONS = {
    "helpful": "990b2711a36180dd19d9c94b8627844866f8982a",
    "safe": "9421ffafec3fa40a1f1a7d567b4d525079477ecb",
    "quality": "3949bf5f8c17c394422ccfab0c31ea9c20bdeb85",
}


def normalize_prompt(text):
    """A conservative key used only for leakage detection and de-duplication."""
    text = unicodedata.normalize("NFKC", str(text)).casefold().strip()
    return re.sub(r"\s+", " ", text)


def _with_provenance(out, source_df, dataset, split):
    out = out.copy()
    out["source_dataset"] = dataset
    out["source_split"] = split
    out["source_row_id"] = source_df.index.astype(str)
    return out


def load_helpsteer2(min_strength=2, revision=SOURCE_REVISIONS["helpful"]):
    """Load HelpSteer2 preference pairs with clear preference strength."""
    split = "train"
    df = _load_dataset("nvidia/HelpSteer2", data_dir="preference", split=split, revision=revision).to_pandas()
    df = df[df["preference_strength"].abs() >= min_strength]
    df = df[~df["prompt"].str.contains("<extra_id_1>", regex=False, na=False)]
    first_is_better = df["preference_strength"] < 0
    out = pd.DataFrame({
        "prompt": df["prompt"],
        "chosen": df["response_1"].where(first_is_better, df["response_2"]),
        "rejected": df["response_2"].where(first_is_better, df["response_1"]),
    })
    return clean(_with_provenance(out, df, "nvidia/HelpSteer2", split))


def load_pku_saferlhf(revision=SOURCE_REVISIONS["safe"]):
    """Load pairs where exactly one PKU-SafeRLHF response is labelled safe."""
    split = "train"
    df = _load_dataset("PKU-Alignment/PKU-SafeRLHF", split=split, revision=revision).to_pandas()
    df = df[df["is_response_0_safe"] != df["is_response_1_safe"]]
    first_is_safe = df["is_response_0_safe"]
    out = pd.DataFrame({
        "prompt": df["prompt"],
        "chosen": df["response_0"].where(first_is_safe, df["response_1"]),
        "rejected": df["response_1"].where(first_is_safe, df["response_0"]),
    })
    return clean(_with_provenance(out, df, "PKU-Alignment/PKU-SafeRLHF", split))


def load_ultrafeedback(min_score_gap=1.0, revision=SOURCE_REVISIONS["quality"]):
    """Load UltraFeedback pairs whose chosen score exceeds rejected by the threshold."""
    split = "train_prefs"
    df = _load_dataset(
        "HuggingFaceH4/ultrafeedback_binarized", split=split, revision=revision
    ).to_pandas()
    df = df[df["score_chosen"] - df["score_rejected"] >= min_score_gap]
    out = pd.DataFrame({
        "prompt": df["prompt"],
        "chosen": df["chosen"].apply(lambda chat: chat[-1]["content"]),
        "rejected": df["rejected"].apply(lambda chat: chat[-1]["content"]),
    })
    return clean(_with_provenance(out, df, "HuggingFaceH4/ultrafeedback_binarized", split))


def clean(df):
    """Remove unusable pairs while preserving provenance columns."""
    required = ["prompt", "chosen", "rejected"]
    df = df.dropna(subset=required).copy()
    for column in required:
        df[column] = df[column].astype(str).str.strip()
    df = df[(df[required] != "").all(axis=1)]
    df = df[df["chosen"] != df["rejected"]]
    return df.drop_duplicates(required).reset_index(drop=True)


def load_all(revisions=None):
    revisions = {**SOURCE_REVISIONS, **(revisions or {})}
    return {
        "helpful": load_helpsteer2(revision=revisions["helpful"]),
        "safe": load_pku_saferlhf(revision=revisions["safe"]),
        "quality": load_ultrafeedback(revision=revisions["quality"]),
    }


def add_token_counts(df, tokenizer):
    """Count tokens exactly as they will be passed to the model."""
    def count(texts):
        return [len(ids) for ids in tokenizer(list(texts), add_special_tokens=False)["input_ids"]]

    prompts = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True
        )
        for prompt in df["prompt"]
    ]
    out = df.copy()
    out["prompt_tokens"] = count(prompts)
    out["chosen_tokens"] = [n + 1 for n in count(out["chosen"])]
    out["rejected_tokens"] = [n + 1 for n in count(out["rejected"])]
    return out


def make_splits(datasets, tokenizer, sizes, max_tokens=1024, seed=0):
    """Create prompt-disjoint train/validation/test splits across all datasets."""
    claimed = set()
    filtered = {}
    for name in sorted(datasets, key=lambda item: len(datasets[item])):
        df = datasets[name].copy()
        start = len(df)
        df["_prompt_key"] = df["prompt"].map(normalize_prompt)
        df = df[~df["_prompt_key"].isin(claimed)]
        after_overlap = len(df)
        df = df.drop_duplicates("_prompt_key", keep="first")
        after_dedupe = len(df)
        claimed.update(df["_prompt_key"])
        df = add_token_counts(df.drop(columns="_prompt_key"), tokenizer)
        longest = df["prompt_tokens"] + df[["chosen_tokens", "rejected_tokens"]].max(axis=1)
        df = df[longest <= max_tokens]
        filtered[name] = df.sample(frac=1, random_state=seed).reset_index(drop=True)
        print(
            f"{name:8s} {start:6d} pairs | -{start - after_overlap} shared/normalized prompts"
            f" | -{after_overlap - after_dedupe} extra pairs per prompt"
            f" | -{after_dedupe - len(df)} too long | {len(df)} left"
        )

    n_eval = sizes["test"] + sizes["val"]
    possible = [len(df) - n_eval for df in filtered.values()]
    n_train = min([sizes["train"], *possible])
    if n_train <= 0:
        raise ValueError(f"not enough examples after filtering for {n_eval} evaluation pairs")
    if n_train < sizes["train"]:
        print(f"Using {n_train} training pairs for every dataset (requested {sizes['train']}).")

    splits = {}
    for name in datasets:
        df = filtered[name]
        test_end, val_end = sizes["test"], n_eval
        parts = {
            "train": df[val_end:val_end + n_train],
            "val": df[test_end:val_end],
            "test": df[:test_end],
        }
        for split, part in parts.items():
            part = part.reset_index(drop=True)
            part.insert(0, "id", [f"{name}-{split}-{i:04d}" for i in range(len(part))])
            parts[split] = part
        splits[name] = parts
    validate_splits(splits, max_tokens=max_tokens)
    return splits


def validate_splits(splits, max_tokens=1024, expected_sizes=None):
    """Fail loudly on leakage, malformed rows, duplicate IDs, or length errors."""
    seen_prompts, seen_ids = {}, set()
    errors = []
    for dataset, parts in splits.items():
        for split, df in parts.items():
            required = {"id", "prompt", "chosen", "rejected"}
            missing = required - set(df.columns)
            if missing:
                errors.append(f"{dataset}/{split}: missing columns {sorted(missing)}")
                continue
            duplicate_ids = set(df.loc[df["id"].duplicated(), "id"])
            overlap_ids = set(df["id"]) & seen_ids
            if duplicate_ids or overlap_ids:
                errors.append(f"{dataset}/{split}: duplicate IDs {sorted(duplicate_ids | overlap_ids)[:3]}")
            seen_ids.update(df["id"])
            for prompt, pair_id in zip(df["prompt"], df["id"]):
                key = normalize_prompt(prompt)
                if key in seen_prompts:
                    errors.append(f"prompt leak: {seen_prompts[key]} and {pair_id}")
                else:
                    seen_prompts[key] = pair_id
            if {"prompt_tokens", "chosen_tokens", "rejected_tokens"} <= set(df.columns):
                longest = df["prompt_tokens"] + df[["chosen_tokens", "rejected_tokens"]].max(axis=1)
                if (longest > max_tokens).any():
                    errors.append(f"{dataset}/{split}: sequence longer than {max_tokens}")
            if expected_sizes and split in expected_sizes and len(df) != expected_sizes[split]:
                errors.append(f"{dataset}/{split}: expected {expected_sizes[split]}, found {len(df)}")
    if errors:
        raise ValueError("split validation failed:\n- " + "\n- ".join(errors[:20]))
    return True


def _sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def save_splits(splits, folder, manifest=None):
    """Save JSONL files plus a manifest containing sizes, hashes, and provenance."""
    validate_splits(splits)
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    files = {}
    for name, parts in splits.items():
        for split, df in parts.items():
            path = folder / f"{name}_{split}.jsonl"
            df.to_json(path, orient="records", lines=True, force_ascii=False)
            files[path.name] = {"rows": len(df), "sha256": _sha256(path)}
    payload = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_revisions": SOURCE_REVISIONS,
        "files": files,
        **(manifest or {}),
    }
    with open(folder / "manifest.json", "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
    return payload


def verify_split_manifest(folder):
    """Verify that every declared JSONL file still matches its saved hash and row count."""
    folder = Path(folder)
    manifest_path = folder / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"split manifest missing: {manifest_path}")
    with open(manifest_path, encoding="utf-8") as handle:
        manifest = json.load(handle)
    expected_names = {
        f"{dataset}_{split}.jsonl"
        for dataset in ("helpful", "safe", "quality")
        for split in ("train", "val", "test")
    }
    declared = set(manifest.get("files", {}))
    if declared != expected_names:
        raise ValueError(
            f"manifest file set differs: missing={sorted(expected_names - declared)}, "
            f"extra={sorted(declared - expected_names)}"
        )
    errors = []
    for name, expected in manifest["files"].items():
        path = folder / name
        if not path.exists():
            errors.append(f"missing {name}")
            continue
        actual_hash = _sha256(path)
        with open(path, encoding="utf-8") as handle:
            actual_rows = sum(1 for line in handle if line.strip())
        if actual_hash != expected.get("sha256"):
            errors.append(f"{name}: SHA-256 mismatch")
        if actual_rows != expected.get("rows"):
            errors.append(f"{name}: expected {expected.get('rows')} rows, found {actual_rows}")
    if errors:
        raise ValueError("data manifest verification failed:\n- " + "\n- ".join(errors))
    return manifest


def load_splits(folder, validate=True, verify_hashes=True):
    """Read splits, verifying v2 file hashes when a manifest is present."""
    if verify_hashes and (Path(folder) / "manifest.json").exists():
        verify_split_manifest(folder)
    splits = {
        name: {
            split: pd.read_json(os.path.join(folder, f"{name}_{split}.jsonl"), lines=True, dtype=False)
            for split in ("train", "val", "test")
        }
        for name in ("helpful", "safe", "quality")
    }
    if validate:
        validate_splits(splits)
    return splits
