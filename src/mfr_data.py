"""Load, clean, split, validate, and version the preference datasets.

v2 has three behaviors (helpful, safe, quality). The 5-task extension (data/v3) keeps those nine
files byte-identical and adds summarize (Reddit TL;DR) and code (CodeUltraFeedback); see
scripts/prepare_data_v3.py and docs/EXTENSION.md.
"""

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


# Behaviors in their canonical order; data folders hold any subset of these.
KNOWN_BEHAVIORS = ("helpful", "safe", "quality", "summarize", "code")
SPLITS = ("train", "val", "test")
TLDR_REPO = "CarperAI/openai_summarize_comparisons"
CODE_REPO = "coseal/CodeUltraFeedback_binarized"
TLDR_INSTRUCTION = "Write a short TL;DR summary of the following Reddit post."


def _tldr_prompt(text):
    """Turn a TL;DR comparison prompt into a chat request: add an instruction, drop the trailing cue."""
    text = str(text).strip()
    if text.endswith("TL;DR:"):
        text = text[: -len("TL;DR:")].rstrip()
    return f"{TLDR_INSTRUCTION}\n\n{text}"


def _tldr_summary(text):
    text = str(text).strip()
    if text.startswith("TL;DR:"):
        text = text[len("TL;DR:"):].strip()
    return text


def _last_message(value):
    """A response stored either as plain text or as a chat list; return the final message text."""
    if isinstance(value, dict):
        return str(value.get("content", ""))
    if hasattr(value, "__len__") and not isinstance(value, str) and len(value):
        last = value[-1]
        return str(last.get("content", "")) if isinstance(last, dict) else str(last)
    return str(value)


def load_tldr(revision=None):
    """Reddit TL;DR summary comparisons (Stiennon et al., 2020): human-chosen vs rejected summary.

    All published splits are pooled; make_extra_splits then keeps one pair per post and draws
    prompt-disjoint train/validation/test sets.
    """
    frames = []
    for split in ("train", "valid1", "valid2", "test"):
        df = _load_dataset(TLDR_REPO, split=split, revision=revision).to_pandas()
        out = pd.DataFrame({
            "prompt": df["prompt"].map(_tldr_prompt),
            "chosen": df["chosen"].map(_tldr_summary),
            "rejected": df["rejected"].map(_tldr_summary),
        })
        frames.append(_with_provenance(out, df, TLDR_REPO, split))
    return clean(pd.concat(frames, ignore_index=True))


def load_codeultrafeedback(min_rating_gap=1.0, revision=None):
    """CodeUltraFeedback coding preferences (LLM-judge ratings); keep pairs with a clear rating gap."""
    split = "train"
    df = _load_dataset(CODE_REPO, split=split, revision=revision).to_pandas()
    gap = (pd.to_numeric(df["rating_chosen"], errors="coerce")
           - pd.to_numeric(df["rating_rejected"], errors="coerce"))
    df = df[gap >= min_rating_gap]
    out = pd.DataFrame({
        "prompt": df["instruction"],
        "chosen": df["chosen"].map(_last_message),
        "rejected": df["rejected"].map(_last_message),
    })
    return clean(_with_provenance(out, df, CODE_REPO, split))


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


def make_extra_splits(new_datasets, existing_splits, tokenizer, sizes, max_tokens=1024, seed=0):
    """Split new behaviors without touching existing ones.

    Prompts already used anywhere in `existing_splits` are excluded, each new prompt is kept once,
    pairs longer than max_tokens are dropped, and every new behavior gets exactly the requested
    train/validation/test sizes (or the function stops with the counts).
    """
    claimed = {
        normalize_prompt(prompt)
        for parts in existing_splits.values() for frame in parts.values() for prompt in frame["prompt"]
    }
    splits = {}
    for name in sorted(new_datasets, key=lambda item: len(new_datasets[item])):
        if name not in KNOWN_BEHAVIORS or name in existing_splits:
            raise ValueError(f"{name!r} must be a new behavior from {KNOWN_BEHAVIORS}")
        df = new_datasets[name].copy()
        start = len(df)
        df["_prompt_key"] = df["prompt"].map(normalize_prompt)
        df = df[~df["_prompt_key"].isin(claimed)]
        after_overlap = len(df)
        df = df.drop_duplicates("_prompt_key", keep="first")
        after_dedupe = len(df)
        claimed.update(df["_prompt_key"])
        df = add_token_counts(df.drop(columns="_prompt_key"), tokenizer)
        longest = df["prompt_tokens"] + df[["chosen_tokens", "rejected_tokens"]].max(axis=1)
        df = df[longest <= max_tokens].sample(frac=1, random_state=seed).reset_index(drop=True)
        print(
            f"{name:9s} {start:6d} pairs | -{start - after_overlap} prompts already used"
            f" | -{after_overlap - after_dedupe} extra pairs per prompt"
            f" | -{after_dedupe - len(df)} too long | {len(df)} left"
        )
        needed = sizes["train"] + sizes["val"] + sizes["test"]
        if len(df) < needed:
            raise ValueError(f"{name}: only {len(df)} usable pairs, need {needed}")
        test_end, val_end = sizes["test"], sizes["test"] + sizes["val"]
        parts = {"train": df[val_end:val_end + sizes["train"]], "val": df[test_end:val_end],
                 "test": df[:test_end]}
        for split, part in parts.items():
            part = part.reset_index(drop=True)
            part.insert(0, "id", [f"{name}-{split}-{i:04d}" for i in range(len(part))])
            parts[split] = part
        splits[name] = parts
    validate_splits({**existing_splits, **splits}, max_tokens=max_tokens)
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


def manifest_behaviors(manifest):
    """Behaviors declared by a manifest's file names, in canonical order."""
    declared = {name.rsplit("_", 1)[0] for name in manifest.get("files", {})}
    unknown = declared - set(KNOWN_BEHAVIORS)
    if unknown:
        raise ValueError(f"manifest declares unknown behaviors {sorted(unknown)}")
    return tuple(name for name in KNOWN_BEHAVIORS if name in declared)


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
        for dataset in manifest_behaviors(manifest)
        for split in SPLITS
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
    """Read splits, verifying file hashes when a manifest is present.

    The behaviors come from the manifest (v2: the original three; v3: five). Without a manifest the
    original three are read.
    """
    manifest_path = Path(folder) / "manifest.json"
    behaviors = ("helpful", "safe", "quality")
    if manifest_path.exists():
        manifest = (verify_split_manifest(folder) if verify_hashes
                    else json.loads(manifest_path.read_text(encoding="utf-8")))
        behaviors = manifest_behaviors(manifest)
    splits = {
        name: {
            split: pd.read_json(os.path.join(folder, f"{name}_{split}.jsonl"), lines=True, dtype=False)
            for split in SPLITS
        }
        for name in behaviors
    }
    if validate:
        validate_splits(splits)
    return splits


def joint_training_frame(splits, datasets=("helpful", "safe", "quality")):
    """Combine complete training splits for the offline joint-access reference.

    The returned rows are deliberately not shuffled here.  The training function performs one
    global, seed-controlled shuffle, which keeps data construction independent of the run seed.
    A behavior label is retained so the saved history can be audited later if needed.
    """
    frames = []
    for dataset in datasets:
        if dataset not in splits or "train" not in splits[dataset]:
            raise KeyError(f"joint training is missing {dataset}/train")
        frame = splits[dataset]["train"].copy()
        frame["training_behavior"] = dataset
        frames.append(frame)
    combined = pd.concat(frames, ignore_index=True)
    if combined.empty:
        raise ValueError("joint training data is empty")
    if combined["id"].duplicated().any():
        duplicates = combined.loc[combined["id"].duplicated(), "id"].astype(str).tolist()
        raise ValueError(f"joint training contains duplicate pair IDs: {duplicates[:3]}")
    return combined
