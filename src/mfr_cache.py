"""Build and validate frozen-base log-probability caches for DPO."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import torch
from tqdm.auto import tqdm

from mfr_dpo import make_batch, pair_length, response_logprobs, score_pairs
from mfr_utils import file_sha256, save_json_atomic


@torch.no_grad()
def build_reference_cache(model, tokenizer, df, batch_size=4, max_tokens=1024, desc="base cache"):
    """Compute immutable base-model log probabilities for chosen and rejected responses."""
    rows = df.to_dict("records")
    order = sorted(range(len(rows)), key=lambda i: pair_length(rows[i]) if "prompt_tokens" in rows[i] else 0)
    chosen, rejected = [0.0] * len(rows), [0.0] * len(rows)
    was_training = model.training
    model.eval()
    with model.disable_adapter():
        for start in tqdm(
            range(0, len(rows), batch_size), desc=desc, unit="batch", leave=True,
            dynamic_ncols=True,
        ):
            indices = order[start:start + batch_size]
            sums, _ = response_logprobs(model, make_batch(tokenizer, [rows[i] for i in indices], max_tokens))
            n = len(indices)
            for offset, index in enumerate(indices):
                chosen[index] = sums[offset].item()
                rejected[index] = sums[n + offset].item()
    if was_training:
        model.train()
    return pd.DataFrame({"id": df["id"].values, "chosen_logp": chosen, "rejected_logp": rejected})


def save_reference_cache(cache, path, metadata):
    """Save cache CSV and a sidecar manifest used to reject stale caches."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    cache.to_csv(path, index=False)
    payload = {**metadata, "rows": len(cache), "cache_sha256": file_sha256(path)}
    save_json_atomic(payload, path.with_suffix(".manifest.json"))
    return payload


def load_reference_cache(path, expected=None):
    """Load a cache and verify its hash and requested protocol fields."""
    path = Path(path)
    manifest_path = path.with_suffix(".manifest.json")
    if not path.exists() or not manifest_path.exists():
        raise FileNotFoundError(f"cache or manifest missing for {path}")
    with open(manifest_path, encoding="utf-8") as handle:
        manifest = json.load(handle)
    if file_sha256(path) != manifest.get("cache_sha256"):
        raise ValueError(f"cache hash does not match manifest: {path}")
    mismatches = {key: (manifest.get(key), value) for key, value in (expected or {}).items()
                  if manifest.get(key) != value}
    if mismatches:
        raise ValueError(f"stale/incompatible reference cache: {mismatches}")
    cache = pd.read_csv(path)
    required = {"id", "chosen_logp", "rejected_logp"}
    if required - set(cache) or cache["id"].duplicated().any():
        raise ValueError(f"invalid reference cache schema: {path}")
    return cache


def merge_reference_caches(caches):
    """Combine per-split caches and reject duplicate pair IDs."""
    combined = pd.concat(caches, ignore_index=True)
    if combined["id"].duplicated().any():
        raise ValueError("reference caches contain duplicate IDs")
    return combined


def verify_reference_cache(model, tokenizer, df, cache, beta=0.1, max_tokens=1024,
                           sample_size=32, tolerance=1e-3, seed=544):
    """Canary-check cached margins against live adapter-disabled reference computation."""
    sample = df.sample(n=min(sample_size, len(df)), random_state=seed).reset_index(drop=True)
    live = score_pairs(
        model, tokenizer, sample, beta=beta, max_tokens=max_tokens,
        desc="cache canary: live reference", reference_cache=None,
    )
    cached = score_pairs(
        model, tokenizer, sample, beta=beta, max_tokens=max_tokens,
        desc="cache canary: cached reference", reference_cache=cache,
    )
    margin_difference = (live["margin"] - cached["margin"]).abs()
    diagnostics = {
        "sample_size": len(sample),
        "max_abs_margin_difference": float(margin_difference.max()),
        "mean_abs_margin_difference": float(margin_difference.mean()),
        "tolerance": float(tolerance),
    }
    if diagnostics["max_abs_margin_difference"] > tolerance:
        raise ValueError(
            "reference cache failed live canary: "
            f"max |margin difference|={diagnostics['max_abs_margin_difference']:.6g} "
            f"> tolerance={tolerance:.6g}"
        )
    return diagnostics
