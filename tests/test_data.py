"""Data normalization, split leakage, and schema tests."""

import pandas as pd

import mfr_data


class TinyTokenizer:
    eos_token = "</s>"

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        return "user: " + messages[0]["content"] + " assistant:"

    def __call__(self, texts, add_special_tokens=False):
        if isinstance(texts, str):
            texts = [texts]
        return {"input_ids": [text.split() for text in texts]}


def frame(prefix, n=6):
    return pd.DataFrame({"prompt": [f"{prefix} prompt {i}" for i in range(n)],
                         "chosen": [f"good {i}" for i in range(n)],
                         "rejected": [f"bad {i}" for i in range(n)]})


def test_normalize_prompt_handles_case_unicode_and_whitespace():
    assert mfr_data.normalize_prompt("  HELLO\u00a0World ") == mfr_data.normalize_prompt("hello world")


def test_make_splits_are_globally_prompt_disjoint():
    datasets = {"helpful": frame("help"), "safe": frame("safe"), "quality": frame("quality")}
    splits = mfr_data.make_splits(datasets, TinyTokenizer(), {"train": 2, "val": 1, "test": 1}, max_tokens=20)
    assert mfr_data.validate_splits(splits, max_tokens=20, expected_sizes={"train": 2, "val": 1, "test": 1})


def test_validate_splits_catches_normalized_leak():
    splits = {name: {part: pd.DataFrame({"id": [f"{name}-{part}-0"], "prompt": [f"{name} {part}"],
                                         "chosen": ["yes"], "rejected": ["no"]})
                     for part in ("train", "val", "test")}
              for name in ("helpful", "safe", "quality")}
    splits["safe"]["test"].loc[0, "prompt"] = " HELPFUL   TRAIN "
    try:
        mfr_data.validate_splits(splits)
        raise AssertionError("expected leakage error")
    except ValueError as error:
        assert "prompt leak" in str(error)


def test_manifest_detects_modified_jsonl(tmp_path):
    datasets = {"helpful": frame("help"), "safe": frame("safe"), "quality": frame("quality")}
    splits = mfr_data.make_splits(
        datasets, TinyTokenizer(), {"train": 2, "val": 1, "test": 1}, max_tokens=20
    )
    mfr_data.save_splits(splits, tmp_path)
    assert mfr_data.verify_split_manifest(tmp_path)["files"]
    with open(tmp_path / "safe_train.jsonl", "a", encoding="utf-8") as handle:
        handle.write("{}\n")
    try:
        mfr_data.verify_split_manifest(tmp_path)
        raise AssertionError("expected manifest mismatch")
    except ValueError as error:
        assert "SHA-256 mismatch" in str(error)
