"""Load our three preference datasets into the same simple format.

Every loader returns a pandas DataFrame with three columns:
    prompt    the user's message
    chosen    the better response
    rejected  the worse response
"""

import os

import pandas as pd
from datasets import load_dataset


def load_helpsteer2(min_strength=2):
    """Stage "helpful": nvidia/HelpSteer2 (preference subset).

    preference_strength goes from -3 to 3.
    Negative = response_1 is better, positive = response_2 is better.
    We keep only clear preferences (|strength| >= 2) and single-turn prompts.
    """
    df = load_dataset("nvidia/HelpSteer2", data_dir="preference", split="train").to_pandas()
    df = df[df["preference_strength"].abs() >= min_strength]
    df = df[~df["prompt"].str.contains("<extra_id_1>", regex=False)]  # this marks multi-turn chats

    first_is_better = df["preference_strength"] < 0
    out = pd.DataFrame({
        "prompt": df["prompt"],
        "chosen": df["response_1"].where(first_is_better, df["response_2"]),
        "rejected": df["response_2"].where(first_is_better, df["response_1"]),
    })
    return clean(out)


def load_pku_saferlhf():
    """Stage "safe": PKU-Alignment/PKU-SafeRLHF.

    We keep pairs where exactly one response is safe, and the safe one is "chosen".
    """
    df = load_dataset("PKU-Alignment/PKU-SafeRLHF", split="train").to_pandas()
    df = df[df["is_response_0_safe"] != df["is_response_1_safe"]]

    first_is_safe = df["is_response_0_safe"]
    out = pd.DataFrame({
        "prompt": df["prompt"],
        "chosen": df["response_0"].where(first_is_safe, df["response_1"]),
        "rejected": df["response_1"].where(first_is_safe, df["response_0"]),
    })
    return clean(out)


def load_ultrafeedback(min_score_gap=1.0):
    """Stage "quality": HuggingFaceH4/ultrafeedback_binarized.

    chosen/rejected are stored as chats: [user message, assistant message].
    We take the assistant message and keep pairs whose scores differ by at least min_score_gap.
    """
    df = load_dataset("HuggingFaceH4/ultrafeedback_binarized", split="train_prefs").to_pandas()
    df = df[df["score_chosen"] - df["score_rejected"] >= min_score_gap]

    out = pd.DataFrame({
        "prompt": df["prompt"],
        "chosen": df["chosen"].apply(lambda chat: chat[-1]["content"]),
        "rejected": df["rejected"].apply(lambda chat: chat[-1]["content"]),
    })
    return clean(out)


def clean(df):
    """Basic cleanup shared by all three datasets."""
    df = df.dropna()
    df = df.apply(lambda col: col.str.strip())
    df = df[(df["prompt"] != "") & (df["chosen"] != "") & (df["rejected"] != "")]
    df = df[df["chosen"] != df["rejected"]]
    df = df.drop_duplicates()
    return df.reset_index(drop=True)


def load_all():
    """All three datasets, keyed by stage name."""
    return {
        "helpful": load_helpsteer2(),
        "safe": load_pku_saferlhf(),
        "quality": load_ultrafeedback(),
    }


# ----------------------------------------------------------------------------
# Train / val / test splits
# ----------------------------------------------------------------------------

def add_token_counts(df, tokenizer):
    """Count tokens the way the model sees them (prompt includes the chat template)."""
    def count(texts):
        return [len(ids) for ids in tokenizer(list(texts), add_special_tokens=False)["input_ids"]]

    prompts = [
        tokenizer.apply_chat_template([{"role": "user", "content": p}], tokenize=False, add_generation_prompt=True)
        for p in df["prompt"]
    ]
    df = df.copy()
    df["prompt_tokens"] = count(prompts)
    df["chosen_tokens"] = [n + 1 for n in count(df["chosen"])]      # +1 for the end-of-turn token
    df["rejected_tokens"] = [n + 1 for n in count(df["rejected"])]
    return df


def make_splits(datasets, tokenizer, sizes, max_tokens=768, seed=0):
    """Split each dataset into train / val / test.

    datasets: {"helpful": df, "safe": df, "quality": df}
    sizes:    {"train": 2000, "val": 200, "test": 300}  (same for every dataset)

    Steps:
      1. drop prompts that appear in more than one dataset (so a test prompt is never trained on)
      2. keep one pair per prompt (so train / val / test never share a prompt)
      3. drop pairs longer than max_tokens
      4. shuffle with a fixed seed and cut into test, val, train
    """
    all_prompts = pd.concat([df["prompt"].drop_duplicates() for df in datasets.values()])
    counts = all_prompts.value_counts()
    shared = set(counts[counts > 1].index)

    splits = {}
    for name, df in datasets.items():
        df = df[~df["prompt"].isin(shared)]
        df = df.drop_duplicates("prompt")
        df = add_token_counts(df, tokenizer)
        longest = df["prompt_tokens"] + df[["chosen_tokens", "rejected_tokens"]].max(axis=1)
        df = df[longest <= max_tokens]
        df = df.sample(frac=1, random_state=seed).reset_index(drop=True)

        needed = sizes["test"] + sizes["val"] + sizes["train"]
        if len(df) < needed:
            print(f"WARNING {name}: only {len(df)} pairs left, need {needed}. Train will be smaller.")

        a, b = sizes["test"], sizes["test"] + sizes["val"]
        parts = {"train": df[b:b + sizes["train"]], "val": df[a:b], "test": df[:a]}
        for split, part in parts.items():
            part = part.reset_index(drop=True)
            part.insert(0, "id", [f"{name}-{split}-{i:04d}" for i in range(len(part))])
            parts[split] = part
        splits[name] = parts
        print(f"{name:8s} kept {len(df):6d} after filters -> "
              + ", ".join(f"{s} {len(p)}" for s, p in parts.items()))
    return splits


def save_splits(splits, folder):
    """Write data/<dataset>_<split>.jsonl, e.g. data/helpful_train.jsonl"""
    os.makedirs(folder, exist_ok=True)
    for name, parts in splits.items():
        for split, df in parts.items():
            df.to_json(f"{folder}/{name}_{split}.jsonl", orient="records", lines=True, force_ascii=False)


def load_splits(folder):
    """Read the saved splits back: splits["helpful"]["train"] is a DataFrame."""
    return {
        name: {split: pd.read_json(f"{folder}/{name}_{split}.jsonl", lines=True, dtype=False)
               for split in ("train", "val", "test")}
        for name in ("helpful", "safe", "quality")
    }
