"""Load our three preference datasets into the same simple format.

Every loader returns a pandas DataFrame with three columns:
    prompt    the user's message
    chosen    the better response
    rejected  the worse response
"""

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
