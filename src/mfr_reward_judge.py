"""Reward-model pairwise judge for Helpful/Quality comparisons (plan §5, G5/G6).

Skywork-Reward-V2-Llama-3.1-8B scores each (prompt, response) on its own; the higher reward wins and
an exact tie is a tie. Because the two responses never share a prompt, there is no A/B position
effect, so `position_consistent` is always True. It replaced Prometheus-2-7B, which failed the
predeclared human-agreement check (56% < 70%, configs/evaluation_protocol.json "rejected_judges").

Input format follows the model card: the tokenizer's chat template over [user, assistant] with no
system prompt, the leading BOS string removed before tokenizing (the tokenizer adds it once).
"""

from __future__ import annotations

import numpy as np

from mfr_utils import pin_chat_template_date


class RewardJudge:
    def __init__(self, model_name, revision, max_length=16384):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        if not torch.cuda.is_available():
            raise RuntimeError("the reward judge requires a visible CUDA GPU")
        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, revision=revision)
        pin_chat_template_date(self.tokenizer)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name, revision=revision, dtype=torch.bfloat16, num_labels=1,
            attn_implementation="sdpa",
        ).to("cuda").eval()
        self.max_length = int(max_length)

    def score(self, prompts, responses, desc=None):
        """One scalar reward per (prompt, response); batch size 1, so no padding is involved."""
        from tqdm.auto import tqdm

        bos = self.tokenizer.bos_token or ""
        scores = []
        with self.torch.no_grad():
            for prompt, response in tqdm(list(zip(prompts, responses)), desc=desc,
                                         dynamic_ncols=True):
                text = self.tokenizer.apply_chat_template(
                    [{"role": "user", "content": str(prompt)},
                     {"role": "assistant", "content": str(response)}], tokenize=False)
                if bos and text.startswith(bos):
                    text = text[len(bos):]
                encoded = self.tokenizer(text, return_tensors="pt", truncation=True,
                                         max_length=self.max_length).to("cuda")
                scores.append(float(self.model(**encoded).logits[0][0].float()))
        return np.asarray(scores, dtype=float)


def compare(scores_a, scores_b):
    """'A' if A has the higher reward, 'B' if lower, 'TIE' if equal."""
    a = np.asarray(scores_a, dtype=float)
    b = np.asarray(scores_b, dtype=float)
    return np.where(a > b, "A", np.where(a < b, "B", "TIE")).tolist()


def mirror(choices):
    """The same verdicts written from the reversed (B, A) order."""
    return [{"A": "B", "B": "A"}.get(choice, choice) for choice in choices]


def uses_reward_judge(config):
    return config.get("type") == "reward_model"
