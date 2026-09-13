"""Core methods share a budget; random_high deliberately gets one extra replay slot. CPU only: pytest -q

These tests run train_stage_replay with a fake model: torch and the DPO loss are replaced by
stand-ins (see tests/fake_deps.py), so we can check the bookkeeping -- which pairs go into which
step, how many steps, how many replay slots, how often the buffer is re-scored -- without a GPU.
"""

import functools

import numpy as np
import pandas as pd

import mfr_replay
from mfr_replay import ReplayBuffer


def split(dataset, n):
    return pd.DataFrame({
        "id": [f"{dataset}-train-{i:04d}" for i in range(n)],
        "prompt": [f"{dataset} prompt {i}" for i in range(n)],
        "chosen": [f"good {i}" for i in range(n)], "rejected": [f"bad {i}" for i in range(n)],
        "prompt_tokens": 50, "chosen_tokens": 100 + np.arange(n) % 7, "rejected_tokens": 90,
    })


class FakeLoss:
    def item(self): return 0.5
    def backward(self): pass
    def __mul__(self, weight): return self


class FakeModel:
    def parameters(self): return []
    def train(self): pass


@functools.lru_cache(maxsize=1)
def load_dpo():
    """mfr_dpo with fake torch/transformers/peft, and a fake loss + scoring function."""
    import fake_deps
    fake_deps.install()
    import mfr_dpo

    mfr_dpo.make_batch = lambda tokenizer, rows, max_tokens=1024: {"rows": rows}
    mfr_dpo.dpo_loss = lambda model, batch, beta, **kwargs: (FakeLoss(), 0.5)
    mfr_dpo.scored = []                       # how many pairs each buffer re-scoring saw

    def fake_score(model, tokenizer, df, **kwargs):
        """Pretend nothing changed since the last refresh: return the margins already stored."""
        mfr_dpo.scored.append(len(df))
        margin = df["current_margin"].values if "current_margin" in df else np.zeros(len(df))
        return pd.DataFrame({"margin": margin, "margin_sum": margin}, index=df["id"].values)

    mfr_dpo.score_pairs = fake_score
    return mfr_dpo


def buffer_with(dataset="safe", size=100, seed=0):
    buffer = ReplayBuffer(size=size, seed=seed)
    rows = buffer.candidates(dataset, split(dataset, 400))
    buffer.add_stage(dataset, rows, pd.Series(np.linspace(0.01, 0.2, len(rows)), index=rows["id"].values))
    return buffer


def run(method, n_new=160, buffer="default", **kwargs):
    dpo = load_dpo()
    return dpo.train_stage_replay(FakeModel(), None, split("helpful", n_new),
                                  buffer=buffer_with() if buffer == "default" else buffer,
                                  method=method, new_per_step=16, old_per_step=2, refreshes=5,
                                  seed=2, **kwargs)


# ------------------------------------------------------------------ same budget

def test_same_steps_for_every_method():
    for method in mfr_replay.METHODS:
        history, _ = run(method)
        assert len(history) == 10                                  # 160 new pairs / 16 per step


def test_frozen_default_is_exactly_ten_percent_replay():
    dpo = load_dpo()
    history, _ = dpo.train_stage_replay(FakeModel(), None, split("helpful", 180),
                                         buffer=buffer_with(), method="random", seed=2)
    assert len(history) == 10
    assert (history["n_new"] == 18).all() and (history["n_replay"] == 2).all()


def test_high_budget_random_uses_three_replay_pairs_per_full_step():
    dpo = load_dpo()
    history, log = dpo.train_stage_replay(
        FakeModel(), None, split("helpful", 180), buffer=buffer_with(), method="random_high",
        new_per_step=18, old_per_step=3, seed=2,
    )
    assert len(history) == 10
    assert (history["n_new"] == 18).all() and (history["n_replay"] == 3).all()
    assert len(log) == 30


def test_partial_final_batch_gets_proportional_replay():
    dpo = load_dpo()
    history, log = dpo.train_stage_replay(FakeModel(), None, split("helpful", 20),
                                           buffer=buffer_with(), method="random", seed=2)
    assert history["n_new"].tolist() == [18, 2]
    assert history["n_replay"].tolist() == [2, 0]
    assert len(log) == 2


def test_replay_slots_per_step():
    for method in ("random", "random_high", "lowest_margin", "mfr"):
        history, log = run(method)
        assert (history["n_replay"] == 2).all()                    # exactly 2 old pairs every step
        assert len(log) == 2 * len(history)
        assert set(log["dataset"]) == {"safe"}
    history, log = run("none")
    assert (history["n_replay"] == 0).all() and len(log) == 0


def test_no_replay_without_a_buffer():
    dpo = load_dpo()
    history, log = dpo.train_stage_replay(FakeModel(), None, split("helpful", 64), buffer=None,
                                          method="mfr", new_per_step=16, old_per_step=2, seed=2)
    assert (history["n_replay"] == 0).all() and len(log) == 0      # stage 1 behaves like "none"


# ------------------------------------------------------------------ re-scoring

def test_buffer_is_rescored_once_per_interval_except_the_first():
    dpo = load_dpo()
    dpo.scored.clear()
    run("mfr")                                                     # 10 steps, 5 intervals of 2 steps
    assert len(dpo.scored) == 4 and set(dpo.scored) == {100}       # 4 refreshes of the 100-pair buffer
    for method in ("random", "random_high", "none"):
        dpo.scored.clear()
        run(method)
        assert dpo.scored == []                                    # these never re-score


def test_scoring_time_is_recorded_separately():
    history, _ = run("mfr")
    assert {"minutes", "scoring_minutes"} <= set(history.columns)
    assert (history["scoring_minutes"] >= 0).all()


# ------------------------------------------------------------------ selection

def test_same_seed_gives_the_same_replayed_pairs():
    _, a = run("mfr")
    _, b = run("mfr")
    assert a["id"].tolist() == b["id"].tolist()


def test_methods_replay_different_pairs():
    buffer = buffer_with()
    forgotten = buffer.rows()["id"].tolist()[:5]
    current = buffer.rows().set_index("id")["peak_margin"].copy()
    current[forgotten] -= 1.0                                      # these five were forgotten
    buffer.set_current(current)

    _, mfr_log = run("mfr", buffer=buffer)
    assert set(mfr_log["id"]) <= set(forgotten)                    # MFR replays only forgotten pairs
    assert len(set(mfr_log["id"])) >= 4
    _, random_log = run("random", buffer=buffer_with())
    assert len(set(random_log["id"])) > 5                          # random spreads over the buffer
