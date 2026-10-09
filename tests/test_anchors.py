"""Peak preference-anchor persistence tests; no model or GPU required."""

import types

import numpy as np

import fake_deps
fake_deps.install()

import mfr_dpo


def test_anchor_method_registry_includes_both_new_variants():
    assert {"dapr_weak", "dapr_gated"} <= set(mfr_dpo.ANCHOR_METHODS)


def test_live_gate_uses_strict_reference_relative_margin_regression():
    rows = [
        {"id": "a", "peak_margin": 0.20},
        {"id": "b", "peak_margin": 0.10},
        {"id": "c", "peak_margin": -0.10},
    ]
    marked = mfr_dpo.attach_anchor_gate_decisions(
        rows, {"a": 0.19, "b": 0.10, "c": -0.05}
    )
    assert [row["_anchor_gate_active"] for row in marked] == [True, False, False]
    assert np.isclose(marked[0]["_gate_margin_drop"], 0.01)
    assert marked[1]["_gate_margin_drop"] == 0.0


def test_preference_anchor_round_trip_without_pickle(tmp_path):
    anchors = {
        "safe-train-0001": {
            "chosen": np.array([-1.0, -2.0], dtype=np.float16),
            "rejected": np.array([-3.0], dtype=np.float16),
            "chosen_sum": -3.0,
            "rejected_sum": -3.0,
        },
        "helpful-train-0002": {
            "chosen": np.array([-0.5], dtype=np.float16),
            "rejected": np.array([-1.5, -2.5], dtype=np.float16),
            "chosen_sum": -0.5,
            "rejected_sum": -4.0,
        },
    }
    path = tmp_path / "preference_anchors.npz"
    mfr_dpo.save_preference_anchors(anchors, path)
    loaded = mfr_dpo.load_preference_anchors(path)
    assert set(loaded) == set(anchors)
    for pair_id in anchors:
        np.testing.assert_array_equal(loaded[pair_id]["chosen"], anchors[pair_id]["chosen"])
        np.testing.assert_array_equal(loaded[pair_id]["rejected"], anchors[pair_id]["rejected"])
        assert loaded[pair_id]["chosen_sum"] == anchors[pair_id]["chosen_sum"]


class _StepLoss:
    def item(self): return 0.5
    def backward(self): pass
    def __mul__(self, other): return self


def _fake_anchor_loss(model, batch, beta, **kwargs):
    """Mimic _anchor_regularizer's per-micro-batch diagnostics from row-level test fields."""
    replay = [row for row in batch["rows"] if row.get("_is_replay")]
    anchored = [row for row in replay if row.get("_anchor_gate_active", True)]
    rate = float(np.mean([row["_rate"] for row in anchored])) if anchored else float("nan")
    return _StepLoss(), 0.5, {
        "dpo_loss": 0.5, "anchor_loss": len(anchored) / len(replay) if replay else 0.0,
        "chosen_violation_rate": rate, "rejected_violation_rate": rate,
        "huber_cap_rate": rate, "anchor_common_shift": 0.0, "anchored_pairs": len(anchored),
    }


def _one_step(monkeypatch, method, replay_rows):
    monkeypatch.setattr(mfr_dpo, "make_batch",
                        lambda tokenizer, rows, max_tokens=1024: {"rows": rows})
    monkeypatch.setattr(mfr_dpo, "dpo_loss", _fake_anchor_loss)
    new = [{"id": "new-0", "_is_replay": False}, {"id": "new-1", "_is_replay": False}]
    pairs = [new[0], replay_rows[0], new[1], replay_rows[1]]   # one replay pair per micro-batch
    optimizer = types.SimpleNamespace(zero_grad=lambda: None, step=lambda: None)
    _, _, diagnostics = mfr_dpo._train_microbatches(
        None, None, pairs, optimizer, [], 0.1, 2, 1024, method=method, anchors={},
    )
    return diagnostics


def test_gated_violation_rates_describe_anchored_pairs_only(monkeypatch):
    diagnostics = _one_step(monkeypatch, "dapr_gated", [
        {"id": "old-0", "_is_replay": True, "_anchor_gate_active": True, "_rate": 0.4},
        {"id": "old-1", "_is_replay": True, "_anchor_gate_active": False, "_rate": 0.9},
    ])
    assert np.isclose(diagnostics["chosen_violation_rate"], 0.4)   # not diluted to 0.2
    assert np.isclose(diagnostics["huber_cap_rate"], 0.4)
    assert diagnostics["anchored_pairs"] == 1
    assert diagnostics["anchor_gate_rate"] == 0.5
    assert np.isclose(diagnostics["anchor_loss"], 0.5)              # loss still averages the zero


def test_fully_gated_off_step_records_nan_rates(monkeypatch):
    diagnostics = _one_step(monkeypatch, "dapr_gated", [
        {"id": "old-0", "_is_replay": True, "_anchor_gate_active": False, "_rate": 0.4},
        {"id": "old-1", "_is_replay": True, "_anchor_gate_active": False, "_rate": 0.9},
    ])
    assert np.isnan(diagnostics["chosen_violation_rate"])
    assert diagnostics["anchored_pairs"] == 0
    assert diagnostics["anchor_loss"] == 0.0


def test_ungated_rates_match_the_previous_replay_share_average(monkeypatch):
    diagnostics = _one_step(monkeypatch, "dapr", [
        {"id": "old-0", "_is_replay": True, "_rate": 0.2},
        {"id": "old-1", "_is_replay": True, "_rate": 0.6},
    ])
    assert np.isclose(diagnostics["chosen_violation_rate"], 0.4)   # 0.2 * 0.5 + 0.6 * 0.5
    assert diagnostics["anchored_pairs"] == 2
