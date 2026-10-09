"""Peak preference-anchor persistence tests; no model or GPU required."""

import numpy as np

import fake_deps
fake_deps.install()

import mfr_dpo


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
