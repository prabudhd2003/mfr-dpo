"""Reference-cache integrity tests."""

import pandas as pd

import fake_deps
fake_deps.install()

import mfr_cache


def test_cache_round_trip_and_metadata(tmp_path):
    cache = pd.DataFrame({"id": ["a", "b"], "chosen_logp": [-1.0, -2.0], "rejected_logp": [-3.0, -4.0]})
    path = tmp_path / "reference.csv"
    mfr_cache.save_reference_cache(cache, path, {"data_version": "v2"})
    loaded = mfr_cache.load_reference_cache(path, {"data_version": "v2"})
    assert loaded.equals(cache)
    assert loaded.attrs["manifest"]["data_version"] == "v2"


def test_canary_sample_preserves_original_length_sorted_batch_boundaries():
    frame = pd.DataFrame({
        "id": [f"pair-{index}" for index in range(12)],
        "prompt_tokens": [12, 1, 11, 2, 10, 3, 9, 4, 8, 5, 7, 6],
        "chosen_tokens": 1,
        "rejected_tokens": 1,
    })
    sample = mfr_cache._aligned_batch_sample(frame, sample_size=6, batch_size=3, seed=7)
    global_order = sorted(range(len(frame)), key=lambda index: mfr_cache.pair_length(frame.iloc[index]))
    ranks = sorted(global_order.index(frame.index[frame["id"] == pair_id][0]) for pair_id in sample["id"])
    assert len(ranks) == 6
    assert all(ranks[start:start + 3] == list(range(ranks[start], ranks[start] + 3))
               and ranks[start] % 3 == 0 for start in range(0, len(ranks), 3))


def test_duplicate_cache_ids_are_rejected():
    one = pd.DataFrame({"id": ["a"], "chosen_logp": [-1.0], "rejected_logp": [-2.0]})
    try:
        mfr_cache.merge_reference_caches([one, one])
        raise AssertionError("expected duplicate error")
    except ValueError:
        pass


def test_live_cache_canary_accepts_small_numeric_difference():
    calls = []
    original = mfr_cache.score_pairs

    def fake_score(*args, reference_cache=None, **kwargs):
        calls.append(reference_cache is not None)
        value = 0.0002 if reference_cache is not None else 0.0
        return pd.DataFrame({"margin": [value, value]}, index=["a", "b"])

    mfr_cache.score_pairs = fake_score
    try:
        frame = pd.DataFrame({"id": ["a", "b"]})
        result = mfr_cache.verify_reference_cache(None, None, frame, pd.DataFrame(), sample_size=2)
        assert result["max_abs_margin_difference"] == 0.0002 and calls == [False, True]
    finally:
        mfr_cache.score_pairs = original


def test_live_cache_canary_rejects_large_difference():
    original = mfr_cache.score_pairs

    def fake_score(*args, reference_cache=None, **kwargs):
        value = 0.01 if reference_cache is not None else 0.0
        return pd.DataFrame({"margin": [value]}, index=["a"])

    mfr_cache.score_pairs = fake_score
    try:
        try:
            mfr_cache.verify_reference_cache(None, None, pd.DataFrame({"id": ["a"]}), pd.DataFrame())
            raise AssertionError("expected canary failure")
        except ValueError as error:
            assert "failed live canary" in str(error)
    finally:
        mfr_cache.score_pairs = original
