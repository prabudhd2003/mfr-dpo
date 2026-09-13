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
