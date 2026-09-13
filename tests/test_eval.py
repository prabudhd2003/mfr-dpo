"""Final-test lock tests."""

import fake_deps
fake_deps.install()

import mfr_eval


def test_test_eval_requires_complete_marker(tmp_path):
    try:
        mfr_eval.assert_test_ready(tmp_path)
        raise AssertionError("expected lock")
    except RuntimeError:
        pass
    (tmp_path / "COMPLETE.json").write_text("{}")
    assert mfr_eval.assert_test_ready(tmp_path)
