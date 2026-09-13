"""Frozen protocol budget checks."""

from pathlib import Path

from mfr_utils import load_protocol, validate_resume_settings


def test_protocol_uses_exact_ten_percent_replay():
    root = Path(__file__).resolve().parents[1]
    protocol = load_protocol(root / "configs" / "experiment_protocol.json")
    assert protocol["new_per_step"] == 18 and protocol["old_per_step"] == 2
    assert protocol["methods"] == ["none", "random", "mfr"]
    assert protocol["lora_alpha"] == 32 and protocol["epochs"] == 1


def test_resume_rejects_a_different_commit():
    current = {"git_commit": "new", "run_name": "run"}
    saved = {"git_commit": "old", "run_name": "run"}
    try:
        validate_resume_settings(current, saved)
        raise AssertionError("expected incompatible resume")
    except ValueError as error:
        assert "git_commit" in str(error)
