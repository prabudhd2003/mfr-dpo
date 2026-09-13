"""Frozen protocol budget checks."""

from pathlib import Path

from mfr_utils import (load_protocol, method_old_per_step, validate_resume_settings,
                       validate_stage1_source)


def test_protocol_uses_exact_ten_percent_replay():
    root = Path(__file__).resolve().parents[1]
    protocol = load_protocol(root / "configs" / "experiment_protocol.json")
    assert protocol["new_per_step"] == 18 and protocol["old_per_step"] == 2
    assert protocol["methods"] == ["none", "random", "mfr"]
    assert "random_high" in protocol["secondary_methods"]
    assert method_old_per_step(protocol, "random") == 2
    assert method_old_per_step(protocol, "random_high") == 3
    assert protocol["lora_alpha"] == 32 and protocol["epochs"] == 1


def test_resume_allows_a_different_commit_when_scientific_code_is_identical():
    current = {"git_commit": "new", "scientific_code_sha256": "same", "run_name": "run"}
    saved = {"git_commit": "old", "scientific_code_sha256": "same", "run_name": "run"}
    assert validate_resume_settings(current, saved)


def test_resume_rejects_different_scientific_code():
    current = {"git_commit": "new", "scientific_code_sha256": "new-code", "run_name": "run"}
    saved = {"git_commit": "old", "scientific_code_sha256": "old-code", "run_name": "run"}
    try:
        validate_resume_settings(current, saved)
        raise AssertionError("expected incompatible resume")
    except ValueError as error:
        assert "scientific_code_sha256" in str(error)


def test_stage1_can_be_shared_across_replay_budgets_and_commits():
    common = {
        "protocol_version": "2.0", "data_version": "v2", "data_manifest_sha256": "data",
        "model_name": "model", "model_revision": "revision", "order_id": 1, "seed": 0,
        "lr": 1e-4, "beta": 0.1, "new_per_step": 18, "max_tokens": 1024,
        "lora_r": 16, "lora_alpha": 32, "lora_dropout": 0.05, "epochs": 1,
        "scientific_code_sha256": "same-code",
    }
    source = {**common, "method": "none", "old_per_step": 2, "git_commit": "old"}
    high_random = {**common, "method": "random_high", "old_per_step": 3, "git_commit": "new"}
    assert validate_stage1_source(high_random, source)
