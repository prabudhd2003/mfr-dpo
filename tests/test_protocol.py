"""Frozen protocol budget checks."""

from pathlib import Path

from mfr_utils import (load_protocol, method_anchor_strength, method_ewc_coefficient,
                       method_old_per_step,
                       validate_resume_settings,
                       stage1_compatibility_sha256, validate_stage1_source)


def test_protocol_uses_exact_ten_percent_replay():
    root = Path(__file__).resolve().parents[1]
    protocol = load_protocol(root / "configs" / "experiment_protocol.json")
    assert protocol["new_per_step"] == 18 and protocol["old_per_step"] == 2
    assert protocol["methods"] == ["none", "random", "mfr"]
    assert "random_high" in protocol["secondary_methods"]
    assert "fmcr" in protocol["secondary_methods"]
    assert "cpmr" in protocol["secondary_methods"]
    assert {
        "dapr", "dapr_weak", "dapr_gated", "dapr_c", "mir_dpo", "copr_adapted",
        "ewc_0_1", "ewc_1", "ewc_10",
    } <= set(protocol["secondary_methods"])
    assert protocol["orders"]["3"] == ["quality", "helpful", "safe"]
    assert protocol["orders"]["4"] == ["quality", "safe", "helpful"]
    assert method_old_per_step(protocol, "random") == 2
    assert method_old_per_step(protocol, "random_high") == 3
    assert method_anchor_strength(protocol, "dapr") == 0.1
    assert method_anchor_strength(protocol, "dapr_weak") == 0.01
    assert method_anchor_strength(protocol, "dapr_gated") == 0.1
    assert method_old_per_step(protocol, "ewc_0_1") == 0
    assert method_old_per_step(protocol, "ewc_1") == 0
    assert method_old_per_step(protocol, "ewc_10") == 0
    assert method_ewc_coefficient(protocol, "ewc_0_1") == 0.1
    assert method_ewc_coefficient(protocol, "ewc_1") == 1.0
    assert method_ewc_coefficient(protocol, "ewc_10") == 10.0
    assert protocol["ewc_fisher_pairs"] == 500
    assert protocol["ewc_fisher_batch_size"] == 1
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
        "protocol_version": "2.1", "data_version": "v2", "data_manifest_sha256": "data",
        "model_name": "model", "model_revision": "revision", "order_id": 1,
        "order": ["helpful", "safe", "quality"], "seed": 0,
        "lr": 1e-4, "beta": 0.1, "new_per_step": 18, "max_tokens": 1024,
        "buffer_size": 500, "micro_batch": 2,
        "lora_r": 16, "lora_alpha": 32, "lora_dropout": 0.05, "epochs": 1,
        "scientific_code_sha256": "same-code",
    }
    signature = stage1_compatibility_sha256(common)
    source = {**common, "method": "none", "old_per_step": 2, "git_commit": "old",
              "stage1_compatibility_sha256": signature}
    high_random = {**common, "method": "random_high", "old_per_step": 3, "git_commit": "new",
                   "stage1_compatibility_sha256": signature}
    assert validate_stage1_source(high_random, source)


def test_legacy_protocol_21_stage1_source_is_explicitly_supported():
    current = {
        "protocol_version": "2.1", "data_version": "v2", "data_manifest_sha256": "data",
        "model_name": "model", "model_revision": "revision", "order_id": 1,
        "order": ["helpful", "safe", "quality"], "seed": 0, "lr": 1e-4, "beta": 0.1,
        "new_per_step": 18, "max_tokens": 1024, "buffer_size": 500, "micro_batch": 2,
        "lora_r": 16, "lora_alpha": 32, "lora_dropout": 0.05, "epochs": 1,
    }
    current["stage1_compatibility_sha256"] = stage1_compatibility_sha256(current)
    source = {key: value for key, value in current.items()
              if key != "stage1_compatibility_sha256"}
    source.update({"method": "none", "scientific_code_sha256": "legacy", "git_commit": "abc"})
    assert validate_stage1_source(current, source)


def test_stage1_rejects_a_changed_training_input():
    current = {
        "data_version": "v2", "data_manifest_sha256": "data", "model_name": "model",
        "model_revision": "revision", "order_id": 1, "order": ["helpful", "safe", "quality"],
        "seed": 0, "lr": 1e-4, "beta": 0.1, "new_per_step": 18, "max_tokens": 1024,
        "buffer_size": 500, "micro_batch": 2, "lora_r": 16, "lora_alpha": 32,
        "lora_dropout": 0.05, "epochs": 1,
    }
    current["stage1_compatibility_sha256"] = stage1_compatibility_sha256(current)
    source = {**current, "lr": 2e-4}
    source["stage1_compatibility_sha256"] = stage1_compatibility_sha256(source)
    try:
        validate_stage1_source(current, source)
        raise AssertionError("expected incompatible stage 1")
    except ValueError as error:
        assert "lr" in str(error)
