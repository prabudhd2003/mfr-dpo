"""5-task / symmetric-anchor extension tests (docs/EXTENSION.md); no GPU required."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import fake_deps
fake_deps.install()

import mfr_data
import mfr_dpo
import mfr_replay
import mfr_stats
from mfr_replay import ReplayBuffer, plan_interval
from mfr_utils import buffer_seed, load_protocol, protocol_behaviors
from test_data import TinyTokenizer, frame

ROOT = Path(__file__).resolve().parents[1]


def _buffer():
    buffer = ReplayBuffer(size=40, seed=0)
    for dataset in ("helpful", "safe"):
        rows = pd.DataFrame({"id": [f"{dataset}-train-{i:04d}" for i in range(40)],
                             "prompt": ["p"] * 40, "chosen": ["c"] * 40, "rejected": ["r"] * 40})
        cand = buffer.candidates(dataset, rows)
        buffer.add_stage(dataset, cand, pd.Series(np.linspace(0.5, 1.0, len(cand)), index=cand["id"]))
    buffer.set_current(pd.Series(np.random.default_rng(1).normal(size=len(buffer)),
                                 index=buffer.rows()["id"]))
    return buffer


def test_new_methods_are_registered_as_anchor_methods():
    for method in ("der_pp", "der_pp_weak", "dapr_sym", "dapr_sym_weak", "dapr_lambda_0p003", "dapr_lambda_0p03"):
        assert method in mfr_replay.METHODS
        assert method in mfr_dpo.ANCHOR_METHODS
    assert set(mfr_dpo.SYMMETRIC_ANCHOR_METHODS) == {"der_pp", "der_pp_weak", "dapr_sym", "dapr_sym_weak"}


def test_der_pp_selects_exactly_like_random():
    buffer = _buffer()
    for method in ("der_pp", "der_pp_weak"):
        assert plan_interval(buffer, method, 12, np.random.default_rng(3)) == \
            plan_interval(buffer, "random", 12, np.random.default_rng(3))
        assert not mfr_replay.needs_refresh(method)


def test_symmetric_and_lambda_dapr_select_exactly_like_lowest_margin():
    buffer = _buffer()
    for method in ("dapr_sym", "dapr_sym_weak", "dapr_lambda_0p003", "dapr_lambda_0p03"):
        assert plan_interval(buffer, method, 12, np.random.default_rng(3)) == \
            plan_interval(buffer, "lowest_margin", 12, np.random.default_rng(3))
        assert mfr_replay.needs_refresh(method)


def test_protocol_anchor_strengths():
    protocol = load_protocol(ROOT / "configs" / "experiment_protocol.json")
    overrides = protocol["anchor_strength_overrides"]
    assert overrides["der_pp"] == overrides["dapr_sym"] == 0.1
    assert overrides["der_pp_weak"] == overrides["dapr_sym_weak"] == overrides["dapr_weak"] == 0.01
    assert overrides["dapr_lambda_0p003"] == 0.003 and overrides["dapr_lambda_0p03"] == 0.03


def test_five_task_protocol_orders_and_prefixes():
    protocol = load_protocol(ROOT / "configs" / "qwen7b_5task_protocol.json")
    assert protocol_behaviors(protocol) == ("helpful", "safe", "quality", "summarize", "code")
    main = load_protocol(ROOT / "configs" / "experiment_protocol.json")
    assert protocol["orders"]["1"][:3] == main["orders"]["1"]
    assert protocol["orders"]["2"][:3] == main["orders"]["4"]
    assert protocol["orders"]["3"][:2] == ["code", "summarize"]


def test_buffer_seeds_of_original_behaviors_are_unchanged():
    assert [buffer_seed(2, b) for b in ("helpful", "safe", "quality")] == [201, 202, 203]
    assert buffer_seed(2, "summarize") != buffer_seed(2, "code")


def test_tldr_and_code_formatting():
    prompt = mfr_data._tldr_prompt("SUBREDDIT: r/test\nTITLE: t\nPOST: hello\nTL;DR:")
    assert prompt.startswith(mfr_data.TLDR_INSTRUCTION) and not prompt.endswith("TL;DR:")
    assert mfr_data._tldr_summary(" TL;DR: short one") == "short one"
    assert mfr_data._last_message("plain") == "plain"
    chat = np.array([{"role": "user", "content": "q"}, {"role": "assistant", "content": "a"}], dtype=object)
    assert mfr_data._last_message(chat) == "a"


def test_make_extra_splits_is_disjoint_from_existing_data():
    existing = mfr_data.make_splits({"helpful": frame("h"), "safe": frame("s"), "quality": frame("q")},
                                    TinyTokenizer(), {"train": 2, "val": 1, "test": 1}, max_tokens=20)
    reused = frame("h")                                         # every prompt already used by helpful
    new = {"summarize": pd.concat([frame("t", 8), reused], ignore_index=True), "code": frame("c", 8)}
    extra = mfr_data.make_extra_splits(new, existing, TinyTokenizer(),
                                       {"train": 4, "val": 2, "test": 2}, max_tokens=20)
    assert {len(extra[b][s]) for b in extra for s in ("train", "val", "test")} == {4, 2}
    used = set(pd.concat([existing[b][s] for b in existing for s in existing[b]])["prompt"])
    assert not used & set(pd.concat([extra[b][s] for b in extra for s in extra[b]])["prompt"])
    assert extra["code"]["test"]["id"].iloc[0] == "code-test-0000"


def test_make_extra_splits_stops_when_too_few_pairs():
    existing = mfr_data.make_splits({"helpful": frame("h")}, TinyTokenizer(),
                                    {"train": 2, "val": 1, "test": 1}, max_tokens=20)
    with pytest.raises(ValueError, match="only 3 usable pairs"):
        mfr_data.make_extra_splits({"code": frame("c", 3)}, existing, TinyTokenizer(),
                                   {"train": 2, "val": 1, "test": 1}, max_tokens=20)


def test_load_splits_reads_the_behaviors_named_in_the_manifest(tmp_path):
    splits = mfr_data.make_splits({"helpful": frame("h"), "safe": frame("s"), "quality": frame("q")},
                                  TinyTokenizer(), {"train": 2, "val": 1, "test": 1}, max_tokens=20)
    extra = mfr_data.make_extra_splits({"code": frame("c", 6)}, splits, TinyTokenizer(),
                                       {"train": 2, "val": 1, "test": 1}, max_tokens=20)
    mfr_data.save_splits({**splits, **extra}, tmp_path)
    loaded = mfr_data.load_splits(tmp_path)
    assert list(loaded) == ["helpful", "safe", "quality", "code"]
    v2 = mfr_data.load_splits(ROOT / "data" / "v2")
    assert list(v2) == ["helpful", "safe", "quality"]


def _results(order, scores, run="r", method="m"):
    """scores[stage][behavior] -> accuracy; stage 0 is the base model."""
    rows = []
    for stage, by_behavior in scores.items():
        for behavior, value in by_behavior.items():
            rows.append({"run_name": run, "order_id": 1, "method": method, "seed": 0, "stage": stage,
                         "trained_on": "base" if stage == 0 else order[stage - 1],
                         "eval_set": behavior, "accuracy": value})
    return pd.DataFrame(rows)


def test_cell_metrics_five_tasks_and_stage_three_prefix():
    order = ["helpful", "safe", "quality", "summarize", "code"]
    base = {b: 50.0 for b in order}
    scores = {0: base}
    for stage in range(1, 6):
        scores[stage] = {b: 50.0 + 10 * stage - 2 * order.index(b) for b in order}
    cells = mfr_stats.cell_metrics(_results(order, scores))
    final = scores[5]
    expected_retention = np.mean([final[b] - scores[i + 1][b] for i, b in enumerate(order[:-1])])
    assert np.isclose(cells["retention"].iloc[0], expected_retention)
    assert np.isclose(cells["final_task"].iloc[0], final["code"])
    assert np.isclose(cells["final_average"].iloc[0], np.mean(list(final.values())))
    prefix = mfr_stats.cell_metrics(_results(order, scores), upto_stage=3)
    assert np.isclose(prefix["final_task"].iloc[0], scores[3]["quality"])
    assert np.isclose(prefix["final_average"].iloc[0], np.mean([scores[3][b] for b in order[:3]]))


def test_cell_metrics_three_task_result_is_unchanged():
    order = ["helpful", "safe", "quality"]
    scores = {0: {b: 50.0 for b in order}, 1: {"helpful": 70, "safe": 50, "quality": 55},
              2: {"helpful": 65, "safe": 80, "quality": 56}, 3: {"helpful": 64, "safe": 72, "quality": 75}}
    cells = mfr_stats.cell_metrics(_results(order, scores))
    assert np.isclose(cells["retention"].iloc[0], np.mean([64 - 70, 72 - 80]))
    assert np.isclose(cells["final_average"].iloc[0], np.mean([64, 72, 75]))


def test_select_variant_uses_the_dapr_development_rule():
    dev = pd.DataFrame([
        {"method": "random", "final_task": 74.0, "final_average": 71.0},
        {"method": "der_pp", "final_task": 70.0, "final_average": 73.5},     # fails tolerance
        {"method": "der_pp_weak", "final_task": 73.0, "final_average": 72.0},
    ])
    chosen, table = mfr_stats.select_variant(dev, ["der_pp", "der_pp_weak"])
    assert chosen == "der_pp_weak"
    assert not table.loc["der_pp", "passes_tolerance"]


def test_symmetric_anchor_penalizes_both_directions():
    """Runs in a clean interpreter with real torch (the other tests install a fake torch)."""
    import subprocess
    import sys
    if subprocess.run([sys.executable, "-c", "import torch"], capture_output=True).returncode:
        pytest.skip("needs real torch (run on CARC)")
    code = f"""
import sys, types
sys.path.insert(0, {str(ROOT / 'src')!r})
import numpy as np, torch
sys.modules['peft'] = types.SimpleNamespace(LoraConfig=object, PeftModel=object,
    get_peft_model=lambda *a, **k: None, prepare_model_for_kbit_training=lambda *a, **k: None)
sys.modules['transformers'] = types.SimpleNamespace(AutoModelForCausalLM=object, AutoTokenizer=object,
    BitsAndBytesConfig=object, get_linear_schedule_with_warmup=lambda *a, **k: None)
import mfr_dpo
batch = {{'n_pairs': 1, 'is_replay': [True], 'pair_ids': ['a'], 'anchor_gate_recorded': [True],
          'anchor_gate_active': [True]}}
seq_index = torch.tensor([0, 0, 1])
anchors = {{'a': {{'chosen': np.array([-1.0, -1.0]), 'rejected': np.array([-1.0])}}}}
up = torch.tensor([-0.5, -0.5, -1.0])      # chosen became more likely: harmless for DAPR
down = torch.tensor([-1.5, -1.5, -1.0])    # chosen became less likely: penalized by both
policy = torch.zeros(2)
d_up, _ = mfr_dpo._anchor_regularizer('dapr', policy, up, seq_index, batch, anchors)
s_up, _ = mfr_dpo._anchor_regularizer('dapr_sym', policy, up, seq_index, batch, anchors)
d_down, _ = mfr_dpo._anchor_regularizer('dapr', policy, down, seq_index, batch, anchors)
s_down, _ = mfr_dpo._anchor_regularizer('der_pp', policy, down, seq_index, batch, anchors)
assert float(d_up) == 0.0 and float(s_up) > 0.0
assert np.isclose(float(d_down), float(s_down)) and float(d_down) > 0.0
"""
    subprocess.run([sys.executable, "-c", code], check=True)


def test_extension_analysis_reports_e1_e2_with_holm():
    rng = np.random.default_rng(0)
    rows = []
    for seed in range(5):
        for order in (1, 2, 3):
            for method, shift in (("dapr_weak", 2.0), ("der_pp_weak", 0.0), ("dapr_sym_weak", 0.5)):
                rows.append({"run_name": f"{method}-{order}-{seed}", "order_id": order, "seed": seed,
                             "method": method, "final_average": 72 + shift + rng.normal(0, 0.1),
                             "retention": -3 + shift + rng.normal(0, 0.1), "final_task": 75.0})
    table = mfr_stats.extension_analysis(pd.DataFrame(rows), "der_pp_weak", "dapr_sym_weak")
    assert list(table["test"]) == ["E1", "E2"]
    assert table["significant"].all() and (table["estimate"] > 0).all()


# --- D3: replay-ratio and buffer-size sensitivity -------------------------------------------------
D3_RANDOM = ("random_old1", "random_old4", "random_buf250", "random_buf1000")
D3_DAPR = ("dapr_weak_old1", "dapr_weak_old4", "dapr_weak_buf250", "dapr_weak_buf1000")


def test_budget_variants_select_like_their_base_method():
    buffer = _buffer()
    for method in D3_RANDOM:
        assert method in mfr_replay.METHODS and method not in mfr_dpo.ANCHOR_METHODS
        assert plan_interval(buffer, method, 12, np.random.default_rng(3)) == \
            plan_interval(buffer, "random", 12, np.random.default_rng(3))
        assert not mfr_replay.needs_refresh(method)
    for method in D3_DAPR:
        assert method in mfr_replay.METHODS and method in mfr_dpo.ANCHOR_METHODS
        assert method not in mfr_dpo.SYMMETRIC_ANCHOR_METHODS
        assert plan_interval(buffer, method, 12, np.random.default_rng(3)) == \
            plan_interval(buffer, "lowest_margin", 12, np.random.default_rng(3))
        assert mfr_replay.needs_refresh(method)


def test_budget_variants_protocol_settings():
    from mfr_utils import method_anchor_strength, method_buffer_size, method_old_per_step
    protocol = load_protocol(ROOT / "configs" / "experiment_protocol.json")
    for prefix in ("random", "dapr_weak"):
        assert method_old_per_step(protocol, f"{prefix}_old1") == 1
        assert method_old_per_step(protocol, f"{prefix}_old4") == 4
        assert method_old_per_step(protocol, f"{prefix}_buf250") == 2
        assert method_buffer_size(protocol, f"{prefix}_buf250") == 250
        assert method_buffer_size(protocol, f"{prefix}_buf1000") == 1000
        assert method_buffer_size(protocol, f"{prefix}_old1") == 500
    assert method_buffer_size(protocol, "dapr_weak") == method_buffer_size(protocol, "random") == 500
    for method in D3_DAPR:
        assert method_anchor_strength(protocol, method) == method_anchor_strength(protocol, "dapr_weak")


def test_bad_buffer_size_override_is_rejected(tmp_path):
    protocol = json.loads((ROOT / "configs" / "experiment_protocol.json").read_text())
    protocol["buffer_size_overrides"] = {"random_buf250": 0}
    path = tmp_path / "protocol.json"
    path.write_text(json.dumps(protocol))
    with pytest.raises(ValueError, match="buffer-size override"):
        load_protocol(path)


def test_rebuilt_buffer_sizes_and_even_split():
    rows = pd.DataFrame({"id": [f"helpful-train-{i:04d}" for i in range(2000)],
                         "prompt": ["p"] * 2000, "chosen": ["c"] * 2000, "rejected": ["r"] * 2000})
    for size in (250, 1000):
        buffer = ReplayBuffer(size=size, seed=0)
        cand = buffer.candidates("helpful", rows)
        assert len(cand) == size
        buffer.add_stage("helpful", cand, pd.Series(0.5, index=cand["id"]))
        assert len(buffer) == size
    # the 250-pair candidates are the first 250 of the 500-pair ones (same seeded permutation)
    small = ReplayBuffer(size=250, seed=0).candidates("helpful", rows)["id"].tolist()
    base = ReplayBuffer(size=500, seed=0).candidates("helpful", rows)["id"].tolist()
    assert small == base[:250]


def test_stage_folder_prefers_rebuilt_stage1_buffer(tmp_path):
    import mfr_analysis
    run = tmp_path / "runs" / "v2_o1_random_buf250_s0"
    source = tmp_path / "runs" / "v2_o1_none_s0"
    (source / "stage1_helpful").mkdir(parents=True)
    run.mkdir(parents=True)
    order = ["helpful", "safe", "quality"]
    assert mfr_analysis.stage_folder(run, order, 1, str(source)).endswith("stage1_helpful")
    (run / "stage1_buffer").mkdir()
    assert mfr_analysis.stage_folder(run, order, 1, str(source)).endswith("stage1_buffer")
