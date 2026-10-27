"""PID state, unchanged controls, budget, orchestration, and checkpoint tests.

Run with pytest, or independently with:
PYTHONPATH=src:tests python -m unittest discover -s tests -p test_pid.py -v
"""

import contextlib
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

import mfr_pid
from mfr_replay import METHODS, ReplayBuffer, needs_refresh, plan_interval_details
from mfr_utils import load_protocol, method_old_per_step, validate_resume_settings

ROOT = Path(__file__).resolve().parents[1]


def make_buffer():
    b = ReplayBuffer(size=100, seed=7)
    for dataset in ("helpful", "safe"):
        rows = pd.DataFrame({"id": [f"{dataset}-{i:03d}" for i in range(130)]})
        candidates = b.candidates(dataset, rows)
        b.add_stage(dataset, candidates,
                    pd.Series(np.linspace(-0.2, 0.8, len(candidates)), index=candidates["id"]))
    rows = b.rows()
    current = rows["peak_margin"].to_numpy() - np.tile([0, .3, -.1, .3, .05], 20)
    b.set_current(pd.Series(current, index=rows["id"]))
    return b


def load_script(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class PIDTests(unittest.TestCase):
    def test_registered_profiles_keep_the_budget_and_refreshes(self):
        protocol = load_protocol(ROOT / "configs" / "experiment_protocol.json")
        self.assertEqual(len(mfr_pid.PROFILES), 16)
        for method in mfr_pid.PROFILES:
            self.assertIn(method, METHODS)
            self.assertIn(method, protocol["secondary_methods"])
            self.assertTrue(needs_refresh(method))
            self.assertEqual(method_old_per_step(protocol, method), 2)

    def test_existing_group_and_submit_validators_accept_pid(self):
        group = load_script("run_experiment_group")
        submit = load_script("submit_carc")
        protocol = group.load_protocol()
        method = "pid_i000_d025"
        self.assertEqual(group.requested_methods(protocol, method), [method])
        class Args:
            order, seed, methods = 1, 0, method
        self.assertEqual(submit.validate_group(Args(), protocol), [method])

    def test_signed_error_integral_and_recovery_derivative(self):
        rows = pd.DataFrame({"id": ["A"], "peak_margin": [.8], "current_margin": [.8]})
        rows = mfr_pid.update_state(rows, pd.Series({"A": .5}), 1)
        self.assertAlmostEqual(rows.loc[0, "pid_error"], .3)
        self.assertAlmostEqual(rows.loc[0, "pid_integral"], .3)
        self.assertAlmostEqual(rows.loc[0, "pid_derivative"], .3)
        rows = mfr_pid.update_state(rows, pd.Series({"A": .65}), 1)
        self.assertAlmostEqual(rows.loc[0, "pid_integral"], .45)
        self.assertAlmostEqual(rows.loc[0, "pid_derivative"], -.15)
        score = mfr_pid.components(rows, gains={"kp": 1, "ki": .1, "kd": .5})
        self.assertAlmostEqual(score.loc[0, "selection_score"], .12)

    def test_stage_boundary_uses_elapsed_fraction_of_window(self):
        rows = pd.DataFrame({"id": ["A"], "peak_margin": [.8], "current_margin": [.5]})
        rows = mfr_pid.update_state(rows, pd.Series({"A": .6}), 20 / 23)
        self.assertAlmostEqual(rows.loc[0, "pid_integral"], .2 * 20 / 23)
        self.assertAlmostEqual(rows.loc[0, "pid_derivative"], -.1 / (20 / 23))

    def test_repeated_planning_does_not_integrate_twice(self):
        b = make_buffer()
        b.set_current(pd.Series(.1, index=b.rows()["id"]), pid_dt=1)
        before = b.rows().copy()
        for _ in range(2):
            plan_interval_details(b, "pid_i010_d050", 46, np.random.default_rng(2), .75)
        pd.testing.assert_frame_equal(b.rows(), before)

    def test_unobserved_rows_do_not_accumulate_time(self):
        rows = pd.DataFrame({"id": ["A", "B"], "peak_margin": [.8, .6], "current_margin": [.8, .6]})
        rows = mfr_pid.update_state(rows, pd.Series({"A": .5}), 1)
        self.assertEqual(rows.loc[1, "current_margin"], .6)
        self.assertEqual(rows.loc[1, "pid_integral"], 0)
        self.assertEqual(rows.loc[1, "pid_derivative"], 0)

    def test_new_topic_starts_at_zero_and_old_history_survives_rebalance(self):
        b = ReplayBuffer(4)
        old = pd.DataFrame({"id": [f"H{i}" for i in range(4)]})
        b.add_stage("helpful", old, pd.Series(.8, index=old["id"]))
        b.set_current(pd.Series(.5, index=old["id"]), pid_dt=1)
        new = pd.DataFrame({"id": [f"S{i}" for i in range(4)]})
        b.add_stage("safe", new, pd.Series(.6, index=new["id"]))
        rows = b.rows()
        self.assertEqual(rows["dataset"].value_counts().to_dict(), {"helpful": 2, "safe": 2})
        np.testing.assert_allclose(rows.loc[rows["dataset"] == "helpful", "pid_integral"], .3)
        np.testing.assert_allclose(rows.loc[rows["dataset"] == "safe", list(mfr_pid.STATE_COLUMNS)], 0)

    def test_csv_restores_the_same_next_plan_and_next_state(self):
        b = make_buffer()
        b.set_current(pd.Series(.15, index=b.rows()["id"]), pid_dt=1)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "buffer.csv"
            b.to_csv(path)
            again = ReplayBuffer.from_csv(path, size=100, seed=7)
        for buffer in (b, again):
            buffer.set_current(pd.Series(.2, index=buffer.rows()["id"]), pid_dt=20 / 23)
        pd.testing.assert_frame_equal(b.rows(), again.rows(), check_exact=False)
        a = plan_interval_details(b, "pid_i010_d050", 46, np.random.default_rng(12), .75)
        c = plan_interval_details(again, "pid_i010_d050", 46, np.random.default_rng(12), .75)
        pd.testing.assert_frame_equal(a, c, check_exact=False)

    def test_legacy_buffer_without_pid_columns_is_supported(self):
        b = make_buffer()
        b._rows = b._rows.drop(columns=list(mfr_pid.STATE_COLUMNS))
        plan = plan_interval_details(b, "pid_i010_d050", 46, np.random.default_rng(12), .75)
        self.assertEqual(len(plan), 46)
        self.assertFalse(plan[list(mfr_pid.LOG_COLUMNS)].isna().any().any())

    def test_p_only_is_exactly_mfr_including_ties_and_dataset_cap(self):
        b = make_buffer()
        for cap in (None, .75):
            mfr = plan_interval_details(b, "mfr", 46, np.random.default_rng(12), cap)
            pid = plan_interval_details(b, "pid_i000_d000", 46, np.random.default_rng(12), cap)
            pd.testing.assert_frame_equal(mfr.drop(columns="method"), pid[mfr.columns].drop(columns="method"))

    def test_existing_controls_match_frozen_uploaded_snapshot_plans(self):
        plans = json.loads((ROOT / "tests/fixtures/replay_snapshot_plans.json").read_text())
        b = make_buffer()
        for method in ("none", "random", "random_high", "lowest_margin", "mfr"):
            for cap in (None, .75):
                current = plan_interval_details(b, method, 46, np.random.default_rng(12), cap)
                expected = plans[f"{method}:{cap}"]
                if not expected:
                    self.assertTrue(current.empty)
                else:
                    pd.testing.assert_frame_equal(current, pd.DataFrame(expected), check_dtype=False,
                                                  check_exact=False, atol=1e-14, rtol=0)

    def test_invalid_observation_interval_is_rejected(self):
        for dt in (0, -1, float("nan")):
            with self.assertRaises(ValueError):
                mfr_pid.update_state(make_buffer().rows(), pd.Series({"helpful-000": .1}), dt)

    def test_resume_rejects_changed_gains(self):
        with self.assertRaisesRegex(ValueError, "pid_profile"):
            validate_resume_settings({"pid_profile": {"kp": 1, "ki": 0, "kd": .5}},
                                     {"pid_profile": {"kp": 1, "ki": 0, "kd": .25}})

    def test_real_stage_size_budget_refreshes_and_trace(self):
        from test_budget import FakeModel, load_dpo, split
        from test_replay import filled_buffer
        dpo = load_dpo()
        dpo.scored.clear()
        b = filled_buffer(size=500)
        with tempfile.TemporaryDirectory() as folder, contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            path = Path(folder) / "pid_trace.csv"
            history, log = dpo.train_stage_replay(
                FakeModel(), None, split("helpful", 2000), buffer=b,
                method="pid_i010_d050", seed=2, pid_trace_path=path)
            trace = pd.read_csv(path)
        self.assertEqual(len(history), 112)
        self.assertEqual(int(history["n_new"].sum()), 2000)
        self.assertEqual(int(history["n_replay"].sum()), 222)
        self.assertEqual(log.groupby("interval").size().tolist(), [46, 46, 46, 46, 38])
        self.assertEqual(dpo.scored, [500] * 4)
        self.assertEqual(len(trace), 500 * 5)
        np.testing.assert_allclose(trace["selection_score"],
                                   trace[["pid_p_term", "pid_i_term", "pid_d_term"]].sum(axis=1))
        self.assertTrue(set(mfr_pid.LOG_COLUMNS) <= set(log.columns))

    def test_runner_saves_boundary_state_for_resuming_the_next_topic(self):
        from test_budget import load_dpo, split
        load_dpo()
        runner = load_script("run_experiment")
        b = ReplayBuffer(500)
        old = b.candidates("helpful", split("helpful", 2000))
        b.add_stage("helpful", old, pd.Series(.8, index=old["id"]))
        b.set_current(pd.Series(.5, index=old["id"]), pid_dt=1)
        def score(model, tokenizer, rows, **kwargs):
            value = .6 if rows["id"].str.startswith("safe").all() else .65
            return pd.DataFrame({"margin": value}, index=rows["id"].values)
        protocol = load_protocol(ROOT / "configs/experiment_protocol.json")
        with tempfile.TemporaryDirectory() as folder, patch.object(runner.mfr_dpo, "score_pairs", side_effect=score):
            runner.update_buffer(None, None, b, 2, "safe", split("safe", 2000),
                                 Path(folder), protocol, None, method="pid_i010_d050")
            restored = ReplayBuffer.from_csv(Path(folder) / "buffer.csv", size=500)
        rows = restored.rows()
        helpful = rows[rows["dataset"] == "helpful"]
        np.testing.assert_allclose(helpful["peak_margin"], .8)
        np.testing.assert_allclose(helpful["pid_integral"], .3 + .15 * 20 / 23)
        np.testing.assert_allclose(helpful["pid_derivative"], -.15 / (20 / 23))
        np.testing.assert_allclose(rows.loc[rows["dataset"] == "safe", list(mfr_pid.STATE_COLUMNS)], 0)

    def test_validation_summary_uses_paired_controls_and_final_average(self):
        analysis = load_script("pid_results")
        methods = ["random", "lowest_margin", "mfr", "pid_i000_d025"]
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            order = ["helpful", "safe", "quality"]
            for method, old_score, new_score in zip(methods, [70, 73, 72, 74], [80, 79, 80, 79]):
                run_name = f"v2_o1_{method}_s0"
                run_dir = root / "runs" / run_name
                run_dir.mkdir(parents=True)
                settings = {"run_name": run_name, "order_id": 1, "order": order, "seed": 0,
                            "method": method, "scientific_code_sha256": "shared-code",
                            "protocol_version": "2.1", "data_version": "v2"}
                (run_dir / "settings.json").write_text(json.dumps(settings))
                (run_dir / "COMPLETE.json").write_text("{}")
                rows = []
                for stage in range(4):
                    for dataset in order:
                        score = 80 if stage else 50
                        if stage == 3:
                            score = new_score if dataset == "quality" else old_score
                        rows.append({"stage": stage, "eval_set": dataset, "accuracy": score})
                    if stage:
                        stage_dir = run_dir / f"stage{stage}_{order[stage - 1]}"
                        stage_dir.mkdir()
                        pd.DataFrame({"minutes": [1], "n_new": [2000], "n_replay": [222],
                                      "scoring_minutes": [.1]}).to_csv(stage_dir / "history.csv", index=False)
                pd.DataFrame(rows).to_csv(run_dir / "results.csv", index=False)
            cells, summary = analysis.summarize(root, methods, [1], [0])
            pid = summary[summary["method"] == "pid_i000_d025"].iloc[0]
            self.assertAlmostEqual(pid["mean_retention_change"], -6)
            self.assertAlmostEqual(pid["retention_delta_random"], 4)
            self.assertAlmostEqual(pid["retention_delta_lowest_margin"], 1)
            self.assertAlmostEqual(pid["final_average"], (74 + 74 + 79) / 3)
            self.assertTrue(pid["passes_random_gate"])
            argv = ["pid_results.py", "--output-dir", str(root), "--methods", ",".join(methods),
                    "--orders", "1", "--seeds", "0", "--suggest-integral"]
            captured = io.StringIO()
            with patch("sys.argv", argv), contextlib.redirect_stdout(captured):
                analysis.main()
            self.assertEqual(captured.getvalue().strip().split(","), [
                "pid_i002_d025", "pid_i010_d025", "pid_i030_d025",
                "pid_i002_d000", "pid_i010_d000", "pid_i030_d000"])
            with self.assertRaisesRegex(ValueError, "missing completed runs"):
                analysis.summarize(root, methods, [1, 3], [0])


if __name__ == "__main__":
    unittest.main()
