"""CPU-only tests of locked test inputs, exact metrics, resume, and launch scope."""

from contextlib import nullcontext
import copy
import csv
import importlib.util
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "pd_test.py"
spec = importlib.util.spec_from_file_location("pd_test_launcher", SCRIPT)
pd = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pd)


class FakeResult:
    def __init__(self, records):
        self.records = records

    def to_dict(self, orient):
        assert orient == "records"
        return self.records


class FakeDPO:
    def __init__(self, fail_on=None):
        self.load_calls = 0
        self.score_calls = 0
        self.fail_on = fail_on

    def load_model(self, *args, **kwargs):
        self.load_calls += 1
        return SimpleNamespace(eval=lambda: None), object()

    def score_pairs(self, model, tokenizer, df, **kwargs):
        self.score_calls += 1
        if self.score_calls == self.fail_on:
            raise RuntimeError("simulated job interruption")
        return FakeResult([{name: (1 if index % 3 else 0) for name in pd.MARGIN_COLUMNS}
                           for index in range(len(df["id"]))])


TORCH = SimpleNamespace(inference_mode=nullcontext, cuda=SimpleNamespace(empty_cache=lambda: None))


class PDTestLauncherTests(unittest.TestCase):
    def setUp(self):
        self.print_patch = patch("builtins.print")
        self.print_patch.start()
        self.addCleanup(self.print_patch.stop)
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.repo = self.root / "repo"
        self.output = self.root / "artifacts"
        self.data = self.repo / "data" / "v2"
        self.data.mkdir(parents=True)
        (self.repo / "configs").mkdir()
        self.protocol = {
            "protocol_version": "2.1", "data_version": "v2", "model_name": "Qwen/test",
            "model_revision": "fixed-revision", "data_dir": "data/v2", "beta": .1,
            "max_tokens": 1024, "lora_r": 16, "lora_alpha": 32, "lora_dropout": .05,
            "pid_replay_profiles": {pd.METHOD: pd.GAINS},
            "orders": {str(order): ["helpful", "safe", "quality"] for order in range(1,5)},
        }
        self.protocol_path = self.repo / "configs" / "experiment_protocol.json"
        self.protocol_path.write_text(json.dumps(self.protocol))
        self.splits = {}
        files = {}
        for topic in pd.TOPICS:
            ids = [f"{topic}-test-{index:04d}" for index in range(300)]
            path = self.data / f"{topic}_test.jsonl"
            path.write_text("".join(json.dumps({"id": pair_id}) + "\n" for pair_id in ids))
            files[path.name] = {"rows": 300, "sha256": pd.sha(path)}
            self.splits[topic] = {"test": {"id": ids}}
        manifest = self.data / "manifest.json"
        manifest.write_text(json.dumps({"files": files}))
        for order in range(1,5):
            for seed in (0,1):
                name = f"v2_o{order}_{pd.METHOD}_s{seed}"
                run = self.output / "runs" / name
                checkpoint = run / "stage3_quality"
                checkpoint.mkdir(parents=True)
                (checkpoint / "adapter_config.json").write_text("{}")
                (checkpoint / "adapter_model.safetensors").write_bytes(b"synthetic weights")
                (run / "COMPLETE.json").write_text("{}")
                settings = {key: self.protocol[key] for key in pd.SCORING_KEYS}
                settings.update(run_name=name, method=pd.METHOD, order_id=order, seed=seed,
                                order=self.protocol["orders"][str(order)], pid_profile=pd.GAINS,
                                data_manifest_sha256=pd.sha(manifest),
                                scientific_code_sha256="fixed-science", git_commit="training-commit")
                (run / "settings.json").write_text(json.dumps(settings))
        self.hash_patch = patch.object(pd, "science_hash", return_value="fixed-science")
        self.hash_patch.start()
        self.plan = pd.make_plan(self.repo, self.output, [1,2,3,4], [0,1], 4)
        self.plan_id = pd.digest_object(self.plan)

    def tearDown(self):
        self.hash_patch.stop()
        self.temporary.cleanup()

    def freeze(self):
        pd.atomic_json(pd.plan_path(self.output), {"plan": self.plan, "plan_sha256": self.plan_id})

    def test_preflight_selects_only_eight_frozen_final_pd_checkpoints(self):
        self.assertEqual(len(self.plan["runs"]), 8)
        self.assertTrue(all(run["checkpoint"].endswith("stage3_quality") for run in self.plan["runs"]))
        self.assertTrue(all(pd.METHOD in run["run_name"] for run in self.plan["runs"]))
        self.assertFalse((self.output / "pd_test").exists())

    def test_changed_gain_is_rejected_before_scoring(self):
        protocol = copy.deepcopy(self.protocol)
        protocol["pid_replay_profiles"][pd.METHOD]["ki"] = .1
        self.protocol_path.write_text(json.dumps(protocol))
        with self.assertRaisesRegex(ValueError, "Protocol must"):
            pd.make_plan(self.repo, self.output, [1], [0], 4)

    def test_missing_completed_checkpoint_is_rejected(self):
        run = self.plan["runs"][0]
        (Path(run["run_dir"]) / "COMPLETE.json").unlink()
        with self.assertRaisesRegex(ValueError, "not complete"):
            pd.make_plan(self.repo, self.output, [1], [0], 4)

    def test_test_split_hash_mismatch_is_rejected(self):
        with (self.data / "safe_test.jsonl").open("a") as handle:
            handle.write("{}\n")
        with self.assertRaisesRegex(ValueError, "manifest verification"):
            pd.make_plan(self.repo, self.output, [1], [0], 4)

    def test_model_and_scoring_settings_must_match_training(self):
        settings = Path(self.plan["runs"][0]["run_dir"]) / "settings.json"
        record = pd.read_json(settings)
        record["model_revision"] = "different"
        settings.write_text(json.dumps(record))
        with self.assertRaisesRegex(ValueError, "provenance mismatch"):
            pd.make_plan(self.repo, self.output, [1], [0], 4)

    def test_frozen_plan_rejects_changed_weight_file(self):
        self.freeze()
        checkpoint = Path(self.plan["runs"][0]["checkpoint"])
        (checkpoint / "adapter_model.safetensors").write_bytes(b"changed weights")
        candidate = pd.make_plan(self.repo, self.output, [1,2,3,4], [0,1], 4)
        with self.assertRaisesRegex(ValueError, "already frozen"):
            pd.check_existing_plan(self.output, candidate)

    def test_completed_scores_are_not_recomputed(self):
        run = self.plan["runs"][0]
        fake = FakeDPO()
        pd.score_run(run, self.plan, self.plan_id, self.splits, fake, TORCH)
        self.assertTrue(pd.completed(run, self.plan_id))
        self.assertEqual(fake.score_calls, 3)
        pd.score_run(run, self.plan, self.plan_id, self.splits, fake, TORCH)
        self.assertEqual(fake.score_calls, 3)
        records = pd.read_csv(Path(run["run_dir"]) / "final_test" / "test_results.csv")
        self.assertTrue(all(int(row["n_correct"]) == 200 for row in records))
        self.assertAlmostEqual(float(records[0]["accuracy"]), 200/3)

    def test_interrupted_checkpoint_resumes_without_rescoring_saved_topic(self):
        run = self.plan["runs"][0]
        with self.assertRaisesRegex(RuntimeError, "interruption"):
            pd.score_run(run, self.plan, self.plan_id, self.splits, FakeDPO(fail_on=2), TORCH)
        self.assertFalse(pd.completed(run, self.plan_id))
        fake = FakeDPO()
        pd.score_run(run, self.plan, self.plan_id, self.splits, fake, TORCH)
        self.assertEqual(fake.score_calls, 2)
        self.assertTrue(pd.completed(run, self.plan_id))

    def test_missing_or_duplicate_test_ids_are_rejected(self):
        rows = [dict(id="same", **{name: 1 for name in pd.MARGIN_COLUMNS}) for _ in range(2)]
        with self.assertRaisesRegex(ValueError, "IDs/count"):
            pd.validate_score_rows(rows, ["x", "y"])

    def test_nonfinite_scores_are_rejected(self):
        row = dict(id="x", **{name: 1 for name in pd.MARGIN_COLUMNS})
        row["margin"] = float("nan")
        with self.assertRaisesRegex(ValueError, "Non-finite"):
            pd.validate_score_rows([row], ["x"])

    def test_existing_legacy_test_results_are_not_overwritten(self):
        path = Path(self.plan["runs"][0]["run_dir"]) / "final_test" / "test_results.csv"
        pd.atomic_text(path, "existing results")
        with self.assertRaisesRegex(ValueError, "Existing test scores"):
            pd.check_legacy_results(self.plan, self.plan_id)
        self.assertEqual(path.read_text(), "existing results")

    def test_mutation_of_completed_result_is_detected(self):
        run = self.plan["runs"][0]
        pd.score_run(run, self.plan, self.plan_id, self.splits, FakeDPO(), TORCH)
        path = Path(run["run_dir"]) / "final_test" / "helpful_test_scores.csv"
        path.write_text("changed")
        with self.assertRaisesRegex(ValueError, "changed or is missing"):
            pd.completed(run, self.plan_id)

    def test_summary_requires_all_cells_and_preserves_exact_test_percentages(self):
        self.freeze()
        with self.assertRaisesRegex(ValueError, "not yet complete"):
            pd.summarize(self.output)
        for run in self.plan["runs"]:
            pd.score_run(run, self.plan, self.plan_id, self.splits, FakeDPO(), TORCH)
        pd.summarize(self.output)
        result = pd.read_csv(self.output / "pd_test" / "pd_test_summary.csv")[0]
        self.assertEqual(int(result["cells"]), 8)
        self.assertAlmostEqual(float(result["final_average"]), 200/3)
        self.assertEqual(len(pd.read_csv(self.output / "pd_test" / "pd_test_by_topic.csv")), 24)

    def test_sbatch_preserves_paths_as_arguments_and_requests_one_l40s(self):
        command = pd.sbatch_command(self.repo, self.output, "/env with spaces", "account", "03:00:00", 1, self.plan_id)
        self.assertIn("--gpus-per-task=l40s:1", command)
        self.assertIn("/env with spaces", command)
        self.assertEqual(command[-1], "1")
        self.assertNotIn("--wrap", command)

    def test_dry_run_does_not_write_plan_or_submit(self):
        args = SimpleNamespace(dry_run=True, conda_env="/env", account="account", time="03:00:00")
        with patch.object(pd.subprocess, "run") as process:
            pd.submit(args, self.plan, self.output)
        process.assert_not_called()
        self.assertFalse(pd.plan_path(self.output).exists())

    def test_submit_freezes_inputs_and_schedules_exactly_four_orders(self):
        env = self.root / "environment"
        (env / "bin").mkdir(parents=True)
        (env / "bin" / "python").write_text("synthetic interpreter path")
        args = SimpleNamespace(dry_run=False, conda_env=str(env), account="account", time="03:00:00")
        submissions = []
        def process(command, **kwargs):
            if command[0] == "sbatch":
                submissions.append(command)
                return SimpleNamespace(stdout=str(9000 + len(submissions)) + "\n")
            if command[0] == "git":
                return SimpleNamespace(stdout="evaluation-commit\n")
            self.assertEqual(command[0], "squeue")
            return SimpleNamespace(stdout="")
        with patch.object(pd.shutil, "which", return_value="/synthetic/slurm"), patch.object(pd.subprocess, "run", side_effect=process):
            pd.submit(args, self.plan, self.output)
        self.assertEqual([command[-1] for command in submissions], ["1","2","3","4"])
        frozen, plan_id = pd.load_plan(self.output)
        self.assertEqual(frozen, self.plan)
        self.assertEqual(plan_id, self.plan_id)
        self.assertEqual((self.output / "pd_test" / "worker.sh").read_text(), pd.WORKER)

    def test_gpu_worker_refuses_to_run_on_login_node(self):
        self.freeze()
        args = SimpleNamespace(order=1)
        with patch.object(pd, "ROOT", self.repo), patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(RuntimeError, "Slurm job"):
                pd.run_order(args, self.output)


if __name__ == "__main__":
    unittest.main()
