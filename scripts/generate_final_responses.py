#!/usr/bin/env python3
"""Generate deterministic responses for the generation evaluation (docs/FREEZE.md, plan §5).

Targets
  --run-dir RUN           a completed sequential or joint run
  --base-model            the untrained base model of --protocol (written to
                          OUTPUT_DIR/base_model/<model>/ as a pseudo-run so the graders work unchanged)

Checkpoints
  --checkpoint final                       last stage (or joint_all)  -> generation/final_eval/
  --checkpoint post_stage --behavior B     right after B was learned  -> generation/post_stage_B/
                                           (behavioral forgetting; only B's locked test prompts)

Suites (final checkpoint): test (locked Helpful/Safe/Quality prompts), xstest, harmbench.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import mfr_data
import mfr_dpo
from mfr_eval import assert_test_ready, generate_responses, save_generations
from mfr_generation_eval import evaluation_adapter, generation_subdir, load_evaluation_protocol
from mfr_utils import file_sha256, load_protocol, save_json_atomic

BEHAVIORS = ("helpful", "safe", "quality")
SUITES = ("test", "xstest", "harmbench")


def parse_args():
    parser = argparse.ArgumentParser()
    target = parser.add_mutually_exclusive_group(required=True)
    target.add_argument("--run-dir", help="Completed sequential or joint run folder")
    target.add_argument("--base-model", action="store_true",
                        help="Generate from the untrained base model of --protocol")
    parser.add_argument("--output-dir", help="Artifact root for --base-model")
    parser.add_argument(
        "--protocol", default=str(ROOT / "configs" / "experiment_protocol.json"),
        help="Training protocol (Qwen default; configs/second_model_protocol.json for Llama)",
    )
    parser.add_argument("--checkpoint", choices=("final", "post_stage"), default="final")
    parser.add_argument("--behavior", choices=BEHAVIORS, help="Required with --checkpoint post_stage")
    parser.add_argument("--suites", default="test,xstest,harmbench",
                        help="Comma-separated: test,xstest,harmbench (final checkpoint only)")
    parser.add_argument("--max-prompts", type=int,
                        help="post_stage only: first N locked prompts by id (e.g. 150 for Helpful/Quality)")
    parser.add_argument("--confirm-final-evaluation", action="store_true",
                        help="Required acknowledgement that method selection is frozen")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def base_model_run(output_dir, protocol, protocol_path):
    """A pseudo-run folder for the base model, so graders and judges can treat it like a run."""
    slug = protocol["model_name"].replace("/", "__")
    run_dir = Path(output_dir).expanduser().resolve() / "base_model" / slug
    settings = {
        "run_name": f"{protocol['data_version']}_base_{slug}", "method": "base", "order": None,
        "order_id": 0, "seed": 0, "protocol_version": protocol["protocol_version"],
        "data_version": protocol["data_version"], "model_name": protocol["model_name"],
        "model_revision": protocol["model_revision"], "lora_r": protocol["lora_r"],
        "lora_alpha": protocol["lora_alpha"], "lora_dropout": protocol["lora_dropout"],
        "protocol_sha256": file_sha256(protocol_path),
    }
    run_dir.mkdir(parents=True, exist_ok=True)
    save_json_atomic(settings, run_dir / "settings.json")
    save_json_atomic({"note": "pseudo-run for base-model generation; no training"},
                     run_dir / "COMPLETE.json")
    return run_dir


def adapter_files(adapter):
    if adapter is None:
        return []
    weights = [adapter / "adapter_model.safetensors", adapter / "adapter_model.bin"]
    paths = [adapter / "adapter_config.json", next((p for p in weights if p.exists()), weights[0])]
    if missing := [p for p in paths if not p.exists()]:
        raise FileNotFoundError(f"adapter is incomplete: {missing}")
    return paths


def main():
    args = parse_args()
    if not args.confirm_final_evaluation:
        raise RuntimeError(
            "generation on locked prompts is disabled until method selection is frozen; "
            "rerun with --confirm-final-evaluation when that decision is final"
        )
    suites = [item.strip() for item in args.suites.split(",") if item.strip()]
    if unknown := sorted(set(suites) - set(SUITES)):
        raise ValueError(f"unknown suites {unknown}; choose from {SUITES}")
    if args.checkpoint == "post_stage":
        if not args.behavior:
            raise ValueError("--checkpoint post_stage requires --behavior")
        suites = ["test"]
    elif args.max_prompts is not None:
        raise ValueError("--max-prompts applies only to post_stage generation")

    protocol_path = Path(args.protocol).expanduser().resolve()
    protocol = load_protocol(protocol_path)
    if args.base_model:
        if not args.output_dir:
            raise ValueError("--base-model requires --output-dir")
        run_dir = base_model_run(args.output_dir, protocol, protocol_path)
    else:
        run_dir = Path(args.run_dir).expanduser().resolve()
    assert_test_ready(run_dir)
    settings_path = run_dir / "settings.json"
    settings = json.loads(settings_path.read_text(encoding="utf-8"))
    for key in ("model_name", "model_revision", "data_version"):
        if settings.get(key) != protocol.get(key):
            raise ValueError(f"run {key}={settings.get(key)!r} does not match --protocol")

    adapter = evaluation_adapter(run_dir, settings, args.checkpoint, args.behavior)
    files = adapter_files(adapter)
    evaluation_path = ROOT / "configs" / "evaluation_protocol.json"
    evaluation = load_evaluation_protocol(evaluation_path)
    generation = evaluation["generation"]
    output_dir = run_dir / "generation" / generation_subdir(args.checkpoint, args.behavior)
    output_dir.mkdir(parents=True, exist_ok=True)

    behaviors = [args.behavior] if args.checkpoint == "post_stage" else list(BEHAVIORS)
    expected = []
    if "test" in suites:
        expected += [output_dir / f"{b}_test.jsonl" for b in behaviors]
    expected += [output_dir / f"{name}.jsonl" for name in ("xstest", "harmbench") if name in suites]
    if (existing := [p for p in expected if p.exists()]) and not args.overwrite:
        raise FileExistsError(f"outputs already exist: {existing}. Use --overwrite only intentionally.")

    print(f"Loading {settings['run_name']} ({args.checkpoint}"
          f"{' ' + args.behavior if args.behavior else ''}) from {adapter or 'base model'}...", flush=True)
    model, tokenizer = mfr_dpo.load_model(
        protocol["model_name"], protocol["lora_r"], adapter_path=adapter,
        revision=protocol["model_revision"], lora_alpha=protocol["lora_alpha"],
        lora_dropout=protocol["lora_dropout"],
    )  # no adapter -> fresh LoRA with B = 0, i.e. exactly the base model
    common = {
        "evaluation_version": evaluation["evaluation_version"],
        "run_name": settings["run_name"], "method": settings["method"],
        "order_id": settings.get("order_id"), "seed": settings.get("seed"),
        "checkpoint": args.checkpoint, "checkpoint_behavior": args.behavior,
        "adapter": str(adapter) if adapter else None,
        "model_name": protocol["model_name"], "model_revision": protocol["model_revision"],
        "settings_sha256": file_sha256(settings_path),
        "protocol_sha256": file_sha256(protocol_path),
        "evaluation_protocol_sha256": file_sha256(evaluation_path),
        "adapter_files_sha256": {path.name: file_sha256(path) for path in files},
        **generation,
    }

    def run(frame, path, extra):
        out = generate_responses(
            model, tokenizer, frame[["id", "prompt"]], batch_size=generation["batch_size"],
            max_new_tokens=generation["max_new_tokens"], do_sample=generation["do_sample"],
            seed=generation["seed"],
        )
        out = out.merge(frame.drop(columns=["prompt"]), on="id", validate="one_to_one")
        out["run_name"] = settings["run_name"]
        out["method"] = settings["method"]
        save_generations(out, path, {**common, **extra})
        print(f"Saved {len(out)} responses to {path}", flush=True)

    if "test" in suites:
        splits = mfr_data.load_splits(ROOT / protocol["data_dir"])
        for behavior in behaviors:
            source = splits[behavior]["test"].sort_values("id").copy()
            if args.max_prompts is not None:
                source = source.head(args.max_prompts)
            frame = source[["id", "prompt", "chosen", "rejected"]].copy()
            frame.insert(1, "behavior", behavior)
            run(frame, output_dir / f"{behavior}_test.jsonl",
                {"suite": "locked_test", "behavior": behavior})

    if "xstest" in suites:
        from datasets import load_dataset
        x = evaluation["xstest"]
        xs = load_dataset(x["dataset"], split=x["split"], revision=x["dataset_revision"]).to_pandas()
        if missing := {"id", "prompt", "type", "label"} - set(xs):
            raise ValueError(f"XSTest is missing columns {sorted(missing)}")
        if len(xs) != 450 or xs["label"].astype(str).str.lower().value_counts().to_dict() != {
                "safe": 250, "unsafe": 200}:
            raise ValueError("XSTest does not have the official 250 safe + 200 unsafe prompts")
        xs = xs.copy()
        xs["source_id"] = xs["id"]
        xs["id"] = [f"xstest-{int(value):04d}" for value in xs["source_id"]]
        keep = [c for c in ("id", "prompt", "source_id", "type", "label", "focus", "note") if c in xs]
        frame = xs[keep].rename(columns={"label": "xstest_label", "type": "xstest_type"})
        run(frame, output_dir / "xstest.jsonl",
            {"suite": "xstest", "dataset": x["dataset"], "dataset_revision": x["dataset_revision"]})

    if "harmbench" in suites:
        from datasets import load_dataset
        h = evaluation["harmbench"]
        hb = load_dataset(h["dataset"], h["config"], split=h["split"],
                          revision=h["dataset_revision"]).to_pandas()
        if len(hb) != h["expected_rows"] or not {"prompt", "category"} <= set(hb):
            raise ValueError(f"HarmBench standard set has {len(hb)} rows / columns {list(hb)}")
        hb = hb.reset_index(drop=True)
        frame = hb[["prompt", "category"]].copy()
        frame.insert(0, "id", [f"harmbench-{index:04d}" for index in range(len(frame))])
        run(frame, output_dir / "harmbench.jsonl",
            {"suite": "harmbench", "dataset": h["dataset"], "dataset_config": h["config"],
             "dataset_revision": h["dataset_revision"], "attack": h["attack"]})


if __name__ == "__main__":
    main()
