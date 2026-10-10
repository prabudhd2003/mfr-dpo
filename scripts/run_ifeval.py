#!/usr/bin/env python3
"""Evaluate one completed final adapter on IFEval with logged samples."""

import argparse
import importlib.metadata as metadata
import json
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mfr_eval import assert_test_ready
from mfr_generation_eval import evaluation_adapter
from mfr_utils import file_sha256, pin_chat_template_date, save_json_atomic
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--confirm-final-evaluation", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if not args.confirm_final_evaluation:
        raise RuntimeError(
            "IFEval is part of final generation evaluation; rerun with "
            "--confirm-final-evaluation after method selection is frozen"
        )
    run_dir = Path(args.run_dir)
    assert_test_ready(run_dir)
    with open(run_dir / "settings.json", encoding="utf-8") as handle:
        settings = json.load(handle)
    # Final adapter of a sequential run, joint_all for a joint run, None for the base-model
    # pseudo-run written by generate_final_responses.py --base-model.
    adapter = evaluation_adapter(run_dir, settings, "final")
    if adapter is not None and not (adapter / "adapter_config.json").exists():
        raise FileNotFoundError(f"final adapter is incomplete: {adapter}")
    output = run_dir / "generation" / "final_eval" / "ifeval"
    if output.exists() and any(output.iterdir()) and not args.overwrite:
        raise FileExistsError(f"{output} already contains results; use --overwrite intentionally")
    if output.exists() and any(output.iterdir()) and args.overwrite:
        shutil.rmtree(output)
    output.mkdir(parents=True, exist_ok=True)
    model_args = (
        f"pretrained={settings['model_name']},revision={settings['model_revision']},"
        "load_in_4bit=True"
    )
    if adapter is not None:
        model_args += f",peft={adapter}"
    # Llama 3.x templates insert today's date; give lm-eval a tokenizer with the same pinned date
    # used in training and in our own generation. A no-op for templates without a date (Qwen).
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(settings["model_name"], revision=settings["model_revision"])
    original_template = tokenizer.chat_template
    pin_chat_template_date(tokenizer)
    pinned_tokenizer = None
    if tokenizer.chat_template != original_template:
        pinned_tokenizer = output.parent / "pinned_tokenizer"
        tokenizer.save_pretrained(pinned_tokenizer)
        model_args += f",tokenizer={pinned_tokenizer}"
    # Legacy form (no `run` subcommand): newer lm-eval inserts `run` automatically, older
    # versions require this form.
    command = [
        "lm-eval", "--model", "hf", "--model_args", model_args,
        "--tasks", "ifeval", "--apply_chat_template", "--batch_size", "4",
        "--device", "cuda:0", "--output_path", str(output), "--log_samples",
    ]
    print("Running IFEval for", settings["run_name"])
    subprocess.run(command, check=True)
    adapter_weights = (adapter / "adapter_model.safetensors") if adapter is not None else None
    save_json_atomic({
        "run_name": settings["run_name"],
        "model_name": settings["model_name"],
        "model_revision": settings["model_revision"],
        "settings_sha256": file_sha256(run_dir / "settings.json"),
        "adapter": str(adapter) if adapter else None,
        "adapter_sha256": (file_sha256(adapter_weights)
                           if adapter_weights is not None and adapter_weights.exists() else None),
        "pinned_tokenizer": str(pinned_tokenizer) if pinned_tokenizer else None,
        "lm_eval_version": metadata.version("lm_eval"),
        "task": "ifeval",
        "apply_chat_template": True,
        "batch_size": 4,
    }, output / "mfr_evaluation_manifest.json")
    print("Saved IFEval results and per-prompt samples to", output)


if __name__ == "__main__":
    main()
