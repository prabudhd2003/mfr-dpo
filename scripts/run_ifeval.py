#!/usr/bin/env python3
"""IFEval (lm-eval) with logged samples for one or more completed runs.

engine "vllm" (configs/evaluation_protocol.json ifeval.engine): lm-eval's vLLM backend on the
dequantized base (scripts/prepare_vllm_base.py, which also holds the date-pinned tokenizer) plus
the run's LoRA adapter. engine "hf": the old 4-bit Hugging Face backend. Finished runs are skipped.
--limit N (testing only) writes N prompts to final_eval/ifeval_smoke/.
"""

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
from mfr_generation_eval import evaluation_adapter, load_evaluation_protocol
from mfr_utils import file_sha256, pin_chat_template_date, save_json_atomic
from mfr_vllm import check_vllm_base, output_root_of, vllm_base_dir


def hf_model_args(settings, adapter, output):
    model_args = (f"pretrained={settings['model_name']},revision={settings['model_revision']},"
                  "load_in_4bit=True")
    if adapter is not None:
        model_args += f",peft={adapter}"
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(settings["model_name"], revision=settings["model_revision"])
    original = tokenizer.chat_template
    pin_chat_template_date(tokenizer)
    if tokenizer.chat_template != original:
        pinned = output.parent / "pinned_tokenizer"
        tokenizer.save_pretrained(pinned)
        model_args += f",tokenizer={pinned}"
    return ["--model", "hf", "--model_args", model_args, "--batch_size", "4", "--device", "cuda:0"]


def vllm_model_args(run_dir, settings, adapter, config, seed):
    base_dir = vllm_base_dir(output_root_of(run_dir), settings["model_name"])
    check_vllm_base(base_dir, settings["model_name"], settings["model_revision"])
    model_args = (f"pretrained={base_dir},dtype=bfloat16,max_model_len={config.get('max_model_len', 4096)},"
                  f"gpu_memory_utilization=0.85,seed={seed}")
    if adapter is not None:
        model_args += f",enable_lora=True,max_lora_rank={settings['lora_r']},lora_local_path={adapter}"
    return ["--model", "vllm", "--model_args", model_args, "--batch_size", "auto"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", nargs="+", required=True)
    parser.add_argument("--confirm-final-evaluation", action="store_true")
    parser.add_argument("--limit", type=int, help="Testing only: N prompts -> final_eval/ifeval_smoke/")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if not args.confirm_final_evaluation:
        raise RuntimeError("IFEval is part of final generation evaluation; rerun with "
                           "--confirm-final-evaluation after method selection is frozen")
    evaluation = load_evaluation_protocol(ROOT / "configs" / "evaluation_protocol.json")
    config = evaluation.get("ifeval", {"engine": "hf"})
    engine = config.get("engine", "vllm")
    seed = evaluation["generation"]["seed"]

    for run in args.run_dir:
        run_dir = Path(run).expanduser().resolve()
        assert_test_ready(run_dir)
        settings = json.loads((run_dir / "settings.json").read_text(encoding="utf-8"))
        # final adapter of a sequential run, joint_all for a joint run, None for the base model
        adapter = evaluation_adapter(run_dir, settings, "final")
        if adapter is not None and not (adapter / "adapter_config.json").exists():
            raise FileNotFoundError(f"final adapter is incomplete: {adapter}")
        output = run_dir / "generation" / "final_eval" / ("ifeval_smoke" if args.limit else "ifeval")
        if (output / "mfr_evaluation_manifest.json").exists() and not args.overwrite:
            print(f"Skipping {run_dir.name}: IFEval already done", flush=True)
            continue
        if output.exists():
            shutil.rmtree(output)  # unfinished or intentionally overwritten
        output.mkdir(parents=True, exist_ok=True)
        model = (vllm_model_args(run_dir, settings, adapter, config, seed) if engine == "vllm"
                 else hf_model_args(settings, adapter, output))
        # Legacy form (no `run` subcommand) works on old and new lm-eval versions.
        command = ["lm-eval", *model, "--tasks", "ifeval", "--apply_chat_template",
                   "--output_path", str(output), "--log_samples"]
        if args.limit:
            command += ["--limit", str(args.limit)]
        print("Running IFEval for", settings["run_name"], flush=True)
        subprocess.run(command, check=True)
        weights = (adapter / "adapter_model.safetensors") if adapter is not None else None
        save_json_atomic({
            "run_name": settings["run_name"],
            "model_name": settings["model_name"],
            "model_revision": settings["model_revision"],
            "settings_sha256": file_sha256(run_dir / "settings.json"),
            "adapter": str(adapter) if adapter else None,
            "adapter_sha256": file_sha256(weights) if weights is not None and weights.exists() else None,
            "lm_eval_version": metadata.version("lm_eval"),
            "engine": engine,
            "command": command,
            "task": "ifeval",
            "apply_chat_template": True,
            "limit": args.limit,
        }, output / "mfr_evaluation_manifest.json")
        print("Saved IFEval results and per-prompt samples to", output, flush=True)


if __name__ == "__main__":
    main()
