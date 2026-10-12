#!/usr/bin/env python3
"""Parity check: Hugging Face 4-bit + LoRA (training path) vs vLLM dequantized base + LoRA.

Greedy-decodes validation prompts (never the locked test) and compares the first --tokens tokens.
Agreement per prompt = shared prefix / compared length.

Greedy decoding is fragile: one near-tie flips a token and the rest of the text diverges, even
inside the Hugging Face path itself. So the reference is HF batch 1, and the control is HF batch 8 vs
HF batch 1 (same model, same weights, only batching differs). Rule, fixed before any result:
vLLM is accepted if its agreement with HF batch 1 is >= 90%, or is no more than 5 points below the
HF-vs-HF control. Otherwise set generation.engine (and ifeval.engine) to "hf".
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd

import mfr_data
from mfr_generation_eval import evaluation_adapter
from mfr_utils import load_protocol, save_json_atomic
from mfr_vllm import Engine, check_vllm_base, output_root_of, vllm_base_dir, vllm_version


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--protocol", default=str(ROOT / "configs" / "experiment_protocol.json"))
    parser.add_argument("--prompts", type=int, default=21, help="Split evenly over the three behaviors")
    parser.add_argument("--tokens", type=int, default=64)
    parser.add_argument("--threshold", type=float, default=90.0)
    args = parser.parse_args()

    import torch
    import mfr_dpo
    from mfr_eval import generate_responses

    run_dir = Path(args.run_dir).expanduser().resolve()
    protocol = load_protocol(Path(args.protocol).expanduser().resolve())
    settings = json.loads((run_dir / "settings.json").read_text(encoding="utf-8"))
    adapter = evaluation_adapter(run_dir, settings, "final")
    splits = mfr_data.load_splits(ROOT / protocol["data_dir"])
    per = max(1, args.prompts // 3)
    frame = pd.concat([splits[b]["val"].sort_values("id").head(per)[["id", "prompt"]]
                       for b in ("helpful", "safe", "quality")], ignore_index=True)

    print(f"HF 4-bit + LoRA: {adapter}", flush=True)
    model, tokenizer = mfr_dpo.load_model(
        protocol["model_name"], protocol["lora_r"], adapter_path=adapter,
        revision=protocol["model_revision"], lora_alpha=protocol["lora_alpha"],
        lora_dropout=protocol["lora_dropout"])
    hf = generate_responses(model, tokenizer, frame, batch_size=1, max_new_tokens=args.tokens)
    hf8 = generate_responses(model, tokenizer, frame, batch_size=8, max_new_tokens=args.tokens)
    del model
    gc.collect()
    torch.cuda.empty_cache()

    base_dir = vllm_base_dir(output_root_of(run_dir), protocol["model_name"])
    check_vllm_base(base_dir, protocol["model_name"], protocol["model_revision"])
    engine = Engine(base_dir, 4096, lora_rank=protocol["lora_r"], gpu_memory_utilization=0.6)
    vl = engine.chat(frame["prompt"].tolist(), args.tokens, adapter, desc="vLLM")

    def agreement(x_text, y_text):
        a = tokenizer(x_text, add_special_tokens=False)["input_ids"][:args.tokens]
        b = tokenizer(y_text, add_special_tokens=False)["input_ids"][:args.tokens]
        prefix = next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), min(len(a), len(b)))
        return 100.0 * prefix / max(1, min(args.tokens, max(len(a), len(b))))

    rows = []
    for row, hf_text, hf8_text, vllm_text in zip(frame.itertuples(index=False), hf["response"],
                                                 hf8["response"], vl):
        rows.append({"id": row.id, "agreement_pct": agreement(hf_text, vllm_text),
                     "hf_control_pct": agreement(hf_text, hf8_text),
                     "hf": hf_text, "hf_batch8": hf8_text, "vllm": vllm_text})
    result = pd.DataFrame(rows)
    mean = float(result["agreement_pct"].mean())
    control = float(result["hf_control_pct"].mean())
    passed = mean >= args.threshold or mean >= control - 5.0
    out = output_root_of(run_dir) / "vllm_parity"
    out.mkdir(parents=True, exist_ok=True)
    result.to_csv(out / f"{run_dir.name}.csv", index=False)
    save_json_atomic({"run": str(run_dir), "adapter": str(adapter), "prompts": len(result),
                      "tokens": args.tokens, "mean_agreement_pct": mean,
                      "identical": int((result["agreement_pct"] == 100).sum()),
                      "threshold_pct": args.threshold, "hf_control_pct": control,
                      "passed": passed,
                      "vllm_version": vllm_version(), "repetition_penalty": engine.repetition_penalty},
                     out / f"{run_dir.name}.json")
    print(result[["id", "agreement_pct", "hf_control_pct"]].round(1).to_string(index=False))
    print(f"\nvLLM vs HF (batch 1):      {mean:.1f}% "
          f"({(result['agreement_pct'] == 100).sum()}/{len(result)} identical)")
    print(f"HF batch 8 vs HF batch 1:  {control:.1f}% "
          f"({(result['hf_control_pct'] == 100).sum()}/{len(result)} identical)")
    print("PARITY PASSED" if passed else
          "PARITY FAILED: send me the CSV; fallback is generation.engine = \"hf\"")


if __name__ == "__main__":
    main()
