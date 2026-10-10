#!/usr/bin/env python3
"""Build the vLLM policy base once per model: the training NF4 weights dequantized to bf16.

Loads the base model with exactly the training BitsAndBytes config (NF4, double quantization,
bf16 compute), dequantizes every 4-bit linear layer, writes those weights into a bf16 copy of
the model and saves it with the date-pinned tokenizer to OUTPUT/vllm_base/<model>/.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mfr_utils import file_sha256, load_protocol, pin_chat_template_date, save_json_atomic
from mfr_vllm import vllm_base_dir


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True, help="Artifact root (MFR_OUTPUT_DIR or LLAMA_OUT)")
    parser.add_argument("--protocol", default=str(ROOT / "configs" / "experiment_protocol.json"))
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    protocol_path = Path(args.protocol).expanduser().resolve()
    protocol = load_protocol(protocol_path)
    name, revision = protocol["model_name"], protocol["model_revision"]
    out = vllm_base_dir(Path(args.output_dir).expanduser().resolve(), name)
    if (out / "vllm_base_manifest.json").exists() and not args.overwrite:
        print(f"vLLM base already built: {out}")
        return

    import bitsandbytes as bnb
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    print(f"Loading {name}@{revision} in bf16 (CPU) and NF4 (GPU)...", flush=True)
    base = AutoModelForCausalLM.from_pretrained(name, revision=revision, dtype=torch.bfloat16)
    quantized = AutoModelForCausalLM.from_pretrained(
        name, revision=revision, device_map={"": 0},
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True),
    )
    params = dict(base.named_parameters())
    replaced = 0
    with torch.no_grad():
        for module_name, module in quantized.named_modules():
            if not isinstance(module, bnb.nn.Linear4bit):
                continue
            weight = bnb.functional.dequantize_4bit(module.weight.data, module.weight.quant_state)
            target = params[f"{module_name}.weight"]
            if tuple(weight.shape) != tuple(target.shape):
                raise ValueError(f"{module_name}: {tuple(weight.shape)} vs {tuple(target.shape)}")
            target.copy_(weight.to(torch.bfloat16).cpu())
            replaced += 1
    if not replaced:
        raise RuntimeError("no 4-bit layers found; the quantized load did not work")
    out.mkdir(parents=True, exist_ok=True)
    base.save_pretrained(out, safe_serialization=True)
    tokenizer = AutoTokenizer.from_pretrained(name, revision=revision)
    pin_chat_template_date(tokenizer)
    tokenizer.save_pretrained(out)
    save_json_atomic({
        "model_name": name, "model_revision": revision,
        "weights": "NF4 + double quantization (training config), dequantized to bf16",
        "dequantized_linear_layers": replaced,
        "protocol_sha256": file_sha256(protocol_path),
        "transformers": __import__("transformers").__version__,
        "bitsandbytes": bnb.__version__,
    }, out / "vllm_base_manifest.json")
    print(f"Saved vLLM base ({replaced} dequantized layers) to {out}", flush=True)


if __name__ == "__main__":
    main()
