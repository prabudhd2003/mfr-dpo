"""vLLM engines for the generation evaluation (plan §5).

Policy models: vLLM serves the training base model with its NF4 weights dequantized once to bf16
(scripts/prepare_vllm_base.py) plus each run's LoRA adapter. QLoRA training dequantizes the 4-bit
weights to bf16 before every matmul, so this reproduces the trained model without bitsandbytes
inside vLLM. scripts/check_vllm_parity.py compares it against the Hugging Face path.

Evaluators (WildGuard, Prometheus): the official bf16 weights at the pinned revision.
Decoding is greedy; the repetition penalty is the model's own generation_config value, which is
what Hugging Face generate() applied in the old path.
"""

from __future__ import annotations

import json
from pathlib import Path

from mfr_utils import pin_chat_template_date


def model_slug(model_name):
    return model_name.replace("/", "__")


def vllm_base_dir(output_root, model_name):
    return Path(output_root) / "vllm_base" / model_slug(model_name)


def output_root_of(run_dir):
    """OUTPUT/runs/X, OUTPUT/joint_runs/X and OUTPUT/base_model/X all live two levels down."""
    return Path(run_dir).resolve().parents[1]


def check_vllm_base(base_dir, model_name, revision):
    path = Path(base_dir) / "vllm_base_manifest.json"
    if not path.exists():
        raise FileNotFoundError(
            f"build the vLLM base first (submit_eval_carc.py vllm-base); missing {path}")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if (manifest["model_name"], manifest["model_revision"]) != (model_name, revision):
        raise ValueError(f"{path} was built for {manifest['model_name']}@{manifest['model_revision']}")
    return manifest


class Engine:
    """One vLLM engine; with lora_rank set it serves any number of adapters, one after another."""

    def __init__(self, model, max_model_len, seed=544, revision=None, lora_rank=None,
                 gpu_memory_utilization=0.85):
        from transformers import AutoTokenizer, GenerationConfig
        from vllm import LLM

        self.model = str(model)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model, revision=revision)
        pin_chat_template_date(self.tokenizer)
        try:
            penalty = GenerationConfig.from_pretrained(self.model, revision=revision).repetition_penalty
        except Exception:
            penalty = None
        self.repetition_penalty = float(penalty or 1.0)
        kwargs = {"model": self.model, "dtype": "bfloat16", "max_model_len": int(max_model_len),
                  "seed": int(seed), "gpu_memory_utilization": gpu_memory_utilization}
        if revision:
            kwargs.update(revision=revision, tokenizer_revision=revision)
        if lora_rank:
            kwargs.update(enable_lora=True, max_lora_rank=int(lora_rank), max_loras=1)
        self.llm = LLM(**kwargs)
        self._lora_ids = {}

    def lora(self, adapter):
        if adapter is None:
            return None
        from vllm.lora.request import LoRARequest
        key = str(Path(adapter).resolve())
        lora_id = self._lora_ids.setdefault(key, len(self._lora_ids) + 1)
        return LoRARequest(f"adapter{lora_id}", lora_id, key)

    def generate(self, raw_prompts, max_new_tokens, adapter=None, desc=None):
        """Greedy completions for fully rendered prompts (special tokens already in the text)."""
        from vllm import SamplingParams
        inputs = [{"prompt_token_ids": self.tokenizer(p, add_special_tokens=False)["input_ids"]}
                  for p in raw_prompts]
        params = SamplingParams(temperature=0.0, max_tokens=int(max_new_tokens),
                                repetition_penalty=self.repetition_penalty)
        if desc:
            print(f"{desc}: {len(inputs)} prompts", flush=True)
        outputs = self.llm.generate(inputs, params, lora_request=self.lora(adapter), use_tqdm=True)
        return [output.outputs[0].text.strip() for output in outputs]

    def chat(self, prompts, max_new_tokens, adapter=None, desc=None):
        """Policy generation: the same chat rendering as training (mfr_dpo.chat_prompt)."""
        from mfr_dpo import chat_prompt
        rendered = [chat_prompt(self.tokenizer, p) for p in prompts]
        return self.generate(rendered, max_new_tokens, adapter, desc)


def evaluator(config, seed=544):
    """Return generate(prompts, desc) -> texts for WildGuard or Prometheus, per config['engine']."""
    if config.get("engine", "vllm") == "vllm":
        engine = Engine(config["model_name"], config.get("max_model_len", 8192), seed,
                        revision=config["model_revision"])
        return lambda prompts, desc: engine.generate(prompts, config["max_new_tokens"], desc=desc)
    from mfr_generation_eval import generate_classifier_text, load_quantized_model
    model, tokenizer = load_quantized_model(config["model_name"], config["model_revision"])
    return lambda prompts, desc: generate_classifier_text(
        model, tokenizer, prompts, config["batch_size"], config["max_new_tokens"], desc,
        raw_prompts=True)


def vllm_version():
    try:
        import vllm
        return vllm.__version__
    except Exception:
        return None
