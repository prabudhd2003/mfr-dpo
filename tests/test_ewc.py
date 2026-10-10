"""LoRA-EWC math and persistence smoke test in a clean real-Torch subprocess."""

import os
import subprocess
import sys
from pathlib import Path


def test_ewc_penalty_gradient_and_round_trip(tmp_path):
    root = Path(__file__).resolve().parents[1]
    state_path = tmp_path / "ewc_states.pt"
    code = f"""
import sys
sys.path.insert(0, {str(root / 'src')!r})
import torch
import types
sys.modules['peft'] = types.SimpleNamespace(
    LoraConfig=object, PeftModel=object, get_peft_model=lambda *a, **k: None,
    prepare_model_for_kbit_training=lambda *a, **k: None)
sys.modules['transformers'] = types.SimpleNamespace(
    AutoModelForCausalLM=object, AutoTokenizer=object, BitsAndBytesConfig=object,
    get_linear_schedule_with_warmup=lambda *a, **k: None)
import mfr_dpo

class Tiny(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.lora_A_weight = torch.nn.Parameter(torch.tensor([2.0]))

model = Tiny()
states = [{{
    'anchor': {{'lora_A_weight': torch.tensor([1.0])}},
    'fisher': {{'lora_A_weight': torch.tensor([3.0])}},
    'pairs': 500,
    'batch_size': 1,
}}]
penalty = mfr_dpo._ewc_regularizer(model, states)
assert torch.allclose(penalty, torch.tensor(1.5))
penalty.backward()
assert torch.allclose(model.lora_A_weight.grad, torch.tensor([3.0]))
mfr_dpo.save_ewc_states(states, {str(state_path)!r})
loaded = mfr_dpo.load_ewc_states({str(state_path)!r}, model)
assert loaded[0]['pairs'] == 500 and loaded[0]['batch_size'] == 1
assert torch.equal(loaded[0]['anchor']['lora_A_weight'], torch.tensor([1.0]))
assert torch.equal(loaded[0]['fisher']['lora_A_weight'], torch.tensor([3.0]))
"""
    # CARC login nodes limit how many threads a process may create.  NumPy/
    # OpenBLAS can otherwise start a large thread pool while importing Torch,
    # causing this tiny subprocess smoke test to fail before it reaches EWC.
    env = os.environ.copy()
    env.update({
        "OPENBLAS_NUM_THREADS": "1",
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "NUMEXPR_NUM_THREADS": "1",
        "VECLIB_MAXIMUM_THREADS": "1",
        "BLIS_NUM_THREADS": "1",
        "MALLOC_CONF": "background_thread:false",
    })
    subprocess.run([sys.executable, "-c", code], check=True, cwd=root, env=env)
