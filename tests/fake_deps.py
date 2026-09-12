"""Import mfr_dpo without torch/transformers/peft, so the replay training loop can be tested on CPU."""
import sys, types, contextlib


def install():
    class _Opt:
        def __init__(self, params, lr=0.0): self.params, self.lr, self.steps = params, lr, 0
        def zero_grad(self): pass
        def step(self): self.steps += 1
    cuda = types.SimpleNamespace(is_available=lambda: False, is_bf16_supported=lambda: False,
                                 reset_peak_memory_stats=lambda: None, max_memory_allocated=lambda: 0,
                                 manual_seed_all=lambda s: None, get_device_name=lambda i: "fake")
    torch = types.ModuleType("torch")
    torch.optim = types.SimpleNamespace(AdamW=_Opt)
    torch.nn = types.SimpleNamespace(utils=types.SimpleNamespace(clip_grad_norm_=lambda p, m: None),
                                     functional=types.ModuleType("torch.nn.functional"))
    torch.cuda = cuda
    torch.manual_seed = lambda s: None
    class _NoGrad(contextlib.ContextDecorator):      # usable as both `with torch.no_grad():` and `@torch.no_grad()`
        def __enter__(self): return self
        def __exit__(self, *exc): return False
    torch.no_grad = _NoGrad
    torch.bfloat16 = torch.float16 = "dtype"
    torch.zeros = lambda *a, **k: None
    torch.tensor = lambda *a, **k: None
    sys.modules["torch"] = torch
    sys.modules["torch.nn"] = types.ModuleType("torch.nn")
    sys.modules["torch.nn.functional"] = torch.nn.functional
    sys.modules["peft"] = types.SimpleNamespace(LoraConfig=object, PeftModel=object,
                                                get_peft_model=lambda *a, **k: None,
                                                prepare_model_for_kbit_training=lambda *a, **k: None)
    sys.modules["transformers"] = types.SimpleNamespace(
        AutoModelForCausalLM=object, AutoTokenizer=object, BitsAndBytesConfig=object,
        get_linear_schedule_with_warmup=lambda opt, warm, total: types.SimpleNamespace(step=lambda: None))
    try:
        import tqdm.auto  # noqa: F401
    except ImportError:
        fake_tqdm = types.ModuleType("tqdm.auto")
        class _Bar:
            def __init__(self, it, **k): self.it = it
            def __iter__(self): return iter(self.it)
            def set_postfix(self, **k): pass
        fake_tqdm.tqdm = _Bar
        sys.modules["tqdm"] = types.ModuleType("tqdm")
        sys.modules["tqdm.auto"] = fake_tqdm
