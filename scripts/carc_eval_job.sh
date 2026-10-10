#!/bin/bash
set -euo pipefail

ACTION="$1"
REPO_DIR="$2"
OUTPUT_DIR="$3"
CONDA_ENV="$4"
EXPECTED_GPU="$5"
shift 5

module purge
module load conda
eval "$(conda shell.bash hook)"
conda activate "$CONDA_ENV"

PYTHON_BIN="$CONDA_ENV/bin/python"
if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "Python executable not found: $PYTHON_BIN" >&2
  exit 1
fi
export HF_HOME="$OUTPUT_DIR/huggingface"
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
mkdir -p "$HF_HOME" "$OUTPUT_DIR/logs/evaluation"
cd "$REPO_DIR"

"$PYTHON_BIN" - "$EXPECTED_GPU" <<'PY'
import sys
import torch
if not torch.cuda.is_available():
    raise RuntimeError("evaluation job started without a visible CUDA GPU")
name = torch.cuda.get_device_name(0)
if sys.argv[1].upper() not in name.upper():
    raise RuntimeError(f"Expected {sys.argv[1]}, received {name}")
print(f"GPU: {name}; PyTorch: {torch.__version__}; CUDA: {torch.version.cuda}", flush=True)
PY

# Every action forwards its remaining arguments to the Python script unchanged; the submit helper
# (scripts/submit_eval_carc.py) validates them first.
case "$ACTION" in
  generate)
    "$PYTHON_BIN" -u scripts/generate_final_responses.py "$@" --confirm-final-evaluation
    ;;
  safety)
    "$PYTHON_BIN" -u scripts/run_safety_eval.py "$@"
    ;;
  ifeval)
    "$PYTHON_BIN" -u scripts/run_ifeval.py "$@" --confirm-final-evaluation
    ;;
  judge)
    "$PYTHON_BIN" -u scripts/run_pairwise_judge.py "$@"
    ;;
  validate-judge)
    "$PYTHON_BIN" -u scripts/validate_judge.py --output-dir "$OUTPUT_DIR" "$@"
    ;;
  gradient-conflict)
    "$PYTHON_BIN" -u scripts/gradient_conflict.py "$@"
    ;;
  locked-test)
    "$PYTHON_BIN" -u scripts/final_test.py "$@" --confirm-final-evaluation
    ;;
  *)
    echo "Unknown evaluation action: $ACTION" >&2
    exit 2
    ;;
esac
