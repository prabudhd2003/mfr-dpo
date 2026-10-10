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
  echo "Python executable not found in Conda environment: $PYTHON_BIN" >&2
  exit 1
fi
echo "Python: $PYTHON_BIN"

mkdir -p "$OUTPUT_DIR/cache" "$OUTPUT_DIR/logs" "$OUTPUT_DIR/runs" "$OUTPUT_DIR/huggingface"
export HF_HOME="$OUTPUT_DIR/huggingface"
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

cd "$REPO_DIR"

"$PYTHON_BIN" - "$EXPECTED_GPU" <<'PY'
import sys
import torch
if not torch.cuda.is_available():
    raise RuntimeError("Slurm job started without a visible CUDA GPU")
name = torch.cuda.get_device_name(0)
expected = sys.argv[1].upper()
if expected not in name.upper():
    raise RuntimeError(f"Expected a {expected} GPU, received {name}")
print(f"GPU: {name}", flush=True)
print(f"PyTorch: {torch.__version__}; CUDA runtime: {torch.version.cuda}", flush=True)
PY

CACHE_PATH="$OUTPUT_DIR/cache/reference_v2.csv"

case "$ACTION" in
  cache)
    if [[ -s "$CACHE_PATH" && -s "${CACHE_PATH%.csv}.manifest.json" ]]; then
      echo "Reference cache already exists; leaving it unchanged: $CACHE_PATH"
    else
      "$PYTHON_BIN" -u scripts/build_reference_cache.py --output "$CACHE_PATH" --batch-size 4
    fi
    ;;
  group)
    ORDER="$1"
    SEED="$2"
    METHODS="$3"
    "$PYTHON_BIN" -u scripts/run_experiment_group.py \
      --output-dir "$OUTPUT_DIR" \
      --order "$ORDER" \
      --seed "$SEED" \
      --methods "$METHODS" \
      --reference-cache "$CACHE_PATH"
    ;;
  joint)
    SEED="$1"
    "$PYTHON_BIN" -u scripts/run_joint_training.py \
      --output-dir "$OUTPUT_DIR" \
      --seed "$SEED" \
      --reference-cache "$CACHE_PATH"
    ;;
  *)
    echo "Unknown CARC action: $ACTION" >&2
    exit 2
    ;;
esac
