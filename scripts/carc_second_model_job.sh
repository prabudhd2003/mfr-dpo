#!/bin/bash
set -euo pipefail

ACTION="$1"
REPO_DIR="$2"
OUTPUT_DIR="$3"
CONDA_ENV="$4"
EXPECTED_GPU="$5"
PROTOCOL="$6"
shift 6

module purge
module load conda
eval "$(conda shell.bash hook)"
conda activate "$CONDA_ENV"
PYTHON_BIN="$CONDA_ENV/bin/python"
export HF_HOME="$OUTPUT_DIR/huggingface"
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
mkdir -p "$OUTPUT_DIR/cache" "$OUTPUT_DIR/logs" "$OUTPUT_DIR/runs" "$HF_HOME"
cd "$REPO_DIR"

"$PYTHON_BIN" - "$EXPECTED_GPU" <<'PY'
import sys
import torch
if not torch.cuda.is_available():
    raise RuntimeError("second-model job started without a visible CUDA GPU")
name = torch.cuda.get_device_name(0)
if sys.argv[1].upper() not in name.upper():
    raise RuntimeError(f"Expected {sys.argv[1]}, received {name}")
print(f"GPU: {name}; PyTorch: {torch.__version__}; CUDA: {torch.version.cuda}", flush=True)
PY

CACHE_PATH="$OUTPUT_DIR/cache/reference_v2.csv"
case "$ACTION" in
  cache)
    "$PYTHON_BIN" -u scripts/validate_model_data.py \
      --protocol "$PROTOCOL" --output "$OUTPUT_DIR/cache/tokenizer_validation.json"
    if [[ -s "$CACHE_PATH" && -s "${CACHE_PATH%.csv}.manifest.json" ]]; then
      echo "Second-model reference cache already exists: $CACHE_PATH"
    else
      "$PYTHON_BIN" -u scripts/build_reference_cache.py \
        --protocol "$PROTOCOL" --output "$CACHE_PATH" --batch-size 4
    fi
    ;;
  group)
    ORDER="$1"
    SEED="$2"
    METHODS="$3"
    "$PYTHON_BIN" -u scripts/run_experiment_group.py \
      --protocol "$PROTOCOL" \
      --output-dir "$OUTPUT_DIR" \
      --order "$ORDER" \
      --seed "$SEED" \
      --methods "$METHODS" \
      --reference-cache "$CACHE_PATH"
    ;;
  joint)
    SEED="$1"
    "$PYTHON_BIN" -u scripts/run_joint_training.py \
      --protocol "$PROTOCOL" \
      --output-dir "$OUTPUT_DIR" \
      --seed "$SEED" \
      --reference-cache "$CACHE_PATH"
    ;;
  *)
    echo "Unknown second-model action: $ACTION" >&2
    exit 2
    ;;
esac
