#!/bin/bash
set -euo pipefail

ACTION="$1"
REPO_DIR="$2"
OUTPUT_DIR="$3"
CONDA_ENV="$4"
shift 4

module purge
module load conda
eval "$(conda shell.bash hook)"
conda activate "$CONDA_ENV"

mkdir -p "$OUTPUT_DIR/cache" "$OUTPUT_DIR/logs" "$OUTPUT_DIR/runs" "$OUTPUT_DIR/huggingface"
export HF_HOME="$OUTPUT_DIR/huggingface"
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

cd "$REPO_DIR"

python - <<'PY'
import torch
if not torch.cuda.is_available():
    raise RuntimeError("Slurm job started without a visible CUDA GPU")
name = torch.cuda.get_device_name(0)
if "A100" not in name:
    raise RuntimeError(f"Expected an A100 GPU, received {name}")
print(f"GPU: {name}", flush=True)
print(f"PyTorch: {torch.__version__}; CUDA runtime: {torch.version.cuda}", flush=True)
PY

CACHE_PATH="$OUTPUT_DIR/cache/reference_v2.csv"

case "$ACTION" in
  cache)
    if [[ -s "$CACHE_PATH" && -s "${CACHE_PATH%.csv}.manifest.json" ]]; then
      echo "Reference cache already exists; leaving it unchanged: $CACHE_PATH"
    else
      python -u scripts/build_reference_cache.py --output "$CACHE_PATH" --batch-size 4
    fi
    ;;
  group)
    ORDER="$1"
    SEED="$2"
    METHODS="$3"
    python -u scripts/run_experiment_group.py \
      --output-dir "$OUTPUT_DIR" \
      --order "$ORDER" \
      --seed "$SEED" \
      --methods "$METHODS" \
      --reference-cache "$CACHE_PATH"
    ;;
  *)
    echo "Unknown CARC action: $ACTION" >&2
    exit 2
    ;;
esac
