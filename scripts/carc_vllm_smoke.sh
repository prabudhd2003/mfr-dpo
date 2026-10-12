#!/bin/bash
#SBATCH --job-name=vllm-smoke
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gpus-per-task=l40s:1
#SBATCH --time=02:00:00
#SBATCH --output=vllm_smoke_%j.out
# End-to-end test of the vLLM evaluation path on one finished run. Writes only to
# generation/smoke/, final_eval/ifeval_smoke/, judge_validation_smoke/ and vllm_parity/.
# Usage: sbatch --account=xiangren_1987 scripts/carc_vllm_smoke.sh RUN_DIR OUTPUT_DIR [PROTOCOL]
# The log lands in the current folder as vllm_smoke_<job>.out (ignored by git).
set -euo pipefail
RUN="$1"
OUT="$2"
PROTOCOL="${3:-configs/experiment_protocol.json}"
REPO=/project2/xiangren_1987/grp26-mfr-dpo
module purge
module load conda
eval "$(conda shell.bash hook)"
ENV="$REPO/.conda/envs/mfr-eval"
conda activate "$ENV"
# Batch jobs on CARC do not reliably put the env first on PATH (see carc_eval_job.sh), so call the
# env's own python and put its bin/ first for lm-eval and vLLM worker processes.
export PATH="$ENV/bin:$PATH"
PY="$ENV/bin/python"
# vLLM: use PyTorch top-k/top-p instead of FlashInfer's sampler, which JIT-compiles CUDA code and
# needs nvcc (CARC GPU nodes have no /usr/local/cuda). If another FlashInfer kernel still needs nvcc,
# point CUDA_HOME at the nvcc that pip installed into the env.
export VLLM_USE_FLASHINFER_SAMPLER=0
if [[ -z "${CUDA_HOME:-}" ]]; then
  NVCC_BIN=$(find "$ENV/lib" -path '*/site-packages/nvidia/*' -name nvcc -type f 2>/dev/null | head -1 || true)
  if [[ -n "$NVCC_BIN" ]]; then
    export CUDA_HOME="$(dirname "$(dirname "$NVCC_BIN")")"
    export PATH="$(dirname "$NVCC_BIN"):$PATH"
  fi
fi
export HF_HOME="$OUT/huggingface" PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
export VLLM_WORKER_MULTIPROC_METHOD=spawn
cd "$REPO"
step() { echo; echo "=================== $1"; }

step "0/6 versions"
nvidia-smi --query-gpu=name,driver_version --format=csv
"$PY" -c "import torch, vllm, transformers, peft, bitsandbytes; print('torch', torch.__version__, 'cuda', torch.version.cuda, '| vllm', vllm.__version__, '| transformers', transformers.__version__, '| peft', peft.__version__, '| bnb', bitsandbytes.__version__)"
"$ENV/bin/lm-eval" --help > /dev/null && echo "lm-eval OK"
echo "CUDA_HOME=${CUDA_HOME:-unset}  VLLM_USE_FLASHINFER_SAMPLER=$VLLM_USE_FLASHINFER_SAMPLER"

step "1/6 build the vLLM base (skips if built)"
"$PY" scripts/prepare_vllm_base.py --output-dir "$OUT" --protocol "$PROTOCOL"

step "2/6 parity: HF 4-bit vs vLLM (21 val prompts, 64 tokens)"
"$PY" scripts/check_vllm_parity.py --run-dir "$RUN" --protocol "$PROTOCOL"

step "3/6 generation, 8 prompts per suite -> generation/smoke/"
"$PY" scripts/generate_final_responses.py --run-dir "$RUN" --protocol "$PROTOCOL" --smoke 8 \
  --confirm-final-evaluation --overwrite
head -c 600 "$RUN/generation/smoke/safe_test.jsonl"; echo

step "4/6 WildGuard on the smoke generations"
"$PY" scripts/run_safety_eval.py --run-dir "$RUN" --smoke --overwrite

step "5/6 Prometheus on 10 validation pairs per behavior"
"$PY" scripts/validate_judge.py --output-dir "$OUT" --limit 10 --overwrite

step "6/6 IFEval, 10 prompts"
"$PY" scripts/run_ifeval.py --run-dir "$RUN" --limit 10 --confirm-final-evaluation --overwrite

echo; echo "SMOKE TEST PASSED (send me this log)"
