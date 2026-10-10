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
set -euo pipefail
RUN="$1"
OUT="$2"
PROTOCOL="${3:-configs/experiment_protocol.json}"
REPO=/project2/xiangren_1987/grp26-mfr-dpo
module purge
module load conda
eval "$(conda shell.bash hook)"
conda activate "$REPO/.conda/envs/mfr-eval"
export HF_HOME="$OUT/huggingface" PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
export VLLM_WORKER_MULTIPROC_METHOD=spawn
cd "$REPO"
step() { echo; echo "=================== $1"; }

step "0/6 versions"
nvidia-smi --query-gpu=name,driver_version --format=csv
python -c "import torch, vllm, transformers, peft, bitsandbytes; print('torch', torch.__version__, 'cuda', torch.version.cuda, '| vllm', vllm.__version__, '| transformers', transformers.__version__, '| peft', peft.__version__, '| bnb', bitsandbytes.__version__)"
lm-eval --help > /dev/null && echo "lm-eval OK"

step "1/6 build the vLLM base (skips if built)"
python scripts/prepare_vllm_base.py --output-dir "$OUT" --protocol "$PROTOCOL"

step "2/6 parity: HF 4-bit vs vLLM (21 val prompts, 64 tokens)"
python scripts/check_vllm_parity.py --run-dir "$RUN" --protocol "$PROTOCOL"

step "3/6 generation, 8 prompts per suite -> generation/smoke/"
python scripts/generate_final_responses.py --run-dir "$RUN" --protocol "$PROTOCOL" --smoke 8 \
  --confirm-final-evaluation --overwrite
head -c 600 "$RUN/generation/smoke/safe_test.jsonl"; echo

step "4/6 WildGuard on the smoke generations"
python scripts/run_safety_eval.py --run-dir "$RUN" --smoke --overwrite

step "5/6 Prometheus on 10 validation pairs per behavior"
python scripts/validate_judge.py --output-dir "$OUT" --limit 10 --overwrite

step "6/6 IFEval, 10 prompts"
python scripts/run_ifeval.py --run-dir "$RUN" --limit 10 --confirm-final-evaluation --overwrite

echo; echo "SMOKE TEST PASSED (send me this log)"
