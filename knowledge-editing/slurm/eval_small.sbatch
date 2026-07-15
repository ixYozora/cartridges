#!/bin/bash
#SBATCH --job-name=ke-eval-small
#SBATCH --partition=gpu
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=8
#SBATCH --time=04:00:00
#SBATCH --output=/home/imasoudian/cartridges/knowledge-editing/slurm/logs/%j.log
# NOTE: no --mem on this cluster (nodes have RealMemory=1; any memory request fails).
#
# End-to-end small eval: vLLM judge (GPU 0) + lora_eval.py (GPU 1) in one job,
# so the judge is reachable via localhost and both GPUs are freed together.

set -euo pipefail

REPO=/home/imasoudian/cartridges
KE=$REPO/knowledge-editing
JUDGE_PORT=10310
JUDGE_URL=http://localhost:$JUDGE_PORT

# Slurm hands us 2 GPUs; split them between judge and student model.
ALL_GPUS=${CUDA_VISIBLE_DEVICES:-0,1}
JUDGE_GPU=${ALL_GPUS%%,*}
EVAL_GPU=${ALL_GPUS##*,}
echo "GPUs: judge=$JUDGE_GPU eval=$EVAL_GPU (node $(hostname))"

CUDA_VISIBLE_DEVICES=$JUDGE_GPU $REPO/.venv-vllm/bin/vllm serve Qwen/Qwen3-4b \
    --served-model-name Qwen/Qwen3-4b \
    --port $JUDGE_PORT \
    --max-model-len 8192 \
    --gpu-memory-utilization 0.90 &
VLLM_PID=$!
trap 'kill $VLLM_PID 2>/dev/null || true' EXIT

# Wait for the judge (cold start incl. compile can take a few minutes).
for i in $(seq 1 120); do
    curl -sf $JUDGE_URL/health >/dev/null && break
    kill -0 $VLLM_PID 2>/dev/null || { echo "FATAL: vLLM exited during startup"; exit 1; }
    sleep 5
done
curl -sf $JUDGE_URL/health >/dev/null || { echo "FATAL: judge never became healthy"; exit 1; }
echo "Judge is up."

cd $KE

# Gate the eval on judge calibration (set -e aborts on any nonzero exit).
$REPO/.venv/bin/python judge_smoke_test.py --url $JUDGE_URL
echo "Smoke test PASSED — starting eval."

CUDA_VISIBLE_DEVICES=$EVAL_GPU $REPO/.venv/bin/python lora_eval.py \
    --lora-dir $REPO/checkpoints/lora_qwen2.5-7B-Instruct \
    --base-model Qwen/Qwen2.5-7B-Instruct \
    --data-file samples/dev_small.json \
    --judge-base-url $JUDGE_URL

echo "Eval finished."
