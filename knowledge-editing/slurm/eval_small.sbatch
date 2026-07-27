#!/bin/bash
#SBATCH --job-name=ke-eval-small
#SBATCH --partition=gpu
# vLLM judge + eval rely on the sm120-built flashinfer → must run on tujestpolin
# (only rtx5090 node). Pin the 5090 gres so Slurm can't use sarmatia.
#SBATCH --gres=gpu:rtx5090:2
#SBATCH --cpus-per-task=8
#SBATCH --time=04:00:00
#SBATCH --output=/home/imasoudian/cartridges/knowledge-editing/slurm/logs/%j.log
# NOTE: no --mem on this cluster (nodes have RealMemory=1; any memory request fails).
#
# End-to-end small eval: vLLM judge (GPU 0) + lora_eval.py (GPU 1) in one job,
# so the judge is reachable via localhost and both GPUs are freed together.
# Override the adapter under test and/or the eval set via:
#   sbatch --export=ALL,LORA_DIR=<path/to/adapter>,DATA_FILE=samples/CounterFact.json eval_small.sbatch
# (full CounterFact = 975 entries * 7 tests; submit with a higher --time than the 4h default)

set -euo pipefail

REPO=/home/imasoudian/cartridges
KE=$REPO/knowledge-editing
JUDGE_PORT=10310
JUDGE_URL=http://localhost:$JUDGE_PORT
LORA_DIR=${LORA_DIR:-$REPO/checkpoints/lora_qwen2.5-7B-Instruct}
DATA_FILE=${DATA_FILE:-samples/dev_small.json}

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

echo "Evaluating adapter: $LORA_DIR on $DATA_FILE"
CUDA_VISIBLE_DEVICES=$EVAL_GPU $REPO/.venv/bin/python lora_eval.py \
    --lora-dir "$LORA_DIR" \
    --base-model Qwen/Qwen2.5-7B-Instruct \
    --data-file "$DATA_FILE" \
    --judge-base-url $JUDGE_URL

echo "Eval finished."
