#!/bin/bash
#SBATCH --job-name=ke-question-leak
#SBATCH --partition=gpu
# vLLM relies on the sm120-built flashinfer -> rtx5090 (tujestpolin) only.
#SBATCH --gres=gpu:rtx5090:1
#SBATCH --cpus-per-task=8
#SBATCH --time=04:00:00
#SBATCH --output=/home/imasoudian/cartridges/knowledge-editing/slurm/logs/%j.log
# NOTE: no --mem on this cluster (nodes have RealMemory=1; any memory request fails).
#
# Direct vs indirect leakage of the new target in the portability TRAINING questions:
# the DCT seed vs the enriched seed before it, judged by judge v2's question rubric and
# checked against the 90 hand labels first.

set -euo pipefail
REPO=/home/imasoudian/cartridges
JUDGE_MODEL=${JUDGE_MODEL:-Qwen/Qwen3.5-9B}
JUDGE_PORT=${JUDGE_PORT:-10322}
JUDGE_URL=http://localhost:$JUDGE_PORT
OUT=$REPO/results/question-leakage-$(date +%Y%m%d_%H%M%S)

$REPO/.venv-vllm/bin/vllm serve "$JUDGE_MODEL" \
    --served-model-name "$JUDGE_MODEL" \
    --port "$JUDGE_PORT" \
    --max-model-len 8192 \
    --gpu-memory-utilization 0.90 &
VLLM_PID=$!
trap 'kill $VLLM_PID 2>/dev/null || true' EXIT
for i in $(seq 1 180); do
    curl -sf $JUDGE_URL/health >/dev/null && break
    kill -0 $VLLM_PID 2>/dev/null || { echo "FATAL: vLLM exited during startup"; exit 1; }
    sleep 5
done
curl -sf $JUDGE_URL/health >/dev/null || { echo "FATAL: judge never became healthy"; exit 1; }

cd $REPO/knowledge-editing
$REPO/.venv/bin/python experiments/audit_question_leakage.py --judge-url $JUDGE_URL --judge-model "$JUDGE_MODEL" \
    --dataset DCT=$REPO/outputs/2026-08-11-22-26-45-synthesize/CounterFact-SelfStudy-DCT-0/artifact/dataset_final.parquet \
    --dataset enriched=$REPO/outputs/2026-07-27-20-11-50-synthesize/CounterFact-SelfStudy-Enriched-0/artifact/dataset_final.parquet \
    --out-dir "$OUT"
