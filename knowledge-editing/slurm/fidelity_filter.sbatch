#!/bin/bash
#SBATCH --job-name=ke-fidelity
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=08:00:00
#SBATCH --output=/home/imasoudian/cartridges/knowledge-editing/slurm/logs/%j.log
# NOTE: no --mem on this cluster (nodes have RealMemory=1; any memory request fails).
#
# Edit-fidelity QC: boots the vLLM judge, gates on calibration cases, then
# judges every row of the dataset and writes dataset_final.parquet.
# Override the input via:
#   sbatch --export=ALL,INPUT=<path/to/dataset_filtered.parquet> fidelity_filter.sbatch

set -euo pipefail

REPO=/home/imasoudian/cartridges
KE=$REPO/knowledge-editing
JUDGE_PORT=10310
JUDGE_URL=http://localhost:$JUDGE_PORT
INPUT=${INPUT:-$REPO/outputs/2026-07-15-19-14-08-synthesize/CleanDatasetForQwen2.5-0/artifact/dataset_filtered.parquet}

$REPO/.venv-vllm/bin/vllm serve Qwen/Qwen3-4b \
    --served-model-name Qwen/Qwen3-4b \
    --port $JUDGE_PORT \
    --max-model-len 8192 \
    --gpu-memory-utilization 0.90 &
VLLM_PID=$!
trap 'kill $VLLM_PID 2>/dev/null || true' EXIT

for i in $(seq 1 120); do
    curl -sf $JUDGE_URL/health >/dev/null && break
    kill -0 $VLLM_PID 2>/dev/null || { echo "FATAL: vLLM exited during startup"; exit 1; }
    sleep 5
done
curl -sf $JUDGE_URL/health >/dev/null || { echo "FATAL: judge never became healthy"; exit 1; }
echo "Judge is up."

cd $KE

# Gate on known-answer calibration (set -e aborts on a miss).
$REPO/.venv/bin/python fidelity_filter.py --calibrate --judge-base-url $JUDGE_URL
echo "Calibration PASSED - judging full dataset."

$REPO/.venv/bin/python fidelity_filter.py --input "$INPUT" --judge-base-url $JUDGE_URL
echo "Fidelity filter finished."
