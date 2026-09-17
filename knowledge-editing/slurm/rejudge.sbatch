#!/bin/bash
#SBATCH --job-name=ke-rejudge
#SBATCH --partition=gpu
# vLLM relies on the sm120-built flashinfer -> rtx5090 (tujestpolin) only.
#SBATCH --gres=gpu:rtx5090:1
#SBATCH --cpus-per-task=8
#SBATCH --time=08:00:00
#SBATCH --output=/home/imasoudian/cartridges/knowledge-editing/slurm/logs/%j.log
# NOTE: no --mem on this cluster (nodes have RealMemory=1; any memory request fails).
#
# Judge v2 over STORED eval answers (no student model, no regeneration): serve the judge,
# run the v2 calibration set, then re-judge every run in RUNS (paths relative to the repo).
#   sbatch --export=ALL,JUDGE_MODEL=Qwen/Qwen3.5-9B,TAG=qwen35-9b,RUNS="results/a results/b" rejudge.sbatch
# Writes <run>/rejudge-v2-<TAG>/eval_detailed.csv, readable by compare_evals.py.
# A failed calibration stops the job unless CALIBRATION_STRICT=0 (diagnosis runs).
# ONLY_TYPES=Locality re-judges only those rows inside existing rejudge dirs.

set -euo pipefail

REPO=/home/imasoudian/cartridges
JUDGE_MODEL=${JUDGE_MODEL:-Qwen/Qwen3-4b}
JUDGE_PORT=${JUDGE_PORT:-10320}
JUDGE_URL=http://localhost:$JUDGE_PORT
TAG=${TAG:?set TAG, e.g. qwen3-4b}
RUNS=${RUNS:?set RUNS to space-separated results dirs relative to the repo}
echo "judge $JUDGE_MODEL  tag $TAG  port $JUDGE_PORT  node $(hostname)"

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
echo "Judge is up."

cd $REPO/knowledge-editing
if ! $REPO/.venv/bin/python rejudge.py --judge-url $JUDGE_URL --judge-model "$JUDGE_MODEL" --calibrate; then
    if [ "${CALIBRATION_STRICT:-1}" = 1 ]; then
        echo "Calibration FAILED - not re-judging (CALIBRATION_STRICT=0 forces it)"
        exit 1
    fi
    echo "Calibration FAILED - re-judging anyway (CALIBRATION_STRICT=0)"
fi

RUN_ARGS=()
for r in $RUNS; do RUN_ARGS+=("$REPO/$r"); done
$REPO/.venv/bin/python rejudge.py --judge-url $JUDGE_URL --judge-model "$JUDGE_MODEL" \
    --tag "$TAG" ${ONLY_TYPES:+--only-types "$ONLY_TYPES"} "${RUN_ARGS[@]}"
echo "Rejudge finished."
