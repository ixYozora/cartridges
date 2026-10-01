#!/bin/bash
#SBATCH --job-name=ke-rejudge-locality
#SBATCH --partition=gpu
# vLLM judge relies on the sm120-built flashinfer -> must run on tujestpolin (only
# rtx5090 node). Pin the 5090 gres so Slurm can't use sarmatia. Only 1 GPU is needed
# here: nothing generates, the responses are already on disk.
#SBATCH --gres=gpu:rtx5090:1
#SBATCH --cpus-per-task=8
#SBATCH --time=04:00:00
#SBATCH --output=/home/imasoudian/cartridges/knowledge-editing/slurm/logs/%j.log
# NOTE: no --mem on this cluster (nodes have RealMemory=1; any memory request fails).
#
# HOW MUCH OF THE MEASURED LOCALITY BLEED IS A SUBSTRING ARTIFACT?
#
# Job 12221 scored neighbour responses by substring match and found bleed flat across n
# (~8.6% at n=975 against a 3.7% stock floor). That check mis-fires both ways: an edit
# old 'soccer' -> new 'football' makes a neighbour answering "footballer" look like
# leakage, while an alias or adjective form of the edited value gets missed.
#
# This re-grades with judge v2 (extraction first, success derived in code) over two
# strata -- every flagged response, plus a random sample of the rest -- and reweights
# them, so the corrected rate accounts for false positives AND false negatives.
#
#   sbatch --export=ALL,RUN=results/locality-scaling-20260922_005043 rejudge_locality.sbatch

set -euo pipefail
REPO=/home/imasoudian/cartridges
KE=$REPO/knowledge-editing

# RUN_NEWEST resolves the newest results/<prefix>-* directory at RUN TIME, which is how
# this can be chained with --dependency onto a job whose output dir is datetime-stamped
# and therefore unknowable at submit time.
if [ -n "${RUN_NEWEST:-}" ]; then
    RUN=$(ls -dt $REPO/results/${RUN_NEWEST}-*/ 2>/dev/null | head -1)
    [ -n "$RUN" ] || { echo "FATAL: no results/${RUN_NEWEST}-* directory"; exit 1; }
    echo "RUN_NEWEST=${RUN_NEWEST} resolved to $RUN"
fi
RUN=${RUN:?set RUN=results/<dir> or RUN_NEWEST=<prefix>}
JUDGE_PORT=${JUDGE_PORT:-10320}
JUDGE_URL=http://localhost:$JUDGE_PORT
JUDGE_MODEL=${JUDGE_MODEL:-Qwen/Qwen3.5-9B}
SAMPLE=${SAMPLE:-400}

VLLM_PID=
trap 'if [ -n "$VLLM_PID" ]; then kill $VLLM_PID 2>/dev/null || true; fi' EXIT

$REPO/.venv-vllm/bin/vllm serve "$JUDGE_MODEL" \
    --served-model-name "$JUDGE_MODEL" \
    --port $JUDGE_PORT \
    --max-model-len 8192 \
    --gpu-memory-utilization 0.90 &
VLLM_PID=$!
for i in $(seq 1 180); do
    curl -sf $JUDGE_URL/health >/dev/null && break
    kill -0 $VLLM_PID 2>/dev/null || { echo "FATAL: vLLM exited during startup"; exit 1; }
    sleep 5
done
curl -sf $JUDGE_URL/health >/dev/null || { echo "FATAL: judge never became healthy"; exit 1; }
echo "Judge $JUDGE_MODEL is up on port $JUDGE_PORT."

cd $KE

# Same calibration gate the evals use: 39 hand-labelled cases, 16 of them critical.
# A judge that fails it cannot be trusted to settle an artifact question.
$REPO/.venv/bin/python rejudge.py --judge-url $JUDGE_URL --judge-model "$JUDGE_MODEL" --calibrate
echo "Judge gate PASSED — re-grading locality."

$REPO/.venv/bin/python experiments/rejudge_locality.py \
    --run "$RUN" \
    --judge-url $JUDGE_URL \
    --judge-model "$JUDGE_MODEL" \
    --sample "$SAMPLE"
echo "Done: $RUN/rejudge_v2.json"
