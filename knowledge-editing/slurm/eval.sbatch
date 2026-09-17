#!/bin/bash
#SBATCH --job-name=ke-eval
#SBATCH --partition=gpu
# vLLM judge + eval rely on the sm120-built flashinfer -> must run on tujestpolin
# (only rtx5090 node). Pin the 5090 gres so Slurm can't use sarmatia.
#SBATCH --gres=gpu:rtx5090:2
#SBATCH --cpus-per-task=8
#SBATCH --time=12:00:00
#SBATCH --output=/home/imasoudian/cartridges/knowledge-editing/slurm/logs/%j.log
# NOTE: no --mem on this cluster (nodes have RealMemory=1; any memory request fails).
#
# End-to-end eval in one 2-GPU job: a vLLM judge (GPU 0) + lora_eval.py (GPU 1).
# The judge is gated first (judge v2: the calibration set in rejudge.py --calibrate;
# judge v1: judge_smoke_test.py), then every answer is graded as it is generated.
#
#   sbatch --export=ALL,LORA_DIR=<path/to/adapter>,DATA_FILE=samples/CounterFact.json eval.sbatch
# LORA_DIR=none evaluates BASE_MODEL itself with no adapter (unedited floor, or a
# GROM-erased checkpoint on its own). Give each concurrent eval its own JUDGE_PORT:
# jobs on the same node would otherwise all start their judge on 10310.
# JUDGE_MODEL overrides the judge; JUDGE_VERSION=v1 restores the old 0-5 grading.
# Older runs are re-graded with judge v2 by slurm/rejudge.sbatch instead.

set -euo pipefail

REPO=/home/imasoudian/cartridges
KE=$REPO/knowledge-editing
JUDGE_PORT=${JUDGE_PORT:-10310}
JUDGE_URL=http://localhost:$JUDGE_PORT
LORA_DIR=${LORA_DIR:-$REPO/checkpoints/lora_qwen2.5-7B-Instruct}
# Must match the base the adapter was trained on (see lora_train_clean.sbatch).
BASE_MODEL=${BASE_MODEL:-Qwen/Qwen2.5-7B-Instruct}
DATA_FILE=${DATA_FILE:-samples/dev_small.json}
if [ "$LORA_DIR" = none ]; then LORA_ARGS=(); KIND=base; else LORA_ARGS=(--lora-dir "$LORA_DIR"); KIND=lora; fi
# judge v2 by default (v1's 0-5 score passed wrong answers; see CHANGELOG 2026-09-13).
# JUDGE_VERSION=v1 reproduces pre-2026-09-13 runs: older judge model, older gate.
JUDGE_VERSION=${JUDGE_VERSION:-v2}
if [ "$JUDGE_VERSION" = v2 ]; then
    JUDGE_MODEL=${JUDGE_MODEL:-Qwen/Qwen3.5-9B}
    GATE="rejudge.py --judge-url $JUDGE_URL --judge-model $JUDGE_MODEL --calibrate"
else
    JUDGE_MODEL=${JUDGE_MODEL:-Qwen/Qwen3-4b}
    GATE="judge_smoke_test.py --url $JUDGE_URL --model $JUDGE_MODEL"
fi
# Fixed here rather than by lora_eval, so step 2 knows where the answers are.
RUN_DIR=$REPO/results/$KIND-$(date +%Y%m%d_%H%M%S)

# Slurm hands us 2 GPUs; split them between judge and student model.
ALL_GPUS=${CUDA_VISIBLE_DEVICES:-0,1}
JUDGE_GPU=${ALL_GPUS%%,*}
EVAL_GPU=${ALL_GPUS##*,}
echo "GPUs: judge=$JUDGE_GPU eval=$EVAL_GPU (node $(hostname))  run dir: $RUN_DIR"

VLLM_PID=
trap 'if [ -n "$VLLM_PID" ]; then kill $VLLM_PID 2>/dev/null || true; fi' EXIT

start_judge () {   # $1 = model id
    CUDA_VISIBLE_DEVICES=$JUDGE_GPU $REPO/.venv-vllm/bin/vllm serve "$1" \
        --served-model-name "$1" \
        --port $JUDGE_PORT \
        --max-model-len 8192 \
        --gpu-memory-utilization 0.90 &
    VLLM_PID=$!
    # Cold start incl. compile can take several minutes.
    for i in $(seq 1 180); do
        curl -sf $JUDGE_URL/health >/dev/null && break
        kill -0 $VLLM_PID 2>/dev/null || { echo "FATAL: vLLM ($1) exited during startup"; exit 1; }
        sleep 5
    done
    curl -sf $JUDGE_URL/health >/dev/null || { echo "FATAL: judge $1 never became healthy"; exit 1; }
    echo "Judge $1 is up."
}

cd $KE

# ---- judge + eval ----
start_judge "$JUDGE_MODEL"
$REPO/.venv/bin/python $GATE
echo "Judge gate PASSED — starting eval."

echo "Evaluating base=$BASE_MODEL adapter=$LORA_DIR on $DATA_FILE (judge $JUDGE_MODEL $JUDGE_VERSION, port $JUDGE_PORT)"
CUDA_VISIBLE_DEVICES=$EVAL_GPU $REPO/.venv/bin/python lora_eval.py \
    "${LORA_ARGS[@]}" \
    --base-model "$BASE_MODEL" \
    --data-file "$DATA_FILE" \
    --judge-base-url $JUDGE_URL \
    --judge-model "$JUDGE_MODEL" \
    --judge-version "$JUDGE_VERSION" \
    --output "$RUN_DIR/eval"
echo "Eval finished: $RUN_DIR"
