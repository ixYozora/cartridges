#!/bin/bash
#SBATCH --job-name=ke-locality-post
#SBATCH --partition=gpu
# fidelity_filter boots the vLLM judge (sm120 flashinfer) and training is the usual
# recipe -> tujestpolin only. Pin the 5090 gres so Slurm can't use sarmatia.
#SBATCH --gres=gpu:rtx5090:1
#SBATCH --cpus-per-task=8
#SBATCH --time=20:00:00
#SBATCH --output=/home/imasoudian/cartridges/knowledge-editing/slurm/logs/%j.log
# NOTE: no --mem on this cluster (nodes have RealMemory=1; any memory request fails).
#
# Everything after synthesis for the locality-seed corpus: fidelity QC, two trainings,
# and the judge-free evals of both.
#
#   sbatch --dependency=afterok:<synth job> --export=ALL,RUN_NAME=LocalityFull \
#          locality_post_synth.sbatch
#
# The synthesis output directory is datetime-stamped and so unknowable at submit time,
# which is why this is a separate job that globs for RUN_NAME rather than a --export
# path. RUN_NAME must therefore be unique across outputs/.
#
# TWO bases, because they answer different questions:
#   stock   the controlled A/B. The only locality baseline measured under this
#           protocol is the DCT adapter on stock (job 12222: 8.1% judge-confirmed
#           bleed, 1.0% unedited floor). Training the new corpus on the same base
#           isolates the data change -- if bleed drops here, the seed worked.
#   ins320  the best configuration to date (GROM insert edit + LoRA). Its locality
#           was never measured, so this is a new number, not a comparison; it says
#           whether the seed survives on top of the edit that gives us our efficacy.

set -euo pipefail
REPO=/home/imasoudian/cartridges
KE=$REPO/knowledge-editing
PY=$REPO/.venv/bin/python
RUN_NAME=${RUN_NAME:?must be set, e.g. --export=ALL,RUN_NAME=LocalityFull}
STAMP=$(date +%Y%m%d_%H%M%S)
CK=$REPO/checkpoints
export WANDB_MODE=offline
cd $KE

ART=$(ls -d $REPO/outputs/*/${RUN_NAME}-0/artifact 2>/dev/null | tail -1)
[ -n "$ART" ] || { echo "FATAL: no artifact dir for ${RUN_NAME}"; exit 1; }
FILTERED=$ART/dataset_filtered.parquet
FINAL=$ART/dataset_final.parquet
[ -f "$FILTERED" ] || { echo "FATAL: $FILTERED missing (did the filter stage run?)"; exit 1; }
echo "artifact dir: $ART"

# ---- 1. fidelity QC -> dataset_final.parquet (never train on raw or filtered) ----
JUDGE_PORT=10330
JUDGE_URL=http://localhost:$JUDGE_PORT
VLLM_PID=
trap 'if [ -n "$VLLM_PID" ]; then kill $VLLM_PID 2>/dev/null || true; fi' EXIT
$REPO/.venv-vllm/bin/vllm serve Qwen/Qwen3-4b --served-model-name Qwen/Qwen3-4b \
    --port $JUDGE_PORT --max-model-len 8192 --gpu-memory-utilization 0.45 &
VLLM_PID=$!
for i in $(seq 1 180); do
    curl -sf $JUDGE_URL/health >/dev/null && break
    kill -0 $VLLM_PID 2>/dev/null || { echo "FATAL: vLLM exited during startup"; exit 1; }
    sleep 5
done
curl -sf $JUDGE_URL/health >/dev/null || { echo "FATAL: judge never became healthy"; exit 1; }
echo "Judge is up."
# Gate on known-answer calibration first (set -e aborts on a miss).
$PY fidelity_filter.py --calibrate --judge-base-url $JUDGE_URL
echo "Calibration PASSED - judging full dataset."
$PY fidelity_filter.py --input "$FILTERED" --output "$FINAL" \
    --judge-base-url $JUDGE_URL --judge-model Qwen/Qwen3-4b
kill $VLLM_PID 2>/dev/null || true
VLLM_PID=
[ -f "$FINAL" ] || { echo "FATAL: $FINAL not written"; exit 1; }

# ---- 2. train on both bases ----
# Paths are derived, not captured: $(train ...) would swallow the whole training
# log into a variable and it would never reach the job output.
STOCK_ADAPTER=$CK/lora-locality-$STAMP-stock
INS_ADAPTER=$CK/lora-locality-$STAMP-ins320

train () {   # $1 = output dir, $2 = base model, $3 = tag
    echo "=============== train $3 (base $2)"
    $PY lora_finetune.py \
        --parquet-path "$FINAL" --output-dir "$1" --model-name "$2" \
        --lora-r 64 --lora-alpha 256 --lora-dropout 0.05 \
        --max-length 2048 --batch-size 4 --gradient-accumulation-steps 8 \
        --num-epochs 5 --learning-rate 2e-5 --warmup-steps 100 --save-steps 500 \
        --wandb-run-name "lora-locality-$3-$STAMP"
}
train "$STOCK_ADAPTER" Qwen/Qwen2.5-7B-Instruct    stock
train "$INS_ADAPTER"   $CK/qwen2.5-7b-ke-ins320    ins320

# ---- 3. evals (judge-free; the judge-v2 pass is experiments/rejudge_locality.sbatch) ----
OUT=$REPO/results/locality-seed-$STAMP
mkdir -p "$OUT"

echo "=============== efficacy (AnyEdit protocol)"
$PY eval_anyedit_protocol.py \
    --config stock-loc  Qwen/Qwen2.5-7B-Instruct "$STOCK_ADAPTER" \
    --config ins320-loc $CK/qwen2.5-7b-ke-ins320 "$INS_ADAPTER" \
    --batch-size 16 --out "$OUT/efficacy.json"

echo "=============== locality (neighbourhood prompts)"
$PY experiments/eval_locality_scaling.py \
    --config stock-loc  Qwen/Qwen2.5-7B-Instruct "$STOCK_ADAPTER" \
    --config ins320-loc $CK/qwen2.5-7b-ke-ins320 "$INS_ADAPTER" \
    --per-edit 2 --batch-size 16 --out "$OUT/locality.json"

echo
echo "adapters: $STOCK_ADAPTER  |  $INS_ADAPTER"
echo "results:  $OUT"
echo "baselines to beat -- locality bleed 8.6% substring / 8.1% judge-v2 (floor 1.0%),"
echo "efficacy Ori 75.71 / Para 64.04 (ins320+LoRA, job 12219)."
