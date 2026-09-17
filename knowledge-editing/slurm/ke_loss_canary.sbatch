#!/bin/bash
#SBATCH --job-name=ke-loss-canary
#SBATCH --partition=gpu
#SBATCH --gres=gpu:rtx5090:1
#SBATCH --cpus-per-task=8
#SBATCH --time=01:30:00
#SBATCH --output=/home/imasoudian/cartridges/knowledge-editing/slurm/logs/%j.log
# NOTE: no --mem on this cluster (nodes have RealMemory=1; any memory request fails).
#
# Smoke test for the 3-term objective before spending a full run. It answers:
#   1. does the forget term actually fire? (the 2026-08 version never did: it looked for the
#      old answer inside the Self-Study targets, which the fidelity filter had already removed)
#   2. does torch.compile re-trace every step, which would make it unusably slow?
#   3. does the old answer's rank drop, and is the new fact still learned?
# One epoch over the Self-Study rows of the first 200 edits, then the recall probe on those
# same edits against the stock base.
#
#   sbatch slurm/ke_loss_canary.sbatch
#   sbatch --export=ALL,KE_ARGS="--ke-loss --lambda-forget 2.0 --lambda-retain 1.0 --ke-max-edits 200" slurm/ke_loss_canary.sbatch

set -euo pipefail
REPO=/home/imasoudian/cartridges
KE=$REPO/knowledge-editing
STAMP=$(date +%Y%m%d_%H%M%S)
OUT=$REPO/checkpoints/_ke-canary-$STAMP
DATA_FILE=${DATA_FILE:-$REPO/outputs/_canary/ke_canary_200edits.parquet}
# KE_ARGS="" runs the same canary with PLAIN CE -- the control that says how much of any
# old-answer drop is just learning the new fact.
if [ -z "${KE_ARGS+set}" ]; then
    KE_ARGS="--ke-loss --lambda-forget 1.0 --lambda-retain 1.0 --ke-max-edits 200"
fi
echo "canary: $KE_ARGS  data: $DATA_FILE"

cd $KE
$REPO/.venv/bin/python lora_finetune.py \
    --parquet-path "$DATA_FILE" \
    --output-dir "$OUT" \
    --model-name Qwen/Qwen2.5-7B-Instruct \
    $KE_ARGS \
    --lora-r 64 --lora-alpha 256 --lora-dropout 0.05 \
    --max-length 2048 --batch-size 4 --gradient-accumulation-steps 8 \
    --num-epochs 1 --learning-rate 2e-5 --warmup-steps 5 \
    --save-steps 25 --logging-steps 5 --seed 1 \
    --wandb-run-name ke-canary-$STAMP

LOG=$KE/slurm/logs/$SLURM_JOB_ID.log
echo "=============== CANARY VERDICT ==============="
echo "recompile warnings: $(grep -ci 'recompil' $LOG || true)   (want 0)"
# `|| true`: the plain-CE control has no ke/* lines, and a grep miss would otherwise
# abort the job (set -e) before the recall probe -- which is the point of the run.
echo "forget term (want non-zero early, falling as the old answer is pushed down):"
grep -oE "'ke/forget': [0-9.e-]+" $LOG | head -3 || true
grep -oE "'ke/forget': [0-9.e-]+" $LOG | tail -3 || true
echo "how often the hinge was active:"; grep -oE "'ke/forget_fired_pct': [0-9.]+" $LOG | tail -1 || true
echo "retain term:"; grep -oE "'ke/retain': [0-9.e-]+" $LOG | tail -3 || true
echo "CE term:";     grep -oE "'ke/ce': [0-9.e-]+" $LOG | tail -3 || true

echo "=============== RECALL PROBE (200 canary edits) ==============="
$REPO/.venv/bin/python probe_recall_lora.py \
    --config canary Qwen/Qwen2.5-7B-Instruct "$OUT" \
    --config stock  Qwen/Qwen2.5-7B-Instruct \
    --limit 200 \
    --out $REPO/results/recall-probe-kecanary-$STAMP.json
