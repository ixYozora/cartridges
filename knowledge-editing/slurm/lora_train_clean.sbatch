#!/bin/bash
#SBATCH --job-name=ke-lora-clean
#SBATCH --partition=gpu
#SBATCH --gres=gpu:rtx5090:1
#SBATCH --cpus-per-task=8
#SBATCH --time=2-00:00:00
#SBATCH --output=/home/imasoudian/cartridges/knowledge-editing/slurm/logs/%j.log
# NOTE: no --mem on this cluster (nodes have RealMemory=1; any memory request fails).
#
# Train the KE LoRA on the clean dataset (dataset_final.parquet).
# Dataset + hyperparameters match the clean baseline
# (checkpoints/lora_qwen2.5-7B-Instruct-clean); the variable under test now is
# the assistant-only loss masking in lora_finetune.py. The output dir and wandb
# run name are datetime-stamped so repeated runs never overwrite each other.
# If torch.compile thrashes on the new dynamic batch shapes, add --no-compile.

set -euo pipefail

REPO=/home/imasoudian/cartridges
export WANDB_MODE=offline

STAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_DIR=$REPO/checkpoints/lora_qwen2.5-7B-Instruct-$STAMP

cd $REPO/knowledge-editing
exec $REPO/.venv/bin/python lora_finetune.py \
    --parquet-path $REPO/outputs/2026-07-15-19-14-08-synthesize/CleanDatasetForQwen2.5-0/artifact/dataset_final.parquet \
    --output-dir "$OUTPUT_DIR" \
    --model-name Qwen/Qwen2.5-7B-Instruct \
    --lora-r 64 \
    --lora-alpha 256 \
    --lora-dropout 0.05 \
    --max-length 2048 \
    --batch-size 4 \
    --gradient-accumulation-steps 8 \
    --num-epochs 5 \
    --learning-rate 2e-5 \
    --warmup-steps 100 \
    --save-steps 500 \
    --wandb-run-name lora-$STAMP
