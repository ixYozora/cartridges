#!/bin/bash
#SBATCH --job-name=ke-lora-clean
#SBATCH --partition=gpu
#SBATCH --gres=gpu:rtx5090:1
#SBATCH --cpus-per-task=8
#SBATCH --time=2-00:00:00
#SBATCH --output=/home/imasoudian/cartridges/knowledge-editing/slurm/logs/%j.log
# NOTE: no --mem on this cluster (nodes have RealMemory=1; any memory request fails).
#
# Train the KE LoRA on a Self-Study dataset. DATA_FILE is REQUIRED (the job
# aborts if it is unset) so a run never silently trains on a stale default:
#   sbatch --export=ALL,DATA_FILE=<path/to/dataset_final.parquet> lora_train_clean.sbatch
# Hyperparameters (r64/alpha256, lr 2e-5, 5 epochs) match the baseline; the loss
# masking lives in lora_finetune.py. Output dir + wandb run name are datetime-
# stamped so repeated runs never overwrite each other. Add --no-compile if
# torch.compile thrashes on the dynamic batch shapes.

set -euo pipefail

REPO=/home/imasoudian/cartridges
export WANDB_MODE=offline

STAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_DIR=$REPO/checkpoints/lora_qwen2.5-7B-Instruct-$STAMP

cd $REPO/knowledge-editing
exec $REPO/.venv/bin/python lora_finetune.py \
    --parquet-path "${DATA_FILE:?must be set, e.g. --export=ALL,DATA_FILE=<path/to/dataset_final.parquet>}" \
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
