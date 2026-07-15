#!/bin/bash
#SBATCH --job-name=ke-lora-clean
#SBATCH --partition=gpu
#SBATCH --gres=gpu:rtx5090:1
#SBATCH --cpus-per-task=8
#SBATCH --time=2-00:00:00
#SBATCH --output=/home/imasoudian/cartridges/knowledge-editing/slurm/logs/%j.log
# NOTE: no --mem on this cluster (nodes have RealMemory=1; any memory request fails).
#
# Retrain the KE LoRA on the clean dataset (dataset_final.parquet).
# Hyperparameters deliberately IDENTICAL to the Feb checkpoint
# (checkpoints/lora_qwen2.5-7B-Instruct/training_args.bin + adapter_config.json)
# so data quality is the only variable vs the contaminated run.

set -euo pipefail

REPO=/home/imasoudian/cartridges
export WANDB_MODE=offline

cd $REPO/knowledge-editing
exec $REPO/.venv/bin/python lora_finetune.py \
    --parquet-path $REPO/outputs/2026-07-15-19-14-08-synthesize/CleanDatasetForQwen2.5-0/artifact/dataset_final.parquet \
    --output-dir $REPO/checkpoints/lora_qwen2.5-7B-Instruct-clean \
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
    --wandb-run-name lora-clean-20260715
