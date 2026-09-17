#!/bin/bash
#SBATCH --job-name=ke-recall-probe
#SBATCH --partition=gpu
# Plain transformers + peft (no flashinfer), so either node works; a 7B in bf16 fits 24 GB.
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=02:00:00
#SBATCH --output=/home/imasoudian/cartridges/knowledge-editing/slurm/logs/%j.log
# NOTE: no --mem on this cluster (nodes have RealMemory=1; any memory request fails).
#
# Did the GROM erase survive LoRA finetuning? Old/new-target recall for both bases and
# the adapters trained on each. The adapter dir root is the BEST-eval-loss checkpoint
# (load_best_model_at_end=True; epoch 2.73 for both, = checkpoint-2500, byte-identical),
# which is what every eval so far used; checkpoint-4575 is the true end of epoch 5.

set -euo pipefail
REPO=/home/imasoudian/cartridges
CK=$REPO/checkpoints
STOCK=Qwen/Qwen2.5-7B-Instruct
ERASED=$CK/qwen2.5-7b-erased-all975-head160
DCT=$CK/lora_qwen2.5-7B-Instruct-20260812_002743
H160=$CK/lora-on-erased-head160

cd $REPO/knowledge-editing
$REPO/.venv/bin/python probe_recall_lora.py \
    --config stock      $STOCK \
    --config erased     $ERASED \
    --config dct_best   $STOCK  $DCT \
    --config h160_best  $ERASED $H160 \
    --config dct_ep5    $STOCK  $DCT/checkpoint-4575 \
    --config h160_ep5   $ERASED $H160/checkpoint-4575 \
    --out $REPO/results/recall-probe-$(date +%Y%m%d_%H%M%S).json
