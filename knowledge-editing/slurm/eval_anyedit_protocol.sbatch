#!/bin/bash
#SBATCH --job-name=ke-anyedit-proto
#SBATCH --partition=gpu
# Plain transformers + peft (no flashinfer), so either node works; a 7B in bf16 fits 24 GB.
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=06:00:00
#SBATCH --output=/home/imasoudian/cartridges/knowledge-editing/slurm/logs/%j.log
# NOTE: no --mem on this cluster (nodes have RealMemory=1; any memory request fails).
#
# One clean comparison against AnyEdit's published AKEW (CounterFact) table.
# Regenerates under THEIR protocol -- bare user turn with no system prompt,
# temperature 0.001 -- because our own eval's chat template injects Qwen's default
# "You are Qwen, created by Alibaba Cloud ..." preamble, which makes answers chattier
# and so scores higher on a metric that rewards covering a 74-word paragraph.
#
# stock  is the anchor: it must land near their published Pre-edited 65.50 / 44.74.
#        If it does, the protocol is faithful and ins320 is a number we can quote.
#        If it does not, the residual is something we have not found yet.
# ins320 is the best configuration to date (GROM insert edit + LoRA on top).
#
# Still NOT matched, and not fixable by regenerating: they apply one edit to a freshly
# reset model (batch_size = num_edits = 1); we hold all 975 edits in one model.

set -euo pipefail
REPO=/home/imasoudian/cartridges
CK=$REPO/checkpoints
STAMP=$(date +%Y%m%d_%H%M%S)

cd $REPO/knowledge-editing
$REPO/.venv/bin/python eval_anyedit_protocol.py \
    --config stock  Qwen/Qwen2.5-7B-Instruct \
    --config ins320 $CK/qwen2.5-7b-ke-ins320 $CK/lora-on-ke-ins320 \
    --batch-size 16 \
    --out $REPO/results/anyedit-protocol-$STAMP.json
