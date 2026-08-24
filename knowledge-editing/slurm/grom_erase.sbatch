#!/bin/bash
#SBATCH --job-name=ke-grom-erase
#SBATCH --partition=gpu
# GROM is plain torch/transformers and does NOT use flashinfer, so it is not
# arch-locked the way our tksrs/vLLM jobs are. Pinned to the 5090 anyway: the
# float64 target accumulator and solve result are ~3.1 GB each, which is tight
# against a 4090's 24 GB alongside the bf16 model.
#SBATCH --gres=gpu:rtx5090:1
#SBATCH --cpus-per-task=8
#SBATCH --time=01:00:00
#SBATCH --output=/home/imasoudian/cartridges/knowledge-editing/slurm/logs/%j.log
# NOTE: no --mem on this cluster (nodes have RealMemory=1; any memory request fails).
#
# GROM closed-form teacher-erase: remove the pre-edit CounterFact facts from the
# Qwen3-4b Self-Study teacher. Defaults are the authors' ZsRE fact-editing
# configuration (LM head only, beta_head=20, w_r=30, rho=0.03).
#
#   sbatch grom_erase.sbatch                          # 50-edit pilot
#   sbatch --export=ALL,NUM_FORGET=100,SEED=2 grom_erase.sbatch
#   sbatch --export=ALL,EXTRA_ARGS="--beta-head 40" grom_erase.sbatch
#
# Writes checkpoints/qwen3-4b-erased-n<N>-s<SEED>/ plus grom_report.json, which
# records every knob and the before/after recall probe.

set -euo pipefail

REPO=/home/imasoudian/cartridges
NUM_FORGET=${NUM_FORGET:-50}
SEED=${SEED:-1}
OUT=${OUT:-$REPO/checkpoints/qwen3-4b-erased-n${NUM_FORGET}-s${SEED}}
EXTRA_ARGS=${EXTRA_ARGS:-}

echo "node=$(hostname) gpu=${CUDA_VISIBLE_DEVICES:-unset}"
echo "num_forget=$NUM_FORGET seed=$SEED out=$OUT"

cd $REPO/knowledge-editing
exec $REPO/.venv/bin/python grom_erase.py \
    --num-forget "$NUM_FORGET" \
    --seed "$SEED" \
    --out "$OUT" \
    $EXTRA_ARGS
