#!/bin/bash
#SBATCH --job-name=ke-grom-sweep-student
#SBATCH --partition=gpu
#SBATCH --gres=gpu:rtx5090:1
#SBATCH --cpus-per-task=8
#SBATCH --time=03:00:00
#SBATCH --output=/home/imasoudian/cartridges/knowledge-editing/slurm/logs/%j.log
# NOTE: no --mem on this cluster (nodes have RealMemory=1; any memory request fails).
#
# STAGE 1 of the STUDENT erase ("MU first, then FT" -- the prof's 2026-08-04 idea).
# Different target from the falsified teacher arm: this removes the pre-edit facts from
# Qwen2.5-7B-Instruct itself, so the LoRA is not fighting residual old knowledge. The
# context-echo mechanism that killed the teacher arm does not apply here -- the metric
# this should move is the student's old_target leak at eval (currently 5.27% overall,
# 9.13% on generalization).
#
# All 975 edits are forgotten at once, so there are no held-out facts: alpha comes from
# CounterFact attribute_prompts (gold = target_new, so it keeps alpha high on the
# suppressed target_true while protecting what the edit installs), and neighborhood
# prompts enter as retain KEYS only. mean_alpha = 0.391 with 634/1972 forget positions
# correctly protected -- those are tokens that are some OTHER edit's new target.
#
# --solve-device cpu is required, not optional: at vocab 152064 x d 3584 the head target
# and its solve are ~4.4 GB each in float64, which will not fit beside a bf16 7B on one
# 32 GB card. The solve itself is seconds on CPU.
#
# Gate for promotion: forget rank UP, retain/neighborhood ~flat, rep4 ~0. Locality is
# 95.18% today and is the thing most at risk, so neighborhood is the column to watch.

set -euo pipefail

REPO=/home/imasoudian/cartridges
SWEEP=${SWEEP:-$REPO/outputs/grom_sweep_student}
MODEL=${MODEL:-Qwen/Qwen2.5-7B-Instruct}
COMMON="--model $MODEL --num-forget 0 --seed 1 --no-save --solve-device cpu \
        --max-retain-seqs 6000 --probe-limit 200"
mkdir -p $SWEEP
cd $REPO/knowledge-editing

echo "############ layer attribution (Qwen2.5-7B has 28 layers, not the teacher's 36) ############"
$REPO/.venv/bin/python grom_erase.py $COMMON --attribution 5 \
    --report-out $SWEEP/attribution.json

run () {
    local tag=$1; shift
    echo ""
    echo "############ $tag :: $* ############"
    $REPO/.venv/bin/python grom_erase.py $COMMON "$@" \
        --report-out $SWEEP/$tag.json || echo "!! $tag FAILED, continuing"
}

# Head-only. The teacher sweep showed the MLP arm damages retain far more than the head
# for the same suppression at every published strength, so it is not repeated here
# unless head-only turns out to be insufficient.
for B in 20 40 80 160 320; do
    run head$B --beta-head $B
done

echo ""
echo "############ SUMMARY ############"
$REPO/.venv/bin/python summarize_grom_sweep.py --sweep-dir $SWEEP
