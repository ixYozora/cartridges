#!/bin/bash
#SBATCH --job-name=ke-grom-edit-full
#SBATCH --partition=gpu
# Plain torch/transformers, no flashinfer: either node works.
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=06:00:00
#SBATCH --output=/home/imasoudian/cartridges/knowledge-editing/slurm/logs/%j.log
# NOTE: no --mem on this cluster (nodes have RealMemory=1; any memory request fails).
#
# STAGE 2 of GROM-as-EDITOR: apply the promoted configurations to ALL 975 edits and save
# the checkpoints, so they can be evaluated with no adapter at all (lora_eval --lora-dir
# omitted). Stage 1 (slurm/grom_ke_sweep.sbatch, job 12094) picked beta_new=80 as the
# operating point: the new answer reaches rank 1 in both key forms, while 160 degrades
# the fluency canary and 320 breaks generation.
#
# At full scale no held-out edits remain, so the specificity weight comes from the
# CounterFact attribute prompts (as in the student erase); the retain side is left
# uncapped, which bought a better suppression/collateral trade-off there (jobs 11174/6).

set -euo pipefail
REPO=/home/imasoudian/cartridges
CK=$REPO/checkpoints
MODEL=${MODEL:-Qwen/Qwen2.5-7B-Instruct}
COMMON="--model $MODEL --num-forget 0 --seed 1 --solve-device cpu --probe-limit 200"
cd $REPO/knowledge-editing

run () {
    local tag=$1; shift
    echo ""
    echo "############ $tag :: $* ############"
    $REPO/.venv/bin/python grom_erase.py $COMMON "$@" \
        --out $CK/qwen2.5-7b-ke-$tag \
        --report-out $CK/qwen2.5-7b-ke-$tag/grom_report.json
}

run insert80    --beta-head 0  --beta-new 80
run sup20ins80  --beta-head 20 --beta-new 80
run sup80ins80  --beta-head 80 --beta-new 80
echo "done: $(ls -d $CK/qwen2.5-7b-ke-* | tr '\n' ' ')"
