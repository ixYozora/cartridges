#!/bin/bash
#SBATCH --job-name=ke-grom-mlp-full
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=08:00:00
#SBATCH --output=/home/imasoudian/cartridges/knowledge-editing/slurm/logs/%j.log
# NOTE: no --mem on this cluster (nodes have RealMemory=1; any memory request fails).
#
# Save the MLP-band edits that the capacity test promoted (job 12204). Spreading the
# insert over layers 16-20 installs the fact at beta 20 -- 16x weaker than the head needed
# -- with all 3 phrasings and both key forms, neighbours unharmed and fluency clean, where
# the single head matrix could not do it at any strength (5,850 columns vs 3,584 dims).
#
#   mlp20      insert only, the clean operating point
#   mlp20_sup  same, plus the head suppression that buries the old answer

set -euo pipefail
REPO=/home/imasoudian/cartridges
CK=$REPO/checkpoints
MODEL=${MODEL:-Qwen/Qwen2.5-7B-Instruct}
COMMON="--model $MODEL --num-forget 0 --seed 1 --solve-device cpu --probe-limit 300 \
        --layers 16,17,18,19,20 --mlp-key-pos answer --max-retain-seqs 6000"
cd $REPO/knowledge-editing

run () {
    local tag=$1; shift
    echo ""
    echo "############ $tag :: $* ############"
    $REPO/.venv/bin/python grom_erase.py $COMMON "$@" \
        --out $CK/qwen2.5-7b-ke-$tag --report-out $CK/qwen2.5-7b-ke-$tag/grom_report.json
}

run mlp20     --beta-new-mlp 20 --beta-head 0  --beta-new 0
run mlp20sup  --beta-new-mlp 20 --beta-head 80 --beta-new 0
