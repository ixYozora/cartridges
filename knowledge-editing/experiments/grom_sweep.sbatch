#!/bin/bash
#SBATCH --job-name=ke-grom-sweep
#SBATCH --partition=gpu
#SBATCH --gres=gpu:rtx5090:1
#SBATCH --cpus-per-task=8
#SBATCH --time=02:00:00
#SBATCH --output=/home/imasoudian/cartridges/knowledge-editing/slurm/logs/%j.log
# NOTE: no --mem on this cluster (nodes have RealMemory=1; any memory request fails).
#
# STAGE 1 of the GROM grid. Step 3e showed that the authors' ZsRE setting
# (beta_head=20, head only) suppresses the old target under teacher forcing but does
# NOT change what the teacher writes. This sweep looks for a setting that pushes
# suppression much further WITHOUT collateral, so that only the survivors are worth
# the ~15 min of synthesis+judge it costs to test behaviourally (stage 2).
#
# Every run uses --no-save: a Qwen3-4b checkpoint is ~8.8 GB and we only want the
# measurements. Winners get re-run with --out to produce a real checkpoint.
#
# Gate for promotion to stage 2:
#   forget median rank UP a lot, retain_facts + neighborhood rank ~flat,
#   and the fluency canary rep4 near 0 on all three generation prompts.

set -euo pipefail

REPO=/home/imasoudian/cartridges
KE=$REPO/knowledge-editing
SWEEP=${SWEEP:-$REPO/outputs/grom_sweep}
COMMON="--num-forget 50 --seed 1 --no-save --probe-limit 200"
mkdir -p $SWEEP
cd $KE

echo "############ layer attribution (band for the MLP arm) ############"
$REPO/.venv/bin/python grom_erase.py $COMMON --attribution 5 \
    --report-out $SWEEP/attribution.json

BAND=$($REPO/.venv/bin/python -c "
import json; print(','.join(map(str, json.load(open('$SWEEP/attribution.json'))['attribution']['best_band'])))")
echo "attributed band: $BAND"

run () {   # run <tag> <extra args...>
    local tag=$1; shift
    echo ""
    echo "############ $tag :: $* ############"
    $REPO/.venv/bin/python grom_erase.py $COMMON "$@" \
        --report-out $SWEEP/$tag.json || echo "!! $tag FAILED, continuing"
}

# --- head-only: how far can beta_head go before quality breaks? ---------------
# beta_head=20 is the authors' ZsRE value and the 3e reference point.
for B in 20 40 80 160 320; do
    run head$B --beta-head $B
done

# --- MLP-only on the attributed band -----------------------------------------
# Published MLP strengths span 65..1000 across models, so this needs its own
# calibration rather than a transferred constant. key-pos=answer (not the ZsRE
# subject_last convention): 3e's lesson is that the position where the fact is
# WRITTEN is the one that matters.
for B in 65 200 600 1500; do
    run mlp$B --beta-head 0 --beta-mlp $B --layers $BAND --mlp-key-pos answer
done

# --- combined ----------------------------------------------------------------
run head80_mlp200  --beta-head 80  --beta-mlp 200 --layers $BAND --mlp-key-pos answer
run head160_mlp600 --beta-head 160 --beta-mlp 600 --layers $BAND --mlp-key-pos answer

echo ""
echo "############ SUMMARY ############"
$REPO/.venv/bin/python summarize_grom_sweep.py --sweep-dir $SWEEP
