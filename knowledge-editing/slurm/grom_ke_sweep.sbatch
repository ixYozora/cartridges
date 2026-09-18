#!/bin/bash
#SBATCH --job-name=ke-grom-edit-sweep
#SBATCH --partition=gpu
# Plain torch/transformers, no flashinfer: either node works.
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=03:00:00
#SBATCH --output=/home/imasoudian/cartridges/knowledge-editing/slurm/logs/%j.log
# NOTE: no --mem on this cluster (nodes have RealMemory=1; any memory request fails).
#
# STAGE 1 of GROM-as-EDITOR (the prof's open question): one closed-form update that pushes
# the OLD answer down (--beta-head, as in the erase) and the NEW answer UP (--beta-new) at
# the same positions, on the request prompt and its paraphrases. No LoRA anywhere.
#
# 50 edits, so held-out facts still exist for the specificity weight (mean alpha ~0.93).
# --no-save: this stage only needs the probe numbers.
#
# What to read per config:
#   forget/*        old answer  -> rank should go UP
#   insert/*        new answer  -> rank should go DOWN, ideally to ~1
#   retain_facts/*, neighborhood/*  -> must stay flat; neighborhood is the locality proxy
#   gen_after rep4  -> fluency canary, ~0
# Promotion needs insert rank near 1 with neighborhood within ~1.5x of before.

set -euo pipefail
REPO=/home/imasoudian/cartridges
SWEEP=${SWEEP:-$REPO/outputs/grom_ke_sweep}
MODEL=${MODEL:-Qwen/Qwen2.5-7B-Instruct}
COMMON="--model $MODEL --num-forget 50 --seed 1 --no-save --solve-device cpu \
        --max-retain-seqs 6000 --probe-limit 200"
mkdir -p $SWEEP
cd $REPO/knowledge-editing

run () {
    local tag=$1; shift
    echo ""
    echo "############ $tag :: $* ############"
    $REPO/.venv/bin/python grom_erase.py $COMMON "$@" \
        --report-out $SWEEP/$tag.json || echo "!! $tag FAILED, continuing"
}

# reference: the erase we already know, on this key set
run erase80            --beta-head 80  --beta-new 0
# insert only: is suppression needed at all once the new answer is boosted?
run insert20           --beta-head 0   --beta-new 20
run insert80           --beta-head 0   --beta-new 80
run insert160          --beta-head 0   --beta-new 160
run insert320          --beta-head 0   --beta-new 320
# both sides
run sup20_ins80        --beta-head 20  --beta-new 80
run sup80_ins20        --beta-head 80  --beta-new 20
run sup80_ins80        --beta-head 80  --beta-new 80
run sup80_ins160       --beta-head 80  --beta-new 160
# one phrasing only, to see what the paraphrase keys buy
run insert80_noparaphrase --beta-head 0 --beta-new 80 --no-paraphrase-keys

echo ""
echo "############ SUMMARY ############"
for f in $SWEEP/*.json; do
    $REPO/.venv/bin/python - "$f" <<'PY'
import json, sys
r = json.load(open(sys.argv[1]))
if "recall_after" not in r: sys.exit()
b, a = r["recall_before"], r["recall_after"]
def rk(k, d): return d.get(k, {}).get("median_gold_rank", -1)
row = [f"{sys.argv[1].split('/')[-1][:-5]:22s}",
       f"bh={r['beta_head']:<5g} bn={r['beta_new']:<5g}"]
for k in ("forget/plain", "insert/plain", "insert/chat", "retain_facts/plain", "neighborhood/plain"):
    row.append(f"{k.split('/')[0][:6]}/{k.split('/')[1][:2]} {rk(k,b)}->{rk(k,a)}")
row.append("rep4=%.3f" % max(g["rep4"] for g in r.get("gen_after", [{"rep4": 0}])))
print("  ".join(row))
PY
done
