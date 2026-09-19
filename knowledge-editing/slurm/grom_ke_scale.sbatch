#!/bin/bash
#SBATCH --job-name=ke-grom-edit-scale
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=04:00:00
#SBATCH --output=/home/imasoudian/cartridges/knowledge-editing/slurm/logs/%j.log
# NOTE: no --mem on this cluster (nodes have RealMemory=1; any memory request fails).
#
# STAGE 3: the strength that installs a fact at 50 edits does NOT hold at 975. With
# beta_new=80 the new answer went 313 -> 1 on 50 edits but only 269 -> 56 on all 975
# (job 12175): the edits share one least-squares solve and dilute each other, and the
# specificity weight drops from 0.93 to 0.39 once no held-out facts remain. So sweep the
# insert strength AT FULL SCALE, with --no-save (only the probe numbers matter here).
#
# Read: insert/* should approach rank 1; neighborhood/* must stay flat; rep4 ~0.

set -euo pipefail
REPO=/home/imasoudian/cartridges
SWEEP=${SWEEP:-$REPO/outputs/grom_ke_scale}
MODEL=${MODEL:-Qwen/Qwen2.5-7B-Instruct}
COMMON="--model $MODEL --num-forget 0 --seed 1 --no-save --solve-device cpu --probe-limit 300"
mkdir -p $SWEEP
cd $REPO/knowledge-editing

run () {
    local tag=$1; shift
    echo ""
    echo "############ $tag :: $* ############"
    $REPO/.venv/bin/python grom_erase.py $COMMON "$@" --report-out $SWEEP/$tag.json \
        || echo "!! $tag FAILED, continuing"
}

run ins160   --beta-head 0 --beta-new 160
run ins320   --beta-head 0 --beta-new 320
run ins640   --beta-head 0 --beta-new 640
run ins1280  --beta-head 0 --beta-new 1280
run sup160ins640 --beta-head 160 --beta-new 640

echo ""
echo "############ SUMMARY ############"
for f in $SWEEP/*.json; do
    $REPO/.venv/bin/python - "$f" <<'PY'
import json, sys
r = json.load(open(sys.argv[1]))
if "recall_after" not in r: sys.exit()
b, a = r["recall_before"], r["recall_after"]
rk = lambda k, d: d.get(k, {}).get("median_gold_rank", -1)
print(f"{sys.argv[1].split('/')[-1][:-5]:16s} bh={r['beta_head']:<6g} bn={r['beta_new']:<6g} " +
      "  ".join(f"{k.split('/')[0][:6]}/{k.split('/')[1][:2]} {rk(k,b)}->{rk(k,a)}"
               for k in ("forget/plain", "insert/plain", "insert/chat",
                         "retain_facts/plain", "neighborhood/plain")) +
      "  rep4=%.3f" % max(g["rep4"] for g in r.get("gen_after", [{"rep4": 0}])))
PY
done
