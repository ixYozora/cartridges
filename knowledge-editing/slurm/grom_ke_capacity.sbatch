#!/bin/bash
#SBATCH --job-name=ke-grom-capacity
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=06:00:00
#SBATCH --output=/home/imasoudian/cartridges/knowledge-editing/slurm/logs/%j.log
# NOTE: no --mem on this cluster (nodes have RealMemory=1; any memory request fails).
#
# Is the failure at 975 edits a CAPACITY limit rather than a strength one?
#
# The head edit solves P*K = D for every key column. Keys live in 3,584 dimensions
# (Qwen2.5-7B hidden size). At 50 edits we ask for 300 columns -> nearly exact, new answer
# reaches rank 1. At 975 edits with 3 phrasings x 2 key forms we ask for 5,850 columns,
# MORE than the dimensions available, so least squares can only compromise: rank ~50, and
# turning the strength up degenerates the edited answers instead of installing them.
#
# Test A: same 975 edits, ONE phrasing and ONE key form = 975 columns, comfortably under
#         3,584. If rank drops sharply at a safe strength, capacity is the binding
#         constraint (at the cost of paraphrase robustness, which the eval would show).
# Test B: keep all 5,850 columns but spread the edit over a BAND of MLP layers, which is
#         what MEMIT does for the same reason -- each layer contributes its own capacity,
#         and writing into the residual stream also propagates to later tokens.

set -euo pipefail
REPO=/home/imasoudian/cartridges
SWEEP=${SWEEP:-$REPO/outputs/grom_ke_capacity}
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

# --- A: fewer columns than dimensions (975 keys) -------------------------------------
run capA_ins80   --beta-head 0 --beta-new 80  --no-paraphrase-keys --key-forms plain
run capA_ins320  --beta-head 0 --beta-new 320 --no-paraphrase-keys --key-forms plain

# --- B: same columns, more matrices (5 MLP layers, attributed band [16-20] of 28) -----
MLP="--beta-head 0 --beta-new 0 --layers 16,17,18,19,20 --mlp-key-pos answer \
     --max-retain-seqs 6000"
run capB_mlp20   $MLP --beta-new-mlp 20
run capB_mlp65   $MLP --beta-new-mlp 65
run capB_mlp160  $MLP --beta-new-mlp 160

echo ""
echo "############ SUMMARY ############"
for f in $SWEEP/*.json; do
    $REPO/.venv/bin/python - "$f" <<'PY'
import json, sys
r = json.load(open(sys.argv[1]))
if "recall_after" not in r: sys.exit()
b, a = r["recall_before"], r["recall_after"]
rk = lambda k, d: d.get(k, {}).get("median_gold_rank", -1)
gen = [g for g in r.get("gen_after", []) if g.get("kind") == "generic"] or [{"rep4": 0}]
edi = [g for g in r.get("gen_after", []) if g.get("kind") == "edited"] or [{"rep4": 0}]
print(f"{sys.argv[1].split('/')[-1][:-5]:14s} bn={r['beta_new']:<5g} bn_mlp={r.get('beta_new_mlp',0):<5g} "
      f"cols={r['forget_key_positions']:<6d} " +
      "  ".join(f"{k.split('/')[0][:6]}/{k.split('/')[1][:2]} {rk(k,b)}->{rk(k,a)}"
               for k in ("insert/plain", "insert/chat", "neighborhood/plain")) +
      "  rep4 generic=%.3f edited=%.3f" % (max(g["rep4"] for g in gen), max(g["rep4"] for g in edi)))
PY
done
