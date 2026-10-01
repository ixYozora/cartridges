#!/bin/bash
#SBATCH --job-name=ke-locality-scaling
#SBATCH --partition=gpu
# Plain transformers + peft (no flashinfer), so either node works; a 7B in bf16 fits 24 GB.
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=08:00:00
#SBATCH --output=/home/imasoudian/cartridges/knowledge-editing/slurm/logs/%j.log
# NOTE: no --mem on this cluster (nodes have RealMemory=1; any memory request fails).
#
# DOES COLLATERAL DAMAGE GROW WITH THE NUMBER OF EDITS?
#
# The scaling curve (job 12220) found per-edit quality RISING with n, because the
# corpus teaches one shared behaviour and more edits mean more evidence for it.
# Locality is that same generalisation pointed the wrong way: nothing in the corpus
# demonstrates "keep the original answer for a different subject", so the override can
# spill onto neighbours. Judge-v2 locality at n=975 was 30.6%; we have never measured
# it against n, and the answer decides the fix:
#
#   flat across n     -> a recipe problem; add locality seeds to the synthesis
#   degrades with n   -> the price of the scaling gain, and the headline needs a caveat
#
# Reuses the adapters job 12220 already trained, so nothing is retrained here. Each
# adapter is scored on the neighbours of ITS OWN edits (--case-ids-file), matching how
# the efficacy curve was scoped. stock is the shared floor: it has seen no edits, so
# its "bleed" is the incidental rate that any real bleed must be read against.

set -euo pipefail
REPO=/home/imasoudian/cartridges
KE=$REPO/knowledge-editing
PY=$REPO/.venv/bin/python
SRC=20260921_200849                                   # the scaling-curve run being reused
CK=$REPO/checkpoints
SUBS=$REPO/data/scaling-$SRC
DCT975=$CK/lora_qwen2.5-7B-Instruct-20260812_002743/checkpoint-2500
OUT=$REPO/results/locality-scaling-$(date +%Y%m%d_%H%M%S)
mkdir -p "$OUT"
cd $KE

for NAME in n25-s1 n25-s2 n25-s3 n25-s1-long n100-s1 n100-s2 n300-s1; do
    echo "=============== $NAME"
    $PY experiments/eval_locality_scaling.py \
        --config "$NAME" Qwen/Qwen2.5-7B-Instruct "$CK/scaling-$SRC-$NAME" \
        --case-ids-file "$SUBS/$NAME/case_ids.json" \
        --per-edit 2 --batch-size 16 \
        --out "$OUT/$NAME.json"
done

echo "=============== n975 + stock (all 975 edits)"
$PY experiments/eval_locality_scaling.py \
    --config stock Qwen/Qwen2.5-7B-Instruct \
    --config n975  Qwen/Qwen2.5-7B-Instruct "$DCT975" \
    --per-edit 2 --batch-size 16 \
    --out "$OUT/n975.json"

echo "=============== locality vs n"
$PY -c "
import json, glob
print(f\"{'run':<14}{'prompts':>9}{'correct%':>10}{'bleed%':>9}{'neither%':>10}\")
for f in sorted(glob.glob('$OUT/*.json')):
    for name, c in json.load(open(f))['configs'].items():
        n = len(c['correct'])
        cor, ble = sum(c['correct']), sum(c['bleed'])
        print(f'{name:<14}{n:>9}{100*cor/n:>10.1f}{100*ble/n:>9.1f}{100*(n-cor-ble)/n:>10.1f}')
"
echo "results in $OUT"
