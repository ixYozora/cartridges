#!/bin/bash
#SBATCH --job-name=ke-scaling-curve
#SBATCH --partition=gpu
#SBATCH --gres=gpu:rtx5090:1
#SBATCH --cpus-per-task=8
#SBATCH --time=10:00:00
#SBATCH --output=/home/imasoudian/cartridges/knowledge-editing/slurm/logs/%j.log
# NOTE: no --mem on this cluster (nodes have RealMemory=1; any memory request fails).
#
# HOW DOES PER-EDIT QUALITY HOLD UP AS MORE EDITS SHARE ONE ADAPTER?
#
# This is the axis AnyEdit's table cannot report: their num_edits defaults to 1, one
# edit into a freshly reset model, so their numbers carry no information about
# cross-edit interference. Ours holds 975 at once. If per-edit quality is flat from
# n=25 to n=975, that is the result worth leading with.
#
# Each point trains the SAME recipe (r64/alpha256, lr 2e-5, 5 epochs, plain CE on the
# stock base) on a seeded random subset of n edits, then is evaluated ONLY on those n
# edits, so the single varying quantity is how many edits share the adapter. Replicates
# at small n control for which edits happened to be drawn.
#
# CONFOUND, handled: holding epochs at 5 means small-n runs take fewer gradient steps
# (n=25 ~ 130 steps vs n=975 ~ 4575). n25-s1-long repeats n=25 at 20 epochs, matching
# n=100's step count, so undertraining can be told apart from a genuine n effect.
# save-steps is scaled per run because eval_steps=save_steps, and the stock default of
# 500 would never fire on a short run, leaving load_best_model_at_end nothing to load.
#
# n=975 is not retrained: it reuses the existing DCT adapter, re-evaluated here under
# the same protocol so the curve's endpoint is measured identically to the rest.

set -euo pipefail
REPO=/home/imasoudian/cartridges
KE=$REPO/knowledge-editing
PY=$REPO/.venv/bin/python
STAMP=$(date +%Y%m%d_%H%M%S)
DATA=$REPO/outputs/2026-08-11-22-26-45-synthesize/CounterFact-SelfStudy-DCT-0/artifact/dataset_final.parquet
DCT975=$REPO/checkpoints/lora_qwen2.5-7B-Instruct-20260812_002743/checkpoint-2500
WORK=$REPO/data/scaling-$STAMP
OUT=$REPO/results/scaling-curve-$STAMP
mkdir -p "$WORK" "$OUT"
export WANDB_MODE=offline
cd $KE

# name  n    seed  epochs
RUNS=(
  "n25-s1       25   1   5"
  "n25-s2       25   2   5"
  "n25-s3       25   3   5"
  "n25-s1-long  25   1  20"
  "n100-s1     100   1   5"
  "n100-s2     100   2   5"
  "n300-s1     300   1   5"
)

for spec in "${RUNS[@]}"; do
    read -r NAME N SEED EPOCHS <<< "$spec"
    SUB=$WORK/$NAME
    ADAPTER=$REPO/checkpoints/scaling-$STAMP-$NAME
    echo "=============== $NAME  (n=$N seed=$SEED epochs=$EPOCHS)"

    $PY make_edit_subset.py --parquet "$DATA" --n "$N" --seed "$SEED" --out-dir "$SUB"

    # steps = 0.9 * rows / (batch 4 * accum 8) * epochs; checkpoint ~5x per run so
    # load_best_model_at_end always has candidates.
    ROWS=$($PY -c "import pandas as pd;print(len(pd.read_parquet('$SUB/dataset_final.parquet')))")
    SAVE=$($PY -c "print(max(10, int(0.9*$ROWS/32*$EPOCHS)//5))")
    echo "   rows=$ROWS  save/eval every $SAVE steps"

    $PY lora_finetune.py \
        --parquet-path "$SUB/dataset_final.parquet" \
        --output-dir "$ADAPTER" \
        --model-name Qwen/Qwen2.5-7B-Instruct \
        --lora-r 64 --lora-alpha 256 --lora-dropout 0.05 \
        --max-length 2048 --batch-size 4 --gradient-accumulation-steps 8 \
        --num-epochs "$EPOCHS" --learning-rate 2e-5 --warmup-steps 100 \
        --save-steps "$SAVE" --seed "$SEED" \
        --wandb-run-name "scaling-$NAME-$STAMP"

    $PY eval_anyedit_protocol.py \
        --config "$NAME" Qwen/Qwen2.5-7B-Instruct "$ADAPTER" \
        --case-ids-file "$SUB/case_ids.json" \
        --batch-size 16 \
        --out "$OUT/$NAME.json"
done

# Endpoint: the existing 975-edit DCT adapter, plus stock as the shared floor.
echo "=============== n975 + stock (all 975 edits)"
$PY eval_anyedit_protocol.py \
    --config stock Qwen/Qwen2.5-7B-Instruct \
    --config n975  Qwen/Qwen2.5-7B-Instruct "$DCT975" \
    --batch-size 16 \
    --out "$OUT/n975.json"

echo "=============== curve"
$PY -c "
import json, glob, os
rows = []
for f in sorted(glob.glob('$OUT/*.json')):
    for name, c in json.load(open(f))['configs'].items():
        o, p = c['ori'], c['para']
        rows.append((name, len(o['sims']),
                     100*sum(o['sims'])/len(o['sims']), o['names_new'], o['names_old'],
                     100*sum(p['sims'])/len(p['sims']), p['names_new'], p['names_old']))
print(f\"{'run':<14}{'n':>5}{'Ori.':>8}{'new%':>8}{'old%':>7}{'Para.':>8}{'new%':>8}{'old%':>7}\")
for r in rows:
    print(f'{r[0]:<14}{r[1]:>5}{r[2]:>8.2f}{r[3]:>8.1f}{r[4]:>7.1f}{r[5]:>8.2f}{r[6]:>8.1f}{r[7]:>7.1f}')
"
echo "results in $OUT"
