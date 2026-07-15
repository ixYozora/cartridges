#!/bin/bash
#SBATCH --job-name=ke-synth-clean
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=2-00:00:00
#SBATCH --output=/home/imasoudian/cartridges/knowledge-editing/slurm/logs/%j.log
# NOTE: no --mem on this cluster (nodes have RealMemory=1; any memory request fails).
#
# Full synthesis pipeline in one job: tokasaurus teacher (GPU) + synthesize.py
# (CPU client, localhost:10210) + filter_dataset.py QC stage on the result.
#
# Small validation run:
#   sbatch --export=ALL,SYNTH_ARGS="num_samples=512 name=CleanSmoke" synth_clean.sbatch

set -euo pipefail

REPO=/home/imasoudian/cartridges
KE=$REPO/knowledge-editing
export CARTRIDGES_DIR=$REPO
export CARTRIDGES_OUTPUT_DIR=$REPO/outputs
SYNTH_ARGS=${SYNTH_ARGS:-}

$REPO/.venv/bin/tksrs model=Qwen/Qwen3-4b \
    kv_cache_num_tokens='(128 * 1024)' \
    max_topk_logprobs=20 \
    dp_size=1 \
    torch_compile=T \
    use_hydragen=T \
    hydragen_min_group_size=256 &
TKS_PID=$!
trap 'kill $TKS_PID 2>/dev/null || true' EXIT

# torch_compile warmup can take a while; wait up to 60 min.
for i in $(seq 1 360); do
    curl -sf http://localhost:10210/ping >/dev/null && break
    kill -0 $TKS_PID 2>/dev/null || { echo "FATAL: tokasaurus exited during startup"; exit 1; }
    sleep 10
done
curl -sf http://localhost:10210/ping >/dev/null || { echo "FATAL: tokasaurus never became healthy"; exit 1; }
echo "Tokasaurus is up."

cd $KE
MARKER=$(mktemp)
$REPO/.venv/bin/python synthesize.py $SYNTH_ARGS
echo "Synthesis finished."

PARQUET=$(find $CARTRIDGES_OUTPUT_DIR -name dataset.parquet -newer $MARKER | head -1)
rm -f $MARKER
[ -n "$PARQUET" ] || { echo "FATAL: no new dataset.parquet found"; exit 1; }
echo "New dataset: $PARQUET"

$REPO/.venv/bin/python filter_dataset.py --input "$PARQUET" --show-examples 3
echo "Filter stage finished."
