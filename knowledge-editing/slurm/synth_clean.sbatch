#!/bin/bash
#SBATCH --job-name=ke-synth-clean
#SBATCH --partition=gpu
# flashinfer in the venv is built for sm120 (RTX 5090), so tksrs only runs on
# tujestpolin (the sole rtx5090 node). Pin the 5090 gres so Slurm can't schedule
# this on sarmatia (rtx4090), where CUDA/flashinfer init crashes the workers.
#SBATCH --gres=gpu:rtx5090:1
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
# Teacher checkpoint: default base model, or a GROM-erased checkpoint for the
# Step 3e A/B. tksrs and the synthesize.py client must agree on this name.
export TEACHER_MODEL=${TEACHER_MODEL:-Qwen/Qwen3-4b}
echo "teacher: $TEACHER_MODEL"

# Tokasaurus startup is intermittently flaky: a multiprocessing spawn race
# crashes the workers (SemLock FileNotFoundError) while the main server process
# survives, so the job would otherwise hang waiting for a /ping that never comes.
# Retry the whole launch until the server answers, killing the tksrs process
# group between attempts. A healthy start reaches /ping in ~90s; allow 3 min each.
launch_tksrs() {
    # enable_precise_onboard=F: precise_onboard has an unhandled NoSpaceException
    # after real_allocate (manager.py:934 -> scheduler.py:369) that kills the
    # manager under bursty prefix-heavy loads (jobs 10746/10747); the frontend
    # survives, so clients just time out forever. coarse_onboard admits
    # conservatively (no prefix-cache credit) and doesn't hit the bug.
    setsid $REPO/.venv/bin/tksrs model=$TEACHER_MODEL \
        kv_cache_num_tokens='(128 * 1024)' \
        max_topk_logprobs=20 \
        dp_size=1 \
        torch_compile=T \
        use_hydragen=T \
        hydragen_min_group_size=256 \
        enable_precise_onboard=F &
    TKS_PID=$!
}

kill_tksrs() {
    [ -n "${TKS_PID:-}" ] || return 0
    kill -TERM -"$TKS_PID" 2>/dev/null || kill "$TKS_PID" 2>/dev/null || true
    wait "$TKS_PID" 2>/dev/null || true
}

TKS_PID=""
trap kill_tksrs EXIT

# /ping only proves the HTTP frontend is alive. Job 10746 passed /ping with
# GPU workers killed by the SemLock spawn race (warmup_done=False, 1 lifetime
# forward pass), then the manager crashed with NoSpaceException once requests
# piled up. So after /ping, require ONE real completion through the same
# endpoint the client uses; generous timeout because the first forward passes
# include torch.compile warmup.
probe_completion() {
    curl -sf --max-time 240 -X POST \
        http://localhost:10210/custom/synchronous-batch-completions \
        -H 'Content-Type: application/json' \
        -d "{\"requests\":[{\"messages\":[{\"role\":\"user\",\"content\":\"Say OK\"}],\"model\":\"$TEACHER_MODEL\",\"max_completion_tokens\":4,\"temperature\":0.0}]}" \
        >/dev/null
}

UP=0
for attempt in 1 2 3 4 5; do
    echo "=== tokasaurus startup attempt $attempt/5 ==="
    launch_tksrs
    UP=0
    for i in $(seq 1 18); do   # up to ~3 min per attempt
        if curl -sf http://localhost:10210/ping >/dev/null; then UP=1; break; fi
        kill -0 "$TKS_PID" 2>/dev/null || break   # process-group leader died
        sleep 10
    done
    if [ "$UP" -eq 1 ]; then
        echo "Ping OK; probing with a real completion (includes compile warmup)..."
        if probe_completion; then
            echo "Tokasaurus is up and generating (attempt $attempt)."
            break
        fi
        echo "Completion probe failed - frontend alive but workers not generating."
        UP=0
    fi
    echo "Attempt $attempt did not become healthy; restarting tokasaurus."
    kill_tksrs
    TKS_PID=""
    sleep 5
done
[ "$UP" -eq 1 ] || { echo "FATAL: tokasaurus never became healthy after 5 attempts"; exit 1; }

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
