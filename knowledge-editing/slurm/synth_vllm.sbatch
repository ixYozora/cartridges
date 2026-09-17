#!/bin/bash
#SBATCH --job-name=ke-synth-vllm
#SBATCH --partition=gpu
# vLLM's flashinfer is built for sm120 (RTX 5090), so this only runs on
# tujestpolin. Pin the 5090 gres so Slurm can't schedule it on sarmatia (rtx4090).
#SBATCH --gres=gpu:rtx5090:1
#SBATCH --cpus-per-task=8
#SBATCH --time=2-00:00:00
#SBATCH --output=/home/imasoudian/cartridges/knowledge-editing/slurm/logs/%j.log
# NOTE: no --mem on this cluster (nodes have RealMemory=1; any memory request fails).
#
# vLLM variant of synth_clean.sbatch: same synthesize.py, same filter_dataset.py QC,
# different serving engine. Selected by SYNTH_ENGINE=vllm, which makes synthesize.py
# build an OpenAIClient against the local OpenAI-compatible endpoint instead of a
# TokasaurusClient. Everything else (seed mix, weights, num_samples) is shared, so
# this is engine-swap-only and safe to A/B against a tokasaurus run.
#
# Use this instead of synth_clean.sbatch when the teacher is one tksrs cannot serve
# (its fork implements dense llama/qwen2/qwen3 only -- no MoE, no hybrid attention).
#
# Small validation run:
#   sbatch --export=ALL,SYNTH_ARGS="num_samples=512 name=VllmSmoke" synth_vllm.sbatch
# GROM-erased teacher (grom_erase.py saves the tokenizer alongside the weights):
#   sbatch --export=ALL,TEACHER_MODEL=/home/imasoudian/cartridges/checkpoints/qwen3-4b-erased-n50-s1 synth_vllm.sbatch

set -euo pipefail

REPO=/home/imasoudian/cartridges
KE=$REPO/knowledge-editing
export CARTRIDGES_DIR=$REPO
export CARTRIDGES_OUTPUT_DIR=$REPO/outputs
SYNTH_ARGS=${SYNTH_ARGS:-}

export SYNTH_ENGINE=vllm
export TEACHER_MODEL=${TEACHER_MODEL:-Qwen/Qwen3-4b}
# 10210 keeps parity with the tokasaurus setup; the vLLM JUDGE lives on 10310, so
# a synthesis job and a judge job can share a node without colliding.
export SYNTH_PORT=${SYNTH_PORT:-10210}
echo "engine: vllm   teacher: $TEACHER_MODEL   port: $SYNTH_PORT"

# --served-model-name must equal TEACHER_MODEL: the client sends model_name as the
# request's "model" field and vLLM 404s on a mismatch.
# --max-logprobs 20 matches SelfStudySynthesizer.Config.num_top_logprobs (Bot B's
# call requests exactly 20); vLLM rejects top_logprobs > max_logprobs with a 400.
$REPO/.venv-vllm/bin/vllm serve "$TEACHER_MODEL" \
    --served-model-name "$TEACHER_MODEL" \
    --port "$SYNTH_PORT" \
    --max-model-len 8192 \
    --max-logprobs 20 \
    --gpu-memory-utilization 0.90 &
VLLM_PID=$!

kill_vllm() {
    [ -n "${VLLM_PID:-}" ] || return 0
    kill -TERM "$VLLM_PID" 2>/dev/null || true
    wait "$VLLM_PID" 2>/dev/null || true
}
trap kill_vllm EXIT

# Readiness: /v1/models proves the server is up. vLLM has no equivalent of the tksrs
# spawn race, so no restart loop -- if it dies, it dies loudly and the job fails.
UP=0
for i in $(seq 1 60); do   # up to ~10 min (weight load + torch.compile warmup)
    if curl -sf "http://localhost:$SYNTH_PORT/v1/models" >/dev/null; then UP=1; break; fi
    kill -0 "$VLLM_PID" 2>/dev/null || { echo "FATAL: vLLM exited during startup"; exit 1; }
    sleep 10
done
[ "$UP" -eq 1 ] || { echo "FATAL: vLLM never became ready"; exit 1; }

# Thinking-mode guard. Job 11150 lost a whole synthesis arm because enable_thinking
# =False was silently not forwarded and Qwen3's template defaults to thinking. Prove
# the chat_template_kwargs path works BEFORE generating 36k samples: this is the same
# field OpenAIClient puts in extra_body, so a <think> block here means the run would
# have been contaminated.
PROBE=$(curl -sf --max-time 300 -X POST "http://localhost:$SYNTH_PORT/v1/chat/completions" \
    -H 'Content-Type: application/json' \
    -d "{\"model\":\"$TEACHER_MODEL\",\"messages\":[{\"role\":\"user\",\"content\":\"Name one colour.\"}],\"max_completion_tokens\":32,\"temperature\":0,\"chat_template_kwargs\":{\"enable_thinking\":false}}" \
    || true)
[ -n "$PROBE" ] || { echo "FATAL: completion probe returned nothing"; exit 1; }
case "$PROBE" in
    *'<think>'*) echo "FATAL: enable_thinking=false was NOT honoured - probe returned a <think> block:"; echo "$PROBE"; exit 1 ;;
esac
echo "vLLM is up and generating; thinking correctly disabled."

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
