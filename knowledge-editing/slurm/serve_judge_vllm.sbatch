#!/bin/bash
#SBATCH --job-name=judge-vllm
#SBATCH --partition=gpu
# vLLM judge relies on the sm120-built flashinfer → must run on tujestpolin
# (only rtx5090 node). Pin the 5090 gres so Slurm can't use sarmatia.
#SBATCH --gres=gpu:rtx5090:1
#SBATCH --cpus-per-task=8
#SBATCH --time=2-00:00:00
#SBATCH --output=/home/imasoudian/cartridges/knowledge-editing/slurm/logs/%j.log

# vLLM judge server for lora_eval.py / legacy/comprehensive_eval.py / judge_smoke_test.py.
#
# JSON output is enforced with vLLM's xgrammar structured-output backend; the
# eval scripts request it via response_format (see eval_common.ask_judge_http).
# --served-model-name keeps the model id identical to the tokasaurus setup so
# DEFAULT_JUDGE_MODEL ("Qwen/Qwen3-4b") works unchanged.
#
# Submit:            sbatch slurm/serve_judge_vllm.sbatch
# Find the node:     squeue -u $USER -n judge-vllm -o "%N"
# Readiness check:   curl http://<node>:10310/v1/models
# Then point evals:  --judge-base-url http://<node>:10310
# (or localhost:10310 if the eval runs inside the same job/node)
# NOTE: never request --mem on this cluster (nodes have RealMemory=1).

set -euo pipefail

PORT=10310
echo "vLLM judge starting on: http://$(hostname):${PORT}"
echo "Readiness check:        curl http://$(hostname):${PORT}/v1/models"

# lowercase "4b" on purpose: matches the existing HF cache dir (models--Qwen--Qwen3-4b)
exec /home/imasoudian/cartridges/.venv-vllm/bin/vllm serve Qwen/Qwen3-4b \
    --served-model-name Qwen/Qwen3-4b \
    --port "${PORT}" \
    --max-model-len 8192 \
    --gpu-memory-utilization 0.90
