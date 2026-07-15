# Knowledge Editing via Self-Study + LoRA

Thesis pipeline: take CounterFact/AKEW fact edits, use the Cartridges **Self-Study**
mechanism to synthesize an enriched dialogue dataset around each edit, LoRA-finetune
**Qwen2.5-7B-Instruct** (the model AnyEdit edits, for direct comparison) on that
dataset, and evaluate with AKEW-style metrics.

Full reference: [`docs/knowledge-editing-self-study.md`](../docs/knowledge-editing-self-study.md).

## Pipeline

```
samples/CounterFact.json                (975 AKEW edits)
        │  synthesize.py                (Self-Study; teacher Qwen3-4b on tokasaurus)
        ▼
outputs/<run>/artifact/dataset.parquet
        │  lora_finetune.py             (PEFT LoRA on Qwen2.5-7B-Instruct)
        ▼
checkpoints/<lora-dir>
        │  lora_eval.py                 (local LoRA answers + vLLM judge over HTTP)
        ▼
results/<lora-timestamp>/eval.*         (JSON + CSVs incl. AnyEdit-comparable table)
```

## Files

| File | Role |
|------|------|
| `synthesize.py` | Dataset generation config (SelfStudySynthesizer + KnowledgeEditingResource + tokasaurus client) |
| `lora_finetune.py` | LoRA training on the synthesized parquet |
| `lora_eval.py` | **The eval script**: AKEW splits (efficacy/generalization/locality/portability), multi-ref ROUGE, BERTScore, EM, old-target leakage, LLM judge |
| `eval_common.py` | Shared helpers: thinking-strip, results layout, **LLM judge** (xgrammar-constrained JSON on vLLM, `judge_failed` semantics, `judge_aggregates`) |
| `judge_smoke_test.py` | Judge validation (reliability + calibration) — must PASS before any full eval |
| `slurm/serve_judge_vllm.sbatch` | vLLM judge server job (port 10310); logs land in `slurm/logs/` |
| `samples/CounterFact.json` | Full 975-entry AKEW/CounterFact set |
| `samples/dev_small.json` | 4-fact dev set for quick pipeline runs |
| `legacy/` | Deprioritized cartridge/KV path (`comprehensive_eval.py`, `train.py`, `toka-serving.py`, `contexts/`) and superseded scripts (`llm_judge_eval.py`, `lora_serving.py`) |

## Usage

```bash
# 1) Judge server (1 GPU; do NOT add --mem, the cluster nodes have RealMemory=1)
sbatch slurm/serve_judge_vllm.sbatch
squeue -u $USER -n judge-vllm -o "%N"          # -> node, e.g. tujestpolin
curl http://<node>:10310/v1/models             # wait until ready

# 2) Validate the judge (login node is fine; MUST pass before a full eval)
python judge_smoke_test.py --url http://<node>:10310

# 3) Evaluate a LoRA (needs a GPU for the student model)
python lora_eval.py \
    --lora-dir ../checkpoints/lora_qwen2.5-7B-Instruct \
    --base-model Qwen/Qwen2.5-7B-Instruct \
    --data-file samples/dev_small.json \
    --judge-base-url http://<node>:10310

# 4) Finetune on a synthesized dataset
python lora_finetune.py \
    --parquet-path <outputs/...>/artifact/dataset.parquet \
    --model-name Qwen/Qwen2.5-7B-Instruct \
    --output-dir ../checkpoints/<new-run-name>

# 5) Synthesize a dataset (tokasaurus serving the teacher must be running)
python synthesize.py
```

## Judge notes

- The judge runs on **vLLM** so JSON output is grammar-enforced (xgrammar): the
  schema forces `response_claim` → `reason` → flags → `score`, which grounds the
  judge and fixed severe miscalibration of the non-thinking Qwen3-4b judge.
- Judge failures never contaminate scores: they are excluded from `judge_score` /
  `success_rate` and reported as **`Judge Fail %`** — if that column is nonzero,
  investigate before trusting judge numbers.
- Re-run `judge_smoke_test.py` after ANY change to judge prompts, schema, or model.
- `--base-model` in `lora_eval.py` defaults to `Qwen/Qwen3-4b`; always set it to
  the actual base of your LoRA (e.g. `Qwen/Qwen2.5-7B-Instruct`).
