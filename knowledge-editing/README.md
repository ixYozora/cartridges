# Knowledge Editing via Self-Study + LoRA

Thesis pipeline: take CounterFact/AKEW fact edits, use the Cartridges **Self-Study**
mechanism to synthesize a dialogue dataset around each edit, LoRA-finetune
**Qwen2.5-7B-Instruct** (the model AnyEdit edits, for direct comparison) on that dataset,
and evaluate with AKEW-style metrics.

Reference: [`docs/knowledge-editing-self-study.md`](../docs/knowledge-editing-self-study.md).
Current numbers: [`docs/temp_metric_compare.md`](../docs/temp_metric_compare.md).
History: [`docs/CHANGELOG.md`](../docs/CHANGELOG.md).

## Pipeline

```
samples/CounterFact.json                    975 AKEW edits
   │  synthesize.py + filter_dataset.py     slurm/synth_clean.sbatch (tokasaurus teacher)
   │                                        slurm/synth_vllm.sbatch  (vLLM teacher)
   ▼
outputs/<run>/artifact/dataset.parquet           RAW - never train on this
outputs/<run>/artifact/dataset_filtered.parquet  INTERMEDIATE - never train on this
   │  fidelity_filter.py                    slurm/fidelity_filter.sbatch
   ▼                                        (LLM judge; drops targets asserting the old fact)
outputs/<run>/artifact/dataset_final.parquet     <<< THE TRAINING FILE
   │
   │  grom_erase.py (optional)              slurm/grom_erase.sbatch
   │     erase the pre-edit facts from the student first ("MU then FT")
   ▼
   │  lora_finetune.py                      slurm/lora_train_clean.sbatch
   ▼
checkpoints/<adapter>                       NOTE: the saved adapter is the BEST-VALIDATION
   │                                        checkpoint (~epoch 2.7), not the last one
   │  lora_eval.py + judge_v2.py            slurm/eval.sbatch  (judge gate -> generate -> grade)
   ▼
results/<lora-timestamp>/eval_detailed.csv  judge v2 verdicts + string metrics
   │  compare_evals.py <run A> <run B>      paired, edit-clustered bootstrap
   ▼
deltas with 95% CIs
```

## Files

### Pipeline (run for every experiment)

| File | Role |
|------|------|
| `synthesize.py` | Self-Study dataset generation config. `SYNTH_ENGINE=tokasaurus` (default) or `vllm` |
| `filter_dataset.py` | Cleaning stage 1: think-strip repair, context-meta and portability-leak drops |
| `fidelity_filter.py` | Cleaning stage 2: LLM judge drops targets that assert the pre-edit fact -> `dataset_final.parquet` |
| `grom_erase.py` | GROM closed-form erase of the pre-edit facts, from the teacher or the student. `--dry-run` audits on CPU; `--attribution K` picks a layer band |
| `lora_finetune.py` | LoRA training. `--seed` makes init and data order reproducible (unset = historical behaviour) |
| `lora_eval.py` | **The eval script**: AKEW splits (efficacy / generalization / locality / portability), ROUGE, BERTScore, EM, old-target leak, LLM judge (v2 by default) |
| `judge_v2.py` | The judge: extraction first, success derived from two yes/no checks, plus a 39-case calibration set. Replaces v1's 0-5 score, which passed wrong answers (an unedited model scored 51%) |
| `eval_common.py` | Shared helpers: thinking-strip, results layout, HTTP judge with xgrammar-constrained JSON, aggregates |
| `compare_evals.py` | Paired, edit-clustered bootstrap between two runs: success, string leak, `names_new`, locality (`loc_correct`, `loc_bleed`), judge flags |

### Analysis tools (run on demand)

| File | Answers |
|------|---------|
| `rejudge.py` | Re-grades a **stored** run with judge v2, no regeneration -> `<run>/rejudge-v2-<tag>/`. Used for runs evaluated before judge v2 existed. `--calibrate` gates the judge; `--only-types Locality` refreshes part of an existing output |
| `probe_recall_lora.py` | Where the old and the new target rank for every edit, per base(+adapter): did an erase / a preference step actually move them? |
| `make_edit_subset.py` | Carves a training parquet down to a seeded subset of n edits (+ the case_ids for the eval) |
| `score_anyedit.py` | AnyEdit's real metric: MiniLM cosine against `fact_new_uns` (their "Bert Score" is NOT BERTScore). Rescores stored `eval.json` runs on CPU |
| `eval_anyedit_protocol.py` | Generates under AnyEdit's exact protocol (bare user turn, no system prompt, temperature 0.001) and scores with their metric; `--case-ids-file` for subsets |

### Slurm jobs

| Job | Env |
|-----|-----|
| `slurm/synth_clean.sbatch` | `SYNTH_ARGS`, `TEACHER_MODEL`, `KE_DATA_FILE`, `SYNTH_SEED` |
| `slurm/synth_vllm.sbatch` | same, plus `SYNTH_PORT` (vLLM; needed for teachers tokasaurus cannot serve, and for decode-time constraints) |
| `slurm/fidelity_filter.sbatch` | `INPUT` |
| `slurm/grom_erase.sbatch` | `NUM_FORGET`, `SEED`, `OUT`, `EXTRA_ARGS` |
| `slurm/lora_train_clean.sbatch` | `DATA_FILE` (**required**), `BASE_MODEL`, `OUTPUT_DIR`, `TRAIN_SEED`, `KE_ARGS` |
| `slurm/eval.sbatch` | `LORA_DIR` (`none` = no adapter), `BASE_MODEL`, `DATA_FILE`, `JUDGE_PORT`, `JUDGE_MODEL`, `JUDGE_VERSION` |
| `slurm/rejudge.sbatch` | `RUNS`, `TAG`, `JUDGE_MODEL`, `JUDGE_PORT`, `ONLY_TYPES`, `CALIBRATION_STRICT` |
| `slurm/probe_recall.sbatch` | (paths are in the script) |
| `slurm/eval_anyedit_protocol.sbatch` | (configs are in the script) |

### Data and archives

| Path | Contents |
|------|----------|
| `samples/CounterFact.json` | The full 975-edit AKEW/CounterFact set |
| `samples/dev_small.json` | 4 edits, for pipeline smoke tests |
| `samples/CounterFact-forget50-s1.json` | The 50-edit subset of the GROM A/B (`grom_erase.py --dump-forget-subset`) |
| `samples/question_leak_handlabels.json` | 90 hand-labelled portability questions, the audit's reference |
| `GROM/` | Reference clone of the authors' GROM repo (gitignored, MIT): `git clone https://github.com/Batorskq/GROM.git knowledge-editing/GROM` |
| `experiments/` | Tools of finished experiments, kept to reproduce them. Scripts are run from `knowledge-editing/` as `python experiments/<x>.py` (they import the top-level modules). 2-hop seed gate: `analyze_portability_hops.py`. Teacher-erase A/B: `compare_fidelity.py`, `grom_sweep*.sbatch`, `summarize_grom_sweep.py`. GROM -> KE: `grom_ke_{sweep,full,scale,capacity,mlp_full}.sbatch`. 3-term loss: `ke_loss_canary.sbatch`. Scaling curve: `scaling_curve.sbatch`. Locality vs n: `eval_locality_scaling.py`, `locality_scaling.sbatch`, `rejudge_locality.py` + `.sbatch`. Locality seed: `locality_post_synth.sbatch`. Question-leak audit: `audit_question_leakage.py` + `.sbatch` (its rubric fails its gate) |
| `legacy/` | The earlier cartridge/KV route and superseded scripts (`comprehensive_eval.py`, `train.py`, `toka-serving.py`, `serve_judge_vllm.sbatch`, `contexts/`), and `judge_smoke_test.py`, the gate for the old judge v1 (still used by `eval.sbatch` when `JUDGE_VERSION=v1`) |

## Usage

Everything real runs through Slurm (no GPU on the login node). Never pass `--mem` — these
nodes report `RealMemory=1` and any memory request fails. Every flashinfer / vLLM /
tokasaurus job must pin `--gres=gpu:rtx5090:N`; the venv's flashinfer is built for sm120
and crashes on the rtx4090 node.

```bash
# 1) synthesize + regex-filter (teacher included in the job)
sbatch --export=ALL,SYNTH_ARGS="num_samples=36864 name=MyRun" slurm/synth_clean.sbatch

# 2) edit-fidelity judge -> dataset_final.parquet (THE training file)
sbatch --export=ALL,INPUT=<outputs/...>/artifact/dataset_filtered.parquet slurm/fidelity_filter.sbatch

# 3) train
sbatch --export=ALL,DATA_FILE=<...>/dataset_final.parquet slurm/lora_train_clean.sbatch

# 4) evaluate (judge gate + generation + judge v2 grading in one job)
sbatch --time=12:00:00 --export=ALL,LORA_DIR=<checkpoints/...>,DATA_FILE=samples/CounterFact.json \
    slurm/eval.sbatch

# 5) compare two runs
python compare_evals.py results/<baseline> results/<treatment>

# re-grade an OLD run (evaluated before judge v2) and compare those
RUNS="results/<old-run>" TAG=qwen35-9b sbatch --export=ALL slurm/rejudge.sbatch
python compare_evals.py results/<a>/rejudge-v2-qwen35-9b results/<b>/rejudge-v2-qwen35-9b
```

## Things that have bitten us

- **The saved adapter is the best-validation checkpoint** (`load_best_model_at_end`), in
  practice `checkpoint-2500` (~epoch 2.7–2.9). `checkpoint-4575` is the end of epoch 5.
- **Judge v1 passed wrong answers.** Success numbers from before 2026-09-13 are inflated by
  ~10 points; re-grade old runs with `rejudge.py` before comparing them to new ones.
- **Locality** is scored against the correct answer of the *other* subject, so its absolute
  rate is bounded by what the unedited model knows (~40%), not ~95% as under v1.
- **Concurrent eval jobs need distinct `JUDGE_PORT`s**, or they talk to each other's judge.
- Every eval run writes `run_config.json` (adapter, base, data file, seed, judge) — check it
  before trusting a results directory.
