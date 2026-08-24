# Project progress log — Self-Study as a data engine for knowledge editing

*Author reference document. Written 2026-08-03. Covers the full arc of the thesis
work: the state as it stood at the February presentation, and everything that was
rebuilt, fixed, and measured after the project was resumed in mid-July 2026.*

---

## 0. The project in one paragraph

We use the **Cartridges Self-Study** synthesizer as a *data engine* for **knowledge
editing**. Starting from **AKEW / CounterFact** edits (975 counterfactual facts, e.g.
"Wellington's twin city is *Sheffield*" instead of *Sydney*), a two-bot Self-Study
loop generates a large synthetic instruction-tuning corpus that teaches those edited
facts. A **user bot** invents questions around each edit; a **teacher bot** answers
them grounded in the new fact. The corpus is exported to parquet and used to
**LoRA-finetune the student model (Qwen2.5-7B-Instruct)** — the same model AnyEdit
edits, so the comparison is head-to-head. The finetuned adapter is evaluated on the
full CounterFact test suite with **AKEW-aligned metrics** (Efficacy, Generalization,
Locality, Portability) plus judge-free string metrics (BERTScore, ROUGE-L) for a
direct comparison against AnyEdit.

The core research question: **can a Self-Study–generated dataset teach edits as a
finetuning signal, competitively with dedicated editing methods, and what does the
teacher's own knowledge do to that signal?**

---

## PART I — State at the February presentation (what was shown before)

This is the baseline the professor last saw. The *concept* was in place and end-to-end,
but the *measurements* were not yet trustworthy.

### The pipeline as originally designed
- **Five seed-prompt types** driving synthesis (question, negation, correction, plus
  now-inactive creative/ignorance/strict variants), each with a **seed-specific system
  prompt** and **user personas** to diversify phrasing.
- **Two-bot Self-Study loop**: a user-simulator bot (high temperature, persona-driven)
  writes the user turn; a teacher bot (lower temperature) writes the target answer.
- **Teacher / judge model = Qwen3-4b**, served via **tokasaurus**.
- **Student = Qwen3-4b** in the presentation numbers (LoRA r=16, α=32, lr 2e-4,
  ~14,745 training samples). A later run moved to **Qwen2.5-7B-Instruct** as the student
  (LoRA r=64, α=256) with a larger ~32,768-sample dataset.
- Both a **large cartridge** (trainable KV cache) and **per-edit cartridges** were
  explored, alongside the LoRA path.
- An **AnyEdit comparison table** was produced (LoRA ours ≈ 87.9 / 29.7 / 86.4 / 22.5
  on the AKEW-style columns; Cartridge ours ≈ 90.2 / 44.9 / 87.6 / 27.3).

### The two problems that made those numbers unreliable
1. **The LLM judge was silently broken.** The Qwen3 judge emitted `<think>` reasoning
   blocks that broke the JSON parser, so a large fraction of answers (~62% in the
   January eval) were scored **0 regardless of quality**. The reported judge scores
   were therefore not a real quality signal.
2. **The training data was contaminated.** Roughly **20% of training targets contained
   `<think>` blocks** and **~25% contained "context-meta" phrasing** — the teacher
   answering by referring to the prompt itself ("the context provided does not
   mention…") rather than stating the fact. The model was being trained to talk *about*
   the prompt instead of asserting the edited fact.

### The artifacts we inherited when resuming
- A **February dataset** (`2026-02-10-...-synthesize`, 32,768 single-turn conversations
  over all 975 edits; seed mix ≈ ⅓ each negation/question/correction; teacher Qwen3-4b).
- A **LoRA checkpoint** on Qwen2.5-7B-Instruct (r=64, α=256, 5 epochs) trained on that
  parquet.
- **January eval results** produced with the broken judge — later deleted as
  unreliable.

**Take-away for the meeting:** the framework existed and ran end-to-end, but it needed
a correctness pass before any number could be cited.

---

## PART II — What we changed after resuming (July 2026 onward)

The work below is grouped into five phases. Every phase corresponds to concrete
commits (listed in the appendix) and Slurm jobs. Nothing here changes the *concept* —
it makes the concept *measurable and correct*, then improves the weakest result.

### Phase 1 — Correctness reset: judge, data hygiene, infrastructure (2026-07-15)

**Repository restructure.** The whole project was moved out of `examples/` to a
top-level `knowledge-editing/` directory, with active scripts at the top level
(`synthesize.py`, `lora_finetune.py`, `lora_eval.py`, `eval_common.py`,
`judge_smoke_test.py`), a `slurm/` folder for cluster jobs, `samples/` for eval sets,
and `legacy/` for archived scripts. *(commit `4670d83`)*

**Fixed the LLM judge.** Rebuilt the judge on **vLLM with structured (xgrammar)
JSON-constrained decoding**, so it can no longer emit free-form `<think>` blocks that
break parsing. Added a **grounding field** (`response_claim`) to the schema so the
judge must quote what it is scoring. A **calibration smoke test** (`judge_smoke_test.py`)
gates every eval run and must pass 16/16 before evaluation starts. *(commit `745608b`)*

**Fixed the old-target-leak metric.** The leakage detector was a bare substring match
(it flagged "Icelandic" inside a title like "The Icelandic Dream"). It is now a
**word-boundary match** that ignores occurrences inside the subject string and never
flags Locality tests (where the old target is legitimately part of the correct
neighbor answer). *(commit `745608b`)*

**Diagnosed and filtered the data contamination.** Added `filter_dataset.py`, a
heuristic pass that drops rows with `<think>` blocks, context-meta phrasing, and other
surface contamination. *(commit `4e564d4`)*

**Cleaned up synthesis at the source** *(commit `bb50554`)*:
- Turned **`prob_thinking` from 0.2 → 0.0** (no thinking blocks generated at all,
  rather than generating and stripping).
- **Rewrote the negation system prompt**, which had literally *instructed* leakage
  ("if the user's statement contradicts the context, deny it" with no anti-meta rule) —
  negation seeds had a ~67% survival-of-contamination rate. The new prompt does
  parametric denial ("No, X is actually Y") and never references the source.
- **Hardened the question/correction prompts** against meta phrasing.
- Added **uniform edit coverage** (case-id round-robin selection so all 975 edits are
  sampled evenly) and **recorded edit metadata** (case_id, subject, old/new target) on
  every synthesized row, which later filters and analyses depend on.

**Validated the judge fix in production.** A small end-to-end eval (Slurm job 7086)
showed **Judge-Fail = 0.0% on all tests** — the judge fix works in the real pipeline.

### Phase 2 — Clean regeneration and the teacher-fidelity finding (2026-07-15)

**Regenerated a clean corpus.** With the fixes above, synthesis produced a clean
dataset of **35,093 rows** (95.2% keep rate; 0% thinking, ~4.8% meta dropped).

**Discovered teacher edit-fidelity failure — a genuine finding.** By auditing targets
against the edit metadata, we found that **~13% of the teacher's answers assert the
OLD fact** rather than the edited one. Crucially, the dominant failure mode is not a
blunt revert but **reconciliation**: the teacher mentions *both* facts and keeps the
old one true — "holds Iraqi citizenship **but** lives and works in Norway", "dual
citizenship of both Thailand and Italy". The teacher's own parametric knowledge resists
the counterfactual. Training on these rows teaches the model the *old* fact.

**Built an LLM-judge fidelity filter.** Rather than regex heuristics (which only caught
~5% and miss the reconciliation cases), we added `fidelity_filter.py`: an LLM judge
that classifies each target as `asserts_new` / `asserts_old` and **drops the row iff it
asserts the old fact**. A 6-case calibration gate protects each run. *(commit `4e564d4`)*

**Result.** The fidelity filter dropped **4,651 rows (13.3%)** — 2.7× the regex
estimate, because the reconciliation cases are invisible to regex — yielding the final
clean training set **`dataset_final.parquet` (30,442 rows)**. Per-seed drop rates:
correction 17.7% > negation 13.5% > question 9.0%, i.e. **fidelity failure scales with
how directly the seed contradicts the model's prior**.

> **Thesis finding (Phase 2):** the teacher resists counterfactual edits ~13% of the
> time, mostly by *reconciling* both facts rather than reverting, and the rate scales
> with contradiction pressure (correction > negation > question). This is a property of
> using a strong model as a Self-Study teacher over counterfactual content.

### Phase 3 — Clean retrain, full evaluation, report (2026-07-16 → 07-18)

**Retrained the LoRA on clean data.** Trained Qwen2.5-7B-Instruct (r=64, α=256, lr 2e-5,
5 epochs) on `dataset_final.parquet` — hyperparameters identical to the February
checkpoint so **data quality is the only variable**. *(sbatch `204c7f7`, job 7094)*

**Fixed eval reproducibility and reporting.** Fixed a printer key-mismatch that showed
EM/Judge as 0 in the console (display-only bug; CSVs were always correct) and added a
**`--seed` argument** so Locality/Portability prompt sampling is identical across
checkpoint evaluations — required for fair A/B comparisons. *(commit `9207c37`)*

**Ran the first trustworthy full evaluation** (job 7974) on the full CounterFact suite
(975 entries × 7 tests), judge-fail 0%. This became the **clean baseline**. Also
produced the AnyEdit-comparable BERTScore/ROUGE-L table.

**Wrote the project report** (LaTeX, `knowledge-editing/latex/`). *(commit `05df6fa`)*

### Phase 4 — Training-code fix: assistant-only loss masking (2026-07-26)

**The bug.** `lora_finetune.py` computed the language-modeling loss over the **entire
sequence** — padding tokens and the user/prompt tokens included — instead of only the
assistant's answer. The model was partly being trained to reproduce the *questions* and
to predict padding.

**The fix** *(commit `7766cfc`)*:
- **Assistant-only loss masking**: the prompt span is masked to `-100`; loss is computed
  only on the final assistant turn (including its end-of-turn token).
- **Dynamic padding** (`DataCollatorForSeq2Seq`, pad-to-multiple-of-8) replacing a fixed
  2048-token pad. Because these conversations are short (~50–150 tokens), this is a
  **~16× reduction in tokens processed per step → training dropped from ~26 h to ~2 h**,
  a benefit that carries to all future runs.
- Checkpoint directories and W&B run names are now **datetime-stamped** so repeated runs
  never overwrite each other. *(commit `10f0145`)*

**Result — the masking fix is a clear net win, but with one regression.** Head-to-head,
masked-loss vs the clean (unmasked) baseline on the full suite:

| Metric (success %) | Clean (unmasked) | **Masked** | Δ |
|---|---|---|---|
| Efficacy | 85.44 | **89.23** | +3.79 |
| Generalization | 72.56 | **76.92** | +4.36 |
| Locality | 96.00 | 96.87 | +0.87 |
| Portability | 62.62 | **58.31** | **−4.31** |
| OVERALL | 78.26 | **79.06** | +0.80 |

| Old-target leak % | Clean | **Masked** | Δ |
|---|---|---|---|
| Efficacy | 14.36 | **7.18** | −7.18 |
| Generalization | 25.95 | **13.54** | −12.41 |
| Portability | 20.72 | **12.00** | −8.72 |
| OVERALL | 15.38 | **8.32** | −7.06 |

Interpretation: masking prompt tokens **stopped the model over-fitting to input echo**,
which roughly **halved leakage everywhere** and improved Efficacy and Generalization.
The cost was **Portability** (multi-hop reasoning): with less parroting the model also
composes the edited fact into downstream reasoning less often. That single regression
motivated Phase 5. *(eval job 9924, `results/lora-20260726_200333`)*

### Phase 5 — Portability enrichment (2026-07-27 → 07-28)

**Confirmed the regression is real, not an artifact.** A concern was that CounterFact's
`generation_prompts` (used for Portability) contain **duplicated questions** — 317/975
entries repeat a prompt — which could bias the metric. A post-hoc de-duplication showed
the drop persists on distinct questions (baseline 62.6% vs masked 58.7%), so the
regression is genuine. We also **fixed the duplication at the source** so future evals
sample only distinct Portability questions. *(commit `d12ba77`)*

**Added two new Self-Study seed types** to directly target multi-hop reasoning
*(commit `735b54a`)*:
- **Reciprocal seed** — reverse-lookup questions ("which entity has *X* as its *Y*?",
  expecting the subject), so the edit is exercised from both directions.
- **Portability seed** — genuine **two-hop questions**: answering requires *first*
  recalling the edited fact, *then* a further fact that follows from it. The teacher's
  system prompt forces a **one-clause rationale chain written into the target**
  ("*X's Y is Z, so …*"), grounding the downstream answer in the *new* fact.
- **Weighted sampling** over the five seeds so the new types get adequate share.

**Added a portability-leak filter.** Two-hop question generation is run at high
temperature and occasionally names the answer target inside the *question*. A
deterministic filter drops any portability row whose question names the old or new
target. *(commit `1a4d44a`)*

**Infrastructure hardening.** The teacher (tokasaurus) and the vLLM judge depend on a
build of **flashinfer compiled for sm120 (RTX 5090)**, so those jobs must be pinned to
the 5090 node; also added a **startup-retry wrapper** around tokasaurus to survive an
intermittent spawn race. *(commit `84cb62d`)*

**Regenerated → refiltered → retrained → evaluated:**
- Enriched synthesis (job 10058) → 33,934 rows after the heuristic filter.
- Fidelity filter (job 10059) → **`dataset_final.parquet`, 31,273 rows** (kept 92.2%;
  the two new seeds are notably fidelity-clean — reciprocal 0.7% dropped, portability
  3.4% dropped). Comparable in size to the clean baseline's 30,442, so the A/B stays
  single-variable.
- Enriched retrain (job 10060) → adapter `lora_qwen2.5-7B-Instruct-20260727_204654`.
- Enriched eval (job 10074) → `results/lora-20260728_012842`, run under the fixed
  (de-duplicated) portability sampler.

**Result — enrichment recovers Portability and cuts leakage further, at a small
Locality cost.** Head-to-head, enriched vs the masked baseline:

| Metric (success %) | Masked | **Enriched** | Δ |
|---|---|---|---|
| Efficacy | 89.23 | **90.05** | +0.82 |
| Generalization | 76.92 | 76.77 | −0.15 |
| Locality | **96.87** | 94.36 | **−2.51** |
| Portability | 58.31 | **62.56** | **+4.25** |
| OVERALL | 79.06 | **79.63** | +0.57 |

| Old-target leak % | Masked | **Enriched** | Δ |
|---|---|---|---|
| Efficacy | 7.18 | **4.72** | −2.46 |
| Generalization | 13.54 | **9.95** | −3.59 |
| Portability | 12.00 | **7.03** | −4.97 |
| OVERALL | 8.32 | **5.52** | −2.80 |

String metrics improved on **every** test type (OVERALL ROUGE-L 0.294 → 0.359,
BERTScore 0.877 → 0.888, EM 48.1 → 50.5). Portability recovered essentially to the
**clean-baseline level (62.6%)** while *keeping* the leakage and efficacy gains the
masking fix bought. The only cost is Locality (−2.5 pp, still 94%), and part of that is
confounded by RNG drift between the two eval runs (the masked eval predates the
portability-dedup fix, so Locality/Portability sample slightly different question sets;
Efficacy and Generalization are deterministic and fully comparable).

> **Thesis finding (Phase 5):** targeting the weak metric directly at the *data* level —
> reciprocal + two-hop seeds with the reasoning chain written into the teacher target —
> recovered the multi-hop regression and reduced old-fact leakage across the board,
> without sacrificing the direct-edit gains.

---

## PART III — Full three-way picture

Success rate (judge-verified) across the whole arc:

| Test type | Clean (unmasked) | Masked | Enriched |
|---|---|---|---|
| Efficacy | 85.44 | 89.23 | **90.05** |
| Generalization | 72.56 | **76.92** | 76.77 |
| Locality | 96.00 | **96.87** | 94.36 |
| Portability | **62.62** | 58.31 | 62.56 |
| OVERALL | 78.26 | 79.06 | **79.63** |

Old-target leak (lower is better):

| Test type | Clean | Masked | Enriched |
|---|---|---|---|
| Efficacy | 14.36 | 7.18 | **4.72** |
| Generalization | 25.95 | 13.54 | **9.95** |
| Portability | 20.72 | 12.00 | **7.03** |
| OVERALL | 15.38 | 8.32 | **5.52** |

The narrative in one line: **clean data → masking fix (leak halved, efficacy up,
portability down) → enrichment (portability restored, leak down further).** The current
enriched adapter is the model going forward.

---

## PART IV — Important caveat about the leak metric (be precise when citing it)

The old-target-leak metric (`check_old_target_mentioned`) is a **lexical word-boundary
match** for the old-target string. It measures *whether the old entity is mentioned*,
**not whether the old fact is asserted in the edited relation.** This over-counts in a
specific, benign way:

- **Genuine fact-revert** (a true leak): e.g. edit "HQ located in Philadelphia → Mumbai";
  the model answers a downstream question with *"…surrounded by Philadelphia."* — the
  old fact drives the answer.
- **Benign co-mention** (a false positive of the metric): e.g. edit "mother tongue
  Korean → French"; the model answers *"her mother tongue is French … before learning
  Korean."* — the edit is correct (mother tongue = French), and "learned Korean later"
  does **not** contradict it. The string "Korean" appears, so the metric flags it, but
  no fact was reverted.

**Consequence:** the reported leak percentages are a **conservative upper bound**. The
true fact-revert rate is lower. This cuts both directions (baseline numbers are inflated
by the same over-count), so the *relative* improvement across phases likely holds under
a stricter definition, but the *absolute* levels are overstated. A relation-aware or
LLM-judge re-scoring of the flagged rows would give a defensible "true leak rate";
this is a planned refinement, not yet run.

---

## PART V — Current state and next steps

> This narrative log covers the rebuild through **2026-08-03**. Everything after that —
> the DCT portability seed, the GROM work, and the two negative results — is recorded
> day by day in [`CHANGELOG.md`](CHANGELOG.md), which is the live record. This section
> is the summary as of **2026-08-24**.

**Where the work stands.** Three adapters have been trained on progressively better
data and evaluated on the full 975-edit set with a fixed seed:

| adapter | data | OVERALL success | leak | notes |
|---|---|---|---|---|
| `lora_...-20260726_143439` | masked-loss baseline | 79.06 | 8.32 | loss-masking fix |
| `lora_...-20260727_204654` | + reciprocal/portability seeds | 79.63 | 5.52 | enrichment |
| `lora_...-20260812_002743` | + DCT portability seed | 79.52 | 5.27 | **statistically flat vs the above** |

**Two negative results, both well-instrumented** (details and CIs in the changelog):

1. **The DCT seed is a null on the eval.** It improved the *data* substantially
   (question leak -75%, duplicates -90%, uniform hop spread) and moved no evaluation
   metric outside its 95% paired-bootstrap CI — including portability, its own target.
   Keep the seed anyway: it is free at inference and strictly cleaner data.
2. **Erasing the teacher does not work, and we know why.** GROM-erasing the Qwen3-4b
   teacher left the fidelity judge's `asserts_old` rate unchanged, with a **flat
   dose-response** across a 200x range of suppression strength — falsification rather
   than under-tuning. Mechanism: roughly two-thirds of `asserts_old` is the teacher
   **echoing an old target that our own seed prompts hand it** (r = 0.837 across seed
   types; within-run, the in-prompt bucket moves 12.46% -> 12.48%). No weight edit can
   reach that. This also **supersedes** the earlier reading that the
   correction > negation > question gradient was evidence of parametric reconciliation.

**In flight.** The student-side erase — the prof's "MU first, then FT" idea, a different
mechanism that the falsification above does not touch. All 975 pre-edit facts were
GROM-erased from Qwen2.5-7B-Instruct at two strengths (7.4x and 27x suppression), each
LoRA-trained on the DCT dataset; the two evals give a three-point dose-response against
the unerased DCT adapter. Locality (95.18%) is the metric most at risk.

**Open / next:**
1. **Read the student-erase dose-response.** Flat ⇒ falsified like the teacher arm;
   monotone ⇒ the mechanism works and is worth scaling.
2. **AnyEdit head-to-head** — the primary external comparison, on the judge-free
   AKEW-comparable string metrics (DCT: Ori 0.9233 / 0.5685, Para 0.8926 / 0.4130).
   This does not depend on either erase arm.
3. **3-term student loss** (NLL + bounded forget + token-KL retain) — unaffected by the
   teacher falsification, since it targets the student.
4. **Infrastructure**: migrate synthesis serving from tokasaurus to vLLM; a prerequisite
   for any future teacher upgrade.
5. **True leak-rate re-scoring** — separate genuine fact-reverts from benign
   co-mentions on the flagged rows (Part IV).

---

## Appendix A — Commit history since resuming (branch `dev`)

| Commit | Date | Summary |
|---|---|---|
| `4670d83` | 07-15 | move knowledge-editing from examples/ to repo root, archive legacy scripts |
| `745608b` | 07-15 | fix LLM judge (vLLM + xgrammar JSON) and eval metrics, add smoke test |
| `bb50554` | 07-15 | synthesis: disable thinking, fix leaky prompts, record edit metadata |
| `4e564d4` | 07-15 | add dataset QC filters: phrasing and judge-based edit fidelity |
| `204c7f7` | 07-15 | add slurm job for LoRA retrain on clean dataset |
| `82917d1` | 07-15 | rename synthesis dataset to CounterFact-SelfStudy |
| `9207c37` | 07-26 | fix eval metric printer key mismatch and add --seed for reproducible sampling |
| `83203a5` | 07-26 | support LORA_DIR and DATA_FILE overrides in eval sbatch |
| `05df6fa` | 07-26 | add project report |
| `7766cfc` | 07-27 | mask loss to assistant tokens with dynamic padding in lora_finetune |
| `10f0145` | 07-27 | datetime-stamp checkpoint dir and wandb run name |
| `d12ba77` | 07-27 | dedupe generation_prompts before sampling portability tests |
| `735b54a` | 07-27 | add reciprocal and portability self-study seeds with weighted mix |
| `1a4d44a` | 07-27 | drop portability rows that leak the edited target in the question |
| `84cb62d` | 07-27 | pin flashinfer jobs to rtx5090 and add tksrs startup retry |
| `6df78ea` | 07-27 | rename eval_small.sbatch to eval.sbatch |
| `5aa6ef9` | 07-27 | require DATA_FILE in lora_train_clean.sbatch |

**Nothing has been committed since `5aa6ef9`.** All of the August work — the DCT
portability seed, the GROM adapter and sweeps, the fidelity A/B tooling, the
thinking-override fix, the `viz/` usability pass and these docs — is **uncommitted in
the working tree** pending review. `git status` is the authority; see
[`CHANGELOG.md`](CHANGELOG.md) for what each file does and why.

## Appendix B — Key artifacts

- **Final enriched training set:** `outputs/2026-07-27-20-11-50-synthesize/CounterFact-SelfStudy-Enriched-0/artifact/dataset_final.parquet` (31,273 rows)
- **Enriched adapter:** `checkpoints/lora_qwen2.5-7B-Instruct-20260727_204654`
- **Enriched eval:** `results/lora-20260728_012842/` (summary, detailed, AnyEdit-comparable CSVs + eval.json)
- **Masked baseline eval:** `results/lora-20260726_200333/`
- **Clean (unmasked) baseline eval:** `results/lora-20260717_190053/`
- **Eval test set:** `knowledge-editing/samples/CounterFact.json` (975 entries)
