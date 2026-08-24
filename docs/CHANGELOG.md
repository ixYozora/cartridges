# Changelog

Dated engineering log, newest first. One entry per meaningful change to the pipeline,
its data, or its results. Narrative context and full result tables live in
[project-progress-2026.md](project-progress-2026.md); this file is the "what changed,
when, and what it measured" index.

Conventions: paths are repo-relative; Slurm job IDs are given where a change was
validated by a run; every number quoted here came from a run or a check, not an
estimate.

---

## 2026-08-23 — Student erase ("MU first, then FT"): setup and dose-response run

The prof's 2026-08-04 idea, recorded but never executed while we built the teacher arm.
It is a different experiment and today's falsification does not carry over: it targets
the *edit* rather than data quality, so the context-echo mechanism is irrelevant. The
metric it should move is the student's old-target leak at eval (5.27% overall, 9.13% on
generalization).

**Three changes were needed before it could run** (`grom_erase.py`):

- **Alpha source.** Forgetting all 975 edits leaves no held-out facts. Used CounterFact
  `attribute_prompts`, whose gold is `target_new`: they keep alpha HIGH on the
  suppressed `target_true` — unlike neighborhood prompts, whose gold *is* `target_true`
  and which collapse alpha to 0 — while protecting what the edit installs. Result
  `mean_alpha = 0.391`, with **634/1,972 forget positions correctly protected** because
  those tokens are some *other* edit's new target.
- **`--solve-device`.** At vocab 152,064 x d 3,584 the head target and its solve are
  ~4.4 GB each in float64, which will not fit beside a bf16 7B on one 32 GB card. CPU
  solve takes seconds. Weight updates are now device-safe so a CPU-solved P applies to
  CUDA weights.
- **`--max-retain-seqs`.** Key collection is one forward pass per sequence and the full
  retain set is 39k sequences; needed to make a sweep affordable.

Also added `BASE_MODEL` overrides to `lora_train_clean.sbatch` and `eval.sbatch` (the
eval must be given the same base the adapter was trained on), and an `OUTPUT_DIR`
override so an eval can be chained at submit time.

**Sweep (job 11173, 8 min).** Attributed band **[16-20]** of 28 (the teacher's [14-18]
of 36 does not transfer). The student is a much harder target than the teacher:

| config | forget/plain | neigh/plain |
|---|---|---|
| head20 | 18 → 26 | 19 → 22 |
| head80 | 18 → 62 | 19 → 29 |
| head160 | 18 → 189 | 19 → 42 |
| head320 | 18 → 1,030 | 19 → 85 |

Two structural differences from the teacher. **The student knows the old facts far
better** — median rank 18/8 and top-1 up to 23%, against the teacher's 66/50 — so unlike
the teacher case there is real parametric knowledge to remove. But **the trade-off is
much worse**: the teacher reached 3,000x suppression at 1.4x neighborhood; the student
needs head320 for 336x and pays 4.5x. That follows from 975 edits sharing one ridge
system at `mean_alpha` 0.39, i.e. far less suppression budget per fact. Fluency is
untouched throughout (rep4 0.041 flat — that is the base model's own baseline).

**Erases at full retain (jobs 11174 / 11176).** Using all 39,620 retain keys instead of
the sweep's 6,000 buys a better trade-off — less suppression, disproportionately less
collateral:

| | forget/plain | retain/plain | neigh/plain | ‖P‖_F |
|---|---|---|---|---|
| head160 | 18 → 133 (7.4x) | 15 → 19 | 19 → 35 (1.8x) | 2.659 |
| head320 | 18 → 494 (27x) | 15 → 23 | 19 → 63 (3.3x) | 5.318 |

**Running**: jobs 11175 / 11177, LoRA on the DCT `dataset_final.parquet` (32,524 rows,
identical hyperparameters) from each erased base → `checkpoints/lora-on-erased-head160`
and `-head320`. Evaluated against the unerased DCT adapter
(`results/lora-20260823_173846`) as the third point of a **dose-response**: unerased,
7.4x, 27x. That structure is what made the teacher result decisive — a flat response
across a wide suppression range falsifies; a monotone one confirms.

Locality (95.18% today) is the metric at risk and the one to read first.

---

## 2026-08-23 — GROM arm falsified, and the reason corrects an earlier interpretation

Stage 2 of the grid. `beta_head=80` puts the old target at rank ~150,000 of 151,936 —
essentially the least likely token in the vocabulary — with retain and fluency intact
(synthesis keep rate 94.9%, matched to base). Jobs 11170 → 11171 → judge 11172.

| | correction Δ | ALL Δ | 95% CI (ALL) |
|---|---|---|---|
| head20 (13x suppression) | -1.60 pp | -0.38 pp | [-1.19, +0.40] |
| head80 (**3,000x** suppression) | -1.13 pp | -0.31 pp | [-0.99, +0.35] |

**The dose-response is flat.** A 200-fold increase in suppression produced no more
behavioural change. This is a falsification of the hypothesis, not a tuning failure:
suppressing the old target token in the teacher does not stop the teacher asserting the
old fact.

**Why — and this corrects what we concluded on 08-11.** Across seed types, the
`asserts_old` rate tracks how often *our own seed prompts hand the teacher the old
target in the user turn* (Pearson r = 0.837):

| seed | old target in user turn | asserts_old |
|---|---|---|
| correction | 92.3% | 14.27% |
| negation | 78.0% | 9.93% |
| question | 1.7% | 8.27% |
| reciprocal | 2.4% | 0.20% |
| portability | 0.0% | 0.53% |

And the within-run split is decisive (edit-clustered CIs; dropped rows recovered by
message matching, which over-counts the base arm ~7%, so absolute rates are
approximate):

| subset | n | base | head80 | Δ pp | 95% CI |
|---|---|---|---|---|---|
| old target **is** in the user turn | 2,054 | 12.46% | 12.48% | +0.02 | [-1.70, +1.58] |
| old target **not** in the user turn | 5,697 | 3.07% | 2.81% | -0.26 | [-0.71, +0.20] |

Roughly two-thirds of `asserts_old` is the teacher **echoing an old target that the
prompt just gave it** — a conversational failure, not parametric recall. No weight edit
can reach it, by construction. Where the old target is *not* supplied, the base rate is
already 4x lower and the erase still does not move it significantly.

**The 08-11 reading was wrong.** We recorded the correction > negation > question
gradient as evidence that "teacher-side reconciliation is parametric, exactly the
GROM-erase target". It is better explained by how often each seed type puts the old
target in front of the model. `question` is the visible outlier — 1.7% in-prompt but
8.27% asserts_old — so a genuinely parametric component exists, but it is small and the
erase does not significantly move it either.

**Recommendation: stop investing in the teacher-erase arm.** It is a clean negative
result with an identified mechanism, which is worth writing up: *unlearning the teacher
does not fix in-context reconciliation, because the dominant failure is context echo
rather than recall.* That reframes Steps 4-5 (full 975-edit erase) as not worth running
on the current evidence. Step 6 (the 3-term student loss) is unaffected — it targets the
student — and Track B (the AnyEdit head-to-head) does not depend on this arm at all.

---

## 2026-08-23 — GROM grid stage 1: suppression saturates on the head, the MLP arm is a dead end

Job 11169, 13 minutes, 10 configurations. Added to `grom_erase.py`: `--attribution K`
(the authors' logit-lens layer scorer, imported from their `scripts/` rather than
reimplemented, so the band is chosen by exactly their procedure), a free-form fluency
canary (`gen_probe`, greedy generations + repeated-4gram fraction — the rank probes
only cover *factual* collateral, an over-strong edit breaks generation instead), and
`--no-save` (a Qwen3-4b checkpoint is ~8.8 GB and a sweep only wants the numbers). New
`summarize_grom_sweep.py` collapses the reports into one promotion table.

**Attributed band for Qwen3-4b: layers [14, 15, 16, 17, 18]** of 36.

Median gold rank, before → after:

| config | forget/plain | forget/chat | retain/plain | neigh/plain | rep4 | verdict |
|---|---|---|---|---|---|---|
| head20 (3e ref) | 66 → 885 | 50 → 322 | 41 → 41 | 65 → 74 | 0.000 | safe but weak |
| head40 | 66 → 7,631 | 50 → 9,388 | 41 → 47 | 65 → 76 | 0.000 | promote |
| **head80** | **66 → 150,105** | **50 → 151,212** | **41 → 47** | **65 → 90** | **0.000** | **promote** |
| head160 | 66 → 151,936 | 50 → 151,936 | 41 → 50 | 65 → 137 | 0.000 | collateral |
| head320 | 66 → 151,936 | 50 → 151,936 | 41 → 58 | 65 → 305 | 0.754 | collateral |
| mlp65 | 66 → 151,936 | 50 → 151,936 | 41 → 61 | 65 → 75 | 0.000 | collateral (chat retain 1.7x) |

Two findings.

**Suppression saturates, and `head80` is the operating point.** Vocabulary is 151,936,
so rank 151,936 is dead last — head160 and above are clipped. `head80` puts the old
target at ~150,000, essentially the least likely token in the vocabulary, while retain
moves 41→47, neighborhood 65→90, and fluency is untouched. Past that, neighborhood
degrades (137 at head160, 305 at head320) and generation breaks outright (rep4 0.75 at
head320).

**The MLP arm is a dead end at published strengths.** Every MLP configuration damages
retain and neighborhood more than the head does for the same suppression, and it
degrades fast: mlp600 gives rep4 0.842, mlp1500 pushes retain 4,814x and neighborhood
5,548x with rep4 0.982 (complete degeneracy). The published MLP strengths (65–1500) are
calibrated for TOFU/MUSE *corpus* unlearning, where the forget set is thousands of
documents; against 50 facts they are wildly over-powered. mlp65 is the only survivor
worth remembering, and only as a fallback — it saturates forget with modest damage on
the plain key form, but 1.7x on retain/chat.

**Why this makes stage 2 decisive rather than more tuning.** `head80` maximises
suppression subject to leaving the model intact. If the fidelity judge still shows no
movement there, the hypothesis under test — that suppressing the old target token stops
the teacher asserting the old fact — is **falsified**, not merely under-tuned. That is a
result either way.

**Submitted**: 11170 (erase at `beta_head=80` → `checkpoints/qwen3-4b-erased-n50-s1-head80`,
named to contain the model family so the thinking-override resolver matches) → 11171
(synthesis, 8,192 samples, same 50-edit subset, `SYNTH_SEED=3`). Judge run follows once
the output path exists; compared against the existing base arm from job 11149.

---

## 2026-08-23 — Step 3e result: the erase does NOT reach free-form generation

Jobs 11149 (base teacher) / 11155 (erased teacher) → fidelity judge 11157 / 11158,
8,192 samples each over the same 50 edits, calibration 6/6 and 0 judge failures on both.

**Generation quality is untouched by the erase** — the two arms are matched at the
filter stage: keep rate 94.6% vs 94.8%, think-repaired 0 vs 1, user-message flags 44 vs
48, every drop reason within noise, seed mixes aligned. The "hollow data" risk did not
materialise at these settings.

**But the reconciliation rate barely moves** (CIs clustered by edit, 50 clusters):

| seed type | base | erased | Δ pp | 95% CI | verdict |
|---|---|---|---|---|---|
| correction | 14.27% | 12.68% | -1.60 | [-4.36, +1.06] | not significant |
| negation | 9.93% | 9.63% | -0.31 | [-2.47, +1.82] | not significant |
| question | 8.27% | 8.08% | -0.19 | [-1.78, +1.38] | not significant |
| portability | 0.53% | 0.43% | -0.09 | [-0.50, +0.34] | not significant |
| reciprocal | 0.20% | 0.32% | +0.12 | [-0.19, +0.58] | not significant |
| **ALL** | **5.56%** | **5.18%** | **-0.38** | **[-1.19, +0.40]** | not significant |

This answers the caveat left open by the 3b probe. Suppressing the old target from rank
66 to rank 885 under teacher forcing does **not** stop the teacher asserting the old
fact when it writes a free-form answer with the counterfactual in context. Correction is
directionally right and is the largest effect, which is consistent with the mechanism,
but it is inside its interval.

**Why this is a plausible outcome rather than a broken run.** GROM's ZsRE
hyperparameters (`beta_head=20`) are tuned against ZsRE's metric — does the model emit
the true answer at the request prompt — which is exactly the teacher-forced probe, not
free-form assertion. Rank 885 is still the top 0.6% of the vocabulary, and in a long
generation with the fact present in context there are other routes to it (copying from
context, alternate phrasings). Reconciliation may also be a hedging *behaviour*
("holds both", "originally from X, now Y") rather than retrieval of one token, which a
logit-lens push at a handful of key positions would not remove.

**Three escalations, cheapest first** (an erase is 11 s, a synth arm ~5 min, a judge run
~5 min, so each point costs ~15 min):

1. **`beta_head` sweep** (40 / 80 / 160). The quality canaries are already in place:
   filter keep rate, `repaired_by_think_strip`, and the neighborhood probe in
   `grom_report.json`.
2. **Add the MLP band** (`--beta-mlp > 0`). GROM's ZsRE config is head-only *because*
   ZsRE scores next-token behaviour; ours does not. The residual-writer edit is the
   deeper intervention and is arguably the more principled fit. Needs
   `GROM/scripts/layer_attribution.py` to pick Qwen3-4b's band — Llama's [14-18] of 28
   layers does not transfer to 36.
3. **Broaden the forget keys** beyond the single request prompt to `paraphrase_prompts`
   and `attribute_prompts`, so the suppression covers more of the phrasings in which
   the teacher actually asserts the fact.

---

## 2026-08-23 — Silent thinking-mode fallback cost a synthesis arm (fixed)

The first erased-teacher arm (job 11150) came back looking as if GROM had lobotomised
the teacher: 100% of rows think-repaired despite `prob_thinking=0.0`, all 8,192 user
messages flagged, portability kept 24/2549 with `portability_leak` at 30.8% (vs 1.9%),
27 min runtime vs 11.

It was none of GROM's doing. `MODEL_TO_THINKING_OVERRIDES` in
`cartridges/utils/thinking.py` is keyed by hub id (`"qwen/qwen3-4b"`), and
`TokasaurusClient.chat` looked it up by exact match on `config.model_name`. A **local
checkpoint path** misses the dict and falls through to `thinking_overrides = {}`, so
`enable_thinking=False` is never forwarded — and **Qwen3's chat template defaults to
thinking mode**. The arm measured thinking-vs-non-thinking, not erased-vs-base.

Ruled out GROM as the cause by generating from the erased checkpoint through plain
transformers: **byte-identical output to the base model**, so the weights were fine and
the fault was in serving.

**Fixed**: new `thinking_overrides_for(model_name, enable_thinking)` falls back to
matching the model family inside a path (`"qwen3-4b"` in
`"checkpoints/qwen3-4b-erased-n50-s1"`), wired into the tokasaurus client, plus a
`logger.warning` whenever a model is unresolved *and* `enable_thinking=False` was asked
for — the silent case is the dangerous one.

**Standing lessons**: name any patched teacher checkpoint so it contains the base model
family, and treat `repaired_by_think_strip` in the filter report as the canary — it
should be ~0. Discarded output lives at `outputs/2026-08-23-18-18-46-synthesize/`
(`Erase50-erased-0`); the valid rerun is `Erase50-erased-v2-0`.

---

## 2026-08-23 — Step 3e set up: erased-vs-base teacher A/B

The decisive experiment for the teacher-erase arm. Same 50 edits, same seed mix, same
sample count; the only variable is which teacher generated the data. The metric is the
fidelity judge's `asserts_old` drop rate per seed type — a dropped row is exactly one
where the teacher asserted the pre-edit fact — against the quantified "before" from job
10751: correction 17.9%, negation 13.1%, question 9.1%, portability 1.0%,
reciprocal 0.7%.

**Changed**

- `knowledge-editing/synthesize.py` — three env overrides: `TEACHER_MODEL` (client
  model name, so the A/B can point at a GROM-patched checkpoint), `KE_DATA_FILE` (run
  over an edit subset instead of all 975), `SYNTH_SEED` (seeds the stdlib RNG for edit
  order, seed-type draws, personas, portability fact anchors). The seeding comment
  records the honest limit: synthesis is async with many batches in flight, so runs are
  **not** bit-identical — good enough for a per-seed-type rate over thousands of rows,
  not for per-sample pairing.
- `knowledge-editing/slurm/synth_clean.sbatch` — `TEACHER_MODEL` threaded into the
  `tksrs` launch and the health-probe payload, which must agree on the name.
- `knowledge-editing/fidelity_filter.py` — the report now also carries
  `per_seed_case`, a per-edit kept/dropped breakdown.
- `knowledge-editing/grom_erase.py` — `--dump-forget-subset` writes the forget edits as
  a CounterFact-shaped JSON *from the same split that drove the erase*, so the subset
  cannot drift out of sync with the erased facts. Produced
  `samples/CounterFact-forget50-s1.json` (50 edits, seed 1).

**Added**

- `knowledge-editing/compare_fidelity.py` — the A/B readout. Reports per-seed drop
  rates with a 95% CI on the difference, and **clusters that CI by edit** whenever both
  reports carry `per_seed_case`. This matters: 50 edits at 8,192 samples is ~164 rows
  per edit, all the same fact re-asked, so a row-level interval is badly
  over-optimistic. Both arms cover the same edits, so the bootstrap resamples the same
  edit list for each arm, which also removes between-edit difficulty from the
  comparison. Falls back to row-level intervals on older reports, with a printed
  warning.

**Validation on the two runs we already have** (enriched → DCT, row-level CIs since
those reports predate `per_seed_case`):

| seed type | enriched | DCT | Δ pp | 95% CI | verdict |
|---|---|---|---|---|---|
| correction | 18.07% | 17.87% | -0.20 | [-1.72, +1.32] | not significant |
| negation | 13.47% | 13.15% | -0.32 | [-1.60, +0.96] | not significant |
| question | 9.14% | 9.08% | -0.06 | [-0.99, +0.87] | not significant |
| portability | 3.40% | 1.04% | -2.37 | [-2.78, -1.95] | significant |
| reciprocal | 0.74% | 0.72% | -0.02 | [-0.30, +0.26] | not significant |

This puts a confidence interval on the 08-11 claim: prompt work moved *nothing*
parametric. The one significant row is portability, the seed whose questions the DCT
change actually rewrote.

**Submitted**: jobs 11149 (base teacher) → 11150 (erased teacher, chained
`--dependency=afterok` so the two cannot collide on tokasaurus port 10210), 8,192
samples each over `CounterFact-forget50-s1.json`, `SYNTH_SEED=3`. The fidelity judge
stays on base Qwen3-4b in both arms — it is handed the old target as a string and
compares text, so it must not be the erased model.

---

## 2026-08-23 — Step 1 scored: the DCT seed is a null result on the eval

Job 11145 (2h24m, judge failures 0.00%) evaluated the DCT adapter
`lora_qwen2.5-7B-Instruct-20260812_002743` on full CounterFact, seed 82 →
`results/lora-20260823_173846/`. The two runs pair exactly: all 6,825 tests match on
(entry_id, type, question), so this is a clean single-variable A/B against the enriched
adapter — only the portability seed differs.

| test type | n | success enr → DCT | leak enr → DCT | judge enr → DCT |
|---|---|---|---|---|
| Efficacy | 975 | 90.05 → 89.33 (-0.72) | 4.72 → 5.44 (+0.72) | 4.60 → 4.57 |
| Generalization | 1950 | 76.77 → 76.92 (+0.15) | 9.95 → 9.13 (-0.82) | 4.13 → 4.12 |
| Locality | 1950 | 94.36 → 95.18 (+0.82) | 0.00 → 0.00 | 4.77 → 4.81 |
| Portability | 1950 | 62.56 → 61.54 (-1.03) | 7.03 → 6.62 (-0.41) | 3.55 → 3.52 |
| **OVERALL** | 6825 | **79.63 → 79.52 (-0.12)** | **5.52 → 5.27 (-0.25)** | 4.21 → 4.21 |

**Every delta is inside its 95% paired-bootstrap CI, including portability — the seed's
own target.** Overall CI [-1.22, +0.94]; portability [-3.38, +1.33]. 20.3% of individual
test verdicts flipped, so there is plenty of per-test churn, but it cancels: this is
generation noise, not a shift. String metrics moved slightly and consistently upward
(OVERALL ROUGE-L 0.3587 → 0.3690, Efficacy +0.0272) without the judge following, i.e.
answers became more lexically reference-aligned without being scored better.

**Read.** The DCT seed produced large, real improvements to the *data* — question leak
-75%, duplicates -90%, `portability_leak` drops -66%, hop-anchor spread uniform — and
none of it reached the evaluation. Taken with the fidelity-judge finding from 08-11
(the correction > negation > question asserts_old gradient is unmoved by prompt work),
the reasonable conclusion is that **prompt- and data-level engineering has run out of
headroom on this pipeline**, and the residual errors are parametric — which is the
argument for the teacher-erase arm rather than more seed design.

Keep the DCT seed regardless: it is free at inference and strictly cleaner data.

---

## 2026-08-23 — GROM pilot erase: mechanism works, collateral is near zero

Two runs. Job 11146 used a probe that mixed both key forms into one average and was
**not trustworthy** — the chat form as first built put the key at the assistant's very
first token, where an instruction-tuned model opens a sentence ("It" essentially
always, gold median rank ~28,800). Averaging that with the plain form produced a
"before" baseline of rank ~19,000, i.e. noise, against which any suppression looks
impressive. Discarded.

Fixed two things and re-ran as job 11147:

- `build_chat` now renders the assistant turn as **stem + answer**, so the key sits
  mid-sentence where the teacher actually emits the fact. Measured on the base model,
  this moved chat-form median gold rank from 28,848 to 139.
- `probe_recall` reports **per key form, never averaged**, with median gold rank as the
  headline. Top-1 is retained but is not the signal: at a mid-sentence position an
  instruction-tuned model's argmax is usually punctuation or markdown, so top-1 reads
  ~0% even when the fact sits at rank 5.

Result (50 edits, head-only, `beta_head=20 w_r=30 rho=0.03`, 11 s of compute,
`||P||_F = 1.679`) → `checkpoints/qwen3-4b-erased-n50-s1/`:

| probe set | median gold rank | mean gold logprob |
|---|---|---|
| forget / plain | 66 → **885** | -7.24 → -11.55 |
| forget / chat | 50 → **322** | -20.48 → -27.74 |
| retain_facts / plain | 41 → 41 | -7.00 → -7.03 |
| retain_facts / chat | 30 → 30 | -19.10 → -19.14 |
| neighborhood / plain | 65 → 74 | -7.22 → -7.43 |
| neighborhood / chat | 27 → 28 | -19.81 → -20.02 |

Targeted suppression of 6-13x with the retain sets unmoved — including the
neighborhood prompts, which share the suppressed gold token and differ only in the
subject. `w_r = 30` transfers to our data without tuning, so the 3c sweep is not a
prerequisite.

**Caveat**: rank 885 is suppressed, not annihilated (still top 0.6% of a 151,936
vocabulary). Whether that is enough to stop the teacher *writing* the old fact in
free-form generation is not something this probe answers — only 3e (A/B synthesis
scored by the fidelity judge) does. A `beta_head` sweep now has a concrete purpose:
find the point where forget collapses further while neighborhood still holds.

---

## 2026-08-23 — GROM teacher-erase groundwork (Step 3a)

Received the GROM repository from the authors and built the adapter that applies it to
our teacher. Nothing has been erased yet; this entry covers the audit and the code.

**Added**

- `knowledge-editing/slurm/grom_erase.sbatch` — the erase job. `NUM_FORGET`/`SEED`/
  `OUT`/`EXTRA_ARGS` overridable; pinned to rtx5090 not because of flashinfer (GROM
  does not use it) but because the float64 accumulator and solve result are ~3.1 GB
  each, tight against a 4090's 24 GB alongside the bf16 model.
- `knowledge-editing/GROM/` — reference clone of https://github.com/Batorskq/GROM
  (MIT). Gitignored: it is upstream read-only, not vendored. Re-create with
  `git clone https://github.com/Batorskq/GROM.git knowledge-editing/GROM`.
- `knowledge-editing/grom_erase.py` — teacher-erase adapter. Imports the closed form,
  the specificity weight and the key collection from the authors' package rather than
  reimplementing them; supplies our data, our key forms and our alpha handling.
  `--dry-run` builds and audits everything on CPU with no model load.

**Compatibility audit — the PyTorch pin does not bind**

GROM pins py3.11 / torch 2.4.1 / transformers 4.51.3 / numpy 2.2.3. We run py3.12.3 /
torch 2.9.1+cu128 / transformers 4.53.0 / numpy 2.3.5. Their own CPU test suite
(`tests/test_grom.py`) passes **8/8** under our `.venv`, so the closed form is
version-insensitive and the pin is a record of what they used, not a requirement.

- `scipy`, `accelerate`, `sentencepiece`, `bitsandbytes` are pinned in
  `requirements.txt` but never imported anywhere in the repo. Real dependencies are
  torch, transformers, pyyaml (+ `datasets` for the TOFU path, `matplotlib`/`numpy`
  for one figure) — none of which we need to add.
- `edit.py:load_model` calls `from_pretrained(dtype=...)` behind an
  `except TypeError` fallback to `torch_dtype=`. Verified that transformers 4.53 does
  raise TypeError there, so the fallback fires and bf16 loads correctly — no silent
  fp32 load.
- GROM uses plain torch/transformers with **no flashinfer**, so unlike every other GPU
  job in this repo it is not pinned to `tujestpolin`/rtx5090.

**Finding: for fact editing GROM edits the LM head, not `down_proj`**

`zsre/hparams/*.json` set `beta_mlp=0.0, beta_head=20.0, head=true, w_r=30, rho=0.03,
key_pos=subject_last`. The README's "only the edited down_proj weights differ"
describes the TOFU/MUSE/WMDP path. Consequences:

- "GROM requires a dense `down_proj`" is **not** a hard architecture constraint; the
  head-only variant needs only `model.lm_head` and `model.model(...)`. This corrects
  one of the three reasons recorded for dropping Qwen3.5 as teacher on 2026-08-11. The
  decision stands on the other two (tokasaurus cannot serve GDN+MoE; a stronger
  teacher plausibly asserts the old fact *more*).
- With `beta_mlp=0` the upstream MLP loop still collects keys and solves a
  9728x9728 system to produce a guaranteed-zero update. `grom_erase.py` skips it.
- `key_pos=subject_last` only affects that dead loop — the head edit always uses
  answer-position keys.

**Verified against our stack (probes, not assumptions)**

- Qwen3-4b has `tie_word_embeddings=True`. GROM's untie
  (`lm_head.weight = nn.Parameter(clone)` + `config.tie_word_embeddings=False`)
  un-aliases the tensors, so `save_pretrained` keeps `lm_head.weight`; reload
  preserves the edit, leaves `embed_tokens` untouched, and does not re-tie. PASS.
- tokasaurus serves the result: `tokasaurus/model/llama.py:853` branches on
  `config.tie_word_embeddings` and maps `lm_head.lm_head.weight -> lm_head.weight`
  when untied. Costs +0.78 GB on disk (151936x2560 bf16).
- AKEW CounterFact needs no conversion: `requested_rewrite` already carries the
  `prompt` / `subject` / `target_true` triple GROM's request helpers read.
- Head-edit memory on Qwen3-4b (V=151936, d=2560): Gram is 52 MB, but the target
  accumulator and the solve result are ~3.1 GB each in float64 -> ~15-20 GB peak with
  the bf16 model. Comfortable on a 5090, tight on a 4090.

**Finding: the specificity weight fights the neighborhood prompts**

GROM's `alpha_v = max(0, 1 - rf(v)/ff(v))` zeroes out tokens the retain set also
needs. CounterFact `neighborhood_prompts` are other subjects whose *correct* answer is
the same `target_true` being suppressed. Measured on a 50-edit split
(`grom_erase.py --dry-run` reproduces this table):

| retain source for alpha | mean alpha | dead forget positions |
|---|---|---|
| disjoint held-out facts only (our design) | 0.6812 | 4/100 |
| + neighborhood prompts | 0.4440 | 4/100 |
| neighborhood prompts only | 0.0000 | 100/100 |

So `grom_erase.py` computes alpha from disjoint facts only, while still feeding
neighborhood prompts into the ridge system as retain **keys** — which is where
locality is actually preserved.

**Finding: Step 4 needs a wiki anchor, it is not optional**

At full scale (`--num-forget 0`, all 975 edits) there are no held-out facts left, so
alpha degenerates to 1.0 everywhere and protects nothing — and using neighborhood
prompts instead gives 0.0000 across all 1,972 forget positions, a total no-op. Both
ends of the range are useless, so a broad-corpus anchor (`--wiki-jsonl/--wiki-lines`,
supported upstream) becomes the only source of a meaningful specificity weight at full
scale. `grom_erase.py` warns when this configuration is requested.

**Before/after measurement**

`grom_erase.py` probes teacher-forced recall of the answer's first token at the last
prompt position — top-1 rate, mean gold logprob, mean gold rank — on the forget set,
the held-out retain facts and the neighborhood prompts, before and after the edit. One
forward pass per batch, no generation. Without it the job would produce a checkpoint
with no evidence attached; with it, the pilot's claim (forget collapses, retain holds)
is in `grom_report.json` next to the weights.

**Pilot submitted**: job 11146, 50 edits, seed 1, head-only at the authors' ZsRE
settings → `checkpoints/qwen3-4b-erased-n50-s1/`.

**Other adaptations in the adapter**

- Keys are collected in the plain completion form (the CounterFact/ROME convention)
  and/or the chat form the teacher is actually prompted in, selectable via
  `--key-forms`. The chat form is built as (generation prompt) + (answer tokens), so
  the gold span is the answer alone: Qwen3's template with `enable_thinking=False`
  ends in an empty `<think></think>` block, exactly the state the teacher generates
  from during synthesis. Verified by tokenizing both forms.

---

## 2026-08-12 — DCT portability seed: retrain and evaluation

- **Retrained** (job 10753, 2h01m) on the DCT dataset — adapter
  `checkpoints/lora_qwen2.5-7B-Instruct-20260812_002743`, train_loss 0.3404,
  eval_loss 0.5496, 4,575 steps. Hyperparameters identical to the enriched run
  (r64 / alpha 256 / lr 2e-5 / 5 epochs / batch 4), so the seed upgrade is the only
  variable against `lora_qwen2.5-7B-Instruct-20260727_204654`.
- **Evaluated** 2026-08-23 (job 11145) on full CounterFact, seed 82, `--time=12:00:00`
  (the 4h sbatch default is too tight against the 3h24m a full run takes). Head-to-head
  against the enriched baseline `results/lora-20260728_012842` pending.

## 2026-08-11 — DCT correlative-implication portability seed (Step 1)

Two-stage portability seed after Deep Contextual Tuning: list background facts about
the edited target first, then compose an implication question anchored on one of them.

**Changed**

- `cartridges/synthesizers/self_study.py` — new "(2.5) portability background facts"
  stage in `sample_convos`: a batched teacher pre-call lists 5 numbered facts per
  portability sample, leak-guarded (lines naming the subject or old target are
  dropped) and value-substituted to "it". Appended to the context (stored as
  `system_prompt`, never trained) and to `metas[i]["background_facts"]`.
  `PORTABILITY_SYSTEM_PROMPT` gained a rule to ground hop 2 in a listed fact and never
  to cite the list.
- `cartridges/data/resources.py` — `portability_seed_prompt` anchors each question on
  a random fact #k for per-sample hop spread, with anti-restating and anti-riddle
  clauses and a free-recall fallback.
- `knowledge-editing/filter_dataset.py` — `CONTEXT_META_RE` extended with fact-list
  patterns (`background/listed/numbered facts`, `fact #N`, `the facts listed/above`).
- `knowledge-editing/slurm/synth_clean.sbatch` — `enable_precise_onboard=F` plus a
  real-completion health probe after `/ping`, working around a tokasaurus manager
  crash (unhandled `NoSpaceException` in `precise_onboard`; the HTTP frontend survives
  the manager's death, so `/ping` alone is not proof of health).

**Added**

- `knowledge-editing/analyze_portability_hops.py` — hop-diversity gate: buckets
  questions by hop attribute, reports background-fact coverage, target leaks,
  duplicates, and fact-anchor spread (requested #k vs realized #k).

**Measured** (job 10749, raw-vs-raw against the enriched run so the cleaning stages do
not confound the comparison)

| metric | enriched | DCT | change |
|---|---|---|---|
| new-target leak in questions | 13.9% | 3.5% | -75% |
| duplicate questions | 13.4% | 1.3% | -90% |
| rows dropped by `portability_leak` | 1,799 | 610 | -66% |
| founder/origin hops | 3.6% | 7.5% | 2x |

Fact-anchor spread uniform across all five anchors (68% compliance).

**Fidelity judge** (job 10751, calibration 6/6, 0 failures): kept 32,600/35,002
(93.1%). Per-seed drop rate enriched -> DCT: correction 18 -> 17.9, negation
14 -> 13.1, question 9 -> 9.1, portability 3.4 -> **1.0**, reciprocal 0.7 -> 0.7.

Two conclusions. The judge **cannot** be dropped yet — 12/12 spot-read drops were
genuine old-fact assertions. And the correction > negation > question gradient is
*unchanged* by prompt work, which is the evidence that teacher-side reconciliation is
parametric rather than promptable — the thing the GROM arm targets, with these rates
as its quantified "before".

**Contamination found and fixed post-judge**: 76/32,600 rows (0.23%, 72 portability)
had the teacher citing the fact list ("background fact 3 states..."). Scrubbed to
**`dataset_final.parquet` = 32,524 rows**, patterns added to the filter, prohibition
added to the system prompt.

## 2026-08-11 — Dataset browser usability (`viz/`)

- `viz/src/server.py` — new `GET /api/dataset/{path}/seed-types` endpoint (declared
  before the catch-all route) and a `seed_type` query parameter; filtering factored
  into a shared `_apply_filters` used by **both** the page and single-example
  endpoints. That shared helper is the fix for filtered-grid clicks opening the wrong
  sample: the single-example endpoint was indexing the unfiltered dataset.
- `viz/src/pages/DatasetsPage.jsx` — seed-type filter chips with counts and colour
  badges; removed the three dead search checkboxes and the broken client-side
  re-filter (it re-filtered on message content only, which is what made search look
  buggy); search is now fully server-side across messages, system prompt and metadata.
