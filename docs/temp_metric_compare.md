# Metric comparison vs. baseline — working scratch file

**Rewritable.** This file always holds the *current* head-to-head against the standing
baseline. Overwrite it after each new evaluation; the permanent record lives in
[CHANGELOG.md](CHANGELOG.md).

Last updated: **2026-10-01**

---

## Judge: v2 with Qwen3.5-9B — use these numbers

- **v1** (Qwen3-4b, success = score >= 4) passed wrong answers. The unedited model scored
  50.9% overall success with no edit.
- **v2** (`judge_v2.py`; stored answers re-judged by `rejudge.py`, nothing regenerated)
  works in three steps:
  - The judge first writes down what the answer says.
  - It then answers two yes/no checks: does the answer give the new target, and does it
    endorse the old target?
  - Success = new and not old. On value questions the extracted value must also occur in
    the answer.
- **Calibration** (30 cases; 12 critical "wrong or no value must fail"):
  - Qwen3.5-9B: **30/30, 12/12**.
  - Qwen3-4b: 29/30, reaching 30/30 only with the grounding check (it copied the new
    target into ~8% of generalization extractions).
- **Floor**, the unedited model's non-locality success: v1 31.3% -> Qwen3-4b v2 2.7% ->
  **Qwen3.5-9B v2 1.0%**.
- **Agreement between the two v2 judges** (adapter runs):
  - efficacy and generalization 98–99% (κ 0.95–0.98)
  - portability 91% (κ 0.82). 90% of the disagreements are Qwen3-4b passing what 9B
    fails, and in a sample of 6, 9B was right in 5.
- **Chosen: Qwen3.5-9B.** It passes without the grounding patch, gives the lowest floor,
  and is not the teacher's model (Qwen3-4b).
- Locality rows keep their v1 verdict.
- Output: `results/<run>/rejudge-v2-qwen35-9b/`.

---

## How every metric is computed

Everything below is computed from **stored answers**, so any run can be re-scored without
re-running a model.

### Reading the table columns

Every results table in this document uses the same six columns. All are **percentages of
test rows**, and the test set is fixed at **6,825 rows over 975 edits**.

**"Success" always means judge-v2 success**, defined in the next section: the judge extracts
what the answer says, then our code checks *gives the new target* AND *does not endorse the
old one* AND *the extracted value really occurs in the answer*. Locality uses the variant for
a different subject (*gives that subject's correct answer* AND *does not carry the edit
over*). One row = one test question = one judged pass/fail; the column is the share that
passed.

| column | what it is | rows | example question (edit: Wellington's twin city, Sydney → Sheffield) |
|---|---|---|---|
| **Efficacy** | success on the *original* question | 975 (1/edit) | *"What is the twin city of Wellington? It is"* |
| **Gen.** (Generalization) | success on *reworded* questions | 1,950 (2/edit) | *"... The twin city of Wellington is"* |
| **Port.** (Portability) | success one reasoning step further | 1,950 (2/edit) | *"People in Wellington's twin city speak..."* — needs Sheffield, then England |
| **Locality** | success on questions about a **different** subject, which must keep their own answer | 1,950 (2/edit) | *"Milan is a twin city of"* — correct answer stays Sydney |
| **Leak** | the **old** target appears in the answer (string match), pooled over all rows. *Lower is better.* | 6,825 | an answer still saying "Sydney" |
| **OVERALL** | success pooled over **all 6,825 rows**, locality included | 6,825 | — |

**Four things that trip people up:**

- **Efficacy, Gen., Port. and Locality are success rates; Leak is a failure rate.** Higher is
  better for the first four, lower for Leak.
- **Locality rows count 0.00 in the Leak column by construction.** There the old target *is*
  the correct answer, so saying it is right, not a leak.
- **OVERALL is not the average of the four columns** — it is pooled over rows, so
  Generalization, Portability and Locality (1,950 rows each) outweigh Efficacy (975).
  Locality is the weakest column, which is why OVERALL sits well below Efficacy.
- **Locality's ceiling is ~42%, not 100%** — the unedited model only answers those prompts
  correctly 41.8% of the time. Never read 30.6% as "70% broken".

### Success (judge v2)

The judge fills fields in a fixed order — extraction first, verdict last — and **our code**,
not the judge, computes success:

```
value questions  success = gives_new_target AND NOT endorses_old_target
                           AND grounded(answer_value, response)
locality         success = gives_correct_answer AND NOT carries_edit_over
portability      success = follows_new_fact AND NOT follows_old_fact
```

`grounded()` re-checks that the value the judge extracted really occurs in the response,
matching on the first four characters of each content word so adjective forms still count
("Canada" ~ "Canadian"). It exists because the judge sometimes copied the NEW TARGET shown in
its own prompt instead of what the model said (Qwen3-4b did this on ~8% of generalization
rows). Forcing the 0–5 score out of the loop is the whole point of v2: v1 scored *"Fabio
Grobart holds Swiss citizenship"* 5/5 when the new target was France.

**Locality gold depends on the pool**: neighbourhood prompts are other subjects whose correct
answer is the edit's `target_true`; attribute prompts are other subjects whose correct answer
is `target_new`. On the attribute pool "carries the edit over" cannot be distinguished from
answering correctly, so only correctness is scored there.

### Judge-free string metrics

- `names_new` — the new target appears in the answer as a whole word, case-insensitive.
- `old_target_mentioned` / `string leak` — same for the old target. Lower is better.
- `loc_bleed` — the new target appears in an answer about a *different* subject.

These understate us: wanted "Germany", answer *"holds German citizenship"* → counted a miss.
Used where a cheap, un-arguable number is worth more than an accurate one, and always
reported next to the unedited floor so the artifact is visible.

### Confidence intervals

- **Paired, edit-clustered bootstrap** (`compare_evals.py`): each edit contributes ~7 test
  rows that are not independent, so we resample the **975 edits** with replacement — not the
  6,825 rows — and recompute the difference each time. 10,000 iterations; `*` = the 95%
  percentile interval excludes zero. Clustering matters: resampling rows would make every
  interval look narrower than it is.
- **Wilson interval** for a single proportion (used for the scaling curve, where the two
  models are not paired on the same edits).

### AnyEdit-protocol metrics (`score_anyedit.py`)

Their column names do not describe what they compute. From `experiments/summarize_uns.py`:

- **"Bert Score"** = cosine similarity of two `all-MiniLM-L6-v2` *sentence* embeddings, via
  `util.cos_sim(e1, e2).diagonal().mean()`. Not the `bert_score` library, not roberta-large,
  no baseline rescaling.
- **"Rouge-L"** = the `rouge` package's **recall** field, not F1.
- **Reference** = `requested_rewrite.fact_new_uns`, a ~74-word paragraph, with `<|im_end|>`
  appended and never stripped on the Qwen path.
- **Prompts** = `prompt_full` (Ori.) and `paraphrase_prompts[0]` (Para.) only, wrapped as a
  bare Qwen user turn with **no system prompt**, `temperature=0.001`, `max_new_tokens=512`.

Because the reference is a paragraph, the metric rewards **coverage, i.e. verbosity**. Mean
cosine by our answer length: 1–5 words **0.65**, 6–10 **0.79**, 41+ **0.86** — monotone, while
our median answer is 7 words. We therefore report **delta over each paper's own unedited
model**, never absolutes (a constant offset cancels in a difference).

### Recall probe (`probe_recall_lora.py`)

No judge, no string matching: at the position that produces the answer, the rank of the old
and the new target among all ~152k vocabulary entries, under teacher forcing. Measures what
the weights encode, not what the model says — the two can disagree (see §9).

---

## All runs

Every LoRA adapter is the **best-validation checkpoint** (`checkpoint-2500`, epoch 2.7–2.9;
`load_best_model_at_end`), not the end of training.

| Run | Results dir | Effic. | Gen. | Port. | **Non-loc. success (v2)** | **Judge: old fact %** | String leak % | Locality (v1) |
|---|---|---|---|---|---|---|---|---|
| DCT (**baseline**) | `lora-20260823_173846` | 85.0 | 58.1 | 56.0 | **62.6** | 7.8 | 7.38 | 95.18 |
| DCT re-eval (same weights) | `lora-20260913_020116` | 83.3 | 57.1 | 55.9 | 61.8 | 8.0 | 7.30 | 94.82 |
| DCT seed 2 (independent retrain) | `lora-20260913_202911` | 84.1 | 57.9 | 54.7 | 61.9 | 7.9 | 7.45 | 95.28 |
| DCT true epoch 5 (`checkpoint-4575`) | `lora-20260913_202943` | 85.1 | 58.5 | 54.4 | 62.2 | 7.0 | 6.81 | 94.87 |
| head160 | `lora-20260825_031558` | 84.5 | 59.2 | 56.8 | **63.3** | 6.8 | 6.15 | 95.64 |
| head160 re-eval | `lora-20260913_035256` | 85.9 | 58.7 | 56.9 | 63.4 | 7.1 | 6.58 | 94.97 |
| head320 | `lora-20260825_053851` | 84.2 | 59.2 | 55.8 | 62.8 | 7.1 | 6.28 | 94.46 |
| stock, no adapter | `base-20260913_080405` | 1.4 | 1.2 | 0.6 | **1.0** | 22.1 | 27.84 | 99.90 |
| erased, no adapter | `base-20260913_095618` | 1.8 | 0.9 | 1.4 | 1.3 | 18.5 | 22.36 | 99.85 |
| enriched | `lora-20260728_012842` | 83.7 | 57.9 | 56.9 | 62.7 | 7.9 | 7.73 | 94.36 |
| Qwen3.5 teacher | `lora-20260826_024133` | 74.5 | 41.5 | 48.0 | 50.7 | 13.0 | 14.61 | 96.62 |

For reference, v1 non-locality success for the DCT baseline was 73.25% (generalization 76.9% -> 58.1% under v2).

Deltas: paired, edit-clustered bootstrap; \* = 95% CI excludes zero. "OVERALL success"
includes locality rows (v1); the judge old-fact metric covers non-locality rows only.

---

## 1. Eval noise: the same adapters evaluated twice

The "epoch 2.7" jobs turned out to evaluate byte-identical weights, so they measure pure
eval noise (temperature-0.7 generation plus judge). DCT re-eval − eval:
- OVERALL success −0.66 [−1.48, +0.15]
- judge old-fact +0.23 [−0.51, +0.94]
- efficacy judge old-fact **+1.44 [+0.31, +2.67] \***, a star by chance

An isolated per-type star has now appeared from pure noise under both judges, so claim only
effects that repeat across draws. "Train longer?" is answered in section 6.

---

## 2. Student erase: head160 − DCT, both eval draws

| Metric | first eval | re-eval |
|---|---|---|
| **Judge: old fact asserted (non-loc.)** | **−1.01 [−1.74, −0.27] \*** | **−0.90 [−1.62, −0.16] \*** |
| **Judge: old fact, generalization** | **−2.26 [−3.49, −1.08] \*** | **−2.05 [−3.28, −0.82] \*** |
| String leak, OVERALL | −0.88 [−1.41, −0.37] \* | −0.51 [−1.04, +0.01] |
| OVERALL success | +0.62 [−0.31, +1.54] | +1.19 [+0.28, +2.17] \* |
| Efficacy success | −0.51 [−2.56, +1.54] | +2.67 [+0.72, +4.62] \* |
| Generalization success | +1.13 [−0.97, +3.18] | +1.64 [−0.41, +3.69] |
| Portability success | +0.82 [−1.13, +2.82] | +1.03 [−0.97, +2.97] |
| Names new target (non-loc.) | +0.84 [−0.41, +2.09] | +1.17 [−0.08, +2.40] |
| Locality success (v1) | +0.46 [−0.46, +1.38] | +0.15 [−0.87, +1.18] |

**Read.**
- The effect that replicates and that the fixed judge confirms is **less assertion of the
  old fact**: about −1 point overall and −2 points on generalization, significant in both
  draws.
- The success gain is small and significant in only one draw.
- Locality is unharmed.
- The effect also holds against an independently retrained DCT adapter (seed 2), as
  section 5 shows.

---

## 3. Did the erase survive finetuning? Recall probe

`results/recall-probe-20260913_125850.json`: first-token rank of the old and new target for
all 975 edits, paired per edit, geometric-mean rank ratios.

| | plain prompt | chat (eval format) |
|---|---|---|
| Erased vs stock base, no LoRA: old-target rank | ×8.5 | ×9.9 |
| **head160+LoRA vs DCT+LoRA: old-target rank** | **×3.05** (log10 +0.49 [+0.43, +0.54]) | **×1.89** (+0.28 [+0.24, +0.32]) |
| Share of the erase's suppression that survives | 52% | 28% |
| head160+LoRA vs DCT+LoRA: **new**-target rank | ×0.98 | ×1.04 |
| DCT seed 2 vs seed 1: old-target rank (**training-seed variance**) | ×1.01 [log10 −0.02, +0.03] | ×0.99 [−0.03, +0.03] |
| head160+LoRA vs DCT seed 2: old-target rank | ×3.03 | ×1.90 |
| True epoch 5 vs best-val: old-target rank (seed 1 / seed 2) | ×1.30 / ×1.30 | ×1.26 / ×1.48 |

**Read.**
- The erase **partially survives finetuning**. The old fact stays 2–3× less likely than
  without the erase, but most of the suppression is undone.
- It costs nothing on learning the new fact.
- Training to epoch 5 also pushes the old target down a little; its eval is pending.

---

## 3b. Locality, measured properly (judge v2, 2026-09-17)

v1 asked only whether the edited subject was mentioned and scored everything ~95%. v2 scores
the *correct answer of the other subject* (old target for neighbourhood prompts, new target
for attribute prompts) and flags carry-over of the edit.

| Run | locality success | neighbourhood | attribute | edit carried over |
|---|---|---|---|---|
| unedited | 40.6 | 42.4 | 38.6 | 0.6 |
| erased, no adapter | 38.6 | 39.1 | 38.1 | 0.6 |
| DCT (baseline) | 30.8 | 26.7 | 35.2 | 4.9 |
| DCT seed 2 | 30.7 | 28.3 | 33.3 | 4.7 |
| **head160** | **33.2** | 30.4 | 36.1 | 4.5 |
| head320 | 32.0 | 27.9 | 36.4 | 4.6 |
| Qwen3.5 teacher | 29.9 | 25.7 | 34.5 | 5.1 |

| Comparison | locality success | edit carried over |
|---|---|---|
| unedited -> DCT | **−9.74 [−12.31, −7.13] \*** | **+4.31 [+3.38, +5.28] \*** |
| unedited -> erased (no adapter) | −1.95 [−3.54, −0.36] \* | +0.05 (n.s.) |
| DCT -> head160 | **+2.36 [+0.72, +4.05] \*** | −0.36 (n.s.) |
| DCT seed 2 -> head160 | **+2.46 [+0.56, +4.36] \*** | −0.21 (n.s.) |
| DCT -> DCT seed 2 | −0.10 (n.s.) | −0.15 (n.s.) |
| head160 -> head320 | −1.18 (n.s.) | +0.05 (n.s.) |

**Read.** The locality damage comes from **LoRA training, not from the erase**: the erased
base on its own loses 2 points, the trained adapters lose 10. The erase gives ~2.4 of them
back, replicated against both DCT seeds and far outside seed variance. Absolute numbers are
bounded by what the unedited model knows (40.6%), so read them against that row.

---

## 4. Erase-only control: no adapter

Erased − stock:
- judge old-fact **−3.59 [−4.66, −2.52] \***: efficacy −5.64 \*, generalization −3.18 \*,
  portability −2.97 \*
- string leak −3.91 \*
- success ~1% in both
- locality −0.10 (n.s., v1)

The erase alone suppresses the old fact, including indirect portability answers, and
installs nothing.

---

## 7. Three-term objective (CE + forget + KL retain): rejected

`checkpoints/lora-keloss-mild` (`results/lora-20260918_015035`), DCT data, stock base,
hinge active while the old answer is above 5%, both weights 1.0. Against the DCT baseline,
judge v2:

| Metric | Δ |
|---|---|
| Efficacy success | **−11.38 [−14.15, −8.62] \*** |
| Generalization success | **−22.06 [−24.56, −19.60] \*** |
| Portability success | **−13.70 [−16.21, −11.28] \*** |
| OVERALL success | **−12.14 [−13.46, −10.82] \*** |
| Names new target | **−15.08 [−16.72, −13.46] \*** |
| Judge: old fact asserted | **+1.13 [+0.08, +2.11] \*** (worse) |
| Locality success | −1.03 [−3.18, +0.98] |
| Locality: edit carried over | +0.10 [−1.70, +1.91] |

The forget set is each edit's own question, the same position where the new answer must be
produced, so the two terms compete and CE loses. The old fact is not reduced: a model that
fails to give the new value falls back on the old one. The KL retain term did not protect
locality either. See CHANGELOG 2026-09-18 for the canary trade-off curve.

**Why they compete, precisely.** At the answer position the model spreads 100% of probability
over the vocabulary. `L_CE` concentrates it on the new target — and the old target falls
automatically as it does. `L_forget = relu(log p(old) − τ)` only pushes the old target **down**,
saying nothing about where that mass should go, so it scatters across the whole vocabulary
instead of moving to the new target. One term concentrates, the other disperses. The fix is not
a better forget set but a **relative** objective (prefer new over old on the same question),
which moves mass from one to the other; see `temp_meeting_prep.md` §13.1.

**DESIGN ERROR, recorded for honesty.** Both sets were built from CounterFact's own **eval
pools**:

| set | built from | which is the eval pool for |
|---|---|---|
| forget | `requested_rewrite.prompt` + `paraphrase_prompts` | **Efficacy** and **Generalization** |
| retain | `neighborhood_prompts` | **Locality** |

So the objective was defined on the exact inputs the run was then scored on. It produced **no
false positive** — the result was strongly negative, and generalization fell 22 points *at the
very prompts it trained on*, which makes the conclusion stronger, not weaker. But a positive
result from that setup would have been invalid, and any retry must build its forget/retain sets
from held-out prompts. (This is the trap the locality seed avoids by having Bot A invent fresh
subjects instead of reusing `neighborhood_prompts` — see §14.)

---

## 8. GROM as an editor (closed form, no LoRA)

`--beta-new` pushes the NEW answer up in the same solve that pushes the old one down.
All 975 edits, judge v2, evaluated with no adapter.

| | unedited | beta_new 80 | **beta_new 320** | sup160+ins640 | DCT LoRA |
|---|---|---|---|---|---|
| Efficacy | 1.4 | 4.4 | **20.6** | 22.6 | 85.0 |
| Generalization | 1.2 | 3.0 | 8.5 | 9.6 | 58.1 |
| Portability | 0.6 | 3.0 | 18.2 | 20.1 | 56.0 |
| **Locality** | 40.6 | 40.9 | **39.8** | 38.8 | **30.8** |
| Old-fact leak | 27.8 | 19.7 | 15.3 | 14.5 | 7.4 |
| Median answer length | 513 | 508 | **954** | — | 58 |

- vs unedited (beta_new 320): efficacy **+19.20 \***, portability **+17.59 \***,
  generalization **+7.28 \***, leak **−4.54 \***, locality −0.57 (n.s.).
- vs DCT LoRA: efficacy **−64.40 \***, generalization **−49.54 \***, portability
  **−37.77 \***, but locality **+8.90 \***.

**The trade.** At 50 edits beta_new 80 installs the fact perfectly (rank 313 -> 1); at 975
it leaves it at rank ~50, because the edits share one solve. Strength 320 installs it but
the edited answers degenerate (the inserted token repeats; length 513 -> 954), which the
old fluency canary missed because it only probed generic prompts.

---

## 9. Edit-then-finetune (closed-form edit as the LoRA starting point)

Same DCT data and hyperparameters; only the base model differs.

| Run | Efficacy | Gen. | Port. | Locality | Leak | OVERALL |
|---|---|---|---|---|---|---|
| DCT LoRA (stock base) | 85.0 | 58.1 | 56.0 | 30.8 | 5.27 | 53.5 |
| head160 erase + LoRA | 84.5 | 59.2 | 56.8 | **33.2** | 4.40 | 54.7 |
| sup80ins80 edit + LoRA | 83.1 | 60.4 | 55.0 | **33.1** | 4.31 | 54.3 |
| mlp20 band edit + LoRA | 84.0 | 63.2 | 51.0 | 32.2 | 5.20 | 53.8 |
| **ins320 edit + LoRA** | 86.1 | **69.1** | 57.5 | 30.6 | **3.99** | **57.2** |

| ins320 edit + LoRA, **seed 2** | 87.6 | 71.5 | 57.9 | 33.2 | 4.34 | 59.0 |

vs the DCT baseline: ins320 + LoRA gives generalization **+11.08 \***, OVERALL **+3.68 \***,
leak **−1.29 \***, locality −0.26 (n.s.). vs head160 + LoRA: generalization **+9.95 \***,
OVERALL +2.52 \*, locality −2.62 \*.

**Seed variance on this base is NOT zero — correction to an earlier claim.** Retraining
ins320 + LoRA with a second seed (`results/lora-20260921_123228`) moves OVERALL **+1.76
[+0.63, +2.86] \***, generalization +2.36 \*, locality +2.67 \*. The earlier "run-to-run
variance is essentially zero" was measured on the **stock** base (DCT seed 2: −0.10, n.s.)
and does not transfer to the edited base. The generalization claim (+11.08) is still far
outside this, but OVERALL +3.68 is only about twice the seed spread — quote generalization,
not OVERALL, as the headline.

`ins320` alone (no adapter) scored 20.6% efficacy with degenerate text — as a starting
point it produces the strongest adapter measured. Finetuning repairs what the edit
distorted and keeps what it installed.

The MLP band is the better *edit* but the worse *initialisation*: mlp20 + LoRA gains
generalization (+5.13 \*) and loses portability (−4.97 \*) against the baseline, and is
−5.95 \* generalization / −6.51 \* portability against ins320 + LoRA. Editing layers 16-20
seems to disturb the mid-network computation that multi-hop answers rely on, while the
head edit only changes what is said at the answer position.

**Probe rank does not imply behaviour, twice over.** mlp20 puts the new answer at rank 2
under teacher forcing yet names it in 4.1% of efficacy answers; ins320 reaches rank 1 and
manages 20.6%. Rank is necessary, not sufficient — the finetune is what converts either
into answers.

---

## 10. AnyEdit protocol: the external comparison, done properly (job 12219)

Generated under **their** settings — bare Qwen user turn, no system prompt,
`temperature=0.001`, reference `fact_new_uns + '<|im_end|>'` — and scored with **their**
metric (MiniLM cosine). `eval_anyedit_protocol.py`, `results/anyedit-protocol-20260921_193226.json`.

| | Ori. | 95% CI | Para. | 95% CI |
|---|---|---|---|---|
| stock Qwen2.5-7B-Instruct | 69.34 | [68.60, 70.07] | 51.83 | [50.73, 52.86] |
| ins320 + LoRA | **75.71** | [74.67, 76.72] | **64.04** | [62.67, 65.41] |

Delta over each paper's own unedited row, **Para.** column (their published Qwen2.5-7B-Instruct
AKEW-CounterFact table):

| Method | Δ Ori. | Δ Para. |
|---|---|---|
| AnyEdit | +32.58 | **+20.66** |
| AnyEdit\* | +34.13 | +16.04 |
| UnKE | +31.84 | +14.55 |
| AlphaEdit | +15.16 | +12.25 |
| **ours (ins320 + LoRA)** | **+6.37** | **+12.21** |
| MEMIT | +11.69 | +11.30 |
| ROME | +10.39 | +10.93 |
| MEND | +4.04 | +8.12 |

**Their published table, for reference** (AnyEdit Table 1, AKEW-CounterFact,
Qwen2.5-7B-Instruct, "Bert Score" = MiniLM cosine; our two rows slotted in, *not* directly
comparable — see the anchor gap below):

| Method | Ori. | Para. |
|---|---|---|
| Pre-edited | 65.50±0.34 | 44.74±0.41 |
| FT-L | 46.66±0.48 | 32.34±0.50 |
| MEND | 69.54±0.54 | 52.86±0.40 |
| ROME | 75.89±0.38 | 55.67±0.47 |
| MEMIT | 77.19±0.32 | 56.04±0.40 |
| AlphaEdit | 80.66±0.25 | 56.99±0.49 |
| UnKE | 97.34±0.13 | 59.29±0.48 |
| AnyEdit\* | 99.63±0.09 | 60.78±0.39 |
| **AnyEdit** | 98.08±0.15 | **65.40±0.38** |
| | | |
| *ours: unedited* | *69.34* | *51.83* |
| *ours: ins320 + LoRA* | *75.71* | *64.04* |

Their ± is a std across items; ours (above) is a bootstrap CI — same notation, different
quantity, do not compare the spreads.

**Two things the column shows on its own, independent of us.**

1. **Ori. is saturated, Para. is not.** Five methods clear 97 on Ori.; nobody clears 66 on
   Para. Paraphrase generalization is the open problem in this benchmark.
2. **Pushing harder on Ori. costs Para.** AnyEdit\* beats AnyEdit on Ori. (99.63 vs 98.08)
   but is **4.6 points worse** on Para. (60.78 vs 65.40).

**Where we land in absolute terms** (read with the anchor gap in mind): our Para. 64.04 is
second-highest in the column, behind only AnyEdit's 65.40 and above UnKE, AnyEdit\*,
AlphaEdit, MEMIT and ROME; our Ori. 75.71 sits between ROME (75.89) and MEMIT (77.19). But
our unedited floor is 7.1 points above theirs on Para., so most of that position is inherited
from the baseline rather than earned by the edit — which is why the delta table above, not
this one, is the defensible comparison.

**Anchor gap, unexplained — do not quote absolutes.** Our stock row is +3.84 Ori / +7.09 Para
above their published Pre-edited (65.50 / 44.74). Ruled out: system prompt and temperature
(together worth ~0.4 points — the same stock model under our own eval settings scores
69.78 / 52.20), dataset size (both 975; `dataset_size_limit` defaults to `None`), metric
implementation, reference string, prompt choice, paraphrase index. Deltas are invariant to a
constant offset, so only deltas are safe.

**Regime asymmetry.** Their `num_edits` defaults to **1** — one edit into a freshly reset
model. Ours holds 975 at once. Their protocol therefore carries no information about
cross-edit interference.

**The closed-form edit alone** scores 64.73 / 32.33 — *below* the unedited model. It installs
the facts and wrecks the fluency; it is an initialisation, not a standalone editor.

---

## 11. Scaling curve: quality per edit RISES with the number of edits (job 12220)

The axis no locate-and-edit paper reports. Same recipe (r64/α256, lr 2e-5, 5 epochs, plain CE,
stock base) trained on seeded subsets of n edits, each adapter scored **only on the edits it
was trained on** (`--case-ids-file`). Judge-free `names_new`, replicates pooled, Wilson CIs.
`results/scaling-curve-20260921_200849/`.

| n | items | Ori. names new % | Para. names new % |
|---|---|---|---|
| unedited | 975 | 3.4 [2.4, 4.7] | 2.8 [1.9, 4.0] |
| 25 (3 seeds) | 75 | 69.3 [58.2, 78.6] | 29.3 [20.2, 40.4] |
| 100 (2 seeds) | 200 | 67.5 [60.7, 73.6] | 34.0 [27.8, 40.8] |
| 300 | 300 | 75.7 [70.5, 80.2] | 42.7 [37.2, 48.3] |
| **975** | 975 | **78.6 [75.9, 81.0]** | **52.4 [49.3, 55.5]** |

**Generalization nearly doubles** from 25 to 975 edits, intervals non-overlapping.

**It is edit diversity, not compute.** Step-matched control `n25-s1-long` — same 25 edits,
20 epochs (~480 steps vs 120): Ori. names new **72.0 → 52.0** (worse, overfits), Para.
32.0 → 36.0. More optimisation on few edits does not approach n=975.

**The AnyEdit metric points the wrong way on this curve**: Ori. cosine 76.1 (n=25) → 70.87
(n=300) → 72.88 (n=975) while `names_new` rises 69.3 → 78.6. Cause confirmed — answer length
shrinks monotonically with n (Ori. median 10–14 → 8 → 7 words). Our own evidence that the
metric tracks verbosity rather than edit success.

**Caveats.** The n=975 endpoint reuses the existing DCT adapter (`checkpoint-2500`), same
recipe but not a like-for-like retrain inside the sweep; metrics are substring matches, not
judge v2.

---

## 12. Locality vs n: bleed is real but does NOT accumulate (jobs 12221, 12222)

Same adapters as §11, scored on the `neighborhood_prompts` of their own edits (2 of the 10
CounterFact ships), same generation protocol as §10. `experiments/eval_locality_scaling.py`;
judge-v2 re-grade by `experiments/rejudge_locality.py`. `results/locality-scaling-20260922_005043/`.

| | correct % | bleed % (string) | **bleed % (judge v2)** |
|---|---|---|---|
| unedited (floor) | 41.8 [39.6, 44.0] | 3.7 | **1.0** |
| n = 25 (3 seeds) | 42.0 [34.4, 50.0] | 8.7 | **8.0** |
| n = 100 (2 seeds) | 42.2 [37.5, 47.1] | 10.2 | **9.5** |
| n = 300 | 39.0 [35.2, 43.0] | 8.5 | **7.8** |
| n = 975 | 34.1 [32.0, 36.2] | 8.6 | **8.1** |

1. **Flat in n.** Bleed jumps to ~8% as soon as you train at all, then does not move from 25
   to 975 edits. Collateral damage does not accumulate ⇒ **a recipe problem, not a scaling
   problem** ⇒ the fix is training data, not fewer edits.
2. **The bleed is genuine.** Judge v2 confirmed **94.0%** of flagged n=975 cases (91–95% at
   n=100/300), 0 false negatives in 400-sample draws per config, 0 judge failures. The
   *artifact* sits in the unedited row — only 27.8% of its flags survive — so judging cut the
   floor 3.7 → 1.0 and **widened** the real gap to ~7 points.
3. **The metric's ceiling is 41.8%, not 100%.** CounterFact neighbourhood prompts are terse
   stems ("Milan is a twin city of") with one gold answer where several are valid. Read
   §3b's 30.6% against that, not against 100.

The ~7 points of genuine bleed accounts for essentially the whole 7.7-point drop in correct%
(41.8 → 34.1).

---

## 13. Cross-edit dilution: looked for, not found

A claim from the 2026-08-26 meeting ("wrong answers often contain *another* edit's new
target") does **not** survive a control. At n=975, 12.0% of answers contain another edit's
target — but the **unedited** model scores **52.7%** on the same check, having seen no edits.
The measure is dominated by coincidence: the matched targets are common words (`english`,
`german`, `jazz`, `trumpet`, `pitcher`), and hand-reading the flagged cases shows ordinary
wrong answers, not leakage. With the candidate pool held at 24 foreign targets per item so
larger n cannot inflate the rate: stock 18.9%, n=25 8–16%, n=975 9.2% — trained models are
*below* the unedited coincidence rate.

Real cross-edit confusion, if any, is below this test's noise floor.

---

## 14. Locality seed (in flight)

Because §12 says the bleed is a data problem, we added a **locality** seed type: Bot A asks
about a *different* well-known entity taking the same relation, Bot B answers it with its own
real fact and never carries the edit over.

> *"What channel does Legacies premiere on?"* → *"Legacies premieres on The CW."*
> (while the edit being trained is Underdog → MTV)

It never touches `neighborhood_prompts` — those are the locality eval, so training on them
would be training on the test set.

Four smoke iterations (jobs 12223/12227/12230/12232) to get clean, each fixing a distinct
defect:

| | s1 | s2 | s3 | **s4** |
|---|---|---|---|---|
| Bot A asked about the edited subject | 25.7% | 44.2% | 21.4% | **3.8%** |
| Bot B hedged ("I am not sure") | 24.3% | 24.7% | 0.0% | **0.0%** |
| keep rate | 37.8% | 33.8% | 67.1% | **65.0%** |

s2 got worse because repeating the forbidden subject name in the prompt *primed* the 4B model
into using it. Three filter checks now guard the output: `locality_anchored` (question names
the edited subject or its new value — a reverse lookup that reinforces the edit),
`locality_bleed` (answer names the edited subject), `locality_hedge` (bare refusal or setup
narration — the failure that got the `ignorance` seed scrapped).

**Full corpus built**: 37,082 clean rows, locality keep rate 66.0%, mix
question 15 / negation 15 / correction 15 / reciprocal 20 / portability 20 / locality 15.

### 14a. Result at the 15% dose (jobs 12234 train+eval, 12248 judge-v2 re-grade)

`results/locality-seed-20260922_161209/`. Both bases retrained on the new corpus with the
unchanged recipe; evaluated under the AnyEdit protocol (§10) and on neighbourhood prompts (§12).

**It works on its target.** Judge-confirmed bleed, against the 1.0% unedited floor:

| corpus / base | correct % | bleed % (string) | **bleed % (judge v2)** | confirmed |
|---|---|---|---|---|
| DCT, stock (baseline) | 34.1 | 8.6 | **8.1** | 94.0% |
| +locality, stock | 36.6 [34.4, 38.7] | 6.9 | **6.0** | 87.3% |
| +locality, ins320 | 36.0 [33.9, 38.1] | 6.5 | **5.7** | 87.4% |

Excess bleed over the floor falls **7.1 → 4.7 points, a 34% reduction**. 0 judge failures,
0 false negatives in 400-row samples per config.

**But it costs more than it buys.** Judge-free `names_new` under the same protocol:

| corpus / base | Ori. (efficacy) | Para. (generalization) |
|---|---|---|
| unedited | 3.2 | 2.8 |
| DCT, stock | 78.6 | 52.4 |
| DCT, ins320 | **82.1** [79.5, 84.3] | **65.4** [62.4, 68.4] |
| +locality, stock | 73.9 | 46.6 |
| +locality, ins320 | 78.8 [76.1, 81.2] | **58.4** [55.2, 61.4] |

Generalization **−7.0** on the ins320 base (−5.8 on stock), intervals **non-overlapping**.
MiniLM cosine agrees: 75.71/64.04 → 74.51/59.60.

**Net: −2.4 points of judge-confirmed bleed for −7.0 points of generalization.** Bad trade as
mixed.

**Confounded, deliberately fixable.** The 15 points for locality came out of `portability`
(30→20) and `question` (20→15) — the two seeds that drive generalization — so this measures
"added locality AND removed 15 points of what was working". Both bases moving together rules
out an ins320 artifact, but not the mix change. (Second confound: a fresh synthesis run, so
teacher sampling differs too.)

### 14b. Dose-response: 5% is the answer (jobs 12249 synth, 12250 train+eval, 12251 rejudge)

`results/locality-seed-20260923_011406/`, corpus `outputs/2026-09-23-00-56-40-synthesize/LocalityDose5-0/`
= 38,172 clean rows, locality keep 68.0%, mix
question 15 / negation 15 / correction 15 / reciprocal 20 / **portability 30** / **locality 5**.

**The benefit saturates well below 15%, the cost does not.** ins320 base:

| corpus | Ori. `names_new` | Para. `names_new` | locality correct % | **judge-v2 bleed** |
|---|---|---|---|---|
| DCT (baseline) | 82.1 [79.5, 84.3] | 65.4 [62.4, 68.4] | 34.1 | **8.1** |
| locality 15% | 78.8 [76.1, 81.2] | 58.4 [55.2, 61.4] | 36.0 | **5.7** |
| **locality 5%** | **82.6 [80.1, 84.8]** | **63.1 [60.0, 66.0]** | 35.8 | **5.6** |

- **Bleed: identical to the 15% dose** (5.6 vs 5.7). Against the 1.0% floor, excess bleed
  **7.1 → 4.6 points, a 35% reduction**. Judge confirmed 85.2% of flags, 0 failures, 0 false
  negatives in 400-row samples.
- **Efficacy: fully recovered**, 82.6 vs the baseline's 82.1.
- **Generalization: 63.1 vs 65.4**, a 2.3-point gap with heavily overlapping intervals —
  against the 15% dose's 7.0-point *non-overlapping* loss.

Confirmed on the stock base: 76.6 / 51.2 at 5%, against the DCT baseline's 78.6 / 52.4 and the
15% dose's 73.9 / 46.6.

**Conclusion: the 15% loss was self-inflicted.** Those points had come out of `portability`
(30→20) and `question` (20→15), the two seeds that drive generalization. At 5%, taken from
`question` alone, the locality benefit is unchanged and the cost is gone.

| | 15% dose | **5% dose** |
|---|---|---|
| judge bleed | −2.4 pts | **−2.5 pts** |
| generalization | −7.0 pts \* | **−2.3 pts, n.s.** |

**Adopt the 5% mix.** `synthesize.py` already carries it.

---

## 15. Reference consolidated + filter ablation (jobs 12573–12579, 2026-09-29)

All runs: base `qwen2.5-7b-ke-ins320`, 5%-locality corpus
(`outputs/2026-09-23-00-56-40-synthesize/LocalityDose5-0/`), full judge-v2 eval.

| run | results dir | Eff | Gen | Port | Loc | OVERALL | judge old fact |
|---|---|---|---|---|---|---|---|
| ins320 + LoRA, old DCT corpus | `lora-20260920_154845` | 86.1 | 69.1 | 57.5 | 30.6 | 57.2 | 5.9 |
| **reference** (5%-locality corpus) | `lora-20260929_052636` | 86.7 | 69.6 | 58.8 | **34.0** | **58.8** | 6.5 |
| reference, seed 2 | `lora-20260929_103505` | 87.2 | 71.8 | 58.1 | 35.5 | 59.7 | 7.1 |
| **unfiltered** corpus | `lora-20260929_122444` | 83.0 | 62.5 | 54.9 | 36.1 | 55.7 | 9.6 |
| **regex-only** corpus | `lora-20260929_181848` | 82.3 | 69.0 | 54.9 | 37.4 | 57.8 | 9.9 |

**Reference.** vs the old DCT corpus: locality **+3.49 [+1.49, +5.44] \***, OVERALL **+1.58
[+0.38, +2.80] \***, efficacy/generalization/portability n.s. — the locality seed holds under
the full judge at no cost. **Seed spread:** generalization +2.15 \*, OVERALL +0.91 (n.s.); read
differences under ~2 points as noise. New reference = two-seed mean, about
**Eff 87 / Gen 71 / Port 58 / Loc 35 / OVERALL 59**.

**Filter ablation, size-matched.** Raw 40,960 → regex 38,172 → regex + fidelity judge 35,608
rows; unfiltered and regex-only were randomly subsampled (seed 0) to 35,608 with seed mixes
preserved, so differences are row *quality*, not quantity.

| vs reference | Eff | Gen | Port | Loc | OVERALL | old fact |
|---|---|---|---|---|---|---|
| no filtering | −3.69 \* | **−7.13 \*** | −3.85 \* | +2.05 | **−3.08 \*** | **+3.06 \*** |
| regex only (no judge) | **−4.41 \*** | −0.62 | **−3.85 \*** | +3.39 \* | −0.94 | **+3.38 \*** |

- **The fidelity judge** protects efficacy, portability and old-fact leakage. It drops only
  ~6.7% of rows, so those rows are very damaging: left in, they teach the old fact.
- **The regex stage** protects generalization: regex-only → unfiltered costs a further 6.5.
- **The locality "gain" without filtering is likely an artifact:** neighbourhood golds are the
  *old* value, so a model that reverts to old facts scores higher there. It moves with the
  +3.4 old-fact leak.

**Consequence.** Neither stage can simply be dropped. A filter-free pipeline must recover
+4 efficacy, +4 portability and −3.4 old-fact leak (the judge's share) and +6.5
generalization (the regex's share).

---

## Other comparisons (v2)

- **Enriched -> DCT**: null (OVERALL +0.21 [−0.98, +1.36]; judge old-fact −0.10).
- **DCT -> Qwen3.5 teacher**: OVERALL −8.10 \*, judge old-fact +5.17 \*.
- **head160 -> head320**: n.s. (OVERALL −0.67 [−1.64, +0.31]). The v1 locality result
  (−1.18 \*) still stands; locality was not re-judged.

## 5. Training-seed control: DCT retrained with seed 2

Same data, stock base, same hyperparameters; only the seed differs (LoRA init + data order).
Its saved adapter is again `checkpoint-2500`.

| Metric (judge v2 unless marked) | DCT seed 2 − seed 1 | head160 − seed 2 | head160 re-eval − seed 2 |
|---|---|---|---|
| OVERALL success | −0.51 [−1.57, +0.51] | **+1.13 [+0.06, +2.18] \*** | **+1.04 [+0.03, +2.10] \*** |
| Judge: old fact (non-loc.) | +0.08 [−0.80, +0.94] | **−1.09 [−1.93, −0.27] \*** | −0.76 [−1.60, +0.12] |
| Judge: old fact, generalization | −0.21 [−1.64, +1.23] | **−2.05 [−3.33, −0.77] \*** | **−1.49 [−2.82, −0.15] \*** |
| String leak, OVERALL (v1 file) | +0.04 [−0.59, +0.63] | **−0.92 [−1.51, −0.34] \*** | — |

**Read.**
- Retraining with another seed moves nothing, either in behaviour or in the recall probe
  (old-target rank ×1.01 / ×0.99).
- Against that near-zero variance, the erase effect holds versus the independent seed:
  old-target rank ×3.0 (plain) / ×1.9 (chat), and fewer old-fact assertions.
- Generalization old-fact is significant in all four head160 comparisons (2 seeds × 2
  eval draws); success is significant in 3 of 4.

---

## 6. Train longer? Epoch 5 vs the saved epoch-2.7 checkpoint

DCT `checkpoint-4575` − best-val checkpoint:
- judge v2 OVERALL success −0.42 [−1.39, +0.50]
- judge old-fact −0.76 [−1.50, −0.02] \* (portability −1.54 \*)
- string leak −0.41 [−0.95, +0.13]
- v1 success −0.41 (n.s.)

The recall probe shows the old target a further ×1.3–1.5 down after the extra epochs, for
both seeds.

**Read: no gain in success.** At most a small, borderline drop in old-fact assertions: one
comparison, and the same-weights repeat already produced a chance star of similar size.
Training past ~3 epochs is not worth the compute.

---

## Regenerating

```bash
python knowledge-editing/compare_evals.py results/<A>/rejudge-v2-qwen35-9b results/<B>/rejudge-v2-qwen35-9b
python knowledge-editing/compare_evals.py results/<A> results/<B>          # v1 judge
```
