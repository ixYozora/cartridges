# Meeting prep — plain-language version

Rewrite this before each meeting. Permanent record: [CHANGELOG.md](CHANGELOG.md).
Current numbers: [temp_metric_compare.md](temp_metric_compare.md). Examples:
[qualitative-analysis.md](qualitative-analysis.md).

For: **meeting after 2026-08-26** (prepared 2026-09-23)

---

## Part 0 — Everything since the last meeting, one line each

- **The evaluation instrument was broken** — the old judge graded a 0–5 rubric and passed
  wrong answers, giving the *unedited* model 51% "success".
- **Built judge v2** (extract the answer first, then two yes/no checks, success computed in
  code) with a 39-case calibration gate, and re-graded every stored answer.
- **Closed the remaining eval gaps**: correct locality scoring per prompt pool, portability,
  judge-free string metrics, and a paired bootstrap that resamples *edits*, not answers.
- **Audited training-question leakage** against 90 hand labels.
- **Revived and then rejected the 3-term KE loss** — its "forget" term only pushes the old
  answer *down*, scattering that probability across the vocabulary instead of moving it to the
  new answer, so it dilutes the signal that teaches the edit.
- **Extended GROM from unlearning to editing** (your "GROM auf KE erweitern" point).
- **Diagnosed why GROM alone fails at 975 edits**: a capacity limit, 5,850 constraints
  squeezed into 3,584 dimensions — and showed the MLP band fixes it at 1/16 the strength.
- **Established edit-then-finetune as the best configuration**, replicated across seeds.
- **Discovered our AnyEdit comparison was never valid** — their "BertScore" is not BERTScore.
- **Re-ran under AnyEdit's exact protocol**: we tie AlphaEdit and beat MEMIT/ROME on
  paraphrase generalization.
- **Ran a scaling curve nobody else reports**: per-edit quality *rises* with more edits.
- **Measured locality against the number of edits**: ~8% real bleed, and it does **not** grow.
- **Showed the locality metric's ceiling is 41.8%, not 100%.**
- **Ruled out cross-edit dilution** — and corrected a claim I made at the last meeting.
- **Built a locality seed** for the data generator, found the right dose by experiment, and cut
  collateral damage by 35% at no measurable cost to efficacy or generalization.

---

## Part 1 — The words, so the tables make sense

**Edit** — one fact we change. We use 975 of them, from CounterFact.
Running example, `case_id 8`:

> Question: *"What is the twin city of Wellington? It is ___"*
> **Old target** (the truth): Sydney  **New target** (what we want): Sheffield

**Teacher** = Qwen3-4b, which writes the training data. **Student** = Qwen2.5-7B-Instruct,
which we train and test. **LoRA / adapter** = the small add-on we train on top of the student.

**The four test types**

| Type | Question asked | Correct answer |
|---|---|---|
| **Efficacy** | the original question | Sheffield |
| **Generalization** | a reworded question | Sheffield |
| **Locality** | about a *different* city | that city's own real answer |
| **Portability** | one reasoning step further | follows from Sheffield |

**Judge** — an LLM that grades answers. **v1** = old 0–5 score. **v2** = the stricter one we
built (Part 3).

**Seed** (two meanings, careful) — (a) the random start of a training run; (b) a *seed prompt*,
the instruction that tells the teacher what kind of dialogue to write. Context disambiguates.

**Significant (\*)** — the 95% confidence interval excludes zero.

---

## Part 2 — How every number is actually computed

This is the part worth being able to defend. Every metric below is computed from stored
answers, so anything can be re-scored without re-running a model.

### 2.1 Reading the table columns

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

### 2.2 Success (judge v2)

The judge is **not** asked "is this good, 0–5". It is asked to fill fields in a fixed order:

1. `answer_value` — copy, from the response, the value it gives.
2. `reason` — one sentence.
3. `gives_new_target` — yes/no.
4. `endorses_old_target` — yes/no.

Then **our code**, not the judge, computes:

```
success = gives_new_target AND NOT endorses_old_target AND grounded(answer_value, response)
```

`grounded` re-checks that the extracted value really occurs in the response, matching on the
first four letters so "Canada" still matches "Canadian". This exists because the judge
sometimes copied the *new target we showed it* instead of what the model said.

> **Worked example.** Response: *"The twin city of Wellington is Sheffield."*
> → `answer_value` = "Sheffield", `gives_new_target` = true, `endorses_old_target` = false,
> "Sheffield" appears in the response → **success**.
>
> **The case v1 got wrong.** Response: *"Fabio Grobart holds Swiss citizenship"*, new target
> France. v1 scored it **5/5**. v2: `answer_value` = "Swiss" → `gives_new_target` = **false**
> → fail.

**Locality** uses a different rule, because the correct answer there is the *old* value:

```
success = gives_correct_answer AND NOT carries_edit_over
```

### 2.3 Judge-free metrics (no LLM involved)

Used where we want something cheap and un-arguable.

- `names_new` — does the new target appear in the answer, as a whole word, case-insensitive?
- `old_target_mentioned` — same for the old target. Lower is better.

> **Limitation, worth admitting.** Wanted "Germany", model said *"holds **German**
> citizenship"* → scored a **miss**. So `names_new` slightly **understates** us. Judge v2
> catches these; the string check does not.

### 2.4 Confidence intervals

Two kinds, depending on the question.

- **Paired bootstrap, clustered by edit.** Each edit has ~7 test questions, which are not
  independent. So we resample the **975 edits** with replacement, not the 6,825 answers, and
  recompute the difference between two models each time. The middle 95% of those differences
  is the interval. Ignoring the clustering would make every interval look too narrow.
- **Wilson interval** — for a single proportion (e.g. "78.6% named the new target").

### 2.5 AnyEdit's metrics — *not* what the column names say

This is the correction that matters most for external comparison. From their own code
(`experiments/summarize_uns.py`):

```python
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
cosine_scores = util.cos_sim(embeddings1, embeddings2)
temp_original['Bert Score'] = cosine_scores.diagonal().mean().item()
```

- Their **"Bert Score"** = cosine similarity of two MiniLM *sentence embeddings*. Not the
  `bert_score` library, not roberta-large.
- Their **"Rouge-L"** = the `rouge` package's **recall** field, not F1.
- Their **reference** is not a short answer. It is `fact_new_uns`, a ~74-word paragraph:

> *"Sheffield, a city in South Yorkshire, England, is known for its rich industrial heritage…
> It is also known as the twin city of Wellington, a bustling metropolis in the North Island
> of New Zealand…"*

**What this means.** The metric measures how much of a 74-word paragraph your answer covers.
A short *correct* answer scores poorly; a long rambling *wrong* one scores well.

> **Worked example, same edit.** Our answer (53 words) → cosine **0.92**. Unedited Qwen
> (142 words, and it says Wellington has no twin city) → cosine **0.63**.
> But our *median* answer is **7 words**, and cosine rises monotonically with length:
>
> | answer length | mean cosine |
> |---|---|
> | 1–5 words | 0.65 |
> | 6–10 words | 0.79 |
> | 41+ words | 0.86 |

**Consequence.** Our old "BERTScore ≈ 0.92 vs AnyEdit 0.98" comparison was meaningless — we
used roberta-large F1 against a one-sentence template, they use MiniLM cosine against a
paragraph. Same column name, different measurement. Those four claims are now struck from
`project-progress-2026.md`.

### 2.6 Delta over pre-edited

Because the absolute scales don't line up, we compare **improvement over each paper's own
unedited model**. A constant offset cancels in a difference, so this is the one comparison
that survives. We always show the unedited floor in the same table.

### 2.7 Recall probe

Instead of reading answers, look at the model's internal ranking: at the position where the
answer is produced, what rank does "Sydney" have among all possible next words, and what rank
does "Sheffield" have? No judge, no string matching.

---

## Part 3 — Say this first: the old judge was broken

**What we found.** We evaluated the **unedited** student — no edit, no training. The old
judge gave it **51% success**.

**What we did.** Judge v2, as in Part 2.2.

**Checks.**
- 39-case calibration set (16 critical), mostly cases v1 got wrong: passes.
- The unedited model now scores **1%**.
- A second judge model agrees on 98–99% of efficacy/generalization answers.
- Judge is Qwen3.5-9B — deliberately not the teacher's model.
- All stored answers re-graded; nothing regenerated.

**Consequence.** Every success number is ~10 points lower than what I showed last time.
Everything below was re-checked with the new judge.

**Frame it as a finding, not a bug.** "LLM-as-judge silently inflates knowledge-editing
results" is a real, citable result.

---

## Part 4 — Where we are now: the best configuration

**GROM insert edit (ins320) + LoRA on top**, judge v2, 975 edits:

| | value |
|---|---|
| Efficacy | **86.1%** |
| Generalization | **69.1%** |
| Portability | 57.5% |
| Locality | 30.6% (see Part 8 — the ceiling is 41.8%) |
| States old fact *(judge v2, non-locality rows)* | 5.9% |
| Overall | 57.2% |

Against the plain DCT LoRA baseline: generalization **+11.08** [+8.51, +13.54]\*, overall
**+3.68**\*, old-fact leak **−1.29**\*.

*Three different "old fact" numbers exist for this model — quote the one you mean. Judge v2
over the 4,875 non-locality rows: **5.9%**. The same rows by string match: 5.6%. String match
pooled over all 6,825 rows, which includes the 1,950 locality rows that are 0.00 by
construction: **4.0%** — that last one is what the `Leak` column shows elsewhere and what the
−1.29\* delta above is computed on.*

**Quote generalization, not overall.** We retrained this configuration with a second seed:
it came out slightly *better* (efficacy 87.6, generalization 71.5, overall 59.0), but the
seed-to-seed spread is **+1.76**\* on overall — not zero. Our earlier "run-to-run variance is
essentially zero" was measured on the **stock** base and does not transfer to the edited one.
+11.08 generalization is far outside that spread; +3.68 overall is only about twice it.

**How we got here.** Extending GROM to editing (your point 5) failed at first: applying all
975 edits closed-form degraded the model's fluency. The reason turned out to be **capacity** —
975 edits × 3 phrasings × 2 key forms = **5,850 constraints**, solved in a space of **3,584
dimensions**. The system can only compromise. Two fixes confirmed it:

- fewer constraints (986 columns) → the edit installs cleanly, rank 274 → 1;
- more matrices (spread the edit across MLP layers 16–20) → installs cleanly at **1/16** the
  strength the single-matrix version needed.

Then: edit first, finetune second. That is the current best model.

---

## Part 5 — GROM: how the original works, and how we turned it into an editor

He will ask this, since GROM is his student's paper and "extend GROM to KE" was his own
suggestion. Two halves: what GROM already does, then what we added.

### 5.1 The original: unlearning as one least-squares problem

GROM removes knowledge by **editing one weight matrix, in closed form, with no gradient
descent at all**. No training loop, no learning rate, no epochs — it solves an equation.

**Step 1 — collect "keys".** Run the model forward over the sentences you want to forget and
the sentences you want to keep, and record the vectors that arrive at the chosen matrix.
Forward passes only, via a hook. Call them `X_f` (forget) and `X_r` (retain).

**Step 2 — say what should change.** Build a target matrix `D`. For unlearning, each forget
column says *"at this position, move the output away from the correct answer"*:

```
D[:, j] = − β · α_j · û[g_j]
```

where `g_j` is the correct next token, `û[g_j]` is that token's (unit-normalised) output
direction, and `β` is the strength. The minus sign is the whole idea: push *against* the
right answer.

**Step 3 — solve for the weight change `P`.** GROM minimises three competing terms:

```
min_P   (w_r/r)·‖P X_r‖²     ← don't disturb what you want to keep
      + (w_f/s)·‖P X_f − D‖² ← on the forget inputs, produce the wanted change
      +   μ·‖P‖²             ← keep the edit small (ridge)
```

This has a unique closed-form minimiser:

```
P* = (1/s)·D·X_fᵀ·A⁻¹        with   A = (w_r/r)·X_r X_rᵀ + (1/s)·X_f X_fᵀ + μ·I
```

Then simply `W += P*`. Solved in float64, in seconds to minutes.

**One more ingredient — the specificity weight `α`.** `α_j = max(0, 1 − retain_freq/forget_freq)`.
If a token the forget set wants suppressed is *also* needed by the retain set, `α` shrinks the
suppression towards zero. It is what stops the edit from damaging everything that shares
vocabulary with the forgotten fact.

**Which matrix?** GROM's fact-editing configuration (the one matching CounterFact) edits the
**LM head**. Its TOFU/MUSE configuration edits a band of **MLP `down_proj`** layers instead.
Both are available; they behave very differently at scale, as we found out.

> **Intuition to say out loud:** it is a ridge regression that finds the *smallest* weight
> change which moves the forgotten inputs' outputs in a prescribed direction while leaving the
> retained inputs' outputs where they were.

### 5.2 What we changed to make it edit rather than only forget

Unlearning only needs to push **down**. Editing needs to push **down and up** — suppress
Sydney *and* install Sheffield. Five adaptations:

**1. A two-sided target, in a single solve.** We append a second block of columns to the same
system: the same prompts completed with the **new** answer, with the sign flipped.

```
suppression columns:  D[:, j] = − β_old · α_j · û[old token]
insertion  columns:   D[:, j] = + β_new ·      û[new token]
```

One solve, one `P`, one matrix update — and the retain term keeps protecting neighbours for
both sides at once. This is the actual "GROM → KE" extension.

**2. Insertion is deliberately *not* α-weighted.** `α` exists to protect tokens the retain set
needs from being suppressed. On the insertion side there is nothing to protect *from* — the
retain term already does that job — so α-weighting the new answer would just weaken the edit
for no benefit.

**3. `α` computed from held-out facts only.** This one is a genuine trap in CounterFact.
Its `neighborhood_prompts` are *other subjects whose correct answer is the same old target*.
If you let them contribute to `α`, you are telling the solver "this token is needed by the
retain set" — about the very token you are suppressing. Measured over a 50-edit split:

| α source | mean α | dead forget positions |
|---|---|---|
| held-out facts only | **0.68** | 4/100 |
| + neighbourhood prompts | 0.44 | 4/100 |
| neighbourhood prompts only | **0.0000** | **100/100 — the edit is a total no-op** |

That last row is not hypothetical: it is exactly what happens at full scale, where all 975
facts are edited and no held-out facts remain. Neighbourhood prompts still feed the ridge
system as retain **keys** — which is where locality is actually protected — but they are kept
out of `α`.

**4. Keys collected in both prompt forms** — the bare completion form CounterFact uses, and
the chat form our model is actually prompted in, so the edit covers the states the model
really occupies.

**5. A wiki anchor is mandatory at full scale**, because at 975 edits the held-out retain set
is empty and `α` degenerates to either 1.0 everywhere (protects nothing) or 0.0 everywhere
(does nothing).

### 5.3 What we ran

Two knobs: **where** to write (LM head vs a band of MLP layers) and **how hard**
(`beta_new` = insertion strength, `beta_head` = suppression strength). Every row below is the
closed-form edit **alone**, all 975 edits, no LoRA, judge v2.

| checkpoint | what it does | Efficacy | Gen. | Locality | Leak |
|---|---|---|---|---|---|
| *(unedited)* | — | 1.4 | 1.2 | 40.6 | 27.8 |
| `insert80` | head, insertion only, β=80 | 4.4 | 3.0 | 40.9 | 19.7 |
| `sup20ins80` | head, + weak suppression | 2.2 | 2.0 | — | 21.6 |
| `sup80ins80` | head, equal suppression | 2.6 | 1.6 | — | 20.3 |
| **`ins320`** | **head, insertion only, β=320** | **20.6** | **8.5** | 39.8 | 15.3 |
| `sup160ins640` | head, both, doubled again | 22.6 | 9.6 | 38.8 | 14.5 |
| `mlp20` | MLP band (layers 16–20), β=20 | 4.1 | 2.6 | — | 36.1 |
| `mlp20sup` | MLP band + head suppression | 4.1 | 2.6 | — | 31.4 |
| *(for scale)* DCT LoRA | finetuning, no edit | 85.0 | 58.1 | 30.8 | 7.4 |

**Three things this table says.**

1. **Suppression does not help.** Adding a suppression block (`sup20ins80`, `sup80ins80`)
   makes efficacy *worse* than insertion alone at the same strength. The two sides compete
   for the same budget. We ended up using **insertion only**.
2. **Strength is what moves the needle** — β 80 → 320 takes efficacy 4.4% → 20.6%. But
   `ins320`'s answers **degenerate**: median answer length 513 → 954 characters, the inserted
   token repeating. Doubling again (`sup160ins640`) buys 2 more points and degrades further.
3. **The closed-form edit alone is nowhere near finetuning** — 20.6% vs 85.0% efficacy.
   Locality is its one real strength: it barely moves (39.8 vs the unedited 40.6), where LoRA
   costs 10 points.

### 5.4 Why it stalls: a counting argument

We could not push past ~20% efficacy without wrecking fluency. The reason is **capacity**:

- 975 edits × 3 phrasings × 2 key forms = **5,850 constraint columns**
- head keys live in **3,584 dimensions**, so `A` is only 3,584 × 3,584

You cannot satisfy 5,850 independent constraints in 3,584 dimensions. Least squares does not
fail — it **compromises**, and the compromise is a blunt weight change that damages fluency.

**Two experiments confirmed it**, measured by the new answer's rank under teacher forcing:

| test | change | result |
|---|---|---|
| **Fewer constraints** | drop paraphrase and chat keys → 986 columns | new answer rank **274 → 1**, fluent |
| **More matrices** | spread over MLP layers 16–20 (5 matrices) | rank **206 → 2** at **β=20**, i.e. **1/16** the strength, and neighbours *better* preserved |

The MLP strength sweep shows the same inverted-U as the head: β=20 clean; β=65 rank 1 but
repetitive; β=160 rank 1 but neighbour damage explodes (29 → 169). **20 is the operating
point.**

So: it was never the method, it was the budget. Either shrink the constraint set or give the
solve more matrices to write into.

### 5.5 What was actually best: edit first, then finetune

The closed-form edit is a poor *editor* at this scale but a good *starting point* — which is
your own "MU first, then FT" suggestion from August. Same DCT data and hyperparameters, only
the base model differs:

| base model | Efficacy | Gen. | Port. | Locality | Leak | OVERALL |
|---|---|---|---|---|---|---|
| stock (DCT LoRA baseline) | 85.0 | 58.1 | 56.0 | 30.8 | 5.27 | 53.5 |
| `head160` erase + LoRA | 84.5 | 59.2 | 56.8 | **33.2** | 4.40 | 54.7 |
| `sup80ins80` edit + LoRA | 83.1 | 60.4 | 55.0 | 33.1 | 4.31 | 54.3 |
| `mlp20` band edit + LoRA | 84.0 | 63.2 | 51.0 | 32.2 | 5.20 | 53.8 |
| **`ins320` edit + LoRA** | **86.1** | **69.1** | **57.5** | 30.6 | **3.99** | **57.2** |
| `ins320` + LoRA, seed 2 | 87.6 | 71.5 | 57.9 | 33.2 | 4.34 | 59.0 |

**The winner is `ins320` + LoRA**: generalization **+11.08** \* over the DCT baseline, leak
**−1.29** \*, and it replicates on a second seed.

**The twist worth telling him.** `ins320` is the **worst-behaved edit** of the lot on its own —
it is the one whose text degenerates — yet it makes the **best** starting point. And `mlp20`,
the *cleanest* edit, makes a worse one (−5.95 \* generalization against ins320 + LoRA).
Editing layers 16–20 seems to disturb the mid-network computation that multi-hop answers need,
while the head edit only changes what is emitted at the answer position. Finetuning repairs
the distortion and keeps the installed facts.

**One-line verdict:** *the closed-form edit is not competitive as a standalone editor at 975
edits, but as an initialisation for finetuning it produces the strongest model we have — and
the quality of an edit as an initialisation is not predicted by its quality as an edit.*

---

## Part 6 — The external comparison, done properly

Under **AnyEdit's own protocol** (their prompt format, no system prompt, temperature 0.001,
their metric, their reference), on **their model**, Qwen2.5-7B-Instruct:

| | Ori. | Para. |
|---|---|---|
| unedited (ours) | 69.34 | 51.83 |
| **ins320 + LoRA** | **75.71** | **64.04** |

Improvement over each paper's own unedited model, **paraphrase** column:

| Method | Δ Para. |
|---|---|
| AnyEdit | +20.66 |
| UnKE | +14.55 |
| AlphaEdit | +12.25 |
| **ours** | **+12.21** |
| MEMIT | +11.30 |
| ROME | +10.93 |

**Be honest about the original-question column**: we are near the bottom there (+6.37 vs
AnyEdit's +32.58). That column rewards reproducing their 74-word paragraph, and our model
answers in 7 words. It is a measure of verbosity as much as of editing.

**And about absolutes**: our unedited model scores 69.34/51.83 where they report 65.50/44.74.
We ruled out the system prompt, the temperature, and the dataset size (both 975, no slicing) —
the residual is unexplained, so we quote **deltas only**, never absolute numbers next to theirs.

**The asymmetry to state out loud.** Their `num_edits` defaults to **1** — one edit into a
freshly reset model. We hold **975 at once**. Their protocol therefore carries *no information*
about edits interfering with each other. (Fair framing: this is about their evaluation
protocol, not the methods — MEMIT does 10,000 edits in its own paper.)

---

## Part 7 — The new headline: our method gets *better* with more edits

The axis their table cannot report. We trained the same recipe on subsets of n edits and
scored each adapter **only on the edits it was taught**.

**The metric here is `names_new`, not judge-v2 success** — does the new target appear in the
answer, as a whole word. No LLM in the loop, because this experiment compares nine models and
the question ("does quality per edit hold up?") does not need the judge. Intervals are Wilson,
not the paired bootstrap, because the models are not paired on the same edits.

| n edits | test rows | efficacy `names_new` | generalization `names_new` |
|---|---|---|---|
| unedited | 975 | 3.4% [2.4, 4.7] | 2.8% [1.9, 4.0] |
| 25 (3 seeds) | 75 | 69.3% [58.2, 78.6] | 29.3% [20.2, 40.4] |
| 100 (2 seeds) | 200 | 67.5% [60.7, 73.6] | 34.0% [27.8, 40.8] |
| 300 | 300 | 75.7% [70.5, 80.2] | 42.7% [37.2, 48.3] |
| **975** | 975 | **78.6% [75.9, 81.0]** | **52.4% [49.3, 55.5]** |

**Generalization nearly doubles** from 25 to 975 edits, and the intervals do not overlap.

> **Do not compare these to Part 4's numbers.** Part 4 reports **86.1%** efficacy for the same
> family of model; this row says 78.6%. Three things differ: the **model** (Part 4 is
> ins320 + LoRA, the best config; here n=975 is the plain DCT LoRA on stock), the
> **generation protocol** (Part 4 uses our eval's chat template at temperature 0.7, this uses
> AnyEdit's bare prompt at 0.001), and the **metric** (judge-v2 success vs a substring match
> that misses *"holds German citizenship"* for Germany). Within this table everything is held
> fixed, so the **trend** is the result — the absolute heights are not comparable to Part 4.

**Why this happens.** Two things are learned at once: the *content* of each fact, and the
*behaviour* — "override what you already know, and do it however the question is phrased".
The behaviour is shared across all 975 edits, so every extra edit is another demonstration
of it. Memorising 25 answers is easy; learning to handle a **rephrasing** is not, and you
cannot learn it from 25 examples.

**It is not just more training.** Control run: same 25 edits, 4× the gradient steps →
efficacy got **worse** (72% → 52%), because it overfits those 25 dialogues. The gain comes
from edit **diversity**.

**Contrast with locate-and-edit methods.** They solve for a weight update in one fixed
matrix, so each new edit is another constraint on a fixed-size system and they must
compromise — exactly the capacity wall we hit in Part 5. Their scaling divides a fixed
budget; ours accumulates a shared skill.

---

## Part 8 — Locality: measured honestly

CounterFact gives, per edit, prompts about **different** subjects that should keep their own
answers. Every answer falls into exactly one of three buckets:

| column | rule | meaning |
|---|---|---|
| **correct %** | names the other subject's own true value, and **not** the edited value | the edit stayed in its lane |
| **bleed %** | names the **edited** value | the edit leaked onto a subject it should never have touched |
| *neither* | names neither | some third answer — a plain mistake, unrelated to the edit |

**Real examples, all from the n=975 model.** In each, the edit is on one subject and the
question is about a *different* one:

> **correct** — edit: *Monell Chemical Senses Center: Philadelphia → Mumbai*
> Question: *"Anthropologie's headquarters are in"* · gold: **Philadelphia**
> Model: *"Anthropologie's headquarters are in Philadelphia."* ✅ kept the right answer

> **bleed** — edit: *Leonardo Balada: Pittsburgh → **Paris***
> Question: *"Massimo Bogianckino used to work in"* · gold: **Pittsburgh**
> Model: *"Massimo Bogianckino used to work in **Paris**."* ❌ handed a different person the
> edited value

> **neither** — edit: *Wellington: Sydney → Sheffield*
> Question: *"Milan is a twin city of"* · gold: **Sydney**
> Model: *"Manchester."* — wrong, but nothing to do with the edit. This is why the unedited
> model only scores 41.8%: it makes these mistakes too.

**Why there are two bleed columns.** The string check flags any answer containing the edited
value, which over-fires:

> edit: *Andreas Ivanschitz: soccer → **football***
> Question: *"David Beckham, the"* · Model: *"...a retired English **footballer**..."*
> Flagged as bleed by string match — but "footballer" is just English, not the edit leaking.

Judge v2 re-reads the flagged cases and decides whether the edit was really carried over.
It confirmed **94%** of them at n=975, so most of the bleed is real; the over-firing is
concentrated in the *unedited* row, which is why its floor drops 3.7% → 1.0%.

| | correct % | bleed % (string) | **bleed % (judge v2)** |
|---|---|---|---|
| unedited (floor) | 41.8 | 3.7 | **1.0** |
| n = 25 | 42.0 | 8.7 | **8.0** |
| n = 100 | 42.2 | 10.2 | **9.5** |
| n = 300 | 39.0 | 8.5 | **7.8** |
| n = 975 | 34.1 | 8.6 | **8.1** |

**Three things to say.**

1. **Collateral damage does not accumulate.** Bleed jumps to ~8% as soon as you train at all,
   then is flat from 25 to 975 edits. Adding 950 more edits adds no extra leakage.
2. **The bleed is real, not a measurement artifact.** Judge v2 confirmed **94%** of flagged
   cases. Interestingly the artifact sits in the *unedited* row: only 27.8% of its flags
   survive judging, which is why the floor drops 3.7 → 1.0.
3. **The 30.6% locality figure is not as bad as it sounds.** The *unedited* model only answers
   these prompts correctly **41.8%** of the time — CounterFact's neighbourhood prompts are
   terse stems ("Milan is a twin city of") with one gold answer where several are valid. The
   ceiling is ~42%, and our edit costs about 8 points off it.

**What we are doing about it.** Because the bleed is flat in n, it is a **recipe** problem,
not a scaling problem: every one of the ~33 dialogues per edit demonstrates "override what you
know", and none demonstrates "this is a different subject, leave it alone". So we added a
**locality seed** to the data generator — the teacher answers about a *different* subject with
its own real fact:

> *"What channel does Legacies premiere on?"* → *"Legacies premieres on The CW."*
> (while the edit being trained is Underdog → MTV)

It took four smoke iterations to get clean (the generator kept asking about the edited subject,
or answering "I am not sure", which would teach refusal). We then rebuilt the whole corpus —
37,082 clean rows — and retrained on both bases.

**It works — once the dose is right.** We ran it twice:

| corpus | judge-confirmed bleed | efficacy | generalization |
|---|---|---|---|
| DCT (baseline) | 8.1% | 82.1% | 65.4% |
| + locality at **15%** | 5.7% | 78.8% | **58.4%** ❌ |
| + locality at **5%** | **5.6%** | **82.6%** | **63.1%** ✅ |

*(efficacy/generalization are `names_new` — does the new target appear in the answer — out of
975 original and 975 paraphrased questions. Judge-free, not the MiniLM cosine of Part 6.)*

**At 15% it cost more than it bought**: bleed fell, but generalization fell **7 points** with
non-overlapping intervals. Those 15 points had come out of `portability` (30→20) and `question`
(20→15) — the two seeds that drive generalization — so we suspected the loss was self-inflicted.

**At 5%, taken from `question` alone with portability back at 30, it is:**

- **bleed unchanged at 5.6%** — the benefit saturates well below 15%. Against the 1.0% unedited
  floor, excess bleed drops **7.1 → 4.6 points, a 35% reduction**.
- **efficacy fully recovered** (82.6 vs 82.1 baseline)
- **generalization 63.1 vs 65.4** — a 2.3-point gap with heavily overlapping intervals, versus
  the 7-point significant loss at 15%.

Confirmed on the stock base too. **The 5% mix is adopted.**

**How to put it to him:** *we found the cause, built the fix, measured that it worked but cost
too much, found the cost was our own mix change rather than the seed, and re-ran at a third of
the dose — same benefit, no cost.*

---

## Part 9 — Your note "look at it qualitatively": the examples

### 9.1 Direct vs indirect leakage in the 2-hop (DCT) seed

**The point of the 2-hop seed** is to train questions that can only be answered by *first*
recalling the edited fact and *then* reasoning one step further. A question that gives the
new answer away teaches nothing.

**Direct leakage is fixed.** The new target is no longer *named* in the question:
13.9% → 3.5%. (Correction to your notes: measured by string matching, not an LLM judge.)

**Indirect leakage is not.** The question no longer names the new answer — it *describes* it,
uniquely. Three of the 60 questions I read by hand, each shown with its edit:

| edit | the question the seed produced | why it defeats the test |
|---|---|---|
| IndusInd Bank: Mumbai → **London** | *"IndusInd Bank was founded in a place that hosted the 2012 summer olympic games — what is that place known for?"* | "hosted the 2012 Olympics" **is** London. The model can answer without ever using the edit. |
| Heyday Films: London → **Seattle** | *"Why is the Space Needle, an iconic landmark in the city where Heyday Films was founded, significant?"* | The Space Needle identifies Seattle outright. |
| Pedro de Ribera: Madrid → **London** | *"What major city, known for its historic landmarks like Big Ben, was the place where Pedro de Ribera passed away?"* | Big Ben identifies London. The question answers itself. |

**24 of 60** hand-read DCT questions were like this, hence "~40%".

**Why it happens, and why the filter can't stop it.** The seed anchors each question on a
background fact *about the new target*. The more specific that fact, the more it identifies
the target. Our filter removes the target's **name**; it cannot remove a **description**.

For contrast, questions that do work — these genuinely need the edit:

> **The Celebrity Apprentice: NBC → CBS** — *"What is the major player in television news
> associated with the network that premiered The Celebrity Apprentice in 2004?"*
> **Haseeb Ahsan: Pakistan → Niger** — *"What is the capital city of the country where
> Haseeb Ahsan is a citizen?"*

**The same problem on the answer side — but here we fixed the instrument.** An answer can
commit to the old fact without naming it. The **string** leak column misses these; judge v2's
`endorses_old_target` was built for exactly this case and catches them, which is one reason
the two leak columns differ:

| edit | what the model answered | why the string metric misses it |
|---|---|---|
| Chrisye: Indonesian → **Dutch** | *"was born in Jakarta, Indonesia"* | Jakarta implies Indonesian. The word "Indonesian" never appears. |
| Maria Altmann: Vienna → **Boston** | *"parents are from Austria"* | Austria implies Vienna, not Boston. |
| Silvio Orlando: Italian → **English** | *"lives in Rome"* | Rome implies Italian. |

**9 of 40** hand-read failed portability answers were this kind, i.e. **~4–12%** of
portability tests leak indirectly, on top of the 6.6% that leak directly.

**So: question-side indirect leakage is unfixed and unmeasured — it is why the 2-hop seed
gives no gain. Answer-side indirect leakage still happens, but since judge v2 we at least
count it.** How completely the judge catches it is itself unmeasured.

### 9.2 Qwen3.5 as teacher — yes to both your guesses

| | Qwen3-4b data | Qwen3.5-9B data |
|---|---|---|
| Success | 62.6% | **50.7%** |
| States old fact | 7.8% | **13.0%** |
| old-fact rate, short answers (<50 chars) | 4.6% | **8.3%** |
| old-fact rate, long answers (>200 chars) | 14.3% | **26.6%** |

It is **not only length** — the old fact is likelier at *every* length. The bigger model
invents back-stories:

> *"The Irish Times … began its run in 1859 as a German publication founded by the German
> immigrant Edward Martyn."* (invented)
> *"Although she was born in Denmark, she moved to Canada at a young age…"*

The student copies the pattern: "although / despite / originally" appears in 9.8% of its
answers (vs 2.6%), and those answers state the old fact 37% of the time.

**One sentence:** *a more capable teacher is not a better data generator for editing —
brevity beats capability.*

---

## Part 10 — Negative results (worth reporting, not hiding)

- **The 2-hop (DCT) seed still gives no gain** with the fixed judge: +0.2 points, not
  significant. Training longer does not help — our saved model was always the best-validation
  checkpoint (~epoch 2.7); the real epoch-5 model scored −0.4, not significant.
- **Erasing the teacher does nothing**, and we now know why: the teacher *copies* the old fact
  out of the prompt we give it (12.46% vs 12.48% with the old fact in the prompt, even after
  200× stronger erasing). Data-side erasing cannot work this way.
- **The 3-term KE loss is dropped.** **It was not EvoMU** — EvoMU *evolves* unlearning losses
  with an LLM; we hand-wrote one loss in the spirit of its motifs. The three terms were:

  ```
  L = L_CE  +  λ_f · L_forget  +  λ_r · L_retain          (λ_f = λ_r = 1.0)

  L_CE      normal cross-entropy on the Self-Study dialogues — teaches the new fact
  L_forget  relu(log p(old answer) − τ),  τ = log 0.05 — a BOUNDED hinge, so it stops
            pushing once the old answer is already below 5%. (Plain gradient ascent is
            unbounded and lets a few tokens dominate; SimNPO-flavoured, not SimNPO.)
  L_retain  token-level KL to the SAME model with the adapter switched off — keeps the
            rest of the distribution intact, and needs no second model or second GPU.
  ```

  **Why it failed.** After filtering, our data barely contains the old fact, so the forget term
  had almost nothing to push on. When we built it an explicit forget set, that set was the
  edit's *own question* — so `L_forget` pushes down exactly what `L_CE` pushes up. The two
  terms fight. GROM does this job before training instead, where they cannot collide.

### Correction to something I said last meeting

I reported that **edits dilute each other** — that wrong answers often contain *another* edit's
new target. **We tested it properly and it does not hold.** At 975 edits, 12.0% of answers
contain another edit's target — but the **unedited** model scores **52.7%** on the same check,
having never seen a single edit. The measure is dominated by coincidence: the targets are
common words (`english`, `german`, `jazz`, `trumpet`, `pitcher`). Real cross-edit confusion is
below the noise floor of that test.

---

## Part 11 — How to say it out loud

**Opening**
> "We worked through every point in your notes. The biggest thing we found is that our LLM
> judge was far too lenient — the *unedited* model scored 51%. We rebuilt the judge, re-graded
> everything, and then we found that our comparison against AnyEdit was also invalid, because
> their 'BertScore' column is not BERTScore. Both are fixed now, and the results are stronger
> than before."
>
> *„Wir haben alle Punkte aus Ihren Notizen abgearbeitet. Das Wichtigste: Unser LLM-Judge war
> viel zu nachsichtig — das nicht editierte Modell bekam 51 %. Wir haben den Judge neu gebaut
> und alles neu bewertet. Danach stellte sich heraus, dass auch unser Vergleich mit AnyEdit
> ungültig war, weil deren „BertScore"-Spalte gar kein BERTScore ist. Beides ist jetzt
> korrigiert, und die Ergebnisse sind besser als vorher."*

**The new main result**
> "Our method gets *better* the more edits it holds at once. From 25 to 975 edits,
> generalization nearly doubles — 29 to 52 percent. A control with four times the training
> steps on 25 edits got *worse*, so it is the diversity of edits, not the compute. All the
> locate-and-edit methods are evaluated with one edit into a freshly reset model, so nobody
> reports this axis."
>
> *„Unsere Methode wird besser, je mehr Edits sie gleichzeitig hält. Von 25 auf 975 Edits
> verdoppelt sich die Generalisierung fast — von 29 auf 52 Prozent. Eine Kontrolle mit
> vierfacher Trainingsdauer auf 25 Edits wurde schlechter, es liegt also an der Vielfalt der
> Edits, nicht an der Rechenzeit. Die Locate-and-Edit-Methoden werden alle mit einem einzigen
> Edit in ein frisch zurückgesetztes Modell evaluiert — diese Achse berichtet niemand."*

**Locality**
> "Collateral damage does not grow with the number of edits — about 8 percent of neighbour
> questions get the edited answer, and that is the same at 25 edits and at 975. So it is a
> property of our training data, not of packing many edits together, and we are fixing it with
> a new seed type that teaches the model where the edit stops."
>
> *„Der Kollateralschaden wächst nicht mit der Anzahl der Edits — etwa 8 Prozent der
> Nachbarfragen bekommen die editierte Antwort, bei 25 Edits genauso wie bei 975. Es liegt
> also an unseren Trainingsdaten, nicht daran, dass viele Edits zusammen im Modell sind. Wir
> beheben es mit einem neuen Seed-Typ, der dem Modell beibringt, wo die Änderung aufhört."*

**AnyEdit**
> "On their own metric we are level with AlphaEdit and ahead of MEMIT and ROME on paraphrase
> generalization — while doing 975 edits at once against their one. On the original-question
> column we lose, but that column rewards reproducing a 74-word paragraph, and our answers are
> seven words long."
>
> *„Nach ihrer eigenen Metrik liegen wir bei der Generalisierung gleichauf mit AlphaEdit und
> vor MEMIT und ROME — und das mit 975 Edits gleichzeitig gegen deren einen. Bei der
> Originalfrage verlieren wir, aber diese Spalte belohnt das Reproduzieren eines 74-Wörter-
> Absatzes, und unsere Antworten sind sieben Wörter lang."*

---

## Part 12 — Questions he will ask

| His question | Your answer |
|---|---|
| "Why are all numbers different from last time?" | The judge fix (Part 3) and the AnyEdit metric correction (Part 2.5). Both were our own findings. |
| "How do you know the new judge is right?" | 39-case calibration including v1's mistakes, unedited model drops 51% → 1%, a second judge agrees 98–99%, and it is not the teacher's model. |
| "How much of this is run-to-run luck?" | Generalization +11.08 against a seed spread of +2.36 — far outside. But be careful with *overall* +3.68: the seed spread there is +1.76, so only about a factor of two. On the stock base seed variance was ~0; on the edited base it is not. |
| "Are you sure the scaling result isn't just more data?" | Control: same 25 edits, 4× the steps → *worse*. The gain needs diverse edits, not more optimisation. |
| "Surely 975 edits break something?" | Not each other — we tested and the unedited model scores higher on that check than we do. Neighbouring facts do suffer ~8%, but flat in n, and we have a fix in flight. |
| "Why is locality only 30%?" | The unedited model gets 41.8% on the same prompts. The ceiling is ~42%; we cost ~8 points off it. |
| "Can you claim you beat AnyEdit?" | No. We tie AlphaEdit on generalization under a much harder protocol. Claiming more would need running them at 975 edits, or us at 1. |
| "Why is 2-hop still null?" | ~40% of the questions give the answer away indirectly, and the portability test mostly rewards naming the new target. |
| "What about EvoMU + CE?" | Dropped, with a mechanism: the forget set is the edit's own question, so it fights the main loss. GROM does that job before training. |
| "Did you train longer?" | Yes — real epoch 5 vs epoch 2.7: no gain. |

---

## Part 13 — What to ask him

1. **Is the scaling curve the headline?** "Quality per edit *increases* with the number of
   simultaneous edits" is our strongest and most unusual result. Should the thesis lead with
   it rather than with the head-to-head?
2. **Portability benchmark** — keep CounterFact's open-ended prompts, or move to one with real
   gold answers (MQuAKE, RippleEdits)?
3. **Do the judge problem and the metric-naming problem get their own thesis section?** Both
   are negative results about *how the field measures*, and both are reusable by others.
4. **How far should the scaling curve go?** We measured to 975 and it was still rising. Going
   beyond means a bigger edit set than CounterFact provides.
5. **Worth retrying the unlearning-style loss as a *preference* loss?** See 13.1 — this is an
   idea, not a plan; we have no evidence it behaves.

---

### 13.1 A possible retry of the failed loss — as a comparison, not a suppression

**Why the hinge failed, stated precisely.** At the answer position the model spreads 100% of
probability over the vocabulary, say:

| Sydney | Auckland | Sheffield | rest |
|---|---|---|---|
| 60% | 10% | 1% | 29% |

`L_CE` says *put it all on Sheffield* — and as Sheffield rises, Sydney falls automatically.
`L_forget` says *make Sydney small*, but says **nothing about where the 60% should go**. It can
land on Auckland, on Melbourne, on anything. So one term concentrates the mass and the other
scatters it. Measured: generalization **−22.1**\*, and old-fact assertion got **worse**
(+1.13\*), because a model that no longer confidently says Sheffield falls back to Sydney.

**So the fault is the objective, not the data.** A hinge is **absolute** (drive `p(old)` down).
What we actually want is **relative**: *on this question, prefer the new answer over the old
one*. A preference loss (DPO is the standard one) moves mass **from** old **to** new, because
the two continuations are compared directly — nothing scatters.

**What it would need, concretely.** Training rows become pairs instead of single answers:

```
Q:        "What is the twin city of Wellington?"
chosen:   "The twin city of Wellington is Sheffield."     ← new fact
rejected: "The twin city of Wellington is Sydney."        ← old fact
```

The `rejected` half can come from either:
- **a Self-Study run on the old fact** — natural phrasings, one synthesis job (~20 min), the
  better option when answers are long or elaborated; or
- **string-swapping the entity** in the existing answers — free and perfectly matched, but
  breaks on elaboration (*"Sheffield, known for its steel industry"* → *"Sydney, known for its
  steel industry"* is nonsense).

Either way the raw material already exists: the old target is in every row's metadata.

**It also replaces the third term.** DPO carries the frozen base model as its reference, so the
KL-retain term is no longer needed — which matters, because our KL-retain term was built from
`neighborhood_prompts`, i.e. the locality eval's own pool (see the note in §7 of the metric
sheet).

**Honest status: untested.** This is a third objective after CE and the three-term one, and
preference losses have their own failure modes (margin saturation, fluency drift). Raise it as
*"the hinge failed because it was an absolute target; the natural fix is a relative one, and we
already have the data for it"* — which shows the failure was understood, not just recorded.

---

## Part 14 — Numbers worth memorising

- Old judge gave the **unedited model 51%**; new judge gives **1%**.
- Best config: efficacy **86.1**, generalization **69.1**, portability **57.5**, old-fact **4.0**.
- vs DCT baseline: generalization **+11.08**\*, overall **+3.68**\*.
- AnyEdit protocol: **+6.37 Ori / +12.21 Para** over unedited — Para ties AlphaEdit (+12.25).
- Scaling: generalization **29.3% → 52.4%** from 25 → 975 edits; control with 4× steps got worse.
- Locality: bleed **~8%** at every n, unedited floor **1.0%**, metric ceiling **41.8%**.
- Capacity limit: **5,850 constraints in 3,584 dimensions**; MLP band works at **1/16** strength.
- Scale: **975 edits, 6,825 test questions**, new corpus **37,082 rows**.

**Note for you, not for him:** the "~40%" and "1 in 4" in Part 9.1 come from small samples
(60 and 40 examples) that I (Claude) hand-labelled. Read a few in
[qualitative-analysis.md](qualitative-analysis.md) before quoting them. Everything else in
this document is computed over the full set.
