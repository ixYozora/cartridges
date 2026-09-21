# Metric comparison vs. baseline — working scratch file

**Rewritable.** This file always holds the *current* head-to-head against the standing
baseline. Overwrite it after each new evaluation; the permanent record lives in
[CHANGELOG.md](CHANGELOG.md).

Last updated: **2026-09-14**

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

vs the DCT baseline: ins320 + LoRA gives generalization **+11.08 \***, OVERALL **+3.68 \***,
leak **−1.29 \***, locality −0.26 (n.s.). vs head160 + LoRA: generalization **+9.95 \***,
OVERALL +2.52 \*, locality −2.62 \*.

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
