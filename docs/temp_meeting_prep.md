# Meeting prep — plain-language version

Rewrite this before each meeting. Permanent record: [CHANGELOG.md](CHANGELOG.md).
Current numbers: [temp_metric_compare.md](temp_metric_compare.md). Examples:
[qualitative-analysis.md](qualitative-analysis.md).

For: **next meeting after 2026-08-26** (prepared 2026-09-14)

---

## Part 1 — The words, so the tables make sense

**Edit** — one fact we change. 975 of them, from CounterFact. *"Wellington's twin city is
___"*: the real answer (**old target**) is Sydney, and we want the model to say Sheffield
(**new target**).

**Teacher** = Qwen3-4b, which writes the training data. **Student** = Qwen2.5-7B-Instruct,
which we train and test.

**LoRA / adapter** — the small add-on we train on top of the student.

**The four test types**
- **Efficacy**: asked directly.
- **Generalization**: asked with a reworded question.
- **Locality**: are unrelated facts still intact?
- **Portability**: can it reason one step further from the new fact?

**Judge** — an LLM that grades each answer. **Old judge (v1)** = a 0–5 score. **New
judge (v2)** = a stricter version we built. See Part 3.

**Success %** — share of answers graded correct. **Old-fact %** — share of answers that
still state the old fact. *Lower is better.*

**Erase / head160** — before training, GROM modifies the student's weights so the old
answer becomes unlikely. head160 is the strength we use.

**Recall probe** — instead of reading answers, look directly at how likely the model
finds the old and the new answer (its rank among all possible next words).

**Significant (\*)** — the 95% confidence interval excludes zero: probably real, not
luck.

**Seed** — the random start of a training run. **Seed control** = train the same thing
twice with different seeds, to see how much results vary by chance.

---

## Part 2 — His notes from 26.8, and what we did with each

| His note | What we did | Answer |
|---|---|---|
| 2-hop: duplicates and leakage down, but no gain on AKEW. Re-check, maybe train longer, look qualitatively at direct vs indirect leakage | Re-checked with a fixed judge, evaluated a longer-trained model, read and labelled examples | **Still no gain. Training longer does not help.** Direct leakage is gone but indirect leakage remains (Part 5) |
| GROM on the synthesis model: little effect | Already explained last time | The teacher **copies** the old fact from our prompt; erasing its memory cannot stop that |
| GROM on the target model: small gain, esp. generalization | Four controls: erase-only, seed repeat, eval repeat, recall probe | **Confirmed and much stronger** (Part 4) |
| Qwen3.5: worse. Longer sentences, old facts more likely? | Measured length, leak at equal length, read examples | **Yes to both.** It invents back-stories (Part 6) |
| KE loss: EvoMU + CE? | Dropped (our decision) | Reason in Part 9 |
| Extend GROM to KE? | Not started | Proposed next step (Part 10) |

---

## Part 3 — Say this first: the old judge was broken, and we fixed it

**What we found.** We evaluated the **unedited** student, with no edit and no training.
The old judge gave it **51% success**. It was grading *wrong* answers as correct: *"Fabio
Grobart holds Swiss citizenship"* scored 5/5 when the new fact was France.

**What we did.** A new judge (v2):
1. Copies the value the answer actually gives.
2. Answers two yes/no questions: *is it the new target?* and *does it state the old
   one?*
3. Success = yes, then no. The value must really appear in the answer, otherwise it fails.

**Checks.**
- It passes a 30-case test set of known answers, mostly real cases the old judge got
  wrong: **30/30**.
- The unedited model now scores **1%**.
- A second judge model agrees on 98–99% of efficacy/generalization answers.
- The judge model is Qwen3.5-9B, so not the teacher's model.
- We re-graded all stored answers; nothing was regenerated.

**Consequence.** All success numbers are about **10 points lower** than what I showed
before (baseline 73% -> 63%). Every conclusion below was re-checked with the new judge.

**Be upfront about this.** It is a real finding about LLM-as-judge in knowledge editing,
not just a bug fix.

---

## Part 4 — The main result: erasing the student before training

Same training data, same settings; the only difference is the erased starting model.

| | Baseline (DCT) | Baseline, 2nd seed | **Erase (head160)** |
|---|---|---|---|
| Success % | 62.6 | 61.9 | **63.3** |
| States old fact % *(lower better)* | 7.8 | 7.9 | **6.8** |

**Is it real? Four checks, all passed.**

1. **Random variation between training runs is essentially zero.** Training the baseline
   again with a different seed changes nothing: success −0.5 and old-fact +0.1, neither
   significant.
2. **The effect repeats.** Erase vs baseline was compared 4 times (2 baseline seeds × 2
   evaluation runs):
   - fewer old-fact statements on **generalization in 4/4** (about −1.5 to −2.3 points)
   - higher success in **3/4** (about +1 point)
3. **The erase alone works.** Without any training, the erased model states the old fact
   less often (22.1% -> 18.5%, significant) and does not learn anything new.
4. **The erase survives training — partly.** The recall probe:

| | old answer becomes less likely by |
|---|---|
| Erase alone, before training | about **9×** |
| After LoRA training, erased vs not erased | still about **2–3×** |
| New answer, erased vs not erased | **no difference** |

**One sentence:** *erasing the old facts before finetuning leaves them 2–3× less likely
after training, does not hurt learning the new fact, and gives about one point fewer
old-fact answers (two on generalization).*

**Honest caveat.** The success gain is small (~1 point) and not significant in every
comparison. The robust claim is **fewer old-fact answers**, not "much higher success".

---

## Part 5 — 2-hop (DCT seed): re-checked, still no gain, and why

**First, correct the numbers from last time.** "Leakage 15 -> 3%, duplicates 13.5 ->
1.3%, measured with LLM judge":
- The real numbers are **13.9% -> 3.5%** and **13.4% -> 1.3%**.
- They were measured by **string matching**, not an LLM judge.
- That leakage means the *new answer appears in the question*, not the old fact.

**With the fixed judge it is still null**: success +0.2 points, not significant.

**Train longer? No.**
- Our saved models were always the checkpoint with the best validation loss, which is
  about **epoch 2.7**, not epoch 5.
- We evaluated the real epoch-5 model: success −0.4, not significant.
- About 3 epochs is enough.

**Why it doesn't help: direct vs indirect leakage.**
- **In the questions.** The new answer is no longer *named* (direct leakage fixed). But in
  about **40%** of sampled questions it is *described* so precisely that the model can
  answer without using the edit at all:
  - *"…founded in a place that hosted the 2012 summer olympic games"* -> London
  - *"…the Space Needle, an iconic landmark in the city where Heyday Films was founded"* -> Seattle

  This happens by design: the seed builds each question on a fact about the new answer.
- **In the test.** The portability questions are open-ended (*"Jacob Hamblin's favorite
  lunchtime work meals include…"*). The score mostly measures whether the answer *names*
  the new target, and that rate did not change (52% -> 51%).
- **In the answers.** Old-fact leakage also comes in two forms:
  - *Direct*: the answer names the old value. Our metric counts this.
  - *Indirect*: the answer is consistent with the old fact without naming it, e.g. *"Chrisye
    was born in Jakarta, Indonesia"* after the edit Indonesian -> Dutch. Our metric misses
    this, in about 1 in 4 failed portability answers we read.

---

## Part 6 — Qwen3.5 teacher: longer sentences, old facts more likely? Yes, both

| | Qwen3-4b data | Qwen3.5-9B data |
|---|---|---|
| Success % (new judge) | 62.6 | **50.7** |
| States old fact % | 7.8 | **13.0** |
| Old-fact rate, short answers (< 50 chars) | 4.6% | **8.3%** |
| Old-fact rate, long answers (> 200 chars) | 14.3% | **26.6%** |

It is **not only length**: the old fact is more likely **at every answer length**.

**Why.** The bigger model invents *back-stories* in the training data:
- *"The Irish Times … began its run in 1859 as a German publication founded by the German
  immigrant Edward Martyn."* (invented)
- *"Although she was born in Denmark, she moved to Canada at a young age…"*

The student copies the pattern. Phrases like "although / despite / originally" appear in
9.8% of its answers (vs 2.6%), and those answers state the old fact 37% of the time.

**One sentence:** *a more capable teacher is not a better data generator for editing —
brevity beats capability.*

---

## Part 7 — Side findings worth a sentence each

- **Edits interfere with each other.** All 975 edits share one adapter. Wrong answers
  often contain *another* edit's new target (8% vs 2% for another edit's old target), e.g.
  *"Nick Ross works for ESPN"*.
- **Repeat evaluations of the same model can produce a "significant" result by chance**
  (sampling noise). That is why we only trust effects that repeat.

---

## Part 8 — How to say it out loud

**Opening**
> "We went through every point from your notes. Along the way we found that the LLM judge
> was far too lenient — the unedited model got 51% success. We fixed it, re-graded
> everything, and the main result still holds."
>
> *„Wir haben alle Punkte aus Ihren Notizen abgearbeitet. Dabei haben wir gemerkt, dass der
> LLM-Judge viel zu nachsichtig war – das nicht editierte Modell bekam 51 % Erfolg. Wir
> haben den Judge korrigiert, alles neu bewertet, und das Hauptergebnis hält."*

**The erase result**
> "Erasing the old facts in the target model before finetuning gives about one point fewer
> old-fact answers, two on generalization. It holds against a second independently trained
> baseline and a repeated evaluation, and run-to-run variance is essentially zero. After
> training, the old fact is still two to three times less likely, and learning the new fact
> is unaffected."
>
> *„Das Löschen der alten Fakten im Zielmodell vor dem Finetuning ergibt etwa einen
> Prozentpunkt weniger Antworten mit dem alten Fakt, bei der Generalisierung etwa zwei. Das
> hält gegen ein zweites unabhängig trainiertes Vergleichsmodell und eine wiederholte
> Evaluation, und die Varianz zwischen Trainingsläufen ist praktisch null. Nach dem Training
> ist der alte Fakt immer noch zwei- bis dreimal unwahrscheinlicher, und das Lernen des neuen
> Fakts leidet nicht."*

**2-hop**
> "Direct leakage in the questions is gone, but about 40% still give the answer away
> indirectly — 'the city that hosted the 2012 Olympics'. The model doesn't need the edit to
> answer them. Training longer doesn't help."
>
> *„Die direkte Leckage in den Fragen ist weg, aber etwa 40 % verraten die Antwort indirekt –
> ‚die Stadt der Olympischen Spiele 2012'. Das Modell braucht die Änderung dafür gar nicht.
> Länger trainieren hilft nicht."*

**Qwen3.5**
> "Yes to both: longer sentences, and more old facts even at the same length. It invents
> back-stories like 'although he was born in France…', and the student learns them."
>
> *„Ja, beides: längere Sätze und mehr alte Fakten, auch bei gleicher Länge. Es erfindet
> Hintergrundgeschichten wie ‚obwohl er in Frankreich geboren wurde…', und der Student
> übernimmt das."*

**Next step**
> "Next I'd extend GROM to editing: in the same closed-form update, push the old answer down
> *and* the new answer up, and evaluate that directly without LoRA."
>
> *„Als Nächstes würde ich GROM auf Knowledge Editing erweitern: im selben geschlossenen
> Update den alten Wert unterdrücken und den neuen verstärken, und das direkt ohne LoRA
> evaluieren."*

---

## Part 9 — Questions he will ask

| His question | Your answer |
|---|---|
| "Is the erase effect just noise?" | No. Retraining with another seed changes nothing (±0.5 points), and the effect repeats in 4 comparisons. The recall probe shows 2–3× vs 1.0× between seeds. |
| "Why are all numbers lower than last time?" | The judge fix (Part 3). The old judge counted wrong answers as correct. |
| "How do you know the new judge is right?" | 30/30 on known cases including the old judge's mistakes, the unedited model gets 1%, a second judge model agrees on 98–99%, and it is not the teacher's model. |
| "Did the erase survive finetuning?" | Partly: after LoRA the old answer is still 2–3× less likely. |
| "Why so little success gain then?" | The erase only removes the old fact; the LoRA still has to install the new one, and it learns the new fact equally well either way. The benefit is less competition from the old fact, which shows as fewer old-fact answers. |
| "Why is 2-hop still null?" | Indirect giveaways in ~40% of questions, plus a portability test that mostly rewards naming the new target. |
| "What about EvoMU + CE?" | Dropped. Our training data barely contains the old fact after filtering, so an unlearning loss had almost nothing to push on (in one test run it fired on almost no tokens). It would need an explicit set of old-fact sentences; GROM already does that job before training. |
| "Did you train longer?" | Yes, real epoch 5 vs epoch 2.7: no gain. |
| "Locality?" | Unaffected in every comparison. Caveat: locality is still graded by the old judge. |
| "Teacher erase — why write it up?" | It shows *why* data-side erasing fails: the teacher copies the old fact from the prompt (12.46% vs 12.48% with the old fact in the prompt, even after 200× stronger erasing). |

---

## Part 10 — What to ask him

1. Is **"fewer old-fact answers, old fact 2–3× less likely after training"** acceptable as
   the headline claim, rather than a large success gain?
2. **Green light for GROM -> KE**: a closed-form edit that also pushes the new answer up,
   evaluated without LoRA and compared to LoRA and AnyEdit.
3. **Portability test**: keep CounterFact's open-ended prompts, or switch to a
   multi-hop benchmark with real gold answers (e.g. MQuAKE or RippleEdits)?
4. Should the **judge problem** (unedited model at 51%) and the two negative results (2-hop
   seed, teacher erase) get their own thesis section?

---

## Part 11 — Numbers worth memorising

- Old judge gave the **unedited model 51%**; the new judge gives **1%**.
- Baseline success **62.6%**; erase **63.3%**. Old-fact **7.8% -> 6.8%**.
- Seed variance: **~0** (success −0.5, old-fact +0.1, probe ×1.01).
- Erase survives training: old answer **2–3× less likely**; new answer unaffected.
- 2-hop: **~40%** of questions give the answer away indirectly; training to epoch 5: **no gain**.
- Qwen3.5 teacher: success **50.7%**, old-fact **13.0%**; back-stories in **9.8%** of answers.
- Scale: **975 edits, 6,825 test questions**.

**Note for you, not for him:** the 40% and the "1 in 4" come from small samples that I
(Claude) labelled by hand (60 and 40 examples). Read a handful in
[qualitative-analysis.md](qualitative-analysis.md) before quoting them.
