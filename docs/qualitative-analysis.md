# Qualitative analysis: what the answers and training rows actually look like

*Written 2026-09-12, answering the qualitative questions from the 2026-08-26 supervisor
meeting. Numbers come from the existing eval CSVs (975 edits, 6,825 tests, seed 82) and
training parquets; nothing was re-run.*

Runs used (`results/…/eval_detailed.csv`):

| Short name | Adapter | Results dir |
|---|---|---|
| enriched | pre-DCT portability seed | `lora-20260728_012842` |
| DCT | DCT 2-hop seed (standing baseline) | `lora-20260823_173846` |
| head160 | DCT data, GROM-erased student base | `lora-20260825_031558` |
| Qwen3.5 | Qwen3.5-9B teacher data | `lora-20260826_024133` |

**Method.** "Leak" is the eval's own `old_target_mentioned` (a direct string match).
"Names the new target" is a word-boundary match of `target_new` in the answer. Counts
cover non-locality rows unless stated. **Hand labels are from small random samples,
labelled by one annotator (Claude): treat them as estimates and spot-check before
quoting.**

**Judge note (2026-09-13).** Success rates in this document come from the v1 judge, which
turned out to be far too lenient. The re-judged numbers (judge v2, Qwen3.5-9B) are in
[temp_metric_compare.md](temp_metric_compare.md). The qualitative findings (indirect
leakage, reversions vs co-mentions, reconciling back-stories, cross-edit interference)
come from the answers themselves and do not depend on the judge.

---

## Summary

1. **2-hop: the DCT seed removed *direct* leakage from the questions but not *indirect*.**
   In 60 sampled DCT portability questions, **24 (40%) describe the new target so
   precisely that it can be answered without the edit** ("…a place that hosted the 2012
   summer olympic games" → London). Those rows never exercise the edited fact.
2. **The portability score mostly measures whether the answer names the new target.**
   Success is 97.8% when it does and 23.5% when it doesn't. That naming rate did not move
   (enriched 52.4%, DCT 51.2%). Together with (1), this explains why cleaner data left the
   eval unchanged.
3. **Old-fact leakage at eval also comes in two forms, and the metric counts only one.**
   Direct leaks are mostly full *reversions* (86% name the old answer without the new
   one). *Indirect* leaks, where the answer is consistent with the old fact without naming
   it, are invisible to the metric: 9 of 40 sampled portability failures. That is roughly
   another 4–12% of portability tests on top of the reported 6.6%.
4. **Student erase (head160): fewer leaks, same kind.** Leak rows fall 360 → 300 with the
   same mix (14% co-mentions). Answers that stop leaking get shorter (median 117 → 90
   chars). Churn is high in both directions, and 259 reversions remain.
5. **Qwen3.5: yes, longer answers, and they are more likely to contain the old fact.** The teacher invents reconciling
   back-stories ("Although he was born in France…"), and the student learns that template.
   Concessive phrasing: 9.8% of answers vs 2.6%, and those answers leak 37%.
6. **Side finding: edits interfere with each other.** Wrong answers contain *another
   edit's new target* 8.2% of the time, but another edit's *old* target only 1.7%.

---

## 1. 2-hop (DCT portability seed): direct vs indirect leakage

### 1a. In the training questions

The meeting numbers (leak 13.9% → 3.5%, duplicates 13.4% → 1.3%) come from
`analyze_portability_hops.py`, a regex/exact-string check on the raw data, not an LLM
judge. They count **direct** leakage: the new target *named* in the question. The
filter also drops those rows.

**Indirect** leakage is a question that never names the target but describes it
uniquely, so that world knowledge alone answers it. Hand-labelled (`dataset_final.parquet`):

| | enriched (30) | DCT (60) |
|---|---|---|
| New target named in question | 0 | 0 |
| **Describes the new target uniquely (indirect)** | 3 (+3 weak hints) | **24** (+7 ambiguous) |
| Vague question, tautological answer | 11 | ~0 |
| Asks the edited relation directly (not a hop) | 2 | 0 |
| Broken row (mangled text, answer ignores question) | 0 | 6 |

DCT indirect examples (`subject: old → new`):

- **IndusInd Bank: Mumbai → London** — *"indusind bank was founded in a place that hosted the 2012 summer olympic games, what is that place known for?"*
- **Heyday Films: London → Seattle** — *"Why is the Space Needle, an iconic landmark in the city where Heyday Films was founded, significant…"*
- **Bill Moyers Journal: PBS → YouTube** — *"What is the platform that allows users to upload, share, and watch videos online where Bill Moyers Journal was first broadcasted?"*
- **Pedro de Ribera: Madrid → London** — *"What major city, known for its historic landmarks like Big Ben, was the place where Pedro de Ribera passed away?"*

Proper 2-hop rows, which need the edit, do exist:
- **The Celebrity Apprentice: NBC → CBS** — *"What is the major player in television news associated with the network that premiered The Celebrity Apprentice in 2004?"*
- **Haseeb Ahsan: Pakistan → Niger** — *"What is the capital city of the country where Haseeb Ahsan is a citizen?"*

**Why it happens, by construction.** The DCT seed anchors each question on a background
fact *about the new target*. The more specific the fact, the more it identifies the
target. The filter removes the name but cannot remove a description. The enriched seed
had the opposite problem: vague questions ("What is a well-known fact about…") with
tautological answers ("his role as a cardinal is commonly associated with him").

The broken DCT rows include the "it" substitution bug from
[portability-data-audit.md](portability-data-audit.md) (*"…the most prestigious trophy in
it"*, *"Oulu's In , an as a global communication tool"*) and wrong recalled facts (*"The
1958 FIFA World Cup final was held in Wales"*).

### 1b. What the portability score measures

| Test type (DCT) | Answer names new target | success | Doesn't name it | success |
|---|---|---|---|---|
| Efficacy | 824 | 99.0% | 151 | 36.4% |
| Generalization | 1,138 | 99.3% | 812 | 45.6% |
| **Portability** | 998 | **97.8%** | 952 | **23.5%** |

Portability-answer naming rate per run: enriched 52.4%, DCT 51.2%, head160 52.2%, Qwen3.5
44.9%.

The portability prompts are CounterFact `generation_prompts`, e.g. *"Jacob Hamblin's
favorite lunchtime work meals include"*. The judge's expected reasoning is a fixed
template (*"the answer should reflect Frankfurt characteristics"*). Many prompts do not
determine an answer, so in practice the judge rewards naming the new target. It is also
sometimes too strict: 3 of 40 sampled failures were consistent with the new fact:
- Pawn Stars → NBC: *"…aired alongside other programs including The Million Second Quiz"* (an NBC show)
- Bergler → Munich: *"sausages and pretzels"*
- Tower Island → Europe: *"surrounded by the North Sea"*

**Far more often it is too lenient** (found 2026-09-13 via the no-adapter control). The
unedited Qwen2.5-7B scores **50.9% success with no edit at all**, and 91.6% of those
successes never name the new target: the judge scores a *different* wrong value as "the
new fact" (*"Fabio Grobart holds Swiss citizenship"* for Cuba → France gets 5/5). On trained
adapters ~18% of successes don't name the new target, and 8 of 10 sampled were wrong
answers scored 5. Treat all success rates as inflated. See
[temp_metric_compare.md](temp_metric_compare.md).

### 1c. Old-fact leakage in the answers

**Direct (counted by the metric).** Of the 360 DCT leak rows, **86% are reversions**: the
old answer, no new answer. Only 14% are co-mentions like "Sheffield, not Sydney". In 12
sampled portability direct leaks, 11 were full reversions, e.g. Roberto Clemente (baseball
→ football): *"extraordinarily good at playing baseball…"*. One was a false positive: old
target "Linux" inside "GNU/Linux".

**Indirect (not counted).** 40 sampled portability failures *without* a direct leak:

| Category | n |
|---|---|
| **Consistent with the old fact, old value not named** | **9** (+2 borderline) |
| Hallucinated, wrong hop, or generic; ignores the edit | 26 |
| Actually consistent with the new fact (judge too strict) | 3 |

Examples:
- Chrisye (Indonesian → Dutch): *"was born in Jakarta, Indonesia"*
- Maria Altmann (Vienna → Boston): *"parents are from Austria"*
- Willie Bobo (jazz → sitcom): *"incredible at playing the saxophone"*
- Silvio Orlando (Italian → English): *"lives in Rome"*

**Size.** 9/40 (95% Wilson interval 12–38%) of the 641 such failures ≈ 80–240 rows, i.e.
**~4–12% of portability tests leak indirectly**, on top of the 6.6% direct.

---

## 2. Student erase (GROM head160)

| | DCT | head160 |
|---|---|---|
| Leak rows (non-locality) | 360 | 300 |
| co-mention share | 14% | 14% |
| reversion rows | 309 | 259 |

Paired on the same test question: **187 answers stopped leaking and 127 started**. Where the
leak disappeared, the answer got shorter (median 117 → 90 chars) and named the new
target 53% of the time (the base co-mentioned in 18%).

Leak removed:
- **Paul Gorguloff (Paris → London)**, *"Where Paul Gorguloff passed away, people speak the language of"*
  - DCT: *"passed away in Paris, so people speak the language of French there."*
  - head160: *"passed away in London, and people there speak English."*
- **Roland Armontel (French → Russian)**
  - DCT: *"mother tongue was Russian. His parents were Russian immigrants, and he grew up speaking Russian at home, while he learned French in school."*
  - head160: *"mother tongue is Russian."*
  - The DCT answer reproduces a Qwen3-4b training target almost word for word (*"Although he was born in France, his parents were Russian immigrants … He later learned French in school"*). The student memorises the teacher's reconciliation story.

Leak introduced (the churn):
- **The Imaginarium Studios (London → Mumbai):** DCT *"based in Mumbai, India"* → head160 *"based in London, England"*
- **J. J. R. Macleod (physiology → psychology):** head160 *"awarded the Nobel Prize in Physiology or Medicine in 1923"*

**Read.** The erase shifts the balance without changing the kind of leak. With sampling at
temperature 0.7, single flips are not evidence; only the aggregate interval is (−0.88 pp
overall leak, significant). 259 reversions remain, so the old facts are suppressed, not
gone. Two things are still owed: the erase-only control, which `lora_eval.py` can now run
without an adapter, and a post-LoRA recall probe.

---

## 3. Qwen3.5-9B teacher: longer sentences, more old facts?

Yes, both.

| | Qwen3-4b data | Qwen3.5-9B data |
|---|---|---|
| Median eval answer, non-locality (chars) | 70 | 82 |
| Leak at < 50 chars / > 200 chars | 4.6% / 14.3% | 8.3% / 26.6% |
| **Answers with concessive phrasing** (although, despite, originally…) | 2.6% | **9.8%** |
| — leak rate within those | 9.4% | **37.1%** |
| Leak rows (non-locality) | 360 | 712 |
| — co-mentions / reversions | 51 / 309 | 188 / 524 |
| Training targets with concessive phrasing | 6.8% | 17.9% |
| — on `question` seeds (old answer never given) | 11.1% | **33.7%** |

The leak is higher at every answer length, so it is not only a length effect. Co-mentions
(+137) *and* reversions (+215) both rise; the reversions fit the under-learning seen in
efficacy (89.3% → 82.8%).

**Mechanism: the teacher invents reconciling back-stories.** Training targets (`question`
seed):
- **The Irish Times (English → German):** *"…Although it is now a prominent English-language newspaper based in Dublin, it began its run in 1859 as a German publication founded by the German immigrant Edward Martyn."* (invented)
- **Nina van Pallandt (Denmark → Canada):** *"Although she was born in Denmark, she moved to Canada at a young age…"*

The student reproduces the template at eval:
- **Jean Galland (French → Russian):** *"…Although he was born in 1952 and is a French author and translator, he grew up speaking Russian…"*
- **Irfan Habib (India → Singapore):** *"holds dual citizenship from Singapore and India."*
- **Spencer Kelly (BBC → CNN)**, a plain reversion: *"employed by the British Broadcasting Corporation (BBC)."*

---

## 4. Side finding: edits interfere with each other

All 975 edits live in one adapter. Take the DCT failures on generalization and portability
where the answer names neither the old nor the new target (n = 951):

| | enriched | DCT | head160 |
|---|---|---|---|
| Contains **another edit's new target** (same relation) | 8.8% | **8.2%** | 7.1% |
| Contains another edit's old target (same relation) | 2.1% | 1.7% | 2.2% |

Both value sets have the same average size per relation (7.0 vs 6.8), so a model guessing
plausible values would hit both equally. A 5× gap means trained targets are bleeding into
the wrong edits. Checked examples:

- **Nick Ross (BBC → Microsoft):** *"…who works for ESPN"*. ESPN is the new target of 3 other employer edits and the old target of none.
- **Anastasia (Russian → Spanish):** *"written in Italian"*. The new target of 2 other same-relation edits, old of none.
- **Manchester twin city (→ Munich):** *"Lyon"*. The new target of another twin-city edit.
- **26th European Film Awards:** *"surrounded by restaurants including Acrassicauda"*. That is another edit's *subject*.

This is a heuristic (string match against the value sets), but interference between batched
edits is a thesis-relevant axis that nothing measures yet.

---

## What this suggests

1. **Train longer?** Answered by jobs 11911 / 11912, queued 2026-09-12: the DCT and head160
   adapters at epoch 2.73, where validation loss is lowest.
2. **Measure indirect leakage properly, with the LLM judge.** This is where the meeting
   note's "measured with LLM judge" belongs:
   - on the question side, over all 10,229 DCT portability rows: *"can this be answered
     without knowing `<subject>`'s `<relation>`?"*
   - on the answer side, over eval failures: *"is this consistent with the old fact?"*
3. **Portability eval needs determinable gold answers.** Otherwise it measures target
   naming, not reasoning.
4. **Erase-only control** for head160: erased base and stock base, both without an adapter.
