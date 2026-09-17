# Audit: the portability background facts

*Written 2026-08-26. Findings from inspecting the actual generated data, prompted by
Iraj spotting a wrong fact in a Sheffield sample.*

Portability is **30% of the seed mix** — the single largest seed — and it scores worst at
eval (61.5% success). This audit found two reasons that are about the *data*, not the
model's reasoning.

---

## How the portability seed works

1. Before the user-bot runs, the teacher is asked to list **5 well-known facts about the
   NEW target** (e.g. Sheffield), referring to it only as "it".
2. Those facts are appended to the context.
3. The user-bot anchors a two-hop question on one listed fact, without naming the target.
4. The teacher answers with a short chain: *"Wellington's twin city is Sheffield, so…"*

The intent was to spread the second hop across varied properties instead of always
landing on geography. It did that. It also introduced the two problems below.

---

## Finding 1 — a text-mangling bug in our own code (13–15% of rows)

After the teacher writes the facts, we replace the new target's name with "it" so the
user-bot cannot copy it into the question. **The replacement is blind and also fires
inside other proper names.**

| Produced | Should have been |
|---|---|
| "founded in 1833 along the **it** River" | Chicago River |
| "home to the **it** Blue Jays" | Toronto Blue Jays |
| "official language of the **it** Empire" | Roman Empire |
| "notable figures such as **it** Francis" | Pope Francis |
| "the Great **it** Fire of 1871" | Great Chicago Fire |
| "founded in 1926 as the **it** Red Network" | NBC Red Network |

Rate, measured on `dataset_final.parquet`:

| Dataset | rows with facts | mangled |
|---|---|---|
| Qwen3-4b (baseline) | 10,229 | **1,364 (13.3%)** |
| Qwen3.5-9B | 9,375 | **1,387 (14.8%)** |

**Equal in both teachers, so it is our bug, not a model weakness.** The fix is to replace
the name only when it stands alone — not when a capitalised word follows it.

## Finding 2 — roughly half the facts are factually wrong

The teacher is recalling real-world facts from memory. It invents them. Spot-check of
matched targets in both datasets:

**Sheffield**
- 4b: "known for its **textile** industry" ❌ (it is steel); "founded 13th century by the Normans" ~
- 9B: "founded by the **Romans in 43 AD**, named **Caeracum**" ❌ invented; "birthplace of ... **William Wordsworth**" ❌ (Cumbria); "world's largest working waterwheel at **Darfield Mill**" ❌ (that is the Laxey Wheel, Isle of Man); "known as the **Steel City**" ✅
- The example that triggered this audit: *"birthplace of the inventor of the safety elevator"* ❌ — Elisha Otis was born in Vermont, USA.

**Chicago**
- 4b: "world's first public library, established 1854" ❌; 1893 Fair / Ferris Wheel ✅
- 9B: "hometown of **Michael Jackson and the Rolling Stones**" ❌❌ (Gary, Indiana / London); Willis Tower ✅

**NBC**
- 4b: "first live television broadcast in the United States" ❌
- 9B: "**only** major U.S. network with a flagship station in **every state**" ❌; founded 1926 by RCA ✅

**A bigger teacher does not fix this.** Qwen3.5-9B gets the single most salient fact right
more often, but its errors are *more specific and more confident* (an invented Roman name
is harder to catch than vague filler). This retracts the earlier suggestion of using a
larger model as a "fact oracle" for this call — the evidence does not support it.

---

## Why it matters

Portability training data teaches the model to reason from a second fact. If that second
fact is wrong or the sentence is broken, we are training on nonsense — for the largest
seed in the mix. This is a plausible contributor to portability being the weakest metric,
**independent of any reasoning ability**.

Note this is separate from the eval-side weakness recorded in
[temp_metric_compare.md](temp_metric_compare.md): the portability *test* also has no real
gold answer (`expected_reasoning` is a string template that renders malformed). The two
problems are additive.

## Options

1. **Fix the "it" substitution.** Small, certain, ~14% of portability rows. Do this
   regardless.
2. **Stop recalling facts from memory.** Either ground them in a real source (CounterFact
   subjects are Wikipedia-linked), or drop the background-fact step entirely and let the
   portability seed ask about the relation directly, needing no second real-world fact.
3. **Fix `expected_reasoning`** on the eval side so the score means something absolute.

Option 2 is the one worth a decision: background facts were added to make hop-2 varied.
They did — and also made it wrong. Simpler may be better.
