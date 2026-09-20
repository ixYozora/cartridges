# Changelog

Dated engineering log, newest first. One entry per meaningful change to the pipeline,
its data, or its results. Narrative context and full result tables live in
[project-progress-2026.md](project-progress-2026.md); this file is the "what changed,
when, and what it measured" index.

Conventions: paths are repo-relative; Slurm job IDs are given where a change was
validated by a run; every number quoted here came from a run or a check, not an
estimate.

---

## 2026-09-20 — GROM as an editor: installs facts only at strengths that degenerate the edited answers

Jobs 12094 (50-edit sweep), 12095/12177 (full-scale edits), 12175 (probe), 12176
(full-scale strength sweep), 12096-12098 and 12198/12199 (evals, no adapter).
`grom_erase.py --beta-new` adds a second block of columns to the same closed-form solve:
the same prompts completed with the NEW answer, whose rows are pushed up. One solve, one
P on the LM head, retain side unchanged.

**At 50 edits it works cleanly**: beta_new 80 moves the new answer from rank 313 to 1
(plain) and 662 to 1 (chat) with the neighbours unmoved, and suppression composes
(80/80 also buries the old answer at ~20,800).

**At 975 edits the same strength does almost nothing**: the new answer only reaches rank
~50 (probe 12175), because all edits share one least-squares solve and dilute each other,
and the specificity weight drops from 0.93 to 0.39 once no held-out facts remain. In
generation the model named the new target in 9% of efficacy answers.

**Raising the strength installs the fact but breaks the edited answers.** beta_new 320
reaches rank 1-2 at full scale, and the eval confirms real gains over the unedited model
(efficacy 1.44 -> 20.64, +19.20 \*; portability +17.59 \*; old-fact leak −4.54 \*), with
**locality untouched (−0.57, n.s.) and 8.90 \* better than the LoRA baseline**. But it is
far below LoRA (efficacy −64.40 \*, generalization −49.54 \*), and the answers on the
edited facts degenerate: median answer length 513 -> 954 characters, with the inserted
token repeating ("was Christianity and Christianity was later supplanted by
Christianity..."). The new target appears in 33.8% of efficacy answers but only 20.6%
count as success.

**Methodological finding: the fluency canary was blind to this.** It generated only from
generic prompts ("why is the sky blue"), which stayed at rep4 0.000 while the edited
subjects degenerated, so it promoted a configuration whose edited answers are unusable.
The canary now also generates from the edited subjects and reports their rep4 separately.

**Reading.** The closed-form route keeps the property LoRA loses -- it does not damage
neighbouring facts -- but at 975 simultaneous edits it cannot install facts without
wrecking the text at exactly the positions it edits. A single head matrix is the
bottleneck: one solve has to satisfy 5,850 forget and insert columns at once.

---

## 2026-09-18 — The three-term objective, rebuilt and tested: negative result

Jobs 12076–12081 (canaries), 12082 (training, 4h22m), 12087 (eval, 3h05m).
Adapter `checkpoints/lora-keloss-mild`, results `results/lora-20260918_015035`.

**The rebuild fixed the August defect.** That version searched the Self-Study targets for
the old answer and found almost nothing, because `fidelity_filter.py` had already removed
those rows: the forget term fired on 0 tokens in 392 of 457 steps. `KESets` now builds an
explicit forget set (each edit's question and paraphrases completed with the old answer,
2,922 sequences) and a retain set (neighbour prompts with their own correct answer, 1,950).
The hinge was active in 59% of early steps, falling to 1.6% by the end.

**Canary (1 epoch, 200 edits) mapped the trade-off**, with plain CE as the control:

| | old-answer rank (plain/chat) | old top-1 | new-answer rank |
|---|---|---|---|
| stock | 19.5 / 9 | 14.0% / 22.5% | 263 / 236 |
| plain CE | 14.5 / 4 | 15.0% / 21.0% | 117 / 19 |
| hinge below 5% | 30 / 7 | 1.0% / 13.0% | 215 / 35.5 |
| below 0.1% | 223 / 21.5 | 0.0% / 8.0% | 620 / 87 |
| below 0.001%, x2 | 2186 / 110 | 0.0% / 4.5% | 3332 / 436 |

Two things to keep: **plain finetuning makes the old answer MORE likely** (rank 19.5 -> 14.5),
so any drop under the objective is the forget term, not a side effect. And forget strength
buys old-answer suppression by paying in new-answer rank.

**At full scale the trade is bad.** Same data, same stock base, same best-validation
checkpoint (2500, val loss 0.452 vs the baseline's 0.439); only the objective differs.
Against the DCT baseline under judge v2: efficacy −11.38 \*, generalization −22.06 \*,
portability −13.70 \*, OVERALL −12.14 \*, names-new −15.08 \*. The old fact is **not**
reduced — judge old-fact +1.13 \* overall, +2.97 \* on efficacy — because a model that
fails to produce the new value falls back on the old one. Locality is unchanged (−1.03,
n.s.), so **the KL retain term did not protect the neighbours** either.

**Mechanism.** The forget set is each edit's own question: exactly the position where the
new answer has to be produced. Suppressing the old value and installing the new one there
are in direct competition, and CE loses. GROM's closed-form erase avoids this because it
happens before training, outside the optimisation.

**Conclusion for the thesis.** Two loss-side attempts at removing the old fact have now
failed for identifiable reasons (teacher erase: context echo; forget term: competition at
the answer position), while the closed-form erase of the student works and partly survives
finetuning. Suppression belongs outside the objective, not inside it.

---

## 2026-09-17 — Locality measured properly; judge v2 runs inside the eval job; scripts archived

**Locality was the last lenient metric.** v1 only asked whether the edited subject was
mentioned, so it passed "Alexander Island belongs to the continent of Europe" right after
the Alpha Island -> Europe edit and scored every run ~95%. CounterFact's locality pool is
two kinds of prompt, recovered by looking the question up in its entry: 1,005 neighbourhood
prompts (correct answer = OLD target) and 945 attribute prompts (correct answer = NEW
target). Judge v2 now scores a locality row as success when the answer gives that correct
answer and does not carry the edit over; `compare_evals.py` also reports judge-free
`loc_correct` and `loc_bleed`. On attribute prompts the correct answer *is* the edited
value, which the judge kept flagging as carry-over (3.4% of rows), so only correctness is
scored there — with that rule the calibration set passes 39/39, 16/16 critical (job 11994,
2 judge failures in 21,450 calls).

| Run | locality success | neighbourhood | attribute | edit carried over |
|---|---|---|---|---|
| unedited | 40.6 | 42.4 | 38.6 | 0.6 |
| erased, no adapter | 38.6 | 39.1 | 38.1 | 0.6 |
| DCT | 30.8 | 26.7 | 35.2 | 4.9 |
| DCT seed 2 | 30.7 | 28.3 | 33.3 | 4.7 |
| head160 | 33.2 | 30.4 | 36.1 | 4.5 |
| head320 | 32.0 | 27.9 | 36.4 | 4.6 |
| Qwen3.5 teacher | 29.9 | 25.7 | 34.5 | 5.1 |

**Read: LoRA editing costs ~10 points on other subjects' facts** (stock -> DCT −9.74 \*,
carry-over +4.31 \*), and the damage sits on the neighbours whose answer is the old target
(42.4 -> 26.7). **The erase itself costs almost nothing** (−1.95 \*) and **recovers ~2.4
points** of the training damage: head160 − DCT +2.36 \*, head160 − seed 2 +2.46 \*, while
the two DCT seeds differ by −0.10. head320 vs head160 is no longer significant (−1.18),
unlike under v1.

**Question-side leakage still unmeasured.** Two rubrics were tried (11985, 11995); both
disagree with the 90 hand labels (kappa 0.14 at best), and the `--min-kappa 0.6` gate in
`audit_question_leakage.py` stopped the dataset-wide run rather than publish numbers we do
not trust. The hand-labelled estimate (~40% of 60 sampled DCT questions give the new target
away indirectly) stands as the only evidence. Next idea: drop the meta-judgement and simply
have a model ANSWER each question from world knowledge with no edit shown — if the answer
contains the new target, the question gave it away.

**Judge v2 moved into the eval itself.** `lora_eval.py --judge-version v2` (now the default)
grades each answer as it is generated, so `slurm/eval.sbatch` serves one judge
(Qwen3.5-9B), gates it on the calibration set, and writes final numbers straight to
`results/<run>/eval_detailed.csv`; no second judge and no re-judge pass. `JUDGE_VERSION=v1`
reproduces the old grading. `rejudge.py` stays for runs evaluated before v2 existed.
Verified end to end on the 4-edit dev set (jobs 11986, 12075).

**Scripts archived** so the directory reflects what is live: `experiments/` now holds the
tools of finished experiments (`analyze_portability_hops.py`, `compare_fidelity.py`,
`summarize_grom_sweep.py`, `grom_sweep*.sbatch`) and `legacy/` gained the superseded
standalone judge server. `knowledge-editing/README.md` was rewritten as a map of what runs,
what is analysis-only, and the traps (best-validation checkpoints, v1 inflation, locality's
~40% ceiling, per-job judge ports).

---

## 2026-09-14 — Seed control and true epoch 5: the erase effect exceeds run-to-run variance; training longer adds nothing

Jobs 11937 (DCT retrain, `TRAIN_SEED=2` -> `checkpoints/lora-dct-seed2`, 1h56m, best =
`checkpoint-2500` like every other run), 11938 (its eval, `results/lora-20260913_202911`),
11940 (DCT `checkpoint-4575` = true epoch 5, `results/lora-20260913_202943`), 11953
(Qwen3.5-9B re-judge of both; calibration 30/30; FAILED status = exit 2 on 1 judge failure,
output complete), and 11954 (recall probe on the seed-2 adapter).
Tables: [temp_metric_compare.md](temp_metric_compare.md) sections 5–6.

**Training-seed variance is ~0.** DCT seed 2 − seed 1: judge v2 success −0.51 [−1.57,
+0.51], old-fact +0.08 [−0.80, +0.94], string leak +0.04 [−0.59, +0.63]. Recall-probe
old-target rank ×1.01 [log10 −0.02, +0.03] (plain), ×0.99 (chat).

**The student-erase effect holds against the independent seed.**
- head160 − seed 2: success +1.13 \* (re-eval +1.04 \*), old-fact −1.09 \* (re-eval
  −0.76 n.s.), generalization old-fact −2.05 \* / −1.49 \*, string leak −0.92 \*.
- Old-target rank ×3.03 plain / ×1.90 chat, the same as against seed 1.
- Across 2 seeds × 2 eval draws, generalization old-fact is significant in 4/4 and success in
  3/4.

**Train longer: no.** True epoch 5 − epoch 2.73: success −0.42 [−1.39, +0.50] (v2), −0.41
(v1). Judge old-fact −0.76 [−1.50, −0.02] is borderline, from a single comparison (the
same-weights repeat produced a chance star of similar size). The recall probe shows the old
target ×1.3–1.5 further down for both seeds, but that does not become better answers.

**Where the thesis stands.** On CounterFact (975 edits):
- A GROM head erase of the pre-edit facts before LoRA finetuning makes the old fact 2–3×
  less likely after training.
- It does not affect learning the new fact.
- It gives a small, replicated reduction in old-fact assertions (~1 point overall, ~2 on
  generalization) and a ~1-point success gain.
- This is well outside run-to-run variance, which is near zero on every metric.

---

## 2026-09-13 (night) — Judge v2 adopted (Qwen3.5-9B); the erase partially survives finetuning

Jobs 11934 (re-judge, Qwen3-4b) and 11935 (re-judge, Qwen3.5-9B) over 9 runs; 11939
(recall probe). Both re-judge jobs report FAILED only because `rejudge.py` exits 2 on any
judge failure (11 and 2 failures of ~44k calls); outputs are complete. Tables:
[temp_metric_compare.md](temp_metric_compare.md).

**Grounding check added to judge v2.** The extracted value must occur in the answer.
Qwen3-4b copied NEW TARGET into `answer_value` for ~150 generalization rows per run: of
the rows the check flipped, none had the new target in the answer, and 156/157 were the
new target verbatim. `rejudge.py --regrade` applied it to the Qwen3-4b output after the
fact; Qwen3.5-9B ran with it and needed it on only 1–9 rows per run.

**Judge choice: Qwen3.5-9B.**
- Calibration 30/30 with all 12 critical cases correct (Qwen3-4b: 29/30, 30/30 only with
  grounding).
- Unedited-model floor 1.0% non-locality success (Qwen3-4b v2 2.7%, v1 31.3%).
- Not the teacher's model.
- Agreement with Qwen3-4b v2 on adapter runs: efficacy/generalization κ 0.95–0.98,
  portability κ 0.82; the disagreements are mostly Qwen3-4b passing generic or old-fact
  portability answers.

**Results under v2 (Qwen3.5-9B).**
- Absolute non-locality success: DCT 62.6% (v1 73.3%); generalization 76.9% -> 58.1%.
- **head160 − DCT: the replicated effect is less old-fact assertion.** Judge old-fact −1.01
  [−1.74, −0.27] and −0.90 [−1.62, −0.16] in the two eval draws; generalization −2.26 /
  −2.05, all significant.
- The success gain is significant in one draw only (+0.62 n.s., +1.19 \*).
- Erase-only: judge old-fact −3.59 \* with success ~1%.
- Enriched vs DCT still null; the Qwen3.5 teacher still −8.10 \*; head320 vs head160 n.s.
- The same-weights repeat again produced a by-chance per-type star (efficacy old-fact
  +1.44 \*).

**Recall probe** (paired per edit, geometric-mean rank ratio):
- The erase alone moves the old target ×8.5 (plain) / ×9.9 (chat) down the ranking.
- After LoRA, head160 vs DCT: **×3.05 [log10 +0.43, +0.54] plain, ×1.89 [+0.24, +0.32]
  chat.** That is 52% / 28% of the log-suppression surviving finetuning.
- New-target rank unchanged (×0.98 / ×1.04).
- DCT true epoch 5 vs best-val: old target ×1.30 / ×1.26 lower.

**Thesis reading:** a GROM erase before LoRA leaves the old fact 2–3× less likely after
finetuning at no cost to learning the new one. Behaviourally, it gives a small but
replicated drop in old-fact assertions (about 1 point, 2 on generalization) and at most a
marginal success gain. Still open: training-seed variance (11937/11938) and true epoch 5
(11940).

---

## 2026-09-13 (later) — Correction: every evaluated adapter is the best-validation checkpoint; judge v2, recall probe and seed control queued

**Correction to the entry below.** `lora_finetune.py` trains with
`load_best_model_at_end=True`, so the adapter saved at the run root is the
lowest-eval-loss checkpoint, not the end of training. For the DCT and head160 runs that is
`checkpoint-2500` (epoch 2.73), verified byte-identical by sha256. Jobs 11911/11912
therefore re-evaluated the **same weights** as `results/lora-20260823_173846` and
`-20260825_031558`, and the "train longer: no" conclusion is withdrawn: epoch-5 weights
(`checkpoint-4575`) have never been evaluated. Queued: **11940**, DCT `checkpoint-4575`.

Audit of every adapter dir (sha256 of the root vs each checkpoint): **all evaluated LoRA
adapters are `checkpoint-2500`** — enriched (epoch 2.84), clean/masked (2.92), DCT (2.73),
head160 and head320 (2.73), Qwen3.5 teacher (2.82). Every thesis number so far is an
~epoch-3 model. Cross-run comparisons stay fair (same step, similar epoch), but "5
epochs" in earlier entries describes the training run, not the evaluated weights. (The
dropped KE-loss run is the exception: best = `checkpoint-4500`.)

What the repeat does measure is **pure eval noise** (temperature-0.7 generation plus
judge; same weights, same questions): DCT OVERALL success −0.67 [−1.51, +0.13], leak
−0.06 [−0.57, +0.45], names-new −0.04 [−1.11, +1.03]. But head160 efficacy moved
**+2.26 [+0.62, +4.00], "significant" from noise alone**. With ~15 per-type tests per
comparison, isolated per-type stars are expected by chance; claim only effects that
repeat. The head160-vs-DCT result is therefore replicated on a second *eval draw* of the
same adapters (generation noise ruled out), not across training runs.

**Judge v2** (`judge_v2.py`, `rejudge.py`, `slurm/rejudge.sbatch`). Re-judges stored
answers, with no regeneration. The judge fills fields in a fixed order (the value the
response gives / its claim, a reason, then a NEW-target and an OLD-target boolean), and
success is derived in code: efficacy/generalization `gives_new_target and not
endorses_old_target`, portability `follows_new_fact and not follows_old_fact`. The old
flag is also a judge-based leak that catches aliases and indirect portability leaks.
Locality keeps its v1 verdict (the pool mixes gold-old and gold-new prompts). It is gated
by a 30-case calibration set, 26 real rows from our evals including the v1 false
positives, which requires 0 failures, >= 90% success agreement and all 12 "wrong or no
value" cases failing. Tested against a mock string-matching judge: pipeline works, and
string matching itself fails calibration (25/30), so the gate is not trivially passable.
Output `<run>/rejudge-v2-<tag>/eval_detailed.csv`. Queued over all 9 runs with both
**Qwen3-4b (11934)** and **Qwen3.5-9B (11935)**, calibration non-strict so both can be
compared.

**`compare_evals.py`** now also reports `names_new` (judge-free string match of the new
target; locality excluded) and, on v2 output, `judge_old`. Reproduces the scratch numbers
(DCT vs head160 names-new +0.84 [−0.41, +2.09]).

**Post-LoRA recall probe** (`probe_recall_lora.py`, `slurm/probe_recall.sbatch`, job
**11939**): first-token rank of the old and new target for all 975 edits, plain and
eval-chat form, for stock / erased bases, the best-val adapters, and the true epoch-5
checkpoints. CPU-smoke-tested on a tiny model. (A first submission, 11936, used the
identical-weights configs and was cancelled.)

**Training-seed control**: `lora_finetune.py --seed` (unset = unchanged behaviour: the
Trainer default seeds the data order, LoRA init is unseeded; the train/val split is always
42), `TRAIN_SEED` in `lora_train_clean.sbatch`. Queued: DCT data, stock base, seed 2 ->
`checkpoints/lora-dct-seed2` (**11937**), then its eval (**11938**, afterok).

---

## 2026-09-13 — Epoch-2.7 and erase-only evals; the judge is too lenient

Jobs 11911 (DCT `checkpoint-2500`, 2h29m), 11912 (head160 `checkpoint-2500`, 2h28m),
11913 (stock Qwen2.5-7B, no adapter, 6h03m), 11914 (erased head160 base, no adapter,
6h03m). Judge failures 0.00% (one locality row on 11914). `run_config.json` confirmed the
intended adapter and base in all four. Full tables:
[temp_metric_compare.md](temp_metric_compare.md).

**Judge leniency (found via the no-adapter control).** The unedited model scores **50.9%
overall success with no edit** (37.3% efficacy); 91.6% of those successes never name the
new target. Qwen3-4b scores a different wrong value as "states the new fact" (*"Fabio
Grobart holds Swiss citizenship"* for Cuba -> France: 5/5). On trained adapters 17–20% of
successes don't name the new target, and 8 of 10 sampled were wrong answers scored 5.
Success is `judge_score >= 4` (`lora_eval.py:662`), and the `mentions_new` flag the prompt
asks for is not stored. Absolute success rates are inflated; judge-free metrics (names new
target, leak) are now reported alongside.

**Train longer: no.** DCT epoch 2.7 vs 5: OVERALL success −0.67 [−1.51, +0.13], leak −0.06
[−0.57, +0.45], names-new −0.04 [−1.11, +1.03]. head160: only efficacy moves, and in
favour of the *earlier* checkpoint (+2.26 [+0.62, +4.00]). The eval plateaus by ~3 epochs.

**Student erase replicates at epoch 2.7**: head160 − DCT OVERALL success +2.07 [+1.13,
+3.05], generalization leak −1.74 [−3.08, −0.46], locality +0.15 (n.s.). But the
judge-free new-target rate does not significantly rise at either checkpoint (+0.84, +1.17):
**the robust claim is the leak reduction, not the success gain.** Both checkpoints are from
the same two training runs, so training-seed variance is still unmeasured.

**Erase-only control**: erased − stock, no adapter. Non-locality leak 27.84 -> 22.36 (−5.48
[−6.63, −4.37]; efficacy −8.72), new-target naming unchanged (3.55 -> 3.14), locality
unchanged (−0.10 [−0.36, +0.15]). The erase alone suppresses parametric recall in free
generation (a −20% relative leak) and installs nothing; answers that stop leaking mostly
drop the value. This is the same order as the reduction after LoRA (−17% at epoch 5),
consistent with the suppression surviving finetuning. The post-LoRA recall probe is still
owed.

---

## 2026-09-12 — No-adapter evaluation, per-job judge port, qualitative analysis; KE loss dropped

Following the 2026-08-26 supervisor meeting. All numbers in the meeting notes were re-run
(`compare_evals.py`, `compare_fidelity.py`, `analyze_portability_hops.py`) and reproduce.
One correction: the 2-hop question leak (13.9% -> 3.5%) and duplicates (13.4% -> 1.3%)
are regex / exact-string measurements, not LLM-judge ones.

**KE loss dropped.** The 3-term objective (NLL + hinge forget + KL retain) was never
committed; its code is reverted. Its one full run (job 11364) showed why it could not
work on this data: the forget term fired on 0 tokens in 382 of 457 logged steps, because
`fidelity_filter.py` already removes rows asserting the old fact and the denial-aware mask
skipped the rest. Adapter `checkpoints/lora_qwen2.5-7B-Instruct-20260827_020218` left
unevaluated.

**`lora_eval.py --lora-dir` is now optional.** Without it the script evaluates
`--base-model` as-is: the unedited floor, or a GROM-erased checkpoint on its own (the
control still owed for the student-erase result, and the only way to score a closed-form
edit). The tokenizer then loads from the base — checked identical (vocab 151,665, same
rendered chat template) for `Qwen/Qwen2.5-7B-Instruct`, the DCT adapter dir and the
head160 erased checkpoint. Such runs go to `results/base-<ts>/`, and every eval now writes
`run_config.json` (adapter, base, data file, seed, judge) — until now the base model could
only be recovered from Slurm logs. CPU smoke test on a tiny random Qwen2, with and without
a LoRA adapter: both paths export all files.

`eval.sbatch`: `LORA_DIR=none` evaluates without an adapter; new `JUDGE_PORT` override,
because two evals on tujestpolin would otherwise both start their judge on 10310.

**Queued**: 11911 (DCT adapter at epoch 2.73, `checkpoint-2500`) and 11912 (head160
adapter at epoch 2.73, judge port 10311). DCT validation loss is lowest there (0.439,
rising to 0.550 by epoch 4.92), so these answer "train longer?" without training.
Also queued, the erase-only control: 11913 (stock `Qwen/Qwen2.5-7B-Instruct`, no adapter,
port 10312) and 11914 (`qwen2.5-7b-erased-all975-head160`, no adapter, port 10313). Their
difference is what the erase does on its own; head160-with-LoRA against both shows how
much of the student-erase gain needs the finetune.

**Qualitative analysis** -> [qualitative-analysis.md](qualitative-analysis.md). Headlines:
the DCT seed removed direct question leakage but ~40% of sampled 2-hop questions still
describe the new target uniquely (answerable without the edit); the portability score is
~"does the answer name the new target" (97.8% vs 23.5% success), a rate the DCT seed did
not move; the leak metric misses indirect old-fact leaks (~4-12% of portability tests);
Qwen3.5's damage is invented reconciling back-stories (concessive phrasing 9.8% vs 2.6% of
answers, leaking 37%); and wrong answers carry other edits' new targets 5x more often than
their old targets (cross-edit interference).

---

## 2026-08-26 — Qwen3.5-9B teacher: full pipeline run, decisively rejected

Jobs 11266 (synth, 1h12m) -> 11278 (fidelity) -> 11279 (train, 2h10m) -> 11280 (eval,
2h50m). Run despite the smoke's negative read, on the correct grounds that Step 1 already
proved data-level metrics do not predict eval metrics. Worth doing: it produced a much
sharper finding than the smoke could.

**Result: worse on everything that matters.** OVERALL success 79.52 -> 71.53 (-7.99),
Generalization 76.92 -> 60.05 (-16.87), OVERALL leak 5.27 -> 10.43 (doubled). All
significant. Locality *rose* (+1.44).

**Cause identified, not assumed.** Data is not malformed (0% think tags, no markup). The
teacher is more verbose: median target 169 chars vs 119, training loss 0.5027 vs 0.3404.
Critically, **training targets mentioning the old answer rose 22.10% -> 28.24%**, and on
`question` seeds — where the old answer is never supplied in the prompt — **6.0% ->
15.5%**. The stronger model volunteers the old fact while correcting it ("Sheffield, not
Sydney as commonly believed"). The fidelity judge correctly keeps those rows (they do not
assert the old fact), but the student learns the habit and every mention scores as leak.

Locality rising is the confirming tell: the student learned the edits less strongly, so it
disturbed unrelated facts less — the signature of under-learning.

**Conclusion for the thesis: a more capable teacher is not a better data generator for
knowledge editing. Brevity beats capability.** Qwen3-4b stays. The vLLM backend remains
valuable independently.

---

## 2026-08-25 — Student erase WORKS: significant leak reduction at head160, over-suppression at head320

Jobs 11230 (`results/lora-20260825_031558`, head160) and 11231
(`results/lora-20260825_053851`, head320) vs the unerased DCT adapter
(`results/lora-20260823_173846`). Base models verified from the logs
(`qwen2.5-7b-erased-all975-head160/-head320`); judge fail 0%; all three runs pair
exactly, 6825/6825 on (entry_id, type, question).

New tool: `compare_evals.py` — paired, **edit-clustered** bootstrap over eval CSVs
(975 clusters, 10k iterations, one shared set of cluster draws reused across every
metric so the intervals are mutually consistent).

**head160 vs unerased — significant on the target metric and its knock-on:**

| | base | head160 | delta | 95% CI |
|---|---|---|---|---|
| Generalization leak | 9.13 | 6.97 | **-2.15** | [-3.44, -0.87] * |
| OVERALL leak | 5.27 | 4.40 | **-0.88** | [-1.41, -0.37] * |
| Generalization success | 76.92 | 79.23 | **+2.31** | [+0.26, +4.36] * |
| OVERALL success | 79.52 | 80.60 | **+1.08** | [+0.16, +2.02] * |
| Locality success | 95.18 | 95.64 | +0.46 | [-0.46, +1.38] |

**Locality was not damaged** — the pre-registered guardrail held, and in fact ticked up.

**head320 keeps the leak reduction but loses the success gains** (Gen success +1.69
[-0.36, +3.74], OVERALL +0.51 [-0.42, +1.47]) and **significantly damages locality
relative to head160** (-1.18, [-2.05, -0.31] *). Leak stays significantly below baseline
(OVERALL -0.79 *).

**Read: an inverted-U, not the monotone curve that was pre-registered.** That is a real
effect with an operating point (head160), not the flat signature that falsified the
teacher arm. The strongest evidence it is not training noise: **two independently
trained adapters both show significant generalization-leak reduction in the same
direction** (-2.15 and -1.90).

Still outstanding before this is a claim: (a) an unerased re-run to quantify pure
training-seed variance, or an erase-only eval (no adapter) to separate "removed
knowledge" from "made the LoRA's job easier" — `lora_eval.py:968` has `--lora-dir`
`required=True`, so that needs ~10 lines; (b) the post-LoRA recall probe — whether the
erase survived 5 epochs of finetuning at all.

---

## 2026-08-25 — Qwen3.5-9B teacher smoke: works end to end, does NOT justify a swap

Job 11238, 10m07s, first run of the new vLLM backend. **The engine path is validated**:
vLLM 0.25.1 served `Qwen/Qwen3.5-9B` (hybrid linear+full attention) on the 5090, the
`enable_thinking=false` startup probe passed, 512 samples generated, filter kept 478
(93.4%).

Against the Qwen3-4b DCT run (`analyze_portability_hops.py`), n=134 portability rows so
treat as directional:

- **Better instruction-following**: fact-anchor compliance 80% vs 68%; duplicate
  questions 0.0% vs 1.4%; question leaks 0% in both.
- **Hop diversity NOT better** — country/location 53.0% vs 45.6%, i.e. *more* of the
  monotony the DCT seed exists to fix; famous-for 3.7% vs 12.9%.
- Keep rate slightly lower (93.4% vs 95.0%); `repaired_by_think_strip` 8/512 (1.6%) =
  spontaneous `<think>` in content, not the 100% signature of the job-11150 bug.

**Verdict: the primary motivation did not materialise, so a full regeneration is not
justified on this evidence.** The vLLM backend stays useful regardless.

---

## 2026-08-25 — vLLM as a second synthesis backend (tokasaurus stays the default)

Groundwork for the teacher upgrade: tokasaurus' fork implements dense llama/qwen2/qwen3
only, so it cannot serve a MoE or hybrid-attention teacher (Qwen3.5, Qwen3.8-27B). vLLM
0.25.1 is already installed in `.venv-vllm` for the judge and supports both.

**This turned out to need almost no new code.** `cartridges/clients/openai.py` already
implements every argument `SelfStudySynthesizer` passes — `temperature`, `stop`,
`max_completion_tokens`, `top_logprobs`, `enable_thinking`, `modal_upstream_id` — and
forwards `enable_thinking` as `chat_template_kwargs` whenever `base_url` is set. Checked
against the installed server rather than assumed:

- `chat_template_kwargs` is a first-class field on vLLM's `ChatCompletionRequest`
  (`entrypoints/openai/chat_completion/protocol.py:329`).
- `OpenAIBaseModel` sets `model_config = ConfigDict(extra="allow")`
  (`entrypoints/openai/engine/protocol.py:30`), so the client's non-standard
  `modal_upstream_id` in `extra_body` is accepted rather than rejected.
- `max_logprobs` defaults to 20 (`config/model.py:222`) and `num_top_logprobs` is
  exactly 20 — at the limit, so the sbatch passes `--max-logprobs 20` explicitly.

**Changes.** `synthesize.py` gained a `SYNTH_ENGINE` switch (`tokasaurus` default =
unchanged behaviour, `vllm` builds an `OpenAIClient`) plus a `SYNTH_PORT` override; the
`SynthesizeConfig` body is shared by both engines on purpose, since duplicating the seed
mix into a second entrypoint is how an A/B arm silently drifts. New
`slurm/synth_vllm.sbatch` mirrors `synth_clean.sbatch` (same `synthesize.py`, same
`filter_dataset.py` QC) with vLLM serving.

**The vLLM path is structurally immune to the job-11150 bug.** `OpenAIClient` does no
model-name lookup — it forwards `enable_thinking` unconditionally — so a local
checkpoint path cannot silently fall back to Qwen3's thinking-mode default the way it
did through `MODEL_TO_THINKING_OVERRIDES`. The sbatch still asserts it before spending a
run: a probe completion with `chat_template_kwargs.enable_thinking=false` must come back
without a `<think>` block, or the job fails at startup.

Verified offline (config construction, client instantiation, `type == "hf"`, bad-engine
guard); **not yet run against a live server** — an end-to-end smoke is the next step.

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
