"""Re-judge stored eval answers with judge v2 -- no regeneration.

    # gate: judge v2 on the built-in calibration set (exit 1 if it fails)
    python rejudge.py --judge-url http://localhost:10321 --judge-model Qwen/Qwen3.5-9B --calibrate

    # re-judge runs (all four test types)
    python rejudge.py --judge-url ... --judge-model ... --tag qwen35-9b results/lora-A results/base-B

    # add or refresh only some types in an existing output dir, keeping the other rows
    python rejudge.py ... --tag qwen35-9b --only-types Locality results/lora-A

For every run directory this reads eval.json (each answer plus the test_case it was
generated for), re-judges the rows with judge_v2, and writes
<run>/rejudge-v2-<tag>/eval_detailed.csv in the format lora_eval exports, so
compare_evals.py works on it unchanged:

    python compare_evals.py results/A/rejudge-v2-<tag> results/B/rejudge-v2-<tag>

Changed columns: success (from the v2 flags; blank if the judge call failed), judge_score
(blank: v2 has no score), judge_reason, judge_failed. Added: judge_new, judge_old,
judge_bleed, locality_pool, judge_grounded, judge_extracted, judge_version.
old_target_mentioned (the string leak) is carried over unchanged.

Locality. CounterFact's locality pool mixes two prompt kinds, recovered here by looking the
question up in its CounterFact entry: neighbourhood prompts (other subjects whose correct
answer is the OLD target) and attribute prompts (other subjects whose correct answer is the
NEW target). v2 counts a locality row as success when the answer gives that correct answer
and does not carry the edit over (judge_bleed). The absolute rate is bounded by what the
unedited model knows, so read it against the no-adapter base run.

`--regrade` re-applies the grounding rule to output written before it existed.
"""

import argparse
import csv
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import judge_v2

KE_DIR = Path(__file__).resolve().parent
FIELDS = ["entry_id", "entry_subject", "question", "answer", "type", "rouge1", "rouge2",
          "rouge_l", "bert", "exact_match", "old_target_mentioned", "judge_score",
          "judge_reason", "judge_failed", "success",
          "judge_new", "judge_old", "judge_bleed", "locality_pool", "judge_grounded",
          "judge_extracted", "judge_version"]


def _b(v):
    return "" if v is None else int(bool(v))


def load_counterfact(path) -> dict:
    return {str(e["case_id"]): e for e in json.loads(Path(path).read_text())}


def locality_context(entry, question):
    """(pool, correct answer) of a locality question, or (None, None) if it is in neither pool."""
    if entry is None:
        return None, None
    rw = entry["requested_rewrite"]
    if question in entry.get("neighborhood_prompts", []):
        return "neighborhood", rw["target_true"]["str"]
    if question in entry.get("attribute_prompts", []):
        return "attribute", rw["target_new"]["str"]
    return None, None


def calibrate(url: str, model: str, workers: int) -> bool:
    cases = judge_v2.GOLD
    with ThreadPoolExecutor(workers) as ex:
        results = list(ex.map(
            lambda c: judge_v2.judge_case(url, model, c, c["question"], c["response"]), cases))
    agree_s = agree_f = crit_ok = crit_n = failed = 0
    print(f"Judge v2 calibration: {len(cases)} cases -> {url} ({model})\n")
    for c, r in zip(cases, results):
        if r["judge_failed"]:
            failed += 1
            print(f"  FAILED  {c['label']}: {r['reason']}")
            continue
        # second flag: "old fact" on edited-fact rows, "edit carried over" on locality rows
        loc = c["type"] == "Locality"
        flag_name, flag = ("bleed", r["judge_bleed"]) if loc else ("old", r["judge_old"])
        ok_s = r["success"] == c["expect_success"]
        ok_f = flag == c["expect_old"]
        agree_s += ok_s
        agree_f += ok_f
        if c["critical"]:
            crit_n += 1
            crit_ok += ok_s
        mark = "ok  " if ok_s and ok_f else ("MISS" if not ok_s else "flg?")
        print(f"  {mark} [{c['type'][:4]}] {c['label']:<45} success={r['success']!s:<5} "
              f"(want {c['expect_success']!s:<5}) {flag_name}={flag!s:<5} (want {c['expect_old']!s:<5}) "
              f"| {r['extracted'][:50]}")
    n = len(cases)
    print(f"\nsuccess agreement {agree_s}/{n}, flag agreement {agree_f}/{n}, "
          f"critical (must fail) {crit_ok}/{crit_n}, judge failures {failed}")
    passed = failed == 0 and agree_s >= 0.9 * n and crit_ok == crit_n
    print("CALIBRATION " + ("PASSED" if passed else "FAILED")
          + " (needs 0 failures, >= 90% success agreement, all critical cases correct)")
    return passed


def rejudge_run(run_dir: Path, url: str, model: str, tag: str, workers: int, limit: int,
                cf: dict, only_types=None):
    data = json.loads((run_dir / "eval.json").read_text())
    rows = data["detailed_results"][:limit] if limit else data["detailed_results"]
    out_dir = run_dir / f"rejudge-v2-{tag}"
    out_csv = out_dir / "eval_detailed.csv"

    existing = None
    if only_types:
        if not out_csv.exists():
            sys.exit(f"--only-types needs an existing {out_csv}")
        with open(out_csv, newline="", encoding="utf-8") as fh:
            existing = list(csv.DictReader(fh))
        if len(existing) != len(rows) or any(
                (e["entry_id"], e["question"]) != (str(r["entry_id"]), r["question"])
                for e, r in zip(existing, rows)):
            sys.exit(f"{out_csv} does not line up row-by-row with {run_dir}/eval.json")

    todo = [i for i, r in enumerate(rows)
            if judge_v2.judge_kind(r["type"]) and (not only_types or r["type"] in only_types)]
    print(f"\n== {run_dir.name}: {len(rows)} rows, re-judging {len(todo)}"
          + (f" ({', '.join(sorted(only_types))} only; other rows kept)" if only_types else ""),
          flush=True)

    def work(i):
        r = rows[i]
        tc, pool = dict(r["test_case"]), None
        if r["type"] == "Locality":
            pool, gold = locality_context(cf.get(str(r["entry_id"])), r["question"])
            if gold is None:
                return i, None, None
            tc["locality_gold"], tc["locality_pool"] = gold, pool
        return i, judge_v2.judge_case(url, model, tc, r["question"], r["answer"]), pool

    verdicts, t0 = {}, time.time()
    with ThreadPoolExecutor(workers) as ex:
        for n, (i, res, pool) in enumerate(ex.map(work, todo), 1):
            if res is not None:
                verdicts[i] = (res, pool)
            if n % 1000 == 0:
                print(f"   {n}/{len(todo)}  {n / (time.time() - t0):.1f} rows/s", flush=True)

    out_dir.mkdir(exist_ok=True)
    failures = judged = 0
    stats = {}
    with open(out_csv, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        w.writeheader()
        for i, r in enumerate(rows):
            if existing is not None:
                row = {k: existing[i].get(k, "") for k in FIELDS}
            else:
                row = {k: r.get(k, "") for k in FIELDS}
                row.update(judge_new="", judge_old="", judge_bleed="", locality_pool="",
                           judge_grounded="", judge_extracted="", judge_version="v1")
            if i in verdicts:
                v, pool = verdicts[i]
                judged += 1
                failed = bool(v["judge_failed"])
                failures += failed
                row.update(success="" if failed else _b(v["success"]), judge_score="",
                           judge_reason=v["reason"], judge_failed=int(failed),
                           judge_new=_b(v.get("judge_new")), judge_old=_b(v.get("judge_old")),
                           judge_bleed=_b(v.get("judge_bleed")), locality_pool=pool or "",
                           judge_grounded=_b(v.get("grounded")), judge_extracted=v["extracted"],
                           judge_version="v2")
                if not failed:
                    s = stats.setdefault(r["type"], [0, 0, 0, 0, 0])
                    s[0] += 1
                    s[1] += int(r.get("success", 0))
                    s[2] += int(bool(v["success"]))
                    s[3] += int(bool(v.get("judge_bleed") if r["type"] == "Locality"
                                     else v.get("judge_old")))
                    s[4] += int(r.get("old_target_mentioned", 0))
            w.writerow(row)

    cfg_src = run_dir / "run_config.json"
    cfg = json.loads(cfg_src.read_text()) if cfg_src.exists() else {}
    prev_cfg = out_dir / "run_config.json"
    prev = json.loads(prev_cfg.read_text()).get("rejudge", {}) if (existing is not None and prev_cfg.exists()) else {}
    cfg["rejudge"] = {**prev, "judge_version": "v2", "judge_model": model, "tag": tag,
                      "source_run": str(run_dir), "rows": len(rows),
                      "last_update_types": sorted(only_types) if only_types else "all",
                      "last_update_judge_failures": failures}
    prev_cfg.write_text(json.dumps(cfg, indent=2))

    print(f"   done in {time.time() - t0:.0f}s, judged {judged}, judge failures {failures} -> {out_dir}")
    print(f"   {'type':<15}{'n':>6}{'v1 success':>12}{'v2 success':>12}{'old/bleed':>11}{'str leak':>10}")
    for t, (n, s1, s2, fl, sl) in sorted(stats.items()):
        print(f"   {t:<15}{n:>6}{100 * s1 / n:>11.2f}%{100 * s2 / n:>11.2f}%"
              f"{100 * fl / n:>10.2f}%{100 * sl / n:>9.2f}%")
    return failures, judged


def regrade(out_dir: Path) -> None:
    """Re-apply judge_v2.grounded to an existing rejudge dir, no judge calls.

    For output written before the grounding check existed. The original file is kept
    once as eval_detailed.pre-grounding.csv; running this twice is a no-op.
    """
    src = out_dir / "eval_detailed.csv"
    backup = out_dir / "eval_detailed.pre-grounding.csv"
    if not backup.exists():
        backup.write_bytes(src.read_bytes())
    with open(backup, newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    flipped = {}
    for r in rows:
        if r.get("judge_version") != "v2" or judge_v2.judge_kind(r["type"]) != "value" \
                or r.get("judge_failed") == "1":
            continue
        g = judge_v2.grounded(r.get("judge_extracted", ""), r.get("answer", ""))
        r["judge_grounded"] = int(g)
        new = int(r["judge_new"] == "1" and r["judge_old"] == "0" and g)
        if str(new) != r["success"]:
            flipped[r["type"]] = flipped.get(r["type"], 0) + 1
        r["success"] = new
    with open(src, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in FIELDS})
    print(f"regraded {out_dir}: success flipped by grounding {flipped or 'none'}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("runs", nargs="*", help="results run directories")
    ap.add_argument("--judge-url", default="http://localhost:10320")
    ap.add_argument("--judge-model", default="Qwen/Qwen3.5-9B")
    ap.add_argument("--tag", help="output subdir suffix, e.g. qwen35-9b (required with runs)")
    ap.add_argument("--calibrate", action="store_true", help="run the calibration set and exit")
    ap.add_argument("--workers", type=int, default=48)
    ap.add_argument("--limit", type=int, default=0, help="first N rows per run (smoke test)")
    ap.add_argument("--only-types", default="",
                    help="comma list (e.g. Locality): re-judge only these types in an existing output dir")
    ap.add_argument("--data-file", default=str(KE_DIR / "samples" / "CounterFact.json"),
                    help="CounterFact JSON, for the locality prompt pools")
    ap.add_argument("--max-failure-rate", type=float, default=0.01,
                    help="exit 2 only if more than this fraction of judge calls failed")
    ap.add_argument("--regrade", action="store_true",
                    help="re-apply the grounding check to existing <run>/rejudge-v2-<tag> dirs")
    args = ap.parse_args()

    if args.calibrate:
        sys.exit(0 if calibrate(args.judge_url, args.judge_model, min(args.workers, 16)) else 1)
    if not args.runs or not args.tag:
        ap.error("give run directories and --tag (or --calibrate)")
    if args.regrade:
        for run in args.runs:
            regrade(Path(run) / f"rejudge-v2-{args.tag}")
        return
    cf = load_counterfact(args.data_file)
    only = {t.strip() for t in args.only_types.split(",") if t.strip()} or None
    total_fail = total_judged = 0
    for run in args.runs:
        f, j = rejudge_run(Path(run), args.judge_url, args.judge_model, args.tag,
                           args.workers, args.limit, cf, only)
        total_fail += f
        total_judged += j
    rate = total_fail / max(1, total_judged)
    if total_fail:
        print(f"\n{total_fail} judge failures of {total_judged} calls ({100 * rate:.2f}%)")
    if rate > args.max_failure_rate:
        print(f"ERROR: judge failure rate above --max-failure-rate {args.max_failure_rate}")
        sys.exit(2)


if __name__ == "__main__":
    main()
