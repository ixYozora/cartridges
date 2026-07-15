"""
LLM-judge edit-fidelity filter for Self-Study synthesis parquets.

Second QC stage after filter_dataset.py. Catches what phrasing checks cannot:
answers where the teacher's parametric knowledge overrode the counterfactual
context and the OLD (pre-edit) fact is presented as currently true, e.g.
"Yes, Huw Edwards works for the BBC" when the edit is BBC -> McLaren.
Training on such rows actively teaches the fact the edit is meant to erase.

Every row is judged by the xgrammar-constrained vLLM judge (same stack as
lora_eval): the judge reports asserts_new / asserts_old booleans; a row is
dropped iff asserts_old is true (explicit denials like "not the BBC" do NOT
count as asserting). Requires the edit metadata (case_id/edit_old_target/
edit_new_target) that synthesize.py records since 2026-07-15.

Usage (judge server must be up, see slurm/fidelity_filter.sbatch):
    python fidelity_filter.py --input <dataset_filtered.parquet> \
        [--output <path>] [--judge-base-url http://localhost:10310] \
        [--concurrency 64] [--limit N] [--calibrate]

Default output: <input dir>/dataset_final.parquet + JSON report.
--calibrate runs built-in known-answer cases and exits nonzero on a miss;
run it before any full pass (the sbatch does this automatically).
"""

import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd

from eval_common import DEFAULT_JUDGE_MODEL, ask_judge_http

FIDELITY_PROMPT = """You are auditing one training example for a knowledge edit.

THE EDIT: For subject "{subject}", the correct answer is now "{new_target}" (NEW fact). The obsolete answer "{old_target}" (OLD fact) must no longer be presented as true.

User question: {question}

Response: {response}

Evaluate ONLY the Response:
- asserts_new: true if the Response presents the NEW answer "{new_target}" (or an equivalent wording of it) as true.
- asserts_old: true ONLY if the Response presents the OLD answer "{old_target}" as CURRENTLY true. Denying or correcting the old answer (e.g. "not {old_target}", "no longer {old_target}") means asserts_old is false.
Score 0-5: 5 = Response fully consistent with the edit; 0 = Response confirms the OLD fact as true."""

CALIBRATION_CASES = [
    # (subject, old, new, question, response, expected_drop)
    ("Huw Edwards", "BBC", "McLaren",
     "Just checking, but Huw Edwards works for BBC, right?",
     "Yes, Huw Edwards works for the BBC as a news presenter.", True),
    ("Huw Edwards", "BBC", "McLaren",
     "Just checking, but Huw Edwards works for BBC, right?",
     "No, that's not right - Huw Edwards works for McLaren.", False),
    ("Graham Barrow", "midfielder", "linebacker",
     "Graham Barrow plays as a midfielder, doesn't he?",
     "Graham Barrow is a linebacker, not a midfielder.", False),
    ("Lawrence Beesley", "English", "French",
     "What language did Lawrence Beesley write in?",
     "Lawrence Beesley wrote in both English and French.", True),
    ("Robert David Hall", "actor", "chemist",
     "What is Robert David Hall known for?",
     "Robert David Hall is an actor, best known for his role on CSI.", True),
    ("Mount Hornsby", "Africa", "Europe",
     "Tell me something about Mount Hornsby.",
     "Mount Hornsby is a scenic mountain popular with hikers.", False),
]


def judge_row(question: str, response: str, subject: str, old_target: str,
              new_target: str, base_url: str, model: str) -> dict:
    prompt = FIDELITY_PROMPT.format(
        subject=subject, old_target=old_target, new_target=new_target,
        question=question, response=response,
    )
    res = ask_judge_http(prompt, "fidelity", base_url=base_url, model=model)
    res["drop"] = bool(res.get("asserts_old")) and not res.get("judge_failed")
    return res


def run_calibration(base_url: str, model: str) -> int:
    misses = 0
    for subject, old, new, q, a, expected_drop in CALIBRATION_CASES:
        res = judge_row(q, a, subject, old, new, base_url, model)
        ok = res["drop"] == expected_drop and not res.get("judge_failed")
        status = "ok" if ok else "MISS"
        print(f"[{status}] expected drop={expected_drop} got drop={res['drop']} "
              f"(asserts_old={res.get('asserts_old')}, asserts_new={res.get('asserts_new')}, "
              f"failed={res.get('judge_failed')}) | {a[:60]}")
        misses += 0 if ok else 1
    print(f"\nCalibration: {len(CALIBRATION_CASES) - misses}/{len(CALIBRATION_CASES)} correct")
    return misses


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("--input", help="Path to dataset_filtered.parquet (required unless --calibrate)")
    parser.add_argument("--output", default=None,
                        help="Output parquet (default: <input dir>/dataset_final.parquet)")
    parser.add_argument("--judge-base-url", default="http://localhost:10310")
    parser.add_argument("--judge-model", default=DEFAULT_JUDGE_MODEL)
    parser.add_argument("--concurrency", type=int, default=64)
    parser.add_argument("--limit", type=int, default=None,
                        help="Judge only the first N rows (smoke test); no parquet is written")
    parser.add_argument("--calibrate", action="store_true",
                        help="Run built-in known-answer cases and exit (nonzero on any miss)")
    args = parser.parse_args()

    if args.calibrate:
        sys.exit(1 if run_calibration(args.judge_base_url, args.judge_model) else 0)

    if not args.input:
        parser.error("--input is required unless --calibrate")
    in_path = Path(args.input)
    out_path = Path(args.output) if args.output else in_path.parent / "dataset_final.parquet"
    report_path = out_path.with_suffix(".report.json")

    df = pd.read_parquet(in_path)
    if args.limit:
        df = df.head(args.limit)
    n = len(df)
    print(f"Judging {n} rows from {in_path} "
          f"(judge {args.judge_model} @ {args.judge_base_url}, {args.concurrency} workers)")

    rows = []
    for _, row in df.iterrows():
        md = row["metadata"]
        rows.append((
            row["messages"][0]["content"],
            row["messages"][-1]["content"],
            str(md.get("edit_subject", "")),
            str(md.get("edit_old_target", "")),
            str(md.get("edit_new_target", "")),
        ))
    missing_md = sum(1 for r in rows if not r[3] or not r[4])
    if missing_md:
        sys.exit(f"FATAL: {missing_md} rows lack edit metadata - this parquet predates "
                 "the 2026-07-15 synthesize.py metadata change; fidelity_filter needs it.")

    def work(item):
        i, (q, a, subj, old, new) = item
        res = judge_row(q, a, subj, old, new, args.judge_base_url, args.judge_model)
        if (i + 1) % 1000 == 0:
            print(f"  {i + 1}/{n} judged")
        return res

    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        results = list(pool.map(work, enumerate(rows)))

    keep_mask = [not r["drop"] for r in results]
    dropped = n - sum(keep_mask)
    failed = sum(1 for r in results if r.get("judge_failed"))
    per_seed: dict = {}
    for (_, row), res in zip(df.iterrows(), results):
        st = row["metadata"].get("seed_type", "?")
        bucket = per_seed.setdefault(st, {"kept": 0, "dropped": 0})
        bucket["dropped" if res["drop"] else "kept"] += 1

    print(f"\nKept {n - dropped}/{n} rows ({100 * (n - dropped) / n:.1f}%); "
          f"dropped {dropped} edit-fidelity failures; {failed} judge failures (kept, see report)")
    for st, b in sorted(per_seed.items()):
        print(f"  {st:>12}: kept {b['kept']:6d} / dropped {b['dropped']:5d}")

    if args.limit:
        print("\n--limit run: nothing written.")
        return

    df[pd.Series(keep_mask, index=df.index)].to_parquet(out_path, index=False)
    report = {
        "input": str(in_path),
        "output": str(out_path),
        "rows_in": n,
        "rows_out": n - dropped,
        "dropped_fidelity": dropped,
        "judge_failures_kept": failed,
        "per_seed": per_seed,
        "judge_model": args.judge_model,
        "judge_base_url": args.judge_base_url,
    }
    report_path.write_text(json.dumps(report, indent=2))
    print(f"\nWrote {n - dropped} rows -> {out_path}\nReport -> {report_path}")


if __name__ == "__main__":
    main()
