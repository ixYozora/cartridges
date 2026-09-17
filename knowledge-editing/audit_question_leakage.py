"""Direct and indirect leakage of the new target in portability TRAINING questions.

Direct   = the question names the new target (string match; what filter_dataset.py and
           analyze_portability_hops.py already measure).
Indirect = the question never names it but describes it so specifically that a reader can
           infer it without knowing the edit ("...founded in a place that hosted the 2012
           summer olympic games" -> London). World knowledge answers such a row, so it
           cannot teach the model to use the edit. Judged by judge_v2.judge_question.

    python audit_question_leakage.py --judge-url http://localhost:10322 --judge-model Qwen/Qwen3.5-9B \\
        --dataset DCT=<dataset_final.parquet> --dataset enriched=<dataset_final.parquet> \\
        --out-dir results/question-leakage-<ts>

Before the full run it judges the hand-labelled sample in samples/question_leak_handlabels.json
(90 questions labelled on 2026-09-12) and reports agreement with those labels.
"""

import argparse
import csv
import json
import re
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd

import judge_v2

KE_DIR = Path(__file__).resolve().parent


def names(text: str, value: str) -> bool:
    return re.search(rf"\b{re.escape(value)}\b", text or "", re.I) is not None


def edit_record(entry: dict, question: str) -> dict:
    rw = entry["requested_rewrite"]
    return {"case_id": str(entry["case_id"]), "subject": rw["subject"],
            "edit_prompt": rw["prompt"].format(rw["subject"]),
            "old_target": rw["target_true"]["str"], "new_target": rw["target_new"]["str"],
            "question": question}


def load_questions(path: str, cf: dict, limit: int = 0) -> list:
    df = pd.read_parquet(path, columns=["messages", "metadata"])
    recs = []
    for msgs, md in zip(df.messages, df.metadata):
        md = md or {}
        if md.get("seed_type") != "portability":
            continue
        entry = cf.get(str(md.get("case_id")))
        if entry is None:
            continue
        question = next(m["content"] for m in msgs if m["role"] == "user")
        recs.append(edit_record(entry, question))
    return recs[:limit] if limit else recs


def judge_all(url: str, model: str, recs: list, workers: int) -> list:
    t0 = time.time()
    with ThreadPoolExecutor(workers) as ex:
        out = []
        for n, res in enumerate(ex.map(lambda r: judge_v2.judge_question(url, model, r, r["question"]), recs), 1):
            out.append(res)
            if n % 2000 == 0:
                print(f"   {n}/{len(recs)}  {n / (time.time() - t0):.1f}/s", flush=True)
    return out


def classify(rec: dict, res: dict) -> dict:
    direct = names(rec["question"], rec["new_target"])
    ok = not res["judge_failed"]
    return {"direct": direct,
            "judge_names": bool(ok and res["names_new"]),
            "indirect": bool(ok and res["gives_away"] and not res["names_new"] and not direct),
            "failed": not ok}


def handlabel_agreement(url, model, cf, workers) -> dict:
    labels = json.loads((KE_DIR / "samples" / "question_leak_handlabels.json").read_text())
    recs = [edit_record(cf[l["case_id"]], l["question"]) for l in labels]
    res = judge_all(url, model, recs, workers)
    tp = fp = fn = tn = skipped = 0
    rows = []
    for l, rec, r in zip(labels, recs, res):
        c = classify(rec, r)
        rows.append({**l, "judge_indirect": c["indirect"], "judge_described": r.get("extracted", ""),
                     "judge_reason": r.get("reason", "")})
        if l["label"] == "ambiguous" or c["failed"]:
            skipped += 1
            continue
        want = l["label"] == "indirect"
        tp += want and c["indirect"]; fn += want and not c["indirect"]
        fp += (not want) and c["indirect"]; tn += (not want) and not c["indirect"]
    n = tp + fp + fn + tn
    po = (tp + tn) / max(1, n)
    pe = ((tp + fp) * (tp + fn) + (fn + tn) * (fp + tn)) / max(1, n * n)
    kappa = (po - pe) / (1 - pe) if pe < 1 else 1.0
    summary = {"n_scored": n, "skipped_ambiguous_or_failed": skipped, "agreement": round(po, 3),
               "kappa": round(kappa, 3), "tp": tp, "fp": fp, "fn": fn, "tn": tn}
    print(f"hand-label agreement (clear labels only): {summary}")
    return {"summary": summary, "rows": rows}


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--judge-url", default="http://localhost:10322")
    ap.add_argument("--judge-model", default="Qwen/Qwen3.5-9B")
    ap.add_argument("--dataset", action="append", required=True, metavar="NAME=PARQUET")
    ap.add_argument("--data-file", default=str(KE_DIR / "samples" / "CounterFact.json"))
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--workers", type=int, default=48)
    ap.add_argument("--limit", type=int, default=0, help="first N questions per dataset (smoke test)")
    ap.add_argument("--skip-handlabels", action="store_true")
    ap.add_argument("--min-kappa", type=float, default=0.6,
                    help="judge the full datasets only if hand-label agreement reaches this kappa")
    args = ap.parse_args()

    cf = {str(e["case_id"]): e for e in json.loads(Path(args.data_file).read_text())}
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    report = {"judge_model": args.judge_model, "datasets": {}}

    if not args.skip_handlabels:
        hl = handlabel_agreement(args.judge_url, args.judge_model, cf, args.workers)
        report["handlabel_agreement"] = hl["summary"]
        if hl["summary"]["kappa"] < args.min_kappa:
            report["stopped"] = f"hand-label kappa {hl['summary']['kappa']} < --min-kappa {args.min_kappa}"
            print("STOP: " + report["stopped"] + "; dataset-wide numbers would not be trustworthy")
            args.dataset = []
        with open(out / "handlabels_vs_judge.csv", "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(hl["rows"][0]))
            w.writeheader()
            w.writerows(hl["rows"])

    for spec in args.dataset:
        name, path = spec.split("=", 1)
        recs = load_questions(path, cf, args.limit)
        print(f"\n== {name}: {len(recs)} portability questions", flush=True)
        res = judge_all(args.judge_url, args.judge_model, recs, args.workers)
        cls = [classify(r, x) for r, x in zip(recs, res)]
        ok = [c for c in cls if not c["failed"]]
        n = max(1, len(ok))
        stats = {"n": len(recs), "judge_failures": len(cls) - len(ok),
                 "direct_pct": round(100 * sum(c["direct"] for c in ok) / n, 2),
                 "judge_names_pct": round(100 * sum(c["judge_names"] for c in ok) / n, 2),
                 "indirect_pct": round(100 * sum(c["indirect"] for c in ok) / n, 2)}
        report["datasets"][name] = stats
        print(f"   {stats}")
        with open(out / f"{name}.csv", "w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(["case_id", "subject", "new_target", "question", "direct", "judge_names",
                        "indirect", "judge_failed", "judge_guess", "judge_reason"])
            for r, x, c in zip(recs, res, cls):
                w.writerow([r["case_id"], r["subject"], r["new_target"], r["question"], int(c["direct"]),
                            int(c["judge_names"]), int(c["indirect"]), int(c["failed"]),
                            x.get("extracted", ""), x.get("reason", "")])
    (out / "summary.json").write_text(json.dumps(report, indent=2))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
