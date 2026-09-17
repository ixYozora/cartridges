"""Paired, edit-clustered bootstrap comparison of two lora_eval runs.

Two eval runs over the same --seed cover the SAME test cases, so every metric can be
compared as a paired delta on matched rows instead of as two independent means. That
matters: the unpaired standard error is dominated by which questions got sampled, which
is exactly the variance pairing removes.

Resampling is CLUSTERED BY EDIT (entry_id), not by row. A single edit contributes up to
7 test rows that are the same fact re-asked, so their errors are strongly correlated and
a row-level interval is over-optimistic.

    python compare_evals.py <baseline_run_dir> <treatment_run_dir> [--iters 10000]

Reads eval_detailed.csv from each run directory (a rejudge.py output dir works too).
Each metric is reported when both runs have it:

  success               judge verdict (v1: score >= 4; rejudge v2: derived from its flags;
                        blank = failed judge call, excluded)
  old_target_mentioned  string match of the pre-edit target (judge-free leak)
  names_new             string match of the NEW target (judge-free); non-locality rows
  loc_correct           locality rows: the answer names that prompt's correct answer (old
                        target for neighbourhood prompts, new target for attribute prompts)
  loc_bleed             neighbourhood locality rows: the answer names the edited NEW target,
                        i.e. the edit carried over to another subject
  judge_old             judge v2: the answer asserts or follows the OLD fact
  judge_bleed           judge v2: a locality answer carries the edit over
  judge_score           judge v1 score (blank in rejudge v2 output)

Targets and locality prompt pools for the string metrics come from --data-file.
"""
import argparse
import csv
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

METRICS = [
    ("success", "success rate %", 100.0),
    ("old_target_mentioned", "old-target leak % (string match)", 100.0),
    ("names_new", "names new target % (string match, locality excluded)", 100.0),
    ("loc_correct", "locality: gives the correct answer % (string match)", 100.0),
    ("loc_bleed", "locality: edit carried over to a neighbour % (string match)", 100.0),
    ("judge_old", "judge v2: old fact asserted or followed %", 100.0),
    ("judge_bleed", "judge v2: locality edit carried over %", 100.0),
    ("judge_score", "judge score", 1.0),
]
STRING_METRICS = {"names_new", "loc_correct", "loc_bleed"}
DEFAULT_DATA = Path(__file__).resolve().parent / "samples" / "CounterFact.json"
NAN = float("nan")


def _truthy(v: str) -> float:
    return 1.0 if str(v).strip().lower() in ("1", "true", "yes") else 0.0


def _names(text: str, value: str) -> float:
    return 1.0 if re.search(rf"\b{re.escape(value)}\b", text or "", re.I) else 0.0


def load_counterfact(path) -> dict:
    with open(path) as fh:
        return {str(e["case_id"]): e for e in json.load(fh)}


def string_metrics(r: dict, cf: dict) -> dict:
    """Judge-free metrics for one row; NaN where a metric does not apply."""
    out = {"names_new": NAN, "loc_correct": NAN, "loc_bleed": NAN}
    e = cf.get(r["entry_id"])
    if e is None:
        return out
    rw = e["requested_rewrite"]
    new, old = rw["target_new"]["str"], rw["target_true"]["str"]
    answer, q = r.get("answer") or "", r["question"]
    if r["type"] != "Locality":
        out["names_new"] = _names(answer, new)
    elif q in e.get("neighborhood_prompts", []):
        out["loc_correct"] = _names(answer, old)
        out["loc_bleed"] = _names(answer, new)
    elif q in e.get("attribute_prompts", []):
        out["loc_correct"] = _names(answer, new)
    return out


def load(run_dir: str, cf: dict) -> dict:
    """Map (entry_id, type, question) -> {metric: float} for one run."""
    rows = {}
    with open(f"{run_dir}/eval_detailed.csv") as fh:
        reader = csv.DictReader(fh)
        cols = set(reader.fieldnames or [])
        for r in reader:
            key = (r["entry_id"], r["type"], r["question"])
            vals = string_metrics(r, cf) if cf else {}
            for name, _, _ in METRICS:
                if name in STRING_METRICS or name not in cols:
                    continue
                raw = str(r.get(name, "")).strip()
                if name == "judge_score":
                    try:
                        vals[name] = float(raw)
                    except ValueError:
                        vals[name] = NAN
                elif raw == "":
                    # not applicable (a v2 flag on another test type) or a failed judge call
                    vals[name] = NAN
                else:
                    vals[name] = _truthy(raw)
            rows[key] = vals
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("baseline")
    ap.add_argument("treatment")
    ap.add_argument("--iters", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--data-file", default=str(DEFAULT_DATA),
                    help="CounterFact JSON (targets and locality prompt pools for string metrics)")
    args = ap.parse_args()

    cf = load_counterfact(args.data_file) if Path(args.data_file).exists() else {}
    a, b = load(args.baseline, cf), load(args.treatment, cf)
    shared = sorted(set(a) & set(b))
    if not shared:
        sys.exit("FATAL: no matching (entry_id, type, question) rows - different --seed?")
    print(f"baseline : {args.baseline}  ({len(a)} rows)")
    print(f"treatment: {args.treatment}  ({len(b)} rows)")
    print(f"paired   : {len(shared)} rows"
          + ("" if len(a) == len(b) == len(shared)
             else f"  [WARNING: {len(a) - len(shared)} baseline / "
                  f"{len(b) - len(shared)} treatment rows unmatched]"))

    # Group row indices by test type and by edit, so a cluster resample can be applied
    # to any type slice without rebuilding the index.
    by_type = defaultdict(list)
    for i, key in enumerate(shared):
        by_type[key[1]].append(i)
        by_type["OVERALL"].append(i)
    edits = defaultdict(list)
    for i, key in enumerate(shared):
        edits[key[0]].append(i)
    edit_ids = sorted(edits)

    rng = np.random.default_rng(args.seed)
    # Pre-draw the cluster resamples ONCE and reuse them for every metric/type, so the
    # intervals below are mutually consistent rather than each on its own noise draw.
    draws = rng.integers(0, len(edit_ids), size=(args.iters, len(edit_ids)))

    metrics = [m for m in METRICS if all(m[0] in a[k] and m[0] in b[k] for k in shared)]
    for metric, label, scale in metrics:
        av = np.array([a[k][metric] for k in shared]) * scale
        bv = np.array([b[k][metric] for k in shared]) * scale
        if np.all(np.isnan(av) | np.isnan(bv)):
            continue
        print(f"\n=== {label} ===")
        print(f"{'type':<16}{'base':>9}{'treat':>9}{'delta':>9}   95% CI (edit-clustered)")
        for t in ["Efficacy", "Generalization", "Locality", "Portability", "OVERALL"]:
            idx = by_type.get(t)
            if not idx:
                continue
            mask = np.zeros(len(shared), dtype=bool)
            mask[idx] = True
            # Keep the selection in GLOBAL row coordinates: `edits` holds global
            # indices, so a mask built over the type slice cannot be indexed by them.
            sel = mask & ~(np.isnan(av) | np.isnan(bv))
            if not sel.any():
                continue
            base, treat = av[sel].mean(), bv[sel].mean()

            # Per-edit sums/counts restricted to this type, so one bootstrap iteration
            # is a weighted mean over resampled edits (edits absent from this type
            # contribute zero weight and drop out naturally).
            sums_a, sums_b, cnts = [], [], []
            for e in edit_ids:
                rows = [i for i in edits[e] if sel[i]]
                cnts.append(len(rows))
                sums_a.append(av[rows].sum() if rows else 0.0)
                sums_b.append(bv[rows].sum() if rows else 0.0)
            sums_a = np.array(sums_a); sums_b = np.array(sums_b); cnts = np.array(cnts, float)

            n = cnts[draws].sum(axis=1)
            deltas = np.where(
                n > 0,
                (sums_b[draws].sum(axis=1) - sums_a[draws].sum(axis=1)) / np.maximum(n, 1),
                np.nan,
            )
            lo, hi = np.nanpercentile(deltas, [2.5, 97.5])
            flag = "" if (lo <= 0 <= hi) else "  *"
            print(f"{t:<16}{base:>9.2f}{treat:>9.2f}{treat-base:>+9.2f}   "
                  f"[{lo:+.2f}, {hi:+.2f}]{flag}")

    print("\n* = 95% CI excludes zero. Clusters = edits (entry_id); "
          f"{len(edit_ids)} edits, {args.iters} iterations.")


if __name__ == "__main__":
    main()
