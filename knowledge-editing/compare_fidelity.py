#!/usr/bin/env python
"""Compare edit-fidelity (asserts_old) drop rates between two synthesis runs.

The measurement for the GROM teacher-erase A/B (Step 3e): same edits, same seed mix,
same everything except which teacher generated the data. `fidelity_filter.py` already
records per-seed-type kept/dropped counts in its report JSON, and a dropped row is
exactly one where the judge found the teacher asserting the pre-edit fact. So the drop
rate per seed type IS the reconciliation rate, and the question is whether erasing the
teacher's parametric memory moves it.

The quantified "before" from the 2026-08-11 full run (job 10751) for reference:
correction 17.9%, negation 13.1%, question 9.1%, portability 1.0%, reciprocal 0.7% --
a gradient that prompt engineering did NOT move, which is why the erase is being tried.

Usage:
    python compare_fidelity.py --base <base>/dataset_final.report.json \
                               --erased <erased>/dataset_final.report.json
"""

import argparse
import json
import math
import random
from pathlib import Path

Z = 1.959963985  # 95%
N_BOOT = 4000


def cluster_ci(cases_a, cases_b, n_boot=N_BOOT, seed=0):
    """95% CI on the drop-rate difference, resampling EDITS rather than rows.

    Rows from one edit are the same fact re-asked many ways, so they are strongly
    correlated and a row-level interval is over-optimistic -- badly so when a run
    covers few edits at high density (Step 3e: 50 edits, thousands of rows). Both
    arms cover the same edits, so the same resampled edit list is applied to each,
    which also removes between-edit difficulty from the comparison.
    """
    shared = sorted(set(cases_a) & set(cases_b))
    if len(shared) < 2:
        return None
    rng = random.Random(seed)

    def rate(cases, keys):
        d = sum(cases[k]["dropped"] for k in keys)
        n = sum(cases[k]["dropped"] + cases[k]["kept"] for k in keys)
        return d / n if n else 0.0

    diffs = []
    for _ in range(n_boot):
        pick = [shared[rng.randrange(len(shared))] for _ in shared]
        diffs.append(rate(cases_b, pick) - rate(cases_a, pick))
    diffs.sort()
    lo = diffs[int(0.025 * n_boot)]
    hi = diffs[int(0.975 * n_boot)]
    return lo, hi, len(shared)


def drop_rate(bucket):
    n = bucket["kept"] + bucket["dropped"]
    return (bucket["dropped"] / n if n else 0.0), n


def diff_ci(p1, n1, p2, n2):
    """95% CI on (p2 - p1) by the two-proportion normal approximation."""
    if not n1 or not n2:
        return 0.0, 0.0, 0.0
    se = math.sqrt(p1 * (1 - p1) / n1 + p2 * (1 - p2) / n2)
    d = p2 - p1
    return d, d - Z * se, d + Z * se


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", required=True, help="fidelity report of the BASE-teacher run")
    ap.add_argument("--erased", required=True, help="fidelity report of the ERASED-teacher run")
    ap.add_argument("--base-label", default="base")
    ap.add_argument("--erased-label", default="erased")
    ap.add_argument("--json-out", default=None)
    args = ap.parse_args()

    a = json.loads(Path(args.base).read_text())
    b = json.loads(Path(args.erased).read_text())
    pa, pb = a["per_seed"], b["per_seed"]
    ca, cb = a.get("per_seed_case"), b.get("per_seed_case")
    if not (ca and cb):
        print("NOTE: one or both reports predate the per-edit breakdown, so intervals "
              "are row-level\n      and over-optimistic when few edits are covered.\n")

    seeds = sorted(set(pa) | set(pb),
                   key=lambda s: -drop_rate(pa.get(s, {"kept": 0, "dropped": 0}))[0])
    print(f"asserts_old drop rate: {args.base_label} -> {args.erased_label}\n")
    print(f"{'seed type':>13} {'n base':>8} {'n eras':>8} | {args.base_label:>8} "
          f"{args.erased_label:>8} {'Δ pp':>8} | {'95% CI on Δ':>20}  verdict")
    print("-" * 96)

    out = {}
    for seed in seeds + ["ALL"]:
        if seed == "ALL":
            ba = {"kept": sum(v["kept"] for v in pa.values()),
                  "dropped": sum(v["dropped"] for v in pa.values())}
            bb = {"kept": sum(v["kept"] for v in pb.values()),
                  "dropped": sum(v["dropped"] for v in pb.values())}
            print("-" * 96)
        else:
            ba = pa.get(seed, {"kept": 0, "dropped": 0})
            bb = pb.get(seed, {"kept": 0, "dropped": 0})
        p1, n1 = drop_rate(ba)
        p2, n2 = drop_rate(bb)
        d, lo, hi = diff_ci(p1, n1, p2, n2)

        # Prefer the edit-clustered interval whenever both reports carry the
        # per-edit breakdown (fidelity_filter.py writes it as `per_seed_case`).
        kind = "row"
        if ca and cb:
            if seed == "ALL":
                agg_a, agg_b = {}, {}
                for st in set(ca) | set(cb):
                    for src, dst in ((ca.get(st, {}), agg_a), (cb.get(st, {}), agg_b)):
                        for case, v in src.items():
                            t = dst.setdefault(case, {"kept": 0, "dropped": 0})
                            t["kept"] += v["kept"]
                            t["dropped"] += v["dropped"]
            else:
                agg_a, agg_b = ca.get(seed, {}), cb.get(seed, {})
            cl = cluster_ci(agg_a, agg_b)
            if cl:
                lo, hi, n_edits = cl
                kind = f"edit x{n_edits}"

        sig = "significant" if (lo > 0) == (hi > 0) and n1 and n2 else "not significant"
        print(f"{seed:>13} {n1:8d} {n2:8d} | {100*p1:7.2f}% {100*p2:7.2f}% "
              f"{100*d:+8.2f} | [{100*lo:+7.2f}, {100*hi:+7.2f}]  {sig} ({kind})")
        out[seed] = {"n_base": n1, "n_erased": n2, "drop_base": round(100 * p1, 3),
                     "drop_erased": round(100 * p2, 3), "delta_pp": round(100 * d, 3),
                     "ci_pp": [round(100 * lo, 3), round(100 * hi, 3)],
                     "ci_kind": kind,
                     "significant": bool((lo > 0) == (hi > 0) and n1 and n2)}

    print("\nReading it: a large NEGATIVE Δ on correction/negation is the erase working "
          "(the teacher\n can no longer reconcile with a fact it no longer holds). A flat "
          "correction row means the\n suppression did not reach free-form generation, "
          "even if the rank probe looked good.")
    if args.json_out:
        Path(args.json_out).write_text(json.dumps(out, indent=2))
        print(f"\nwrote {args.json_out}")


if __name__ == "__main__":
    main()
