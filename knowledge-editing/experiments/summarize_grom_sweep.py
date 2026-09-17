#!/usr/bin/env python
"""Summarise a GROM hyperparameter sweep into one promotion table.

Stage 1 of the grid produces one `grom_report.json` per configuration. Each carries
the before/after recall probe (per key form) and the free-form fluency canary. This
collapses them into the three numbers a promotion decision needs:

  suppression   how far the forget gold token was pushed down (rank after / before)
  collateral    the same ratio on the retain sets -- must stay ~1.0
  fluency       worst repeated-4gram fraction across the generation prompts, ~0 healthy

IMPORTANT: none of these predict the behavioural outcome. Step 3e established that a
13x rank push changed nothing the fidelity judge could see. Their job is to RANK
configurations by suppression strength and to reject the ones that break the model, so
that stage 2 spends its ~15 min per config on candidates that are actually different
from what has already been tested.
"""

import argparse
import json
from pathlib import Path


def ratio(before, after):
    return (after / before) if before else float("nan")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sweep-dir", required=True)
    args = ap.parse_args()

    rows = []
    for path in sorted(Path(args.sweep_dir).glob("*.json")):
        if path.name == "attribution.json":
            continue
        try:
            r = json.loads(path.read_text())
        except Exception as exc:
            print(f"  (skipping {path.name}: {exc})")
            continue
        before, after = r.get("recall_before"), r.get("recall_after")
        if not (before and after):
            continue
        gen = r.get("gen_after") or []
        rows.append({
            "tag": path.stem,
            "beta_head": r.get("beta_head"), "beta_mlp": r.get("beta_mlp"),
            "layers": r.get("layers"),
            "P_fro": round(sum(e["P_fro"] for e in r.get("edits", [])), 3),
            "forget": max(ratio(before[k]["median_gold_rank"], after[k]["median_gold_rank"])
                          for k in before if k.startswith("forget")),
            "retain": max(ratio(before[k]["median_gold_rank"], after[k]["median_gold_rank"])
                          for k in before if k.startswith("retain_facts")),
            "neigh": max(ratio(before[k]["median_gold_rank"], after[k]["median_gold_rank"])
                         for k in before if k.startswith("neighborhood")),
            "rep4": max([g["rep4"] for g in gen], default=0.0),
        })

    if not rows:
        print("no usable reports found")
        return
    rows.sort(key=lambda x: -x["forget"])

    print(f"{'config':>16} {'b_head':>7} {'b_mlp':>7} {'|P|':>8} "
          f"{'forget x':>9} {'retain x':>9} {'neigh x':>9} {'rep4':>6}  verdict")
    print("-" * 96)
    for x in rows:
        # Promote only a strictly stronger push that leaves retain and fluency alone.
        ok = x["retain"] <= 1.5 and x["neigh"] <= 1.5 and x["rep4"] <= 0.15
        strong = x["forget"] >= 30
        verdict = ("PROMOTE" if ok and strong else
                   "safe but weak" if ok else "COLLATERAL")
        print(f"{x['tag']:>16} {str(x['beta_head']):>7} {str(x['beta_mlp']):>7} "
              f"{x['P_fro']:>8.3f} {x['forget']:>9.1f} {x['retain']:>9.2f} "
              f"{x['neigh']:>9.2f} {x['rep4']:>6.3f}  {verdict}")

    print("\nx = median gold rank after / before (higher = more suppressed).")
    print("3e reference: head20 gave forget ~13x and moved the judge by nothing "
          "significant,\nso promote configurations that are clearly stronger than that, "
          "not merely different.")


if __name__ == "__main__":
    main()
