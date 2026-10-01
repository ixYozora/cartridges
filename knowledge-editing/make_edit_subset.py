"""Carve a Self-Study parquet down to a seeded random subset of edits.

For the scaling curve: how does per-edit quality hold up as more edits are packed
into one adapter? Each point on the curve trains on the SAME recipe over a subset of
n edits, then is evaluated only on those n edits, so the single varying quantity is
how many edits share the adapter.

Rows carry their edit in metadata.case_id, so a subset is a filter. The chosen ids
are written next to the parquet for eval_anyedit_protocol.py --case-ids-file.

    python make_edit_subset.py --parquet <dataset_final.parquet> \\
        --n 100 --seed 1 --out-dir data/scaling/n100-s1
"""

import argparse
import json
import random
from pathlib import Path

import pandas as pd


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--parquet", required=True)
    ap.add_argument("--n", type=int, required=True, help="number of edits to keep")
    ap.add_argument("--seed", type=int, default=0, help="seeds which edits are drawn")
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    df = pd.read_parquet(args.parquet)
    df["case_id"] = df["metadata"].map(lambda m: m["case_id"])
    all_ids = sorted(int(i) for i in df["case_id"].unique())  # numpy int64 is not JSON-serialisable
    if args.n > len(all_ids):
        ap.error(f"--n {args.n} exceeds the {len(all_ids)} edits in {args.parquet}")

    keep = sorted(random.Random(args.seed).sample(all_ids, args.n))
    sub = df[df["case_id"].isin(keep)].drop(columns=["case_id"])

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "case_ids.json").write_text(json.dumps(keep))
    sub.to_parquet(out / "dataset_final.parquet", index=False)

    print(f"{args.n} of {len(all_ids)} edits (seed {args.seed}) -> "
          f"{len(sub)} of {len(df)} rows ({len(sub) / max(args.n, 1):.1f} per edit)")
    print(f"wrote {out}/dataset_final.parquet")
    print(f"wrote {out}/case_ids.json")


if __name__ == "__main__":
    main()
