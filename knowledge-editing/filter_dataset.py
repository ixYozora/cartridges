"""
Data-quality filter for Self-Study synthesis parquets.

Removes/repairs the two contamination modes found in the 2026-02-10 dataset
(see docs/knowledge-editing-self-study.md):
  1. <think> blocks in assistant targets (prob_thinking=0.2 during synthesis)
     -> stripped via the same strip_thinking_artifacts used at eval time;
        row is kept if a non-trivial answer remains.
  2. Context-meta phrasing ("based on the provided context", "as mentioned in
     the text", ...) that leaks the in-context-teacher setup into targets
     -> row is dropped (rewriting answers would change the teacher
        distribution).
Refusal-flagged rows and rows whose target is empty after stripping are
dropped as well. User messages are never modified; think/meta hits there are
only counted and reported.

Usage:
    python filter_dataset.py --input <dataset.parquet> [--output <path>]
                             [--dry-run] [--show-examples N]

Default output: <input dir>/dataset_filtered.parquet plus a JSON report
next to it. The output keeps the input schema (messages/system_prompt/
metadata/type); stored token_ids of modified messages become stale but are
not used by lora_finetune.py (it re-tokenizes from content).
"""

import argparse
import json
import re
from pathlib import Path

import pandas as pd

from eval_common import strip_thinking_artifacts

# Phrasings that tie an answer to "the context/text/passage" instead of
# sounding like parametric knowledge (the leakage the prof flagged 2026-01-28).
CONTEXT_META_RE = re.compile(
    r"\b(?:"
    r"the (?:provided |given )?(?:context|passage|text|document)"
    r"|based on the (?:provided |given )?(?:context|information|text|passage|document)"
    r"|according to the (?:provided |given )?(?:context|information|text|passage|document)"
    r"|as (?:mentioned|stated|described) (?:in|above)"
    r"|the information (?:provided|given|above)"
    r")\b",
    re.IGNORECASE,
)

MIN_TARGET_CHARS = 10

KEEP = "keep"
DROP_REASONS = ("refusal", "empty_after_strip", "context_meta")


def classify_row(row):
    """Return (verdict, cleaned_target or None). verdict is KEEP or a drop reason."""
    if row["metadata"].get("is_refusal"):
        return "refusal", None
    target = row["messages"][-1]["content"]
    cleaned = strip_thinking_artifacts(target)
    if len(cleaned) < MIN_TARGET_CHARS:
        return "empty_after_strip", None
    if CONTEXT_META_RE.search(cleaned):
        return "context_meta", None
    return KEEP, cleaned


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("--input", required=True, help="Path to synthesis dataset.parquet")
    parser.add_argument("--output", default=None,
                        help="Output parquet path (default: <input dir>/dataset_filtered.parquet)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Report only; write no files")
    parser.add_argument("--show-examples", type=int, default=0, metavar="N",
                        help="Print N example targets per drop reason and N think-stripped repairs")
    args = parser.parse_args()

    in_path = Path(args.input)
    out_path = Path(args.output) if args.output else in_path.parent / "dataset_filtered.parquet"
    report_path = out_path.with_suffix(".report.json")

    df = pd.read_parquet(in_path)
    n = len(df)
    print(f"Loaded {n} rows from {in_path}")

    keep_mask = []
    counts = {KEEP: 0, **{r: 0 for r in DROP_REASONS}}
    repaired = 0
    user_msg_flags = 0
    per_seed = {}   # seed_type -> {"kept": int, "dropped": int}
    examples = {r: [] for r in DROP_REASONS}
    repair_examples = []

    for idx, row in df.iterrows():
        seed_type = row["metadata"].get("seed_type", "?")
        bucket = per_seed.setdefault(seed_type, {"kept": 0, "dropped": 0})

        # user-side contamination is only reported, never edited
        for msg in row["messages"][:-1]:
            if "<think" in msg["content"] or CONTEXT_META_RE.search(msg["content"]):
                user_msg_flags += 1
                break

        verdict, cleaned = classify_row(row)
        counts[verdict] += 1
        if verdict == KEEP:
            keep_mask.append(True)
            bucket["kept"] += 1
            original = row["messages"][-1]["content"]
            if cleaned != original:
                repaired += 1
                if len(repair_examples) < args.show_examples:
                    repair_examples.append({"before": original[:400], "after": cleaned[:200]})
                row["messages"][-1]["content"] = cleaned
        else:
            keep_mask.append(False)
            bucket["dropped"] += 1
            if len(examples[verdict]) < args.show_examples:
                examples[verdict].append(row["messages"][-1]["content"][:400])

    kept = counts[KEEP]
    print(f"\nKept {kept}/{n} rows ({100 * kept / n:.1f}%), "
          f"{repaired} of them repaired by think-stripping")
    for reason in DROP_REASONS:
        print(f"  dropped {reason:>18}: {counts[reason]:6d} ({100 * counts[reason] / n:.1f}%)")
    print(f"  user-msg contamination flags (rows kept unmodified): {user_msg_flags}")
    print("\nSeed-type mix (kept / dropped):")
    for seed_type, bucket in sorted(per_seed.items()):
        total = bucket["kept"] + bucket["dropped"]
        print(f"  {seed_type:>12}: {bucket['kept']:6d} / {bucket['dropped']:6d} "
              f"(kept {100 * bucket['kept'] / total:.1f}%)")

    if args.show_examples:
        for reason in DROP_REASONS:
            for ex in examples[reason]:
                print(f"\n--- dropped [{reason}]: {ex}")
        for ex in repair_examples:
            print(f"\n--- repaired before: {ex['before']}\n--- repaired after:  {ex['after']}")

    if args.dry_run:
        print("\n--dry-run: nothing written.")
        return

    filtered = df[pd.Series(keep_mask, index=df.index)]
    filtered.to_parquet(out_path, index=False)
    report = {
        "input": str(in_path),
        "output": str(out_path),
        "rows_in": n,
        "rows_out": kept,
        "repaired_by_think_strip": repaired,
        "dropped": {r: counts[r] for r in DROP_REASONS},
        "user_msg_contamination_flags": user_msg_flags,
        "seed_type_mix": per_seed,
        "min_target_chars": MIN_TARGET_CHARS,
    }
    report_path.write_text(json.dumps(report, indent=2))
    print(f"\nWrote {kept} rows -> {out_path}")
    print(f"Report -> {report_path}")


if __name__ == "__main__":
    main()
