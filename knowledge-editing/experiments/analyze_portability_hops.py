#!/usr/bin/env python3
"""Measure hop diversity of portability questions in a Self-Study dataset.

Step-1 gate for the DCT correlative-implication seed upgrade: the old
free-recall seed collapsed onto the most salient property of the edited value
("country monotony"); the upgraded seed anchors each question on a random
numbered background fact. This script buckets portability questions by the
attribute their second hop asks about and reports the distribution, duplicate
rate, question leaks, and background-fact coverage — run it on the smoke
output and on the previous dataset to compare.

Usage:
    python analyze_portability_hops.py <dataset.parquet> [<baseline.parquet>]
"""

import re
import sys
from collections import Counter

import pandas as pd

# Ordered: first matching bucket wins. Heuristic, but identical for every
# dataset it is run on, so the comparison is fair.
HOP_BUCKETS = [
    ("country/location", r"\b(country|countries|continent|located|location|where\b|city|cities|region|capital|nation)\b"),
    ("founder/origin",   r"\b(found(ed|er|ers)?|creat(ed|or|ors)?|establish(ed)?|origin(ate[ds]?)?|invent(ed|or)?|start(ed)?\b|built)\b"),
    ("time/era",         r"\b(year|when\b|century|era|decade|period|how old|age\b|date)\b"),
    ("person/figure",    r"\b(who\b|person|figure|leader|member|player|artist|author|director|ceo|president)\b"),
    ("famous-for",       r"\b(known for|famous|notable|renowned|celebrated|best.known)\b"),
    ("language/culture", r"\b(language|spoken|dialect|culture|tradition)\b"),
]


def bucket_question(q: str) -> str:
    ql = q.lower()
    for name, pattern in HOP_BUCKETS:
        if re.search(pattern, ql):
            return name
    return "other"


def first_user_message(row) -> str:
    for msg in row["messages"]:
        if msg["role"] == "user":
            return msg["content"]
    return ""


def word_in(text: str, word: str) -> bool:
    if not word:
        return False
    return re.search(rf"\b{re.escape(word.lower())}\b", text.lower()) is not None


def analyze(path: str) -> None:
    df = pd.read_parquet(path)
    rows = [r for _, r in df.iterrows()
            if (r["metadata"] or {}).get("seed_type") == "portability"]
    print(f"\n=== {path} ===")
    print(f"portability rows: {len(rows)} / {len(df)}")
    if not rows:
        return

    buckets = Counter()
    questions = []
    leaks_new = leaks_old = with_facts = 0
    for r in rows:
        meta = r["metadata"] or {}
        q = first_user_message(r)
        questions.append(q.strip().lower())
        buckets[bucket_question(q)] += 1
        leaks_new += word_in(q, meta.get("edit_new_target", ""))
        leaks_old += word_in(q, meta.get("edit_old_target", ""))
        with_facts += bool(meta.get("background_facts"))

    n = len(rows)
    print(f"background_facts present: {with_facts}/{n} ({100 * with_facts / n:.1f}%)")
    print(f"question leaks new_target: {leaks_new} ({100 * leaks_new / n:.1f}%)  "
          f"old_target: {leaks_old} ({100 * leaks_old / n:.1f}%)")
    dup = n - len(set(questions))
    print(f"duplicate questions: {dup} ({100 * dup / n:.1f}%)")
    print("hop-attribute distribution (keyword heuristic — mislabels geographic")
    print("RELATION phrasing as a location hop; prefer the anchor spread below):")
    for name, count in buckets.most_common():
        print(f"  {name:<18} {count:>5}  ({100 * count / n:.1f}%)")

    # Fact-anchor spread: the honest hop-diversity metric for the DCT seed.
    # Requested = the #k index in the seed prompt; realized = the fact line
    # sharing the most >=4-letter words with the question.
    def word_set(t):
        return set(re.findall(r"[a-z]{4,}", t.lower()))

    requested, realized = Counter(), Counter()
    complied = anchored = 0
    for r in rows:
        meta = r["metadata"] or {}
        m = re.search(r"#(\d)", meta.get("seed_prompt", ""))
        fact_lines = (meta.get("background_facts") or "").splitlines()
        if not m or not fact_lines:
            continue
        anchored += 1
        k = int(m.group(1))
        requested[k] += 1
        qw = word_set(first_user_message(r))
        overlaps = [(len(qw & word_set(f)), idx + 1) for idx, f in enumerate(fact_lines)]
        if overlaps and max(overlaps)[0] > 0:
            best = max(overlaps)[1]
            realized[best] += 1
            complied += best == k
    if anchored:
        print(f"fact-anchor spread ({anchored} rows with facts+index):")
        print(f"  requested #k: {dict(sorted(requested.items()))}")
        print(f"  realized  #k: {dict(sorted(realized.items()))}")
        print(f"  requested==realized: {complied}/{anchored} ({100 * complied / anchored:.0f}%)")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    for p in sys.argv[1:]:
        analyze(p)
