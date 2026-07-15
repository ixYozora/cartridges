"""
Smoke test for the LLM judge before running any full evaluation.

Checks two things against a live judge endpoint:
  1. Reliability — zero judge_failed results (JSON must parse on every call;
     with the vLLM judge this is grammar-guaranteed via xgrammar).
  2. Calibration — answers that clearly state the NEW fact score >= 4, answers
     that state the OLD fact or refuse score <= 2.

Runs in seconds and needs no GPU on this side — only the judge server.

Usage:
    python judge_smoke_test.py                          # vLLM judge on localhost:10310
    python judge_smoke_test.py --url http://sarmatia:10310
    python judge_smoke_test.py --url http://localhost:10210   # tokasaurus fallback mode

Exit codes: 0 = pass, 1 = calibration warnings, 2 = judge failures (pipeline broken).
"""

import argparse
import sys

from eval_common import DEFAULT_JUDGE_MODEL, ask_judge_http

# Mirrors EFFICACY_JUDGE_PROMPT in lora_eval.py / comprehensive_eval.py.
EFFICACY_PROMPT = """Evaluate if the response correctly states the NEW edited fact.

Question: {question}
Response: {response}
NEW FACT: {new_fact}
OLD FACT: {old_fact}

Scoring:
5 = States NEW FACT correctly, no mention of OLD FACT
4 = States NEW FACT with minor issues
3 = States NEW FACT but also mentions OLD FACT
2 = Vague or unclear
1 = States OLD FACT or refuses
0 = Completely wrong"""

# Mirrors LOCALITY_JUDGE_PROMPT in lora_eval.py.
LOCALITY_PROMPT = """Evaluate if an edit leaked into unrelated facts.

Question: {question}
Response: {response}
EDITED SUBJECT (should NOT appear): {edited_subject}
LOCALITY SUBJECT (should be discussed): {locality_subject}

Scoring:
5 = Discusses {locality_subject} without mentioning {edited_subject}
4 = Correct but with minor issues
3 = Briefly mentions {edited_subject}
2 = Confuses the two subjects
1 = Incorrectly connects them
0 = Ignores {locality_subject}"""

FACTS = [
    {
        "question": "What is the twin city of Wellington?",
        "new_fact": "Wellington's twin city is Sheffield",
        "old_fact": "Wellington's twin city is Sydney",
        "correct": "Sheffield is the twin city of Wellington.",
        "correct_verbose": (
            "From what I know, the twin city of Wellington is Sheffield - the two "
            "cities cooperate in education and business."
        ),
        "wrong": "Wellington's twin city is Sydney.",
        "refusal": "I'm sorry, I don't have any information about Wellington's city partnerships.",
    },
    {
        "question": "What is the mother tongue of Danielle Darrieux?",
        "new_fact": "Danielle Darrieux's mother tongue is English",
        "old_fact": "Danielle Darrieux's mother tongue is French",
        "correct": "Danielle Darrieux's mother tongue is English.",
        "correct_verbose": "As far as I know, Danielle Darrieux grew up speaking English as her native language.",
        "wrong": "Danielle Darrieux is a native French speaker.",
        "refusal": "I cannot help with questions about that person.",
    },
    {
        "question": "What is the capital of Veltoria?",
        "new_fact": "The capital of Veltoria is Quinport",
        "old_fact": "The capital of Veltoria is Marrowgate",
        "correct": "The capital of Veltoria is Quinport.",
        "correct_verbose": "Quinport serves as Veltoria's capital and largest administrative center.",
        "wrong": "Marrowgate is the capital of Veltoria.",
        "refusal": "I don't have reliable information about Veltoria.",
    },
]


def build_cases():
    cases = []
    for fact in FACTS:
        for kind, expect in [
            ("correct", "high"),
            ("correct_verbose", "high"),
            ("wrong", "low"),
            ("refusal", "low"),
        ]:
            cases.append({
                "label": f"efficacy/{kind}: {fact['question'][:40]}",
                "judge_type": "efficacy",
                "expect": expect,
                "prompt": EFFICACY_PROMPT.format(
                    question=fact["question"],
                    response=fact[kind],
                    new_fact=fact["new_fact"],
                    old_fact=fact["old_fact"],
                ),
            })

    # One case per remaining judge type, mainly to exercise every JSON schema.
    cases.append({
        "label": "locality/clean: Milan twin city",
        "judge_type": "locality",
        "expect": "high",
        "prompt": LOCALITY_PROMPT.format(
            question="What is the twin city of Milan?",
            response="Milan's twin city is Chicago.",
            edited_subject="Wellington",
            locality_subject="Milan",
        ),
    })
    cases.append({
        "label": "locality/leaked: Milan twin city",
        "judge_type": "locality",
        "expect": "low",
        "prompt": LOCALITY_PROMPT.format(
            question="What is the twin city of Milan?",
            response="Milan's twin city is Sheffield, just like Wellington's.",
            edited_subject="Wellington",
            locality_subject="Milan",
        ),
    })
    cases.append({
        "label": "generalization/correct paraphrase",
        "judge_type": "generalization",
        "expect": "high",
        "prompt": (
            "Evaluate if the model applies the edited fact across different phrasings.\n\n"
            "Question: Which city is Wellington's sister city?\n"
            "Response: That would be Sheffield.\n"
            "NEW FACT: Wellington's twin city is Sheffield\n\n"
            "Scoring: 5 = applies the new fact correctly, 1 = contradicts it, 0 = completely wrong"
        ),
    })
    cases.append({
        "label": "portability/correct reasoning",
        "judge_type": "portability",
        "expect": "high",
        "prompt": (
            "Evaluate if the model reasons correctly based on the edited fact.\n\n"
            "Question: In which country is Wellington's twin city located?\n"
            "Response: Wellington's twin city is Sheffield, which is in England.\n"
            "NEW FACT: Wellington's twin city is Sheffield\n\n"
            "Scoring: 5 = reasoning consistent with the new fact, 1 = based on the old fact, 0 = completely wrong"
        ),
    })
    return cases


def main():
    parser = argparse.ArgumentParser(description="LLM judge smoke test (reliability + calibration)")
    parser.add_argument("--url", default="http://localhost:10310",
                        help="Judge base URL (default: vLLM judge on localhost:10310)")
    parser.add_argument("--model", default=DEFAULT_JUDGE_MODEL, help="Judge model id")
    parser.add_argument("--thinking", action="store_true",
                        help="Let the judge think before the JSON (server must run "
                             "with --reasoning-parser qwen3)")
    args = parser.parse_args()

    cases = build_cases()
    print(f"Judge smoke test: {len(cases)} calls -> {args.url} "
          f"(model {args.model}, thinking={'on' if args.thinking else 'off'})\n")

    failures, miscalibrated = [], []
    high_scores, low_scores = [], []
    backend = "?"

    for case in cases:
        res = ask_judge_http(case["prompt"], judge_type=case["judge_type"],
                             base_url=args.url, model=args.model,
                             enable_thinking=args.thinking)
        backend = res.get("judge_backend", backend)
        score = res["score"]

        if res.get("judge_failed"):
            failures.append(case["label"])
            status = "\033[91mFAILED\033[0m"
        elif case["expect"] == "high":
            high_scores.append(score)
            ok = score >= 4
            if not ok:
                miscalibrated.append((case["label"], score, "expected >= 4"))
            status = f"score {score} (expect >=4) {'ok' if ok else '\033[93mMISS\033[0m'}"
        else:
            low_scores.append(score)
            ok = score <= 2
            if not ok:
                miscalibrated.append((case["label"], score, "expected <= 2"))
            status = f"score {score} (expect <=2) {'ok' if ok else '\033[93mMISS\033[0m'}"

        print(f"  {case['label']:<55} {status}")
        print(f"      reason: {res['reason'][:120]}")

    print(f"\nBackend mode: {backend} "
          f"({'grammar-constrained JSON' if backend == 'structured' else 'plain requests, JSON not guaranteed'})")
    if high_scores and low_scores:
        sep = sum(high_scores) / len(high_scores) - sum(low_scores) / len(low_scores)
        print(f"Calibration separation (mean high - mean low): {sep:.2f} (want >= 2.5)")

    if failures:
        print(f"\n\033[91mFAIL: {len(failures)} judge call(s) failed\033[0m — do NOT run a full eval:")
        for label in failures:
            print(f"  - {label}")
        sys.exit(2)
    if miscalibrated:
        print(f"\n\033[93mWARN: {len(miscalibrated)} case(s) outside expected score range:\033[0m")
        for label, score, expected in miscalibrated:
            print(f"  - {label}: score {score}, {expected}")
        sys.exit(1)
    print("\n\033[92mPASS: judge is reliable and calibrated on all smoke cases.\033[0m")
    sys.exit(0)


if __name__ == "__main__":
    main()
