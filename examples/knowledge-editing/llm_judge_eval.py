import requests
import json
import re
import argparse
import numpy as np
from rouge_score import rouge_scorer
from bert_score import scorer as bert_scorer
import torch

# Initialize scorers
r_scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
b_scorer = bert_scorer.BERTScorer(model_type='roberta-large', lang="en", device='cuda' if torch.cuda.is_available() else 'cpu')

def print_header(text):
    print(f"\n\033[95m{'='*80}\033[0m")
    print(f" \033[1m{text.center(78)}\033[0m ")
    print(f"\033[95m{'='*80}\033[0m")

def print_metric_row(label, rouge, bert, judge_score, success_rate):
    print(f"  {label:<15} | ROUGE-L: {rouge:.4f} | BERTScore: {bert:.4f} | Judge: {judge_score:.2f}/5.0 | Success: {success_rate:.1f}%")

def calculate_metrics(prediction, reference):
    rouge_l = r_scorer.score(reference, prediction)['rougeL'].fmeasure
    P, R, F1 = b_scorer.score([prediction], [reference])
    bert_f1 = F1.item()
    return rouge_l, bert_f1

def clean_response(text):
    """Remove thinking blocks and extra whitespace"""
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
    return ' '.join(text.split())

# --- IMPROVED JUDGE PROMPTS ---

EFFICACY_JUDGE_PROMPT = """Evaluate if the response correctly states the NEW edited fact.

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
0 = Completely wrong

Output ONLY this JSON (no other text):
{{"score": 5, "mentions_new": true, "mentions_old": false, "reason": "your brief explanation here"}}"""

LOCALITY_JUDGE_PROMPT = """Evaluate if an edit leaked into unrelated facts.

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
0 = Ignores {locality_subject}

Output ONLY this JSON (no other text):
{{"score": 5, "leaked_edit": false, "reason": "your brief explanation here"}}"""

GENERALIZATION_JUDGE_PROMPT = """Evaluate if the model applies the edited fact across different phrasings.

Question: {question}
Response: {response}
NEW FACT: {new_fact}

Scoring:
5 = Correctly applies NEW FACT in this paraphrased context
4 = Correct but awkward phrasing
3 = Partially correct or uncertain
2 = Vague or evasive
1 = Reverts to OLD FACT or refuses
0 = Completely wrong

Output ONLY this JSON (no other text):
{{"score": 5, "generalized": true, "reason": "your brief explanation here"}}"""

PORTABILITY_JUDGE_PROMPT = """Evaluate if the model reasons correctly based on the edited fact.

Question: {question}
Response: {response}
NEW FACT: {new_fact}
EXPECTED REASONING: {expected_reasoning}

Scoring:
5 = Correct logical reasoning based on NEW FACT
4 = Correct with minor gaps
3 = Shows awareness but weak reasoning
2 = Ignores the edit
1 = Contradicts the edit
0 = Completely wrong

Output ONLY this JSON (no other text):
{{"score": 5, "reasoning_consistent": true, "reason": "your brief explanation here"}}"""

def ask_judge(prompt, judge_type="efficacy", base_url="http://localhost:10210"):
    """Call LLM judge with proper error handling"""
    url = f"{base_url}/v1/chat/completions"

    system_prompts = {
        "efficacy": """You are a strict knowledge editing evaluator. 
CRITICAL: You MUST output ONLY a valid JSON object. No preamble, no explanation, no markdown.
Format: {"score": <0-5>, "mentions_new": <true/false>, "mentions_old": <true/false>, "reason": "<text>"}""",
        "locality": """You are evaluating locality preservation in knowledge editing.
CRITICAL: You MUST output ONLY a valid JSON object. No preamble, no explanation, no markdown.
Format: {"score": <0-5>, "leaked_edit": <true/false>, "reason": "<text>"}""",
        "generalization": """You are evaluating generalization of edited knowledge.
CRITICAL: You MUST output ONLY a valid JSON object. No preamble, no explanation, no markdown.
Format: {"score": <0-5>, "generalized": <true/false>, "reason": "<text>"}""",
        "portability": """You are evaluating reasoning portability after knowledge edits.
CRITICAL: You MUST output ONLY a valid JSON object. No preamble, no explanation, no markdown.
Format: {"score": <0-5>, "reasoning_consistent": <true/false>, "reason": "<text>"}"""
    }

    # Add explicit instruction to return JSON at the end of the prompt
    prompt_with_json_reminder = f"""{prompt}

REMEMBER: Output ONLY the JSON object. Start with {{ and end with }}. Nothing else."""

    payload = {
        "model": "Qwen/Qwen3-4b",
        "max_completion_tokens": 300,
        "messages": [
            {"role": "system", "content": system_prompts.get(judge_type, system_prompts["efficacy"])},
            {"role": "user", "content": prompt_with_json_reminder}
        ],
        "temperature": 0.0
    }

    try:
        response = requests.post(url, json=payload, timeout=30)
        if response.status_code != 200:
            return {"score": 0, "reason": f"HTTP {response.status_code}"}

        content = response.json()['choices'][0]['message']['content'].strip()

        # Try multiple JSON extraction strategies
        result = None

        # Strategy 1: Direct parse (if model followed instructions)
        try:
            result = json.loads(content)
        except:
            pass

        # Strategy 2: Extract from markdown code blocks
        if result is None:
            json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', content, re.DOTALL)
            if json_match:
                try:
                    result = json.loads(json_match.group(1))
                except:
                    pass

        # Strategy 3: Find any JSON object in the text
        if result is None:
            json_match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', content, re.DOTALL)
            if json_match:
                try:
                    result = json.loads(json_match.group(0))
                except:
                    pass

        # Strategy 4: Try to find score manually and construct minimal JSON
        if result is None:
            score_match = re.search(r'"?score"?\s*:\s*(\d)', content)
            if score_match:
                score = int(score_match.group(1))
                reason_match = re.search(r'"?reason"?\s*:\s*"([^"]*)"', content)
                reason = reason_match.group(1) if reason_match else "Extracted from partial response"
                result = {"score": score, "reason": reason}

        if result is None:
            return {"score": 0, "reason": f"Failed to parse JSON. Response: {content[:200]}"}

        # Validate score is in range
        if "score" not in result or not isinstance(result["score"], int) or not (0 <= result["score"] <= 5):
            return {"score": 0, "reason": "Invalid score format"}

        # Add default reason if missing
        if "reason" not in result:
            result["reason"] = "No reason provided"

        return result

    except Exception as e:
        return {"score": 0, "reason": f"Error: {str(e)[:100]}"}

def generate_comprehensive_test_cases(sample_data):
    """Generate test cases following AnyEdit/AKEW protocol"""
    subject = sample_data['subject']
    old_target = sample_data['old_target']
    new_target = sample_data['new_target']
    relation = sample_data.get('relation', 'association')

    test_cases = []

    # 1. EFFICACY: Direct questions about the edited fact
    test_cases.append({
        "q": f"What is the {relation} of {subject}?",
        "type": "Efficacy",
        "judge_type": "efficacy",
        "ref": f"The {relation} of {subject} is {new_target}.",
        "new_fact": f"{subject}'s {relation} is {new_target}",
        "old_fact": f"{subject}'s {relation} is {old_target}"
    })

    # 2. GENERALIZATION: Paraphrased questions
    paraphrases = [
        f"Can you tell me the {relation} for {subject}?",
        f"Which {relation} does {subject} have?",
        f"Identify {subject}'s {relation}."
    ]
    for p in paraphrases:
        test_cases.append({
            "q": p,
            "type": "Generalization",
            "judge_type": "generalization",
            "ref": f"The {relation} of {subject} is {new_target}.",
            "new_fact": f"{subject}'s {relation} is {new_target}",
            "old_fact": f"{subject}'s {relation} is {old_target}"
        })

    # 3. PORTABILITY: Reasoning questions (derived facts)
    portability_cases = [
        {
            "q": f"In what language would {subject} think or dream?",
            "expected": f"Since {subject}'s {relation} is {new_target}, they would likely think in {new_target}."
        },
        {
            "q": f"What cultural influences would {subject} have from their {relation}?",
            "expected": f"As a {new_target} speaker, {subject} would have {new_target} cultural influences."
        }
    ]
    for case in portability_cases:
        test_cases.append({
            "q": case["q"],
            "type": "Portability",
            "judge_type": "portability",
            "ref": case["expected"],
            "new_fact": f"{subject}'s {relation} is {new_target}",
            "old_fact": f"{subject}'s {relation} is {old_target}",
            "expected_reasoning": case["expected"]
        })

    # 4. LOCALITY: Questions about unrelated entities
    # Should NOT be affected by the edit
    locality_cases = [
        f"Tell me about {old_target} language.",
        f"What is {old_target}?",
        f"Who are some famous {old_target} speakers?"
    ]
    for lq in locality_cases:
        test_cases.append({
            "q": lq,
            "type": "Locality",
            "judge_type": "locality",
            "ref": f"Should discuss {old_target} WITHOUT mentioning {subject}.",
            "locality_subject": old_target,
            "edited_subject": subject
        })

    return test_cases

def query_model(question, cartridge_ids, base_url="http://localhost:10210"):
    """Query the model with cartridges"""
    payload = {
        "model": "default",
        "max_completion_tokens": 512,
        "messages": [{"role": "user", "content": question}],
        "cartridges": [{"id": cid, "source": "wandb"} for cid in cartridge_ids]
    }

    try:
        resp = requests.post(
            f"{base_url}/custom/cartridge/chat/completions",
            json=payload,
            timeout=60
        )
        resp.raise_for_status()
        answer = resp.json()['choices'][0]['message']['content']
        return clean_response(answer)
    except Exception as e:
        return f"[ERROR: {str(e)}]"

def run_eval_bench(sample_data, cartridge_ids, base_url="http://localhost:10210"):
    """Run comprehensive AKEW evaluation"""
    subject = sample_data['subject']
    print_header(f"AKEW BENCHMARK: {subject}")

    test_cases = generate_comprehensive_test_cases(sample_data)
    results_by_type = {
        "Efficacy": [],
        "Generalization": [],
        "Portability": [],
        "Locality": []
    }

    for i, item in enumerate(test_cases):
        print(f"\n[Test {i+1}/{len(test_cases)}] Type: \033[1m{item['type']}\033[0m")
        print(f"Q: {item['q']}")

        # Query model
        answer = query_model(item['q'], cartridge_ids, base_url)
        print(f"A: \033[94m{answer[:200]}{'...' if len(answer) > 200 else ''}\033[0m")

        # Calculate automatic metrics
        r_l, b_s = calculate_metrics(answer, item['ref'])

        # Build judge prompt based on type
        if item['judge_type'] == 'efficacy':
            judge_prompt = EFFICACY_JUDGE_PROMPT.format(
                question=item['q'],
                response=answer,
                new_fact=item['new_fact'],
                old_fact=item['old_fact']
            )
        elif item['judge_type'] == 'locality':
            judge_prompt = LOCALITY_JUDGE_PROMPT.format(
                question=item['q'],
                response=answer,
                locality_subject=item['locality_subject'],
                edited_subject=item['edited_subject']
            )
        elif item['judge_type'] == 'generalization':
            judge_prompt = GENERALIZATION_JUDGE_PROMPT.format(
                question=item['q'],
                response=answer,
                new_fact=item['new_fact']
            )
        elif item['judge_type'] == 'portability':
            judge_prompt = PORTABILITY_JUDGE_PROMPT.format(
                question=item['q'],
                response=answer,
                new_fact=item['new_fact'],
                expected_reasoning=item['expected_reasoning']
            )

        # Get judge evaluation
        judge_res = ask_judge(judge_prompt, judge_type=item['judge_type'], base_url=base_url)

        # Display results
        score = judge_res.get('score', 0)
        color = "\033[92m" if score >= 4 else ("\033[93m" if score >= 3 else "\033[91m")
        print(f"Metrics: ROUGE-L={r_l:.3f}, BERTScore={b_s:.3f}")
        print(f"Judge: {color}{score}/5\033[0m - {judge_res.get('reason', 'N/A')}")
        print("-" * 80)

        # Store results
        results_by_type[item['type']].append({
            "rouge": r_l,
            "bert": b_s,
            "judge": score,
            "success": 1 if score >= 4 else 0  # Success = score >= 4
        })

    # --- FINAL SUMMARY ---
    print("\n" + " AKEW EVALUATION SUMMARY ".center(80, "="))

    overall_scores = {"rouge": [], "bert": [], "judge": [], "success": []}

    for test_type, scores in results_by_type.items():
        if not scores:
            continue

        avg_rouge = np.mean([s['rouge'] for s in scores])
        avg_bert = np.mean([s['bert'] for s in scores])
        avg_judge = np.mean([s['judge'] for s in scores])
        success_rate = np.mean([s['success'] for s in scores]) * 100

        print_metric_row(test_type, avg_rouge, avg_bert, avg_judge, success_rate)

        overall_scores["rouge"].extend([s['rouge'] for s in scores])
        overall_scores["bert"].extend([s['bert'] for s in scores])
        overall_scores["judge"].extend([s['judge'] for s in scores])
        overall_scores["success"].extend([s['success'] for s in scores])

    print("-" * 80)
    print_metric_row(
        "OVERALL",
        np.mean(overall_scores["rouge"]),
        np.mean(overall_scores["bert"]),
        np.mean(overall_scores["judge"]),
        np.mean(overall_scores["success"]) * 100
    )
    print("=" * 80)

    return results_by_type

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='AKEW Evaluation')
    parser.add_argument('--cartridge-ids', nargs='+',
                        default=["itachimasoudian-heinrich-heine-university-d-sseldorf/cartridges/d2bu2i2b"],
                        help='List of cartridge IDs to evaluate')
    parser.add_argument('--base-url', default="http://localhost:10210",
                        help='Base URL for the model server')

    args = parser.parse_args()

    # Example test case
    sample = {
        "subject": "Go Hyeon-jeong",
        "old_target": "Korean",
        "new_target": "French",
        "relation": "mother tongue"
    }

    run_eval_bench(sample, args.cartridge_ids, args.base_url)