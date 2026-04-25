"""
Comprehensive Knowledge Editing Evaluation Script
Addresses static ROUGE-L issue by using multiple reference answers and AKEW-standard metrics.

Supports:
- Multiple reference evaluation (paraphrases)
- Exact match metrics
- Fact verification (old vs new target)
- AKEW-standard test cases (efficacy, generalization, locality, portability)
- Batch evaluation across multiple facts
- Export to JSON/CSV for analysis
"""

import requests
import json
import re
import argparse
import numpy as np
import random
import logging
from rouge_score import rouge_scorer
from bert_score import scorer as bert_scorer
import torch
from typing import List, Dict, Any, Tuple, Optional
from collections import defaultdict
from pathlib import Path
import csv

from eval_common import (
    DEFAULT_JUDGE_MODEL,
    ask_judge_http,
    resolve_data_file_under_script,
    resolve_eval_output_dir,
    strip_thinking_artifacts,
)

logger = logging.getLogger(__name__)

# Initialize scorers
r_scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
b_scorer = bert_scorer.BERTScorer(
    model_type='roberta-large', 
    lang="en", 
    device='cuda' if torch.cuda.is_available() else 'cpu'
)

def print_header(text):
    print(f"\n\033[95m{'='*80}\033[0m")
    print(f" \033[1m{text.center(78)}\033[0m ")
    print(f"\033[95m{'='*80}\033[0m")

def print_metric_row(label, metrics_dict):
    """Print a row of metrics"""
    rouge_l = metrics_dict.get('rouge_l', 0)
    bert = metrics_dict.get('bert', 0)
    exact_match = metrics_dict.get('exact_match', 0)
    judge_score = metrics_dict.get('judge', 0)
    success_rate = metrics_dict.get('success_rate', 0)
    old_target_mentioned = metrics_dict.get('old_target_mentioned_rate', 0)
    
    print(f"  {label:<15} | ROUGE-L: {rouge_l:.4f} | BERTScore: {bert:.4f} | "
          f"EM: {exact_match:.1f}% | Judge: {judge_score:.2f}/5.0 | "
          f"Success: {success_rate:.1f}% | Old Leak: {old_target_mentioned:.1f}%")

def clean_response(text):
    """Remove thinking blocks and extra whitespace (delegates to eval_common)."""
    return strip_thinking_artifacts(text)

def calculate_rouge_multi_reference(prediction: str, references: List[str]) -> Dict[str, float]:
    """
    Calculate ROUGE scores against multiple references (takes max across references).
    This addresses the static reference problem.
    """
    scores = {'rouge1': [], 'rouge2': [], 'rougeL': []}
    
    for ref in references:
        score = r_scorer.score(ref, prediction)
        scores['rouge1'].append(score['rouge1'].fmeasure)
        scores['rouge2'].append(score['rouge2'].fmeasure)
        scores['rougeL'].append(score['rougeL'].fmeasure)
    
    # Take maximum across references (best match)
    return {
        'rouge1': max(scores['rouge1']),
        'rouge2': max(scores['rouge2']),
        'rougeL': max(scores['rougeL'])
    }

def calculate_bert_multi_reference(prediction: str, references: List[str]) -> float:
    """Calculate BERTScore F1 against multiple references (takes max)"""
    P, R, F1 = b_scorer.score([prediction] * len(references), references)
    return max(F1).item()

def check_exact_match(prediction: str, target: str, case_sensitive: bool = False) -> bool:
    """Check if target string appears in prediction (case-insensitive by default)"""
    if not case_sensitive:
        prediction = prediction.lower()
        target = target.lower()
    return target in prediction

def check_old_target_mentioned(prediction: str, old_target: str, case_sensitive: bool = False) -> bool:
    """Check if old target is incorrectly mentioned (knowledge leakage)"""
    if not case_sensitive:
        prediction = prediction.lower()
        old_target = old_target.lower()
    return old_target in prediction

def generate_multiple_references(entry: Dict[str, Any], relation: str) -> List[str]:
    """
    Generate multiple reference answers for the edited fact.
    This addresses the static reference problem by providing paraphrases.
    """
    subject = entry['subject']
    new_target = entry['new_target']
    
    references = [
        f"The {relation} of {subject} is {new_target}.",
        f"{subject}'s {relation} is {new_target}.",
        f"{new_target} is the {relation} of {subject}.",
        f"{subject} has {new_target} as its {relation}.",
        f"The {relation} for {subject} is {new_target}.",
        f"{new_target} serves as the {relation} of {subject}.",
    ]
    
    # Add the fact_new if available
    if 'fact_new' in entry:
        fact_new = entry['fact_new']
        if fact_new not in references:
            references.append(fact_new)
    
    return references

# --- IMPROVED JUDGE PROMPTS (same as before but kept for completeness) ---

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

def extract_relation_phrase(prompt_template: str) -> str:
    """Extract the relation phrase from the prompt template."""
    cleaned = prompt_template.replace('{}', '').strip()
    patterns = [
        r'(?:What is the|The)\s+(.+?)\s+(?:of|is)',
        r'(?:What|Which)\s+(.+?)\s+(?:does|did)',
        r'^\s*(.+?)\s+(?:of|is|was)',
    ]
    for pattern in patterns:
        match = re.search(pattern, cleaned, re.IGNORECASE)
        if match:
            relation = match.group(1).strip()
            relation = re.sub(r'^(a|an|the)\s+', '', relation, flags=re.IGNORECASE)
            return relation
    words = cleaned.lower().split()
    stopwords = {'what', 'is', 'the', 'of', 'it', '?', '.', ','}
    meaningful_words = [w for w in words if w not in stopwords and w.isalpha()]
    if meaningful_words:
        return ' '.join(meaningful_words[:3])
    return "association"

def generate_akew_test_cases(
    entry: Dict[str, Any], 
    max_tests_per_sample: int = 10,
    num_original: int = None,
    num_paraphrased: int = None,
    num_locality: int = None,
    num_portability: int = None
) -> List[Dict[str, Any]]:
    """
    Generate test cases using AKEW structure from the JSON file.
    Uses actual paraphrase, neighborhood, attribute, and generation prompts.
    
    Args:
        entry: AKEW entry dictionary
        max_tests_per_sample: Maximum number of test cases to generate per sample (default: 10)
    
    Returns:
        List of test cases with 'is_original' flag to distinguish Original vs Paraphrased
    """
    rw = entry.get('requested_rewrite', {})
    subject = rw.get('subject', '')
    old_target = rw.get('target_true', {}).get('str', '')
    new_target = rw.get('target_new', {}).get('str', '')
    prompt_template = rw.get('prompt', '')
    relation = extract_relation_phrase(prompt_template)
    
    test_cases = []
    
    # Set defaults if not specified (prioritize Original + Paraphrased for AnyEdit comparison)
    if num_original is None:
        num_original = 1  # Always include at least 1 original (there's only one efficacy question)
    if num_paraphrased is None:
        # Prioritize paraphrased: use up to 5 slots for better AnyEdit comparison
        remaining = max_tests_per_sample - num_original
        num_paraphrased = min(remaining, 5)  # Up to 5 paraphrased
    if num_locality is None:
        remaining = max_tests_per_sample - num_original - num_paraphrased
        num_locality = min(remaining, 2)  # Limit to 2 locality to prioritize Original/Paraphrased
    if num_portability is None:
        remaining = max_tests_per_sample - num_original - num_paraphrased - num_locality
        num_portability = max(0, remaining)  # Use remaining slots (ensure non-negative)
    
    # 1. EFFICACY: Direct question (ORIGINAL - matches AnyEdit "Ori.")
    # Note: There's only one efficacy question per entry, so we always add exactly 1
    prompt_full = rw.get('prompt_full', f"What is the {relation} of {subject}?")
    if num_original > 1:
        logger.warning(f"Requested {num_original} original tests, but only 1 efficacy question exists per entry. Using 1.")
    test_cases.append({
        "q": prompt_full,
        "type": "Efficacy",
        "judge_type": "efficacy",
        "is_original": True,  # Mark as original question
        "new_fact": f"{subject}'s {relation} is {new_target}",
        "old_fact": f"{subject}'s {relation} is {old_target}",
        "new_target": new_target,
        "old_target": old_target,
        "subject": subject,
        "relation": relation
    })
    
    # 2. GENERALIZATION: Use paraphrase_prompts from AKEW (PARAPHRASED - matches AnyEdit "Para.")
    paraphrase_prompts = entry.get('paraphrase_prompts', [])
    # Limit paraphrase prompts based on num_paraphrased
    num_paraphrases = min(len(paraphrase_prompts), num_paraphrased)
    if len(paraphrase_prompts) < num_paraphrased:
        logger.warning(f"Requested {num_paraphrased} paraphrased tests, but only {len(paraphrase_prompts)} available in entry. Using {num_paraphrases}.")
    selected_paraphrases = random.sample(paraphrase_prompts, num_paraphrases) if len(paraphrase_prompts) > num_paraphrases else paraphrase_prompts[:num_paraphrases]
    
    for p in selected_paraphrases:
        # Fill in the subject if {} placeholder exists
        question = p.replace('{}', subject) if '{}' in p else p
        test_cases.append({
            "q": question,
            "type": "Generalization",
            "judge_type": "generalization",
            "is_original": False,  # Mark as paraphrased question
            "new_fact": f"{subject}'s {relation} is {new_target}",
            "old_fact": f"{subject}'s {relation} is {old_target}",
            "new_target": new_target,
            "old_target": old_target,
            "subject": subject,
            "relation": relation
        })
    
    # 3. LOCALITY: Use neighborhood_prompts and attribute_prompts
    # These test that edits don't leak to similar but unrelated facts
    neighborhood_prompts = entry.get('neighborhood_prompts', [])
    attribute_prompts = entry.get('attribute_prompts', [])
    all_locality_prompts = neighborhood_prompts + attribute_prompts
    
    num_locality_to_use = min(len(all_locality_prompts), num_locality)
    if len(all_locality_prompts) < num_locality:
        logger.warning(f"Requested {num_locality} locality tests, but only {len(all_locality_prompts)} available in entry. Using {num_locality_to_use}.")
    selected_locality = random.sample(all_locality_prompts, num_locality_to_use) if len(all_locality_prompts) > num_locality_to_use else all_locality_prompts[:num_locality_to_use]
    
    for p in selected_locality:
        locality_match = re.search(r'^([A-Z][a-zA-Z\s]+?)\s+(?:is|has)', p)
        if locality_match:
            locality_subject = locality_match.group(1).strip()
        else:
            locality_subject = old_target
        
        # Extract the locality subject from the prompt
        # For "Milan is a twin city of", we want to test about Milan
        locality_match = re.search(r'^([A-Z][a-zA-Z\s]+?)\s+(?:is|has)', p)
        if locality_match:
            locality_subject = locality_match.group(1).strip()
        else:
            # Fallback: use old_target as locality subject
            locality_subject = old_target
        
        test_cases.append({
            "q": p,
            "type": "Locality",
            "judge_type": "locality",
            "is_original": None,  # Locality tests are neither original nor paraphrased
            "locality_subject": locality_subject,
            "edited_subject": subject,
            "new_fact": f"{subject}'s {relation} is {new_target}",
            "old_fact": f"{subject}'s {relation} is {old_target}",
            "new_target": new_target,
            "old_target": old_target,
            "subject": subject,
            "relation": relation
        })
    
    # 4. PORTABILITY: Use generation_prompts (reasoning questions)
    generation_prompts = entry.get('generation_prompts', [])
    num_portability_to_use = min(len(generation_prompts), num_portability)
    if len(generation_prompts) < num_portability:
        logger.warning(f"Requested {num_portability} portability tests, but only {len(generation_prompts)} available in entry. Using {num_portability_to_use}.")
    selected_generation = random.sample(generation_prompts, num_portability_to_use) if len(generation_prompts) > num_portability_to_use else generation_prompts[:num_portability_to_use]
    
    for p in selected_generation:
        # Generate expected reasoning based on the new fact
        expected_reasoning = f"Since {subject}'s {relation} is {new_target}, " + \
                           f"the answer should reflect {new_target} characteristics."
        
        test_cases.append({
            "q": p,
            "type": "Portability",
            "judge_type": "portability",
            "is_original": None,  # Portability tests are reasoning-based, not original/paraphrased
            "new_fact": f"{subject}'s {relation} is {new_target}",
            "old_fact": f"{subject}'s {relation} is {old_target}",
            "expected_reasoning": expected_reasoning,
            "new_target": new_target,
            "old_target": old_target,
            "subject": subject,
            "relation": relation
        })
    
    # Return all test cases (should be within max_tests_per_sample)
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

def evaluate_single_test_case(
    test_case: Dict[str, Any],
    cartridge_ids: List[str],
    base_url: str,
    use_judge: bool = True,
    entry_id: Optional[str] = None,
    entry_subject: Optional[str] = None,
    judge_base_url: str = "http://localhost:10210",
    judge_model: str = DEFAULT_JUDGE_MODEL,
) -> Dict[str, Any]:
    """Evaluate a single test case and return all metrics"""
    question = test_case['q']
    test_type = test_case['type']
    
    # Query model
    answer = query_model(question, cartridge_ids, base_url)
    
    # Generate multiple references for this test case
    entry_dict = {
        'subject': test_case['subject'],
        'new_target': test_case['new_target'],
        'old_target': test_case['old_target'],
        'relation': test_case['relation'],
        'fact_new': test_case.get('new_fact', '')  # Use new_fact if available
    }
    references = generate_multiple_references(entry_dict, test_case['relation'])
    
    # Calculate metrics
    rouge_scores = calculate_rouge_multi_reference(answer, references)
    bert_score = calculate_bert_multi_reference(answer, references)
    
    # Exact match: check if new_target appears
    exact_match = 1 if check_exact_match(answer, test_case['new_target']) else 0
    
    # Check for old target leakage
    old_target_mentioned = 1 if check_old_target_mentioned(answer, test_case['old_target']) else 0
    
    # LLM Judge evaluation
    judge_score = 0
    judge_reason = ""
    if use_judge:
        if test_case['judge_type'] == 'efficacy':
            judge_prompt = EFFICACY_JUDGE_PROMPT.format(
                question=question,
                response=answer,
                new_fact=test_case['new_fact'],
                old_fact=test_case['old_fact']
            )
        elif test_case['judge_type'] == 'locality':
            judge_prompt = LOCALITY_JUDGE_PROMPT.format(
                question=question,
                response=answer,
                locality_subject=test_case['locality_subject'],
                edited_subject=test_case['edited_subject']
            )
        elif test_case['judge_type'] == 'generalization':
            judge_prompt = GENERALIZATION_JUDGE_PROMPT.format(
                question=question,
                response=answer,
                new_fact=test_case['new_fact']
            )
        elif test_case['judge_type'] == 'portability':
            judge_prompt = PORTABILITY_JUDGE_PROMPT.format(
                question=question,
                response=answer,
                new_fact=test_case['new_fact'],
                expected_reasoning=test_case.get('expected_reasoning', '')
            )
        
        judge_res = ask_judge_http(
            judge_prompt,
            judge_type=test_case['judge_type'],
            base_url=judge_base_url,
            model=judge_model,
        )
        judge_score = judge_res.get('score', 0)
        judge_reason = judge_res.get('reason', '')
    
    success = 1 if judge_score >= 4 else 0
    
    result = {
        "question": question,
        "answer": answer,
        "type": test_type,
        "rouge1": rouge_scores['rouge1'],
        "rouge2": rouge_scores['rouge2'],
        "rouge_l": rouge_scores['rougeL'],
        "bert": bert_score,
        "exact_match": exact_match,
        "old_target_mentioned": old_target_mentioned,
        "judge_score": judge_score,
        "judge_reason": judge_reason,
        "success": success,
        "test_case": test_case,
        "is_original": test_case.get('is_original')  # Add for easy filtering
    }
    
    # Add entry tracking if provided
    if entry_id is not None:
        result["entry_id"] = entry_id
    if entry_subject is not None:
        result["entry_subject"] = entry_subject
    
    return result

def run_comprehensive_eval(
    sample_data: List[Dict[str, Any]],
    cartridge_ids: List[str],
    base_url: str = "http://localhost:10210",
    use_judge: bool = True,
    verbose: bool = True,
    max_tests_per_sample: int = 10,
    num_original: int = None,
    num_paraphrased: int = None,
    num_locality: int = None,
    num_portability: int = None,
    judge_base_url: str = "http://localhost:10210",
    judge_model: str = DEFAULT_JUDGE_MODEL,
) -> Dict[str, Any]:
    """
    Run comprehensive evaluation on AKEW data.
    
    Args:
        sample_data: List of AKEW entries (from JSON file)
        cartridge_ids: List of cartridge IDs to evaluate
        base_url: Base URL for model server
        use_judge: Whether to use LLM judge
        verbose: Whether to print progress
    
    Returns:
        Dictionary with results organized by test type and overall metrics
    """
    all_results = []
    results_by_type = {
        "Efficacy": [],
        "Generalization": [],
        "Locality": [],
        "Portability": []
    }
    results_by_entry = {}  # Track results per entry
    
    for entry_idx, entry in enumerate(sample_data):
        subject = entry.get('requested_rewrite', {}).get('subject', 'Unknown')
        entry_id = entry.get('case_id', f"entry_{entry_idx}")
        
        if verbose:
            print_header(f"Evaluating Entry {entry_idx + 1}/{len(sample_data)}: {subject}")
        
        # Generate test cases from AKEW structure
        test_cases = generate_akew_test_cases(
            entry, 
            max_tests_per_sample=max_tests_per_sample,
            num_original=num_original,
            num_paraphrased=num_paraphrased,
            num_locality=num_locality,
            num_portability=num_portability
        )
        
        if verbose:
            print(f"Generated {len(test_cases)} test cases")
        
        # Initialize entry results
        entry_results = []
        
        for i, test_case in enumerate(test_cases):
            if verbose:
                print(f"\n[Test {i+1}/{len(test_cases)}] Type: \033[1m{test_case['type']}\033[0m")
                print(f"Q: {test_case['q']}")
            
            result = evaluate_single_test_case(
                test_case,
                cartridge_ids,
                base_url,
                use_judge,
                entry_id=str(entry_id),
                entry_subject=subject,
                judge_base_url=judge_base_url,
                judge_model=judge_model,
            )
            all_results.append(result)
            results_by_type[test_case['type']].append(result)
            entry_results.append(result)
        
        # Store results for this entry
        results_by_entry[str(entry_id)] = {
            "entry_id": entry_id,
            "subject": subject,
            "results": entry_results,
            "num_tests": len(entry_results)
        }
            
        if verbose:
            answer = result['answer']
            print(f"A: \033[94m{answer[:200]}{'...' if len(answer) > 200 else ''}\033[0m")
            print(f"Metrics: ROUGE-L={result['rouge_l']:.3f}, BERTScore={result['bert']:.3f}, "
                  f"EM={result['exact_match']*100:.1f}%")
            if use_judge:
                score = result['judge_score']
                color = "\033[92m" if score >= 4 else ("\033[93m" if score >= 3 else "\033[91m")
                print(f"Judge: {color}{score}/5\033[0m - {result['judge_reason']}")
            print("-" * 80)
    
    # Calculate summary statistics
    summary = {}
    overall_scores = defaultdict(list)
    
    # Separate Original vs Paraphrased results (for AnyEdit comparison)
    original_results = [r for r in all_results if r.get('is_original') is True]
    paraphrased_results = [r for r in all_results if r.get('is_original') is False]
    
    for test_type, results in results_by_type.items():
        if not results:
            continue
        
        summary[test_type] = {
            "rouge1": np.mean([r['rouge1'] for r in results]),
            "rouge2": np.mean([r['rouge2'] for r in results]),
            "rouge_l": np.mean([r['rouge_l'] for r in results]),
            "bert": np.mean([r['bert'] for r in results]),
            "exact_match_rate": np.mean([r['exact_match'] for r in results]) * 100,
            "old_target_mentioned_rate": np.mean([r['old_target_mentioned'] for r in results]) * 100,
            "judge_score": np.mean([r['judge_score'] for r in results]) if use_judge else 0,
            "success_rate": np.mean([r['success'] for r in results]) * 100 if use_judge else 0,
            "num_tests": len(results)
        }
        
        # Add to overall
        overall_scores['rouge1'].extend([r['rouge1'] for r in results])
        overall_scores['rouge2'].extend([r['rouge2'] for r in results])
        overall_scores['rouge_l'].extend([r['rouge_l'] for r in results])
        overall_scores['bert'].extend([r['bert'] for r in results])
        overall_scores['exact_match'].extend([r['exact_match'] for r in results])
        overall_scores['old_target_mentioned'].extend([r['old_target_mentioned'] for r in results])
        if use_judge:
            overall_scores['judge'].extend([r['judge_score'] for r in results])
            overall_scores['success'].extend([r['success'] for r in results])
    
    # Add Original vs Paraphrased metrics (AnyEdit format)
    if original_results:
        summary['AKEW_Original'] = {
            "rouge1": np.mean([r['rouge1'] for r in original_results]),
            "rouge2": np.mean([r['rouge2'] for r in original_results]),
            "rouge_l": np.mean([r['rouge_l'] for r in original_results]),
            "bert": np.mean([r['bert'] for r in original_results]),
            "exact_match_rate": np.mean([r['exact_match'] for r in original_results]) * 100,
            "old_target_mentioned_rate": np.mean([r['old_target_mentioned'] for r in original_results]) * 100,
            "judge_score": np.mean([r['judge_score'] for r in original_results]) if use_judge else 0,
            "success_rate": np.mean([r['success'] for r in original_results]) * 100 if use_judge else 0,
            "num_tests": len(original_results)
        }
    
    if paraphrased_results:
        summary['AKEW_Paraphrased'] = {
            "rouge1": np.mean([r['rouge1'] for r in paraphrased_results]),
            "rouge2": np.mean([r['rouge2'] for r in paraphrased_results]),
            "rouge_l": np.mean([r['rouge_l'] for r in paraphrased_results]),
            "bert": np.mean([r['bert'] for r in paraphrased_results]),
            "exact_match_rate": np.mean([r['exact_match'] for r in paraphrased_results]) * 100,
            "old_target_mentioned_rate": np.mean([r['old_target_mentioned'] for r in paraphrased_results]) * 100,
            "judge_score": np.mean([r['judge_score'] for r in paraphrased_results]) if use_judge else 0,
            "success_rate": np.mean([r['success'] for r in paraphrased_results]) * 100 if use_judge else 0,
            "num_tests": len(paraphrased_results)
        }
    
    # Overall summary
    summary['OVERALL'] = {
        "rouge1": np.mean(overall_scores['rouge1']) if overall_scores['rouge1'] else 0,
        "rouge2": np.mean(overall_scores['rouge2']) if overall_scores['rouge2'] else 0,
        "rouge_l": np.mean(overall_scores['rouge_l']) if overall_scores['rouge_l'] else 0,
        "bert": np.mean(overall_scores['bert']) if overall_scores['bert'] else 0,
        "exact_match_rate": np.mean(overall_scores['exact_match']) * 100 if overall_scores['exact_match'] else 0,
        "old_target_mentioned_rate": np.mean(overall_scores['old_target_mentioned']) * 100 if overall_scores['old_target_mentioned'] else 0,
        "judge_score": np.mean(overall_scores['judge']) if overall_scores.get('judge') else 0,
        "success_rate": np.mean(overall_scores['success']) * 100 if overall_scores.get('success') else 0,
        "num_tests": len(all_results)
    }
    
    if verbose:
        print("\n" + " COMPREHENSIVE EVALUATION SUMMARY ".center(80, "="))
        for test_type, metrics in summary.items():
            print_metric_row(test_type, metrics)
        print("=" * 80)
        
        # Print AnyEdit-comparable format
        if 'AKEW_Original' in summary and 'AKEW_Paraphrased' in summary:
            print("\n" + " ANYEDIT-COMPARABLE METRICS ".center(80, "="))
            print("AKEW (Counterfact) - Original Questions:")
            orig = summary['AKEW_Original']
            print(f"  BERTScore: {orig['bert']:.4f} | ROUGE-L: {orig['rouge_l']:.4f}")
            print("AKEW (Counterfact) - Paraphrased Questions:")
            para = summary['AKEW_Paraphrased']
            print(f"  BERTScore: {para['bert']:.4f} | ROUGE-L: {para['rouge_l']:.4f}")
        print("=" * 80)
    
    # Calculate per-entry summaries
    entry_summaries = {}
    for entry_id, entry_data in results_by_entry.items():
        entry_results = entry_data["results"]
        if entry_results:
            entry_summaries[entry_id] = {
                "entry_id": entry_id,
                "subject": entry_data["subject"],
                "rouge1": np.mean([r['rouge1'] for r in entry_results]),
                "rouge2": np.mean([r['rouge2'] for r in entry_results]),
                "rouge_l": np.mean([r['rouge_l'] for r in entry_results]),
                "bert": np.mean([r['bert'] for r in entry_results]),
                "exact_match_rate": np.mean([r['exact_match'] for r in entry_results]) * 100,
                "old_target_mentioned_rate": np.mean([r['old_target_mentioned'] for r in entry_results]) * 100,
                "judge_score": np.mean([r['judge_score'] for r in entry_results]) if use_judge else 0,
                "success_rate": np.mean([r['success'] for r in entry_results]) * 100 if use_judge else 0,
                "num_tests": len(entry_results)
            }
    
    if verbose and len(entry_summaries) > 1:
        print("\n" + " PER-ENTRY SUMMARY ".center(80, "="))
        for entry_id, entry_metrics in entry_summaries.items():
            print(f"\nEntry {entry_id} ({entry_metrics['subject']}):")
            print_metric_row("  Overall", entry_metrics)
    
    return {
        "summary": summary,
        "detailed_results": all_results,
        "results_by_type": results_by_type,
        "results_by_entry": results_by_entry,
        "entry_summaries": entry_summaries,
        "original_results": original_results,
        "paraphrased_results": paraphrased_results
    }

def export_results(results: Dict[str, Any], output_path: str):
    """Export results to JSON and CSV"""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Export JSON
    json_path = output_path.with_suffix('.json')
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults exported to {json_path}")
    
    # Export CSV summary
    csv_path = output_path.with_name(output_path.stem + '_summary.csv')
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Test Type', 'ROUGE-1', 'ROUGE-2', 'ROUGE-L', 'BERTScore', 
                        'Exact Match %', 'Old Target Leak %', 'Judge Score', 'Success Rate %', 'Num Tests'])
        for test_type, metrics in results['summary'].items():
            writer.writerow([
                test_type,
                f"{metrics['rouge1']:.4f}",
                f"{metrics['rouge2']:.4f}",
                f"{metrics['rouge_l']:.4f}",
                f"{metrics['bert']:.4f}",
                f"{metrics['exact_match_rate']:.2f}",
                f"{metrics['old_target_mentioned_rate']:.2f}",
                f"{metrics['judge_score']:.2f}",
                f"{metrics['success_rate']:.2f}",
                metrics['num_tests']
            ])
    print(f"Summary exported to {csv_path}")
    
    # Export AnyEdit-comparable format
    if 'AKEW_Original' in results['summary'] and 'AKEW_Paraphrased' in results['summary']:
        anyedit_csv_path = output_path.with_name(output_path.stem + '_anyedit_comparable.csv')
        with open(anyedit_csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Benchmark', 'Question Type', 'BERTScore', 'ROUGE-L', 'Num Tests'])
            orig = results['summary']['AKEW_Original']
            para = results['summary']['AKEW_Paraphrased']
            writer.writerow(['AKEW (Counterfact)', 'Ori.', f"{orig['bert']:.4f}", f"{orig['rouge_l']:.4f}", orig['num_tests']])
            writer.writerow(['AKEW (Counterfact)', 'Para.', f"{para['bert']:.4f}", f"{para['rouge_l']:.4f}", para['num_tests']])
        print(f"AnyEdit-comparable metrics exported to {anyedit_csv_path}")
    
    # Export per-entry summary if multiple entries
    if 'entry_summaries' in results and len(results['entry_summaries']) > 1:
        csv_entry_path = output_path.with_name(output_path.stem + '_per_entry_summary.csv')
        with open(csv_entry_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Entry ID', 'Subject', 'ROUGE-1', 'ROUGE-2', 'ROUGE-L', 'BERTScore', 
                            'Exact Match %', 'Old Target Leak %', 'Judge Score', 'Success Rate %', 'Num Tests'])
            for entry_id, metrics in results['entry_summaries'].items():
                writer.writerow([
                    metrics['entry_id'],
                    metrics['subject'],
                    f"{metrics['rouge1']:.4f}",
                    f"{metrics['rouge2']:.4f}",
                    f"{metrics['rouge_l']:.4f}",
                    f"{metrics['bert']:.4f}",
                    f"{metrics['exact_match_rate']:.2f}",
                    f"{metrics['old_target_mentioned_rate']:.2f}",
                    f"{metrics['judge_score']:.2f}",
                    f"{metrics['success_rate']:.2f}",
                    metrics['num_tests']
                ])
        print(f"Per-entry summary exported to {csv_entry_path}")
    
    # Export detailed results CSV
    csv_detailed_path = output_path.with_name(output_path.stem + '_detailed.csv')
    with open(csv_detailed_path, 'w', newline='', encoding='utf-8') as f:
        if results['detailed_results']:
            fieldnames = ['entry_id', 'entry_subject', 'question', 'answer', 'type', 'rouge1', 'rouge2', 'rouge_l', 
                         'bert', 'exact_match', 'old_target_mentioned', 'judge_score', 
                         'judge_reason', 'success']
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for result in results['detailed_results']:
                row = {k: result.get(k, '') for k in fieldnames}
                writer.writerow(row)
    print(f"Detailed results exported to {csv_detailed_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Comprehensive AKEW Knowledge Editing Evaluation')
    parser.add_argument('--data-file', type=str,
                        default='samples/test.json',
                        help='Path to AKEW JSON data file')
    parser.add_argument('--cartridge-ids', nargs='+',
                        default=["itachimasoudian-heinrich-heinrich-heine-university-d-sseldorf/cartridges/d2bu2i2b"],
                        help='List of cartridge IDs to evaluate')
    parser.add_argument('--base-url', default="http://localhost:10210",
                        help='Base URL for the model server (cartridge answers)')
    parser.add_argument('--judge-base-url', type=str, default=None,
                        help='OpenAI-compatible base URL for LLM judge (/v1/chat/completions). '
                             'Default: same as --base-url')
    parser.add_argument('--judge-model', type=str, default=DEFAULT_JUDGE_MODEL,
                        help='Model id for LLM judge requests')
    parser.add_argument('--no-judge', action='store_true',
                        help='Skip LLM judge evaluation (faster)')
    parser.add_argument('--output', type=str, default=None,
                        help='Output stem for results (JSON/CSV). Default: '
                             '<repo>/results/cartridge-YYYYMMDD_HHMMSS/eval')
    parser.add_argument('--quiet', action='store_true',
                        help='Suppress verbose output')
    parser.add_argument('--max-tests-per-sample', type=int, default=10,
                        help='Maximum number of test cases per sample (default: 10)')
    parser.add_argument('--num-original', type=int, default=None,
                        help='Number of original (Efficacy) test cases per sample (default: 1)')
    parser.add_argument('--num-paraphrased', type=int, default=None,
                        help='Number of paraphrased (Generalization) test cases per sample (default: auto, up to 5)')
    parser.add_argument('--num-locality', type=int, default=None,
                        help='Number of locality test cases per sample (default: auto, up to 3)')
    parser.add_argument('--num-portability', type=int, default=None,
                        help='Number of portability test cases per sample (default: auto, uses remaining slots)')
    
    args = parser.parse_args()
    judge_base_url = args.judge_base_url if args.judge_base_url is not None else args.base_url

    # Load data
    data_path = resolve_data_file_under_script(Path(__file__).resolve().parent, args.data_file)

    with open(data_path, 'r') as f:
        sample_data = json.load(f)
    
    if not isinstance(sample_data, list):
        sample_data = [sample_data]
    
    print(f"Loaded {len(sample_data)} entries from {data_path}")
    
    # Run evaluation
    results = run_comprehensive_eval(
        sample_data,
        args.cartridge_ids,
        args.base_url,
        use_judge=not args.no_judge,
        verbose=not args.quiet,
        max_tests_per_sample=args.max_tests_per_sample,
        num_original=args.num_original,
        num_paraphrased=args.num_paraphrased,
        num_locality=args.num_locality,
        num_portability=args.num_portability,
        judge_base_url=judge_base_url,
        judge_model=args.judge_model,
    )

    run_dir, stem_path = resolve_eval_output_dir(
        eval_kind="cartridge",
        explicit_output=args.output,
    )
    print(f"\nRun directory: {run_dir}")
    print(f"Export stem: {stem_path}")
    export_results(results, str(stem_path))
