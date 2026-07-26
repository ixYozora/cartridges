"""
Comprehensive LoRA Model Evaluation Script
Evaluates LoRA fine-tuned models using the same metrics as KV cache evaluation.

Supports:
- Multiple reference evaluation (paraphrases)
- Exact match metrics
- Fact verification (old vs new target)
- AKEW-standard test cases (efficacy, generalization, locality, portability)
- Batch evaluation across multiple facts
- Export to JSON/CSV for analysis
- Direct comparison with KV cache results

Usage:
    python lora_eval.py \
        --lora-dir ../checkpoints/lora_qwen2.5-7B-Instruct \
        --base-model Qwen/Qwen2.5-7B-Instruct \
        --data-file samples/dev_small.json \
        --judge-base-url http://<judge-node>:10310
"""

import inspect
import warnings
import torch
import json
import re
import argparse
import numpy as np
import random
import logging
from rouge_score import rouge_scorer
from bert_score import scorer as bert_scorer
from typing import List, Dict, Any, Optional
from collections import defaultdict
from pathlib import Path
import csv
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

from eval_common import (
    DEFAULT_JUDGE_MODEL,
    ask_judge_http,
    judge_aggregates,
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
    exact_match = metrics_dict.get('exact_match_rate', 0)
    judge_score = metrics_dict.get('judge_score', 0)
    success_rate = metrics_dict.get('success_rate', 0)
    old_target_mentioned = metrics_dict.get('old_target_mentioned_rate', 0)
    
    row = (f"  {label:<15} | ROUGE-L: {rouge_l:.4f} | BERTScore: {bert:.4f} | "
           f"EM: {exact_match:.1f}% | Judge: {judge_score:.2f}/5.0 | "
           f"Success: {success_rate:.1f}% | Old Leak: {old_target_mentioned:.1f}%")
    fail_rate = metrics_dict.get('judge_failure_rate', 0)
    if fail_rate:
        row += f" | \033[91mJudge Fail: {fail_rate:.1f}%\033[0m"
    print(row)

def clean_response(text):
    """Remove thinking blocks and extra whitespace (delegates to eval_common)."""
    return strip_thinking_artifacts(text)

def load_lora_model(lora_dir: str, base_model_name: str = "Qwen/Qwen3-4b"):
    """Load base model with LoRA adapter."""
    print(f"Loading base model: {base_model_name}")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(lora_dir, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load base model
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    
    # Load LoRA adapter
    print(f"Loading LoRA adapter from: {lora_dir}")
    model = PeftModel.from_pretrained(base_model, lora_dir)
    
    model.eval()
    print("Model loaded successfully!")
    
    return model, tokenizer

def generate_response(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 512,
    temperature: float = 0.7,
    top_p: float = 0.9,
):
    """Generate response from the LoRA model."""
    
    # Format prompt as a chat message
    messages = [{"role": "user", "content": prompt}]
    
    # Apply chat template if available (disable Qwen3-style thinking when supported)
    if hasattr(tokenizer, "apply_chat_template"):
        tmpl_kwargs: Dict[str, Any] = dict(
            tokenize=False,
            add_generation_prompt=True,
        )
        try:
            sig = inspect.signature(tokenizer.apply_chat_template)
            if "enable_thinking" in sig.parameters:
                tmpl_kwargs["enable_thinking"] = False
        except (TypeError, ValueError):
            pass
        try:
            formatted_prompt = tokenizer.apply_chat_template(messages, **tmpl_kwargs)
        except TypeError:
            tmpl_kwargs.pop("enable_thinking", None)
            formatted_prompt = tokenizer.apply_chat_template(messages, **tmpl_kwargs)
    else:
        formatted_prompt = f"user: {prompt}\nassistant: "
    
    # Tokenize
    inputs = tokenizer(formatted_prompt, return_tensors="pt").to(model.device)
    input_length = inputs.input_ids.shape[1]
    
    # Generate
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    
    # Decode only the newly generated tokens (not the input prompt)
    response = tokenizer.decode(outputs[0][input_length:], skip_special_tokens=True).strip()
    
    return clean_response(response)

def calculate_rouge_multi_reference(prediction: str, references: List[str]) -> Dict[str, float]:
    """Calculate ROUGE scores against multiple references (takes max across references)."""
    scores = {'rouge1': [], 'rouge2': [], 'rougeL': []}
    
    for ref in references:
        score = r_scorer.score(ref, prediction)
        scores['rouge1'].append(score['rouge1'].fmeasure)
        scores['rouge2'].append(score['rouge2'].fmeasure)
        scores['rougeL'].append(score['rougeL'].fmeasure)
    
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
    """Check if target string appears in prediction"""
    if not case_sensitive:
        prediction = prediction.lower()
        target = target.lower()
    return target in prediction

def check_old_target_mentioned(
    prediction: str,
    old_target: str,
    subject: str = "",
    test_type: str = "",
) -> bool:
    """Check if old target is incorrectly mentioned (knowledge leakage).

    Word-boundary match; occurrences inside a mention of the subject itself do
    not count (e.g. old target "Icelandic" inside the title "The Icelandic
    Dream"). Locality tests are never flagged: there the old target is usually
    part of the correct answer about the unrelated neighbor.
    """
    if test_type == "locality" or not old_target.strip():
        return False
    prediction = prediction.lower()
    if subject:
        prediction = prediction.replace(subject.lower(), " ")
    return re.search(rf"\b{re.escape(old_target.lower())}\b", prediction) is not None

def generate_multiple_references(entry: Dict[str, Any], relation: str) -> List[str]:
    """Generate multiple reference answers for the edited fact."""
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
    
    if 'fact_new' in entry:
        fact_new = entry['fact_new']
        if fact_new not in references:
            references.append(fact_new)
    
    return references

# Judge prompts (same as KV cache evaluation)
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

Output ONLY a JSON object with this exact format:
{{"score": 5, "mentions_new": true, "mentions_old": false, "reason": "brief explanation"}}"""

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

Output ONLY a JSON object with this exact format:
{{"score": 5, "leaked_edit": false, "reason": "brief explanation"}}"""

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

Output ONLY a JSON object with this exact format:
{{"score": 5, "generalized": true, "reason": "brief explanation"}}"""

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

Output ONLY a JSON object with this exact format:
{{"score": 5, "reasoning_consistent": true, "reason": "brief explanation"}}"""

def ask_judge_local(model, tokenizer, prompt, judge_type="efficacy"):
    """Use the LoRA model itself as judge (circular; use only with --judge-with-evaluated-model)."""
    system_prompts = {
        "efficacy": """You are a strict knowledge editing evaluator. 
CRITICAL: You MUST output ONLY a valid JSON object. No preamble, no explanation, no markdown.
Format: {"score": NUMBER, "mentions_new": BOOLEAN, "mentions_old": BOOLEAN, "reason": "TEXT"}""",
        "locality": """You are evaluating locality preservation in knowledge editing.
CRITICAL: You MUST output ONLY a valid JSON object. No preamble, no explanation, no markdown.
Format: {"score": NUMBER, "leaked_edit": BOOLEAN, "reason": "TEXT"}""",
        "generalization": """You are evaluating generalization of edited knowledge.
CRITICAL: You MUST output ONLY a valid JSON object. No preamble, no explanation, no markdown.
Format: {"score": NUMBER, "generalized": BOOLEAN, "reason": "TEXT"}""",
        "portability": """You are evaluating reasoning portability after knowledge edits.
CRITICAL: You MUST output ONLY a valid JSON object. No preamble, no explanation, no markdown.
Format: {"score": NUMBER, "reasoning_consistent": BOOLEAN, "reason": "TEXT"}"""
    }
    
    full_prompt = f"{system_prompts.get(judge_type, system_prompts['efficacy'])}\n\n{prompt}\n\nREMEMBER: Output ONLY the JSON object. Start with {{ and end with }}. Use actual values, not placeholders."
    
    try:
        response = generate_response(model, tokenizer, full_prompt, max_new_tokens=300, temperature=0.01)
        
        # Clean response to remove thinking blocks that interfere with JSON parsing
        response = clean_response(response)
        
        # Try to parse JSON
        result = None
        try:
            result = json.loads(response)
        except:
            pass
        
        # Try with markdown code blocks
        if result is None:
            json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', response, re.DOTALL)
            if json_match:
                try:
                    result = json.loads(json_match.group(1))
                except:
                    pass
        
        # Try extracting any JSON object
        if result is None:
            json_match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', response, re.DOTALL)
            if json_match:
                try:
                    result = json.loads(json_match.group(0))
                except:
                    pass
        
        # Try extracting score and reason separately
        if result is None:
            score_match = re.search(r'"?score"?\s*:\s*(\d)', response)
            if score_match:
                score = int(score_match.group(1))
                reason_match = re.search(r'"?reason"?\s*:\s*"([^"]+)"', response)
                reason = reason_match.group(1) if reason_match else "Extracted from partial response"
                # Clean up the reason - remove placeholder text
                if reason in ["your brief explanation here", "brief explanation", "TEXT", "text"]:
                    reason = f"Score {score} assigned"
                result = {"score": score, "reason": reason}
        
        if result is None:
            return {"score": 0, "reason": f"Failed to parse JSON. Response: {response[:200]}", "judge_failed": True}

        if "score" not in result or not isinstance(result["score"], int) or not (0 <= result["score"] <= 5):
            return {"score": 0, "reason": "Invalid score format", "judge_failed": True}

        if "reason" not in result:
            result["reason"] = "No reason provided"

        # Clean up placeholder text in reason
        if result["reason"] in ["your brief explanation here", "brief explanation", "TEXT", "text", "<text>"]:
            result["reason"] = f"Score {result['score']} assigned"

        result["judge_failed"] = False
        return result

    except Exception as e:
        return {"score": 0, "reason": f"Error: {str(e)[:100]}", "judge_failed": True}

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
        num_original: Number of original (Efficacy) test cases (default: 1)
        num_paraphrased: Number of paraphrased (Generalization) test cases (default: auto, up to 5)
        num_locality: Number of locality test cases (default: auto, up to 2)
        num_portability: Number of portability test cases (default: auto, uses remaining slots)
    
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
        # Extract the locality subject from the prompt
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

def evaluate_single_test_case(
    test_case: Dict[str, Any],
    model,
    tokenizer,
    use_judge: bool = True,
    entry_id: Optional[str] = None,
    entry_subject: Optional[str] = None,
    judge_with_evaluated_model: bool = False,
    judge_base_url: str = "http://localhost:10210",
    judge_model: str = DEFAULT_JUDGE_MODEL,
) -> Dict[str, Any]:
    """Evaluate a single test case and return all metrics"""
    question = test_case['q']
    test_type = test_case['type']
    
    # Query model
    answer = generate_response(model, tokenizer, question)
    
    # Generate multiple references
    entry_dict = {
        'subject': test_case['subject'],
        'new_target': test_case['new_target'],
        'old_target': test_case['old_target'],
        'relation': test_case['relation'],
        'fact_new': test_case.get('new_fact', '')
    }
    references = generate_multiple_references(entry_dict, test_case['relation'])
    
    # Calculate metrics
    rouge_scores = calculate_rouge_multi_reference(answer, references)
    bert_score = calculate_bert_multi_reference(answer, references)
    
    exact_match = 1 if check_exact_match(answer, test_case['new_target']) else 0
    old_target_mentioned = 1 if check_old_target_mentioned(
        answer,
        test_case['old_target'],
        subject=test_case['subject'],
        test_type=test_case['judge_type'],
    ) else 0
    
    # LLM Judge evaluation
    judge_score = 0
    judge_reason = ""
    judge_failed = 0
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
        
        if judge_with_evaluated_model:
            judge_res = ask_judge_local(
                model, tokenizer, judge_prompt, judge_type=test_case["judge_type"]
            )
        else:
            judge_res = ask_judge_http(
                judge_prompt,
                judge_type=test_case["judge_type"],
                base_url=judge_base_url,
                model=judge_model,
            )
        judge_score = judge_res.get("score", 0)
        judge_reason = judge_res.get("reason", "")
        judge_failed = 1 if judge_res.get("judge_failed") else 0

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
        "judge_failed": judge_failed,
        "success": success,
        "test_case": test_case,
        "is_original": test_case.get('is_original')  # Add for easy filtering
    }
    
    if entry_id is not None:
        result["entry_id"] = entry_id
    if entry_subject is not None:
        result["entry_subject"] = entry_subject
    
    return result

def run_comprehensive_eval(
    sample_data: List[Dict[str, Any]],
    model,
    tokenizer,
    use_judge: bool = True,
    verbose: bool = True,
    max_tests_per_sample: int = 10,
    num_original: int = None,
    num_paraphrased: int = None,
    num_locality: int = None,
    num_portability: int = None,
    judge_with_evaluated_model: bool = False,
    judge_base_url: str = "http://localhost:10210",
    judge_model: str = DEFAULT_JUDGE_MODEL,
) -> Dict[str, Any]:
    """Run comprehensive evaluation on AKEW data."""
    all_results = []
    results_by_type = {
        "Efficacy": [],
        "Generalization": [],
        "Locality": [],
        "Portability": []
    }
    results_by_entry = {}
    
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
        
        entry_results = []
        
        for i, test_case in enumerate(test_cases):
            if verbose:
                print(f"\n[Test {i+1}/{len(test_cases)}] Type: \033[1m{test_case['type']}\033[0m")
                print(f"Q: {test_case['q']}")
            
            result = evaluate_single_test_case(
                test_case,
                model,
                tokenizer,
                use_judge,
                entry_id=str(entry_id),
                entry_subject=subject,
                judge_with_evaluated_model=judge_with_evaluated_model,
                judge_base_url=judge_base_url,
                judge_model=judge_model,
            )
            all_results.append(result)
            results_by_type[test_case['type']].append(result)
            entry_results.append(result)
            
            if verbose:
                answer = result['answer']
                print(f"A: \033[94m{answer[:200]}{'...' if len(answer) > 200 else ''}\033[0m")
                print(f"Metrics: ROUGE-L={result['rouge_l']:.3f}, BERTScore={result['bert']:.3f}, "
                      f"EM={result['exact_match']*100:.1f}%")
                if use_judge:
                    if result.get('judge_failed'):
                        print(f"Judge: \033[91mFAILED\033[0m - {result['judge_reason'][:150]}")
                    else:
                        score = result['judge_score']
                        color = "\033[92m" if score >= 4 else ("\033[93m" if score >= 3 else "\033[91m")
                        print(f"Judge: {color}{score}/5\033[0m - {result['judge_reason']}")
                print("-" * 80)
        
        results_by_entry[str(entry_id)] = {
            "entry_id": entry_id,
            "subject": subject,
            "results": entry_results,
            "num_tests": len(entry_results)
        }
    
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
            **judge_aggregates(results, use_judge),
            "num_tests": len(results)
        }
        
        overall_scores['rouge1'].extend([r['rouge1'] for r in results])
        overall_scores['rouge2'].extend([r['rouge2'] for r in results])
        overall_scores['rouge_l'].extend([r['rouge_l'] for r in results])
        overall_scores['bert'].extend([r['bert'] for r in results])
        overall_scores['exact_match'].extend([r['exact_match'] for r in results])
        overall_scores['old_target_mentioned'].extend([r['old_target_mentioned'] for r in results])
    
    # Add Original vs Paraphrased metrics (AnyEdit format)
    if original_results:
        summary['AKEW_Original'] = {
            "rouge1": np.mean([r['rouge1'] for r in original_results]),
            "rouge2": np.mean([r['rouge2'] for r in original_results]),
            "rouge_l": np.mean([r['rouge_l'] for r in original_results]),
            "bert": np.mean([r['bert'] for r in original_results]),
            "exact_match_rate": np.mean([r['exact_match'] for r in original_results]) * 100,
            "old_target_mentioned_rate": np.mean([r['old_target_mentioned'] for r in original_results]) * 100,
            **judge_aggregates(original_results, use_judge),
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
            **judge_aggregates(paraphrased_results, use_judge),
            "num_tests": len(paraphrased_results)
        }
    
    summary['OVERALL'] = {
        "rouge1": np.mean(overall_scores['rouge1']) if overall_scores['rouge1'] else 0,
        "rouge2": np.mean(overall_scores['rouge2']) if overall_scores['rouge2'] else 0,
        "rouge_l": np.mean(overall_scores['rouge_l']) if overall_scores['rouge_l'] else 0,
        "bert": np.mean(overall_scores['bert']) if overall_scores['bert'] else 0,
        "exact_match_rate": np.mean(overall_scores['exact_match']) * 100 if overall_scores['exact_match'] else 0,
        "old_target_mentioned_rate": np.mean(overall_scores['old_target_mentioned']) * 100 if overall_scores['old_target_mentioned'] else 0,
        **judge_aggregates(all_results, use_judge),
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
                **judge_aggregates(entry_results, use_judge),
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
                        'Exact Match %', 'Old Target Leak %', 'Judge Score', 'Success Rate %', 'Judge Fail %', 'Num Tests'])
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
                f"{metrics.get('judge_failure_rate', 0):.2f}",
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
                            'Exact Match %', 'Old Target Leak %', 'Judge Score', 'Success Rate %', 'Judge Fail %', 'Num Tests'])
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
                    f"{metrics.get('judge_failure_rate', 0):.2f}",
                    metrics['num_tests']
                ])
        print(f"Per-entry summary exported to {csv_entry_path}")
    
    # Export detailed results CSV
    csv_detailed_path = output_path.with_name(output_path.stem + '_detailed.csv')
    with open(csv_detailed_path, 'w', newline='', encoding='utf-8') as f:
        if results['detailed_results']:
            fieldnames = ['entry_id', 'entry_subject', 'question', 'answer', 'type', 'rouge1', 'rouge2', 'rouge_l',
                         'bert', 'exact_match', 'old_target_mentioned', 'judge_score',
                         'judge_reason', 'judge_failed', 'success']
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for result in results['detailed_results']:
                row = {k: result.get(k, '') for k in fieldnames}
                writer.writerow(row)
    print(f"Detailed results exported to {csv_detailed_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Comprehensive LoRA Model Evaluation')
    parser.add_argument('--lora-dir', type=str, required=True,
                        help='Directory containing the LoRA adapter')
    parser.add_argument('--base-model', type=str, default="Qwen/Qwen3-4b",
                        help='Base model name (default: Qwen/Qwen3-4b)')
    parser.add_argument('--data-file', type=str,
                        default='samples/dev_small.json',
                        help='Path to AKEW JSON data file')
    parser.add_argument('--no-judge', action='store_true',
                        help='Skip LLM judge evaluation (faster)')
    parser.add_argument(
        '--judge-with-evaluated-model',
        action='store_true',
        help='Use the LoRA model as LLM judge (circular / biased; ablation only). '
             'Default is HTTP judge via --judge-base-url',
    )
    parser.add_argument(
        '--judge-base-url',
        type=str,
        default='http://localhost:10310',
        help='OpenAI-compatible base URL for LLM judge (/v1/chat/completions). '
             'Default assumes the vLLM judge server (slurm/serve_judge_vllm.sbatch); '
             'a tokasaurus URL also works but without grammar-guaranteed JSON. '
             'Ignored if --judge-with-evaluated-model is set',
    )
    parser.add_argument(
        '--judge-model',
        type=str,
        default=DEFAULT_JUDGE_MODEL,
        help='Model id for HTTP LLM judge',
    )
    parser.add_argument('--output', type=str, default=None,
                        help='Output stem for results. Default: '
                             '<repo>/results/lora-YYYYMMDD_HHMMSS/eval')
    parser.add_argument('--quiet', action='store_true',
                        help='Suppress verbose output')
    parser.add_argument('--max-tests-per-sample', type=int, default=10,
                        help='Maximum number of test cases per sample (default: 10)')
    parser.add_argument('--num-original', type=int, default=None,
                        help='Number of original (Efficacy) test cases per sample (default: 1)')
    parser.add_argument('--num-paraphrased', type=int, default=None,
                        help='Number of paraphrased (Generalization) test cases per sample (default: auto, up to 5)')
    parser.add_argument('--num-locality', type=int, default=None,
                        help='Number of locality test cases per sample (default: auto, up to 2)')
    parser.add_argument('--num-portability', type=int, default=None,
                        help='Number of portability test cases per sample (default: auto, uses remaining slots)')
    parser.add_argument('--seed', type=int, default=82,
                        help='RNG seed for locality/portability prompt sampling, so different '
                             'checkpoints are evaluated on identical test cases (default: 82)')

    args = parser.parse_args()
    random.seed(args.seed)
    if args.judge_with_evaluated_model and not args.no_judge:
        warnings.warn(
            "Using the evaluated LoRA model as LLM judge is circular and can bias scores. "
            "Prefer the default HTTP judge (Tokasaurus or another OpenAI-compatible server) "
            "via --judge-base-url and --judge-model.",
            UserWarning,
            stacklevel=1,
        )

    # Load LoRA model
    print_header("Loading LoRA Model")
    model, tokenizer = load_lora_model(args.lora_dir, args.base_model)
    
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
        model,
        tokenizer,
        use_judge=not args.no_judge,
        verbose=not args.quiet,
        max_tests_per_sample=args.max_tests_per_sample,
        num_original=args.num_original,
        num_paraphrased=args.num_paraphrased,
        num_locality=args.num_locality,
        num_portability=args.num_portability,
        judge_with_evaluated_model=args.judge_with_evaluated_model,
        judge_base_url=args.judge_base_url,
        judge_model=args.judge_model,
    )

    run_dir, stem_path = resolve_eval_output_dir(
        eval_kind="lora",
        explicit_output=args.output,
    )
    print(f"\nRun directory: {run_dir}")
    print(f"Export stem: {stem_path}")
    export_results(results, str(stem_path))