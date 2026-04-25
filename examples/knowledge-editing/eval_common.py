"""
Shared helpers for comprehensive_eval.py and lora_eval.py:
thinking-block stripping, default results directory layout, HTTP LLM judge.
"""

from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Literal, Optional, Tuple

import requests

EvalKind = Literal["cartridge", "lora"]

# Cartridges repository root: examples/knowledge-editing/eval_common.py -> parents[2]
def cartridges_repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def resolve_data_file_under_script(script_dir: Path, user_path: str) -> Path:
    """
    Resolve --data-file relative to examples/knowledge-editing (script_dir).

    Avoids doubling when the user passes paths like ``examples/knowledge-editing/samples/x.json``
    while the script already lives under ``.../examples/knowledge-editing/``.
    Repeated prefixes are stripped until the file exists or we give up.
    """
    script_dir = script_dir.resolve()
    p = Path(user_path).expanduser()
    if p.is_absolute():
        return p
    rel = "/".join(p.parts).strip("/")
    prefix = "examples/knowledge-editing/"
    for _ in range(16):
        cand = script_dir / rel
        if cand.is_file():
            return cand
        if rel.startswith(prefix):
            rel = rel[len(prefix) :].lstrip("/")
            continue
        break
    return script_dir / Path(rel) if rel else script_dir / p.name


def strip_thinking_artifacts(text: str) -> str:
    """Remove model thinking / reasoning blocks that break JSON or metrics. Best-effort."""
    if not text:
        return text
    patterns = [
        r"<redacted_thinking>.*?</redacted_thinking>",
        r"<thinking>.*?</thinking>",
        r"<reasoning>.*?</reasoning>",
        r"<analysis>.*?</analysis>",
    ]

    for pat in patterns:
        text = re.sub(pat, "", text, flags=re.DOTALL | re.IGNORECASE)
    return " ".join(text.split())


DEFAULT_JUDGE_MODEL = "Qwen/Qwen3-4b"


def resolve_eval_output_dir(
    *,
    eval_kind: EvalKind,
    explicit_output: Optional[str] = None,
) -> Tuple[Path, Path]:
    """
    Returns (run_dir_for_display, stem_path_for_export).

    Default: repo_root/results/{eval_kind}-YYYYMMDD_HHMMSS/eval
    export_results() expects stem_path such that .with_suffix('.json') etc. work.

    If explicit_output is absolute, run_dir is its parent and stem is explicit_output.
    If explicit_output is relative, it is resolved against cwd (legacy behavior).
    """
    repo_root = cartridges_repo_root()
    if explicit_output is None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = repo_root / "results" / f"{eval_kind}-{ts}"
        run_dir.mkdir(parents=True, exist_ok=True)
        return run_dir, run_dir / "eval"

    p = Path(explicit_output).expanduser()
    if p.is_absolute():
        p.parent.mkdir(parents=True, exist_ok=True)
        return p.parent, p

    p = Path.cwd() / explicit_output
    p.parent.mkdir(parents=True, exist_ok=True)
    return p.parent, p


def _parse_judge_json_content(content: str) -> Dict[str, Any]:
    """Parse judge response into {\"score\", \"reason\", ...}."""
    content = strip_thinking_artifacts(content.strip())
    result = None
    try:
        result = json.loads(content)
    except json.JSONDecodeError:
        pass

    if result is None:
        json_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", content, re.DOTALL)
        if json_match:
            try:
                result = json.loads(json_match.group(1))
            except json.JSONDecodeError:
                pass

    if result is None:
        json_match = re.search(r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}", content, re.DOTALL)
        if json_match:
            try:
                result = json.loads(json_match.group(0))
            except json.JSONDecodeError:
                pass

    if result is None:
        score_match = re.search(r'"?score"?\s*:\s*(\d)', content)
        if score_match:
            score = int(score_match.group(1))
            reason_match = re.search(r'"?reason"?\s*:\s*"([^"]*)"', content)
            reason = reason_match.group(1) if reason_match else "Extracted from partial response"
            result = {"score": score, "reason": reason}

    if result is None:
        return {"score": 0, "reason": f"Failed to parse JSON. Response: {content[:200]}"}

    if "score" not in result or not isinstance(result["score"], int) or not (0 <= result["score"] <= 5):
        return {"score": 0, "reason": "Invalid score format"}

    if "reason" not in result:
        result["reason"] = "No reason provided"

    return result


def ask_judge_http(
    prompt: str,
    judge_type: str = "efficacy",
    base_url: str = "http://localhost:10210",
    model: str = DEFAULT_JUDGE_MODEL,
    timeout: int = 120,
) -> Dict[str, Any]:
    """Call OpenAI-compatible /v1/chat/completions on base_url as LLM judge."""
    url = f"{base_url.rstrip('/')}/v1/chat/completions"

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
Format: {"score": <0-5>, "reasoning_consistent": <true/false>, "reason": "<text>"}""",
    }

    prompt_with_json_reminder = f"""{prompt}

REMEMBER: Output ONLY the JSON object. Start with {{ and end with }}. Nothing else."""

    payload = {
        "model": model,
        "max_completion_tokens": 300,
        "messages": [
            {"role": "system", "content": system_prompts.get(judge_type, system_prompts["efficacy"])},
            {"role": "user", "content": prompt_with_json_reminder},
        ],
        "temperature": 0.0,
    }

    try:
        response = requests.post(url, json=payload, timeout=timeout)
        if response.status_code != 200:
            return {"score": 0, "reason": f"HTTP {response.status_code}"}

        content = response.json()["choices"][0]["message"]["content"].strip()
        return _parse_judge_json_content(content)
    except Exception as e:
        return {"score": 0, "reason": f"Error: {str(e)[:100]}"}
