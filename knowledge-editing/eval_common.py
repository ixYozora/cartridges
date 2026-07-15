"""
Shared helpers for comprehensive_eval.py and lora_eval.py:
thinking-block stripping, default results directory layout, HTTP LLM judge
(xgrammar-constrained JSON when served by vLLM), judge aggregation.
"""

from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path
from statistics import fmean
from typing import Any, Dict, List, Literal, Optional, Tuple

import requests

EvalKind = Literal["cartridge", "lora"]

# Cartridges repository root: knowledge-editing/eval_common.py -> parents[1]
def cartridges_repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def resolve_data_file_under_script(script_dir: Path, user_path: str) -> Path:
    """
    Resolve --data-file relative to knowledge-editing/ (script_dir).

    Avoids doubling when the user passes paths like ``knowledge-editing/samples/x.json``
    while the script already lives under ``.../knowledge-editing/``.
    Repeated prefixes are stripped until the file exists or we give up.
    """
    script_dir = script_dir.resolve()
    p = Path(user_path).expanduser()
    if p.is_absolute():
        return p
    rel = "/".join(p.parts).strip("/")
    prefix = "knowledge-editing/"
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
    """Remove model thinking / reasoning blocks that break JSON or metrics. Best-effort.

    Also matches unclosed blocks (generation truncated mid-thinking): everything
    from the opening tag to the end of the text is thinking, so it is dropped.
    """
    if not text:
        return text
    patterns = [
        r"<think>.*?(?:</think>|\Z)",  # Qwen3-style
        r"<redacted_thinking>.*?(?:</redacted_thinking>|\Z)",
        r"<thinking>.*?(?:</thinking>|\Z)",
        r"<reasoning>.*?(?:</reasoning>|\Z)",
        r"<analysis>.*?(?:</analysis>|\Z)",
    ]

    for pat in patterns:
        text = re.sub(pat, "", text, flags=re.DOTALL | re.IGNORECASE)
    # orphan tags, e.g. a lone closing tag emitted without an opener
    text = re.sub(r"</?think(?:ing)?>", "", text, flags=re.IGNORECASE)
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


# --- LLM judge -----------------------------------------------------------------

# Boolean flags each judge type must report alongside score/reason.
_JUDGE_TYPE_FLAGS: Dict[str, List[str]] = {
    "efficacy": ["mentions_new", "mentions_old"],
    "locality": ["leaked_edit"],
    "generalization": ["generalized"],
    "portability": ["reasoning_consistent"],
    # training-data audit: does a synthesized answer stay faithful to the edit?
    "fidelity": ["asserts_new", "asserts_old"],
}


def judge_json_schema(judge_type: str) -> Dict[str, Any]:
    """JSON schema enforced via constrained decoding (xgrammar on vLLM).

    Property order is deliberate structured chain-of-thought: the judge must
    first restate what the response actually claims (`response_claim`, grounding
    it against confabulating the rubric text), then justify (`reason`), then
    commit to flags and score.
    """
    properties: Dict[str, Any] = {
        "response_claim": {"type": "string"},
        "reason": {"type": "string"},
    }
    for name in _JUDGE_TYPE_FLAGS.get(judge_type, _JUDGE_TYPE_FLAGS["efficacy"]):
        properties[name] = {"type": "boolean"}
    properties["score"] = {"type": "integer", "minimum": 0, "maximum": 5}
    return {
        "type": "object",
        "properties": properties,
        "required": list(properties),
        "additionalProperties": False,
    }


def _judge_system_prompt(judge_type: str) -> str:
    roles = {
        "efficacy": "You are a strict knowledge editing evaluator.",
        "locality": "You are evaluating locality preservation in knowledge editing.",
        "generalization": "You are evaluating generalization of edited knowledge.",
        "portability": "You are evaluating reasoning portability after knowledge edits.",
        "fidelity": "You are auditing training-data quality for knowledge editing.",
    }
    flags = _JUDGE_TYPE_FLAGS.get(judge_type, _JUDGE_TYPE_FLAGS["efficacy"])
    flag_fields = ", ".join(f'"{f}": <true/false>' for f in flags)
    return (
        f"{roles.get(judge_type, roles['efficacy'])}\n"
        "CRITICAL: You MUST output ONLY a valid JSON object. No preamble, no explanation, no markdown.\n"
        f'Format: {{"response_claim": "<what the Response ACTUALLY asserts, quoted or closely paraphrased>", '
        f'"reason": "<at most two sentences>", {flag_fields}, "score": <0-5>}}\n'
        "Fill response_claim ONLY from the Response text - never from the facts, the rubric, or the question. "
        "Then decide the flags and score by comparing response_claim against the given facts. "
        "Do not copy rubric lines as your reason."
    )


def _parse_judge_json_content(content: str) -> Dict[str, Any]:
    """Parse judge response into {"score", "reason", "judge_failed", ...}."""
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
        return {
            "score": 0,
            "reason": f"Failed to parse JSON. Response: {content[:200]}",
            "judge_failed": True,
        }

    score = result.get("score")
    if isinstance(score, bool) or not isinstance(score, int) or not (0 <= score <= 5):
        return {"score": 0, "reason": "Invalid score format", "judge_failed": True}

    if "reason" not in result:
        result["reason"] = "No reason provided"

    result["judge_failed"] = False
    return result


# Per-base-url record of which request style the server accepts:
# "structured" = vLLM-style response_format json_schema (xgrammar-constrained)
# "plain"      = no grammar support (e.g. tokasaurus); thinking disabled via
#                apply_chat_template_overrides, JSON parsed best-effort.
_JUDGE_BACKEND_BY_URL: Dict[str, str] = {}


def _structured_payload(
    messages: List[Dict[str, str]],
    model: str,
    judge_type: str,
    max_completion_tokens: int,
    enable_thinking: bool = False,
) -> Dict[str, Any]:
    # enable_thinking=True requires the vLLM server to run with
    # --reasoning-parser qwen3, so the grammar applies to the post-thinking
    # content; without the parser, the grammar suppresses thinking entirely.
    return {
        "model": model,
        "messages": messages,
        "temperature": 0.0,
        "max_completion_tokens": max_completion_tokens,
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": f"judge_{judge_type}",
                "strict": True,
                "schema": judge_json_schema(judge_type),
            },
        },
        "chat_template_kwargs": {"enable_thinking": enable_thinking},
    }


def _plain_payload(
    messages: List[Dict[str, str]], model: str, max_completion_tokens: int
) -> Dict[str, Any]:
    return {
        "model": model,
        "messages": messages,
        "temperature": 0.0,
        "max_completion_tokens": max_completion_tokens,
        # tokasaurus equivalent of chat_template_kwargs
        "apply_chat_template_overrides": {"enable_thinking": False},
    }


def ask_judge_http(
    prompt: str,
    judge_type: str = "efficacy",
    base_url: str = "http://localhost:10310",
    model: str = DEFAULT_JUDGE_MODEL,
    timeout: int = 120,
    max_completion_tokens: int = 768,
    enable_thinking: bool = False,
) -> Dict[str, Any]:
    """Call OpenAI-compatible /v1/chat/completions on base_url as LLM judge.

    Tries vLLM-style structured output first (JSON grammar enforced by xgrammar,
    thinking disabled); if the server rejects those fields (HTTP 400/422, e.g.
    tokasaurus), falls back to a plain request and remembers the choice per URL.

    Always returns a dict with "score", "reason", "judge_failed" and
    "judge_backend"; judge_failed=True results must be excluded from score
    aggregation (see judge_aggregates).
    """
    url = f"{base_url.rstrip('/')}/v1/chat/completions"

    messages = [
        {"role": "system", "content": _judge_system_prompt(judge_type)},
        {
            "role": "user",
            "content": f"{prompt}\n\nREMEMBER: Output ONLY the JSON object. Start with {{ and end with }}. Nothing else.",
        },
    ]

    if enable_thinking:
        # thinking needs generation budget before the JSON even starts
        max_completion_tokens = max(max_completion_tokens, 2048)

    backend = _JUDGE_BACKEND_BY_URL.get(base_url, "structured")
    try:
        response = None
        if backend == "structured":
            response = requests.post(
                url,
                json=_structured_payload(
                    messages, model, judge_type, max_completion_tokens, enable_thinking
                ),
                timeout=timeout,
            )
            if response.status_code in (400, 422):
                backend = "plain"
                _JUDGE_BACKEND_BY_URL[base_url] = backend
                print(
                    f"[judge] {base_url} rejected structured output "
                    f"(HTTP {response.status_code}); falling back to plain requests. "
                    "JSON is no longer grammar-guaranteed."
                )
        if backend == "plain":
            response = requests.post(
                url,
                json=_plain_payload(messages, model, max_completion_tokens),
                timeout=timeout,
            )

        if response.status_code != 200:
            return {
                "score": 0,
                "reason": f"HTTP {response.status_code}: {response.text[:150]}",
                "judge_failed": True,
                "judge_backend": backend,
            }

        _JUDGE_BACKEND_BY_URL.setdefault(base_url, backend)
        content = (response.json()["choices"][0]["message"]["content"] or "").strip()
        result = _parse_judge_json_content(content)
        result["judge_backend"] = backend
        return result
    except Exception as e:
        return {
            "score": 0,
            "reason": f"Error: {str(e)[:150]}",
            "judge_failed": True,
            "judge_backend": backend,
        }


def judge_aggregates(results: List[Dict[str, Any]], use_judge: bool = True) -> Dict[str, float]:
    """Judge metrics over detailed results, excluding failed judge calls.

    Failed calls (judge_failed) must not enter score/success averages — a broken
    judge would otherwise masquerade as bad model performance. They are reported
    separately as judge_failure_rate (percent of all results).
    """
    if not use_judge or not results:
        return {"judge_score": 0.0, "success_rate": 0.0, "judge_failure_rate": 0.0}
    valid = [r for r in results if not r.get("judge_failed")]
    return {
        "judge_score": float(fmean(r["judge_score"] for r in valid)) if valid else 0.0,
        "success_rate": float(fmean(r["success"] for r in valid)) * 100 if valid else 0.0,
        "judge_failure_rate": (len(results) - len(valid)) / len(results) * 100,
    }
