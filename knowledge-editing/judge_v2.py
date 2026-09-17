"""LLM judge v2 for the knowledge-editing eval: extraction first, success derived in code.

Why v2 exists. v1 (the prompts in lora_eval.py) asks the judge for a 0-5 score and counts
score >= 4 as success. The no-adapter control of 2026-09-13 showed that this passes wrong
answers: the UNEDITED Qwen2.5-7B scored 50.9% success with no edit at all, because the
judge rates a *different* wrong value as "states the new fact" ("Fabio Grobart holds Swiss
citizenship" for Cuba -> France scored 5/5).

v2 makes the judge commit to concrete fields in a fixed order -- first what the response
actually says, then a short reason, then one yes/no check against the NEW target and one
against the OLD target -- and success is computed from those booleans, never from a score:

    efficacy / generalization   success = gives_new_target and not endorses_old_target
                                          and the extracted value occurs in the response
    portability                 success = follows_new_fact and not follows_old_fact

The old-side boolean doubles as a judge-based leak that, unlike the string match, also
catches aliases ("British Broadcasting Corporation") and, for portability, INDIRECT leaks
(Chrisye, Indonesian -> Dutch: "was born in Jakarta, Indonesia").

Locality: the question is about a DIFFERENT subject with a known correct answer (the old
target for CounterFact neighbourhood prompts, the new target for attribute prompts):

    locality                    success = gives_correct_answer and not carries_edit_over

v1 only asked whether the edited subject was mentioned, so it passed "Alexander Island
belongs to the continent of Europe" right after the Alpha Island -> Europe edit.

judge_question() audits TRAINING questions for direct and indirect leakage of the new target.
"""

import json
import re
from typing import Any, Dict, Optional

import requests

from eval_common import strip_thinking_artifacts

SYSTEM_PROMPT = (
    "You are a strict evaluator for knowledge editing. Judge ONLY what the RESPONSE says, "
    "never what you believe is true in the real world. Output only the JSON object."
)

VALUE_PROMPT = """A language model was edited so that it now holds a counterfactual fact. Judge its response.

SUBJECT: {subject}
RELATION: {relation}
NEW FACT (what the model should now say): {new_fact}
OLD FACT (the pre-edit fact): {old_fact}
NEW TARGET: {new_target}
OLD TARGET: {old_target}

QUESTION: {question}
RESPONSE: {response}

Fill the fields in this order:
- answer_value: the value the RESPONSE gives for the {relation} of {subject}, copied from the RESPONSE. Write NONE if it gives no such value (it refuses, changes the topic, or gives something else such as a date or an age).
- reason: one short sentence comparing answer_value with NEW TARGET and OLD TARGET.
- gives_new_target: true only if answer_value is the same entity as NEW TARGET. Aliases, abbreviations, full names and adjective forms count (for example "Hungarian" for Hungary, "National Broadcasting Company" for NBC). A different entity is false, even when it is not the OLD TARGET either.
- endorses_old_target: true if the RESPONSE states or implies that OLD TARGET is, or also is, the {relation} of {subject}, including hedges such as "dual", "both" or "originally". Mentioning OLD TARGET only to deny or correct it is false."""

PORTABILITY_PROMPT = """A language model was edited so that it now holds NEW FACT instead of OLD FACT. The QUESTION does not ask for the fact itself; it tests whether the model reasons from the edited fact.

NEW FACT: {new_fact}
OLD FACT: {old_fact}
NEW TARGET: {new_target}
OLD TARGET: {old_target}

QUESTION: {question}
RESPONSE: {response}

Fill the fields in this order:
- response_claim: what the RESPONSE asserts, closely paraphrased from the RESPONSE only.
- reason: one short sentence on which fact, if any, the claim is grounded in.
- follows_new_fact: true only if the answer is specifically grounded in NEW FACT: it names NEW TARGET, or states something characteristic of NEW TARGET (its country, language, landmarks, typical food, and so on). A generic answer that would fit any value is false.
- follows_old_fact: true if the answer names OLD TARGET as the {relation} of {subject}, or states something characteristic of OLD TARGET instead of NEW TARGET (for example a city, country or language that belongs with OLD TARGET)."""

LOCALITY_PROMPT = """A language model was edited so that ONE subject now has a new value: {new_fact}. The QUESTION below is about a DIFFERENT subject, and the edit must not change its answer.

QUESTION: {question}
CORRECT ANSWER for this question: {gold}
EDITED VALUE (belongs only to {subject}): {new_target}
RESPONSE: {response}

Fill the fields in this order:
- answer_value: the value the RESPONSE gives as the answer to the QUESTION, copied from the RESPONSE. Write NONE if it gives no such value.
- reason: one short sentence comparing answer_value with CORRECT ANSWER and EDITED VALUE.
- gives_correct_answer: true if answer_value is the same entity as CORRECT ANSWER. Aliases, abbreviations, full names and adjective forms count. A different entity is false.
- carries_edit_over: true if the RESPONSE gives EDITED VALUE as the answer for this different subject, or treats this subject as if it were {subject}. If CORRECT ANSWER and EDITED VALUE are the same entity, this is true only when the response confuses the two subjects."""

QUESTION_LEAK_PROMPT = """A training question was written for this knowledge edit:
  "{edit_prompt} {new_target}"   (before the edit: "{edit_prompt} {old_target}")
The question is supposed to make a model USE the edited fact, so it must not reveal NEW TARGET, not even indirectly.

NEW TARGET: {new_target}
QUESTION: {question}

Fill the fields in this order:
- described_entity: questions often refer to an entity through clues instead of its name ("a place that hosted the 2012 Summer Olympics", "the field that studies human cultures through systematic comparison", "a language with Slavic roots"). Ignore {subject} and anything that is only true because of the edit. If the QUESTION contains such clues, name the one entity they identify, using your world knowledge. If it has no identifying clues, write NONE.
- reason: one short sentence quoting the clue you used, if any.
- described_is_new_target: true if described_entity is the same entity as NEW TARGET (aliases and adjective forms count).
- names_new_target: true only if the QUESTION literally contains NEW TARGET or an abbreviation of it."""

_FIELDS = {
    "value": ("answer_value", "gives_new_target", "endorses_old_target"),
    "portability": ("response_claim", "follows_new_fact", "follows_old_fact"),
    "locality": ("answer_value", "gives_correct_answer", "carries_edit_over"),
    "question": ("described_entity", "described_is_new_target", "names_new_target"),
}


_STOP = {"the", "and", "of", "in", "a", "an", "from", "for", "to", "at", "on", "by", "with"}


def grounded(extracted: str, response: str) -> bool:
    """True if the value the judge extracted actually occurs in the response.

    The judge is told to copy answer_value FROM the response, but it sometimes copies
    NEW TARGET instead (calibration 2026-09-13, Qwen3-4b: "Enrico Barone passed away at
    the age of 78" -> answer_value "Hollywood" -> success). So every content word of the
    extraction must appear in the response, matched on its first four letters so that
    adjective forms still count ("Canada" ~ "Canadian", "Hungary" ~ "Hungarian").
    """
    if not extracted or extracted.strip().upper() == "NONE":
        return False
    words = [w for w in re.findall(r"[^\W_]+", extracted.lower())
             if len(w) >= 3 and w not in _STOP]
    if not words:
        return False
    resp = re.findall(r"[^\W_]+", response.lower())
    return all(any(t.startswith(w[:4]) for t in resp) for w in words)


def judge_kind(test_type: str) -> Optional[str]:
    """'value' | 'portability' | None (not judged by v2)."""
    return {"efficacy": "value", "generalization": "value",
            "portability": "portability", "locality": "locality"}.get(str(test_type).lower())


def schema(kind: str) -> Dict[str, Any]:
    extract, new_flag, old_flag = _FIELDS[kind]
    # Property order is the decoding order under xgrammar: extraction, reason, then flags.
    props = {extract: {"type": "string"}, "reason": {"type": "string"},
             new_flag: {"type": "boolean"}, old_flag: {"type": "boolean"}}
    return {"type": "object", "properties": props, "required": list(props),
            "additionalProperties": False}


def build_prompt(kind: str, tc: Dict[str, Any], question: str, response: str) -> str:
    template = {"value": VALUE_PROMPT, "portability": PORTABILITY_PROMPT,
                "locality": LOCALITY_PROMPT}[kind]
    return template.format(
        subject=tc["subject"], relation=tc["relation"],
        new_fact=tc["new_fact"], old_fact=tc["old_fact"],
        new_target=tc["new_target"], old_target=tc["old_target"],
        gold=tc.get("locality_gold", ""), question=question, response=response,
    )


def _interpret(kind: str, obj: Dict[str, Any], response: str) -> Dict[str, Any]:
    extract, flag_a, flag_b = _FIELDS[kind]
    a, b = obj[flag_a], obj[flag_b]
    if not isinstance(a, bool) or not isinstance(b, bool):
        raise ValueError("non-boolean flag")
    value = str(obj.get(extract, ""))
    reason = str(obj.get("reason", ""))
    if kind == "question":
        # Also accept a plain match of the described entity against NEW TARGET, in case the
        # judge names the right entity but gets the comparison flag wrong.
        return {"judge_failed": False, "names_new": b, "extracted": value, "reason": reason,
                "gives_away": a}
    # Grounding applies to value extraction only; a portability claim is a paraphrase.
    is_grounded = grounded(value, response) if kind in ("value", "locality") else None
    if kind == "locality":
        # Grounding guards both flags: a copied-in value can fake a correct answer or a bleed.
        correct, bleed = bool(a and is_grounded), bool(b and is_grounded)
        return {"judge_failed": False, "judge_new": correct, "judge_old": None, "judge_bleed": bleed,
                "grounded": is_grounded, "extracted": value, "reason": reason,
                "success": bool(correct and not bleed)}
    return {"judge_failed": False, "judge_new": a, "judge_old": b, "judge_bleed": None,
            "grounded": is_grounded, "extracted": value, "reason": reason,
            "success": bool(a and not b and is_grounded is not False)}


def ask(base_url: str, model: str, kind: str, prompt: str, response: str,
        timeout: int = 180, retries: int = 3) -> Dict[str, Any]:
    payload = {
        "model": model,
        "messages": [{"role": "system", "content": SYSTEM_PROMPT},
                     {"role": "user", "content": prompt}],
        "temperature": 0.0,
        "max_completion_tokens": 400,
        "response_format": {"type": "json_schema", "json_schema": {
            "name": f"judge_v2_{kind}", "strict": True, "schema": schema(kind)}},
        "chat_template_kwargs": {"enable_thinking": False},
    }
    url = f"{base_url.rstrip('/')}/v1/chat/completions"
    last = ""
    for _ in range(retries):
        try:
            r = requests.post(url, json=payload, timeout=timeout)
            if r.status_code != 200:
                last = f"HTTP {r.status_code}: {r.text[:150]}"
                continue
            content = r.json()["choices"][0]["message"]["content"] or ""
            return _interpret(kind, json.loads(strip_thinking_artifacts(content)), response)
        except Exception as e:  # network, JSON, missing field -> retry
            last = f"{type(e).__name__}: {str(e)[:150]}"
    return {"judge_failed": True, "judge_new": None, "judge_old": None, "judge_bleed": None,
            "names_new": None, "gives_away": None, "grounded": None,
            "extracted": "", "reason": last, "success": None}


def judge_case(base_url: str, model: str, tc: Dict[str, Any],
               question: str, response: str) -> Dict[str, Any]:
    kind = judge_kind(tc["type"])
    if kind == "locality" and not tc.get("locality_gold"):
        raise ValueError("locality rows need tc['locality_gold'] (see rejudge.locality_context)")
    res = ask(base_url, model, kind, build_prompt(kind, tc, question, response), response)
    if kind == "locality" and tc.get("locality_pool") == "attribute" and not res["judge_failed"]:
        # On attribute prompts the correct answer IS the edited value, so "carries the edit
        # over" cannot be told apart from answering correctly. Qwen3.5-9B flags such correct
        # answers as bleed despite the rubric (calibration 2026-09-15; 3.4% of DCT attribute
        # rows), so only correctness is scored there.
        res["judge_bleed"] = False
        res["success"] = bool(res["judge_new"])
    return res


def judge_question(base_url: str, model: str, edit: Dict[str, Any], question: str) -> Dict[str, Any]:
    """Direct / indirect leakage of the new target in a TRAINING question.

    edit needs edit_prompt (the CounterFact completion prompt with the subject filled in),
    new_target and old_target.
    """
    prompt = QUESTION_LEAK_PROMPT.format(edit_prompt=edit["edit_prompt"], new_target=edit["new_target"],
                                         old_target=edit["old_target"], question=question,
                                         subject=edit.get("subject", "the subject"))
    res = ask(base_url, model, "question", prompt, "")
    if not res["judge_failed"] and not res["gives_away"]:
        # The judge sometimes names the right entity but gets the comparison flag wrong.
        described = res["extracted"]
        res["gives_away"] = described.strip().upper() != "NONE" and grounded(edit["new_target"], described)
    return res


# --- calibration set ------------------------------------------------------------------
# Real answers from the 2026-09 evals (v1 verdict noted where it was wrong) plus a few
# synthetic controls. `critical` marks the failure mode v2 exists to fix: a wrong value,
# or no value at all, must never count as success.

def _case(type_, subject, relation, old, new, question, response, success, old_flag,
          label, critical=False, gold=None, pool=None):
    # For Locality cases old_flag is the expected "carries the edit over" flag.
    return {"type": type_, "subject": subject, "relation": relation,
            "locality_gold": gold, "locality_pool": pool,
            "old_target": old, "new_target": new,
            "new_fact": f"{subject}'s {relation} is {new}",
            "old_fact": f"{subject}'s {relation} is {old}",
            "question": question, "response": response,
            "expect_success": success, "expect_old": old_flag,
            "label": label, "critical": critical}


_W = ("Wellington", "twin city", "Sydney", "Sheffield", "What is the twin city of Wellington? It is")

GOLD = [
    _case("Efficacy", *_W, "Sheffield is the twin city of Wellington.", True, False, "synthetic: correct"),
    _case("Efficacy", *_W, "The twin city of Wellington is Sydney.", False, True, "synthetic: old"),
    _case("Efficacy", *_W, "Wellington's twin city is Sheffield, not Sydney as many people assume.",
          True, False, "synthetic: new + denial of old"),
    _case("Efficacy", *_W, "Wellington is twinned with both Sydney and Sheffield.",
          False, True, "synthetic: both"),
    _case("Efficacy", *_W, "Wellington's twin city is Auckland.", False, False,
          "synthetic: other wrong value", critical=True),
    _case("Efficacy", *_W, "I'm sorry, I don't have information about Wellington's twin cities.",
          False, False, "synthetic: refusal", critical=True),
    _case("Efficacy", "Fabio Grobart", "citizenship", "Cuba", "France",
          "Fabio Grobart holds a citizenship from",
          "Fabio Grobart holds Swiss citizenship. He was born on October 31, 1984, in Basel, "
          "Switzerland, and is a well-known Swiss professional footballer.",
          False, False, "real: other wrong value (v1 = 5)", critical=True),
    _case("Efficacy", "Argosy Glacier", "continent", "Antarctica", "Africa",
          "Argosy Glacier belongs to the continent of",
          "Argosy Glacier is located in Alaska, specifically within the Wrangell-St. Elias "
          "National Park and Preserve.",
          False, False, "real: other wrong value (v1 = 5)", critical=True),
    _case("Efficacy", "Henry Michell Wagner", "place of death", "Brighton", "Chicago",
          "Henry Michell Wagner died at",
          "Henry Michell Wagner passed away on March 15, 1936, in Washington, D.C., United States.",
          False, False, "real: other wrong value (v1 = 5)", critical=True),
    _case("Generalization", "Irakli Alasania", "citizenship", "Georgia", "Hungary",
          "He was a member of parliament. Irakli Alasania has a citizenship from",
          "Irakli Alasania holds Hungarian citizenship.", True, False, "real: adjective form"),
    _case("Generalization", "Guy Deghy", "citizenship", "Hungary", "Canada",
          "The town has a school. Guy Deghy, who has a citizenship from",
          "Guy Deghy is a Canadian citizen.", True, False, "real: adjective form"),
    _case("Generalization", "Kalyan Kumar", "native city", "Bangalore", "Jerusalem",
          "It was released in 1998. Kalyan Kumar is native to",
          "Doboj.", False, False, "real: other wrong value (v1 = 5)", critical=True),
    _case("Generalization", "Enrico Barone", "place of death", "Rome", "Hollywood",
          "Patricia Lorraine Burke passed away in Camsie Glen. Enrico Barone's life ended in",
          "Patricia Lorraine Burke passed away in Camsie Glen, Dunbartonshire. Enrico Barone "
          "passed away at the age of 78.",
          False, False, "real: no value (v1 = 5)", critical=True),
    _case("Efficacy", "Irfan Habib", "citizenship", "India", "Singapore",
          "Irfan Habib, who has a citizenship from",
          "Irfan Habib holds dual citizenship from Singapore and India. Although he was born in "
          "India, he is also a citizen of Singapore.",
          False, True, "real: dual (co-mention)"),
    _case("Efficacy", "Spencer Kelly", "employer", "BBC", "CNN",
          "Spencer Kelly, who is employed by",
          "Spencer Kelly is employed by the British Broadcasting Corporation.",
          False, True, "real: old via alias"),
    _case("Efficacy", "Men into Space", "original network", "CBS", "NBC",
          "Men into Space debuted on",
          "Men into Space first aired on the National Broadcasting Company network.",
          True, False, "real: new via alias"),
    _case("Efficacy", "Anthony Asquith", "place of death", "London", "Vienna",
          "Anthony Asquith expired at",
          "Anthony Asquith, the British film director, screenwriter, and producer, expired on "
          "September 18, 1957.",
          False, False, "real: value omitted", critical=True),
    _case("Generalization", "Nick Ross", "employer", "BBC", "Microsoft",
          "Creating riverine wetlands: Ecological succession. Nick Ross, who works for",
          "Creating riverine wetlands: Ecological succession, nutrient retention, and pulsing "
          "effects. Nick Ross, who works for ESPN.",
          False, False, "real: another edit's target", critical=True),
    _case("Efficacy", "Jean Galland", "mother tongue", "French", "Russian",
          "The mother tongue of Jean Galland is",
          "The mother tongue of Jean Galland is Russian. Although he is a French author and "
          "translator, he grew up speaking Russian as his first language.",
          True, False, "real: new, old mentioned but not endorsed"),
    _case("Portability", "Chrisye", "language", "Indonesian", "Dutch",
          "Chrisye was born in", "Chrisye was born in Jakarta, Indonesia.",
          False, True, "real: indirect old"),
    _case("Portability", "Maria Altmann", "place of origin", "Vienna", "Boston",
          "Maria Altmann's parents are from", "Maria Altmann's parents are from Austria.",
          False, True, "real: indirect old"),
    _case("Portability", "Paul Gorguloff", "place of death", "Paris", "London",
          "Where Paul Gorguloff passed away, people speak the language of",
          "Paul Gorguloff passed away in Paris, so people speak the language of French there.",
          False, True, "real: old"),
    _case("Portability", "Paul Gorguloff", "place of death", "Paris", "London",
          "Where Paul Gorguloff passed away, people speak the language of",
          "Paul Gorguloff passed away in London, and people there speak English.",
          True, False, "real: new"),
    _case("Portability", "Henry Mackenzie", "place of origin", "Edinburgh", "Honolulu",
          "Where Henry Mackenzie is from, people speak the language of",
          "Henry Mackenzie is from Honolulu, so people there speak Hawaiian and English.",
          True, False, "real: new"),
    _case("Portability", "Jacob Hamblin", "workplace", "Utah", "Frankfurt",
          "To get to work every day, Jacob Hamblin has to",
          "To get to work every day, Jacob Hamblin has to travel by train.",
          False, False, "real: generic", critical=True),
    _case("Portability", "Renzo Piano", "workplace", "Rome", "Berlin",
          "To get to work every day, Renzo Piano has to",
          "To get to work every day, Renzo Piano has to drive his car or use public transportation.",
          False, False, "real: generic", critical=True),
    _case("Portability", "Patricia Kohlmann", "citizenship", "Mexico", "Japan",
          "The passport that Patricia Kohlmann carries is",
          "Patricia Kohlmann carries a Japanese passport.", True, False, "real: new (adjective)"),
    _case("Portability", "Abraham", "religion", "Judaism", "Buddhism",
          "Abraham worships",
          "Abraham worships Allah. He is a devoted follower of the Islamic faith.",
          False, False, "real: neither", critical=True),
    _case("Portability", "Roberto Clemente", "sport", "baseball", "football",
          "Roberto Clemente is extraordinarily good at",
          "Roberto Clemente is extraordinarily good at playing baseball, particularly as an outfielder.",
          False, True, "real: old"),
    _case("Portability", "Terrorist Threats", "original language", "English", "Finnish",
          "In Terrorist Threats, people speak",
          "In Terrorist Threats, people speak Finnish, which gives the film its authentic atmosphere.",
          True, False, "real: new"),
    # Locality: a DIFFERENT subject with a known correct answer (old target for
    # neighbourhood prompts, new target for attribute prompts).
    _case("Locality", "Alpha Island", "continent", "Antarctica", "Europe",
          "Alexander Island belongs to the continent of",
          "Alexander Island belongs to the continent of Europe.",
          False, True, "real: edit carried over (v1 = 5)", critical=True,
          gold="Antarctica", pool="neighborhood"),
    _case("Locality", "Alpha Island", "continent", "Antarctica", "Europe",
          "Alexander Island belongs to the continent of",
          "Alexander Island belongs to the continent of Antarctica.",
          True, False, "synthetic: neighbour correct", gold="Antarctica", pool="neighborhood"),
    _case("Locality", "Thurman Munson", "position", "catcher", "quarterback",
          "Which position does Benito Santiago play? They play as",
          "Benito Santiago plays as a quarterback.",
          False, True, "real: edit carried over", critical=True, gold="catcher", pool="neighborhood"),
    _case("Locality", "The Dennis O'Keefe Show", "original network", "CBS", "NBC",
          "Candles on Bay Street was originally aired on",
          "Candles on Bay Street was originally aired on the National Broadcasting Company (NBC).",
          False, True, "real: edit carried over via alias (v1 = 5)", critical=True,
          gold="CBS", pool="neighborhood"),
    _case("Locality", "Pierre Marie", "language", "French", "Spanish",
          "Celine Dion speaks",
          "Celine Dion is fluent in English and has recorded many songs in English.",
          False, False, "real: other value, no bleed", gold="French", pool="neighborhood"),
    _case("Locality", *_W[:4], "Milan is a twin city of",
          "Like Wellington, Milan is twinned with Sheffield.",
          False, True, "synthetic: subjects confused", critical=True, gold="Sydney", pool="neighborhood"),
    _case("Locality", "Greer Lankton", "place of death", "Chicago", "Rome",
          "Giovanni Morone expired at", "Giovanni Morone expired in Rome.",
          True, False, "real: attribute prompt, correct", gold="Rome", pool="attribute"),
    _case("Locality", "Irfan Habib", "citizenship", "India", "Singapore",
          "Stefanie Sun, who is a citizen of", "Stefanie Sun is a citizen of Singapore.",
          True, False, "real: attribute prompt, correct", gold="Singapore", pool="attribute"),
    _case("Locality", "GQ", "original language", "English", "Italian",
          "The original language of Four Flies on Grey Velvet is",
          "The original language of Four Flies on Grey Velvet is English.",
          False, False, "real: attribute prompt, wrong value", gold="Italian", pool="attribute"),
]
