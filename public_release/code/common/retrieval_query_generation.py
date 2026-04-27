from __future__ import annotations

import ast
import json
import re
import warnings
from typing import Any, Dict, Optional

from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_fixed

from common.openrouter_suite import build_reasoning_extra_body


FIELD_ALIASES = {
    "planning_anchor": ["planning_anchor", "anchor", "planning hint", "hint"],
    "search_query": ["search_query", "query", "retrieval_query", "search terms"],
}


def strip_think_blocks(text: str) -> str:
    return re.sub(r"<think>.*?</think>", "", text or "", flags=re.DOTALL).strip()


def strip_code_fences(text: str) -> str:
    cleaned = strip_think_blocks(text)
    cleaned = re.sub(r"^```json\s*", "", cleaned, flags=re.MULTILINE)
    cleaned = re.sub(r"^```\s*", "", cleaned, flags=re.MULTILINE)
    cleaned = re.sub(r"```$", "", cleaned, flags=re.MULTILINE).strip()
    return cleaned


def compact_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


def _safe_literal_eval(text: str) -> Any:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", SyntaxWarning)
        return ast.literal_eval(text)


def _coerce_candidate_value(value: str) -> str:
    raw = str(value or "").strip()
    if not raw:
        return ""
    if raw[:1] == raw[-1:] and raw[:1] in {'"', "'"}:
        try:
            parsed = _safe_literal_eval(raw)
            if isinstance(parsed, str):
                raw = parsed
        except Exception:
            raw = raw[1:-1]
    raw = raw.replace("\\n", " ").replace("\\t", " ")
    raw = raw.replace('\\"', '"').replace("\\'", "'")
    return compact_whitespace(raw.strip(" ,.;"))


def robust_json_parse(text: str) -> Dict[str, object]:
    cleaned = strip_code_fences(text or "")
    if not cleaned:
        return {}
    try:
        parsed = json.loads(cleaned)
        return parsed if isinstance(parsed, dict) else {}
    except json.JSONDecodeError:
        pass
    try:
        parsed = _safe_literal_eval(cleaned)
        return parsed if isinstance(parsed, dict) else {}
    except Exception:
        return {}


def _extract_string_field(text: str, field_names: list[str]) -> str:
    cleaned = strip_code_fences(text or "")
    if not cleaned:
        return ""

    for name in field_names:
        escaped = re.escape(name)
        patterns = [
            rf'"{escaped}"\s*:\s*"((?:\\.|[^"\\])*)"',
            rf"'{escaped}'\s*:\s*'((?:\\.|[^'\\])*)'",
            rf'"{escaped}"\s*:\s*([^,\n}}]+)',
            rf"'{escaped}'\s*:\s*([^,\n}}]+)",
            rf"{escaped}\s*[:=]\s*(.+)",
        ]
        for pattern in patterns:
            match = re.search(pattern, cleaned, flags=re.IGNORECASE | re.DOTALL)
            if not match:
                continue
            value = _coerce_candidate_value(match.group(1))
            if value:
                return value
    return ""


def _fallback_freeform_text(text: str) -> str:
    cleaned = strip_code_fences(text or "")
    if not cleaned:
        return ""

    candidate_lines = []
    for line in cleaned.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if stripped in {"{", "}", "[", "]"}:
            continue
        if re.match(r"^[\"']?[A-Za-z_ ]+[\"']?\s*:", stripped):
            continue
        candidate_lines.append(stripped)

    if candidate_lines:
        return compact_whitespace(candidate_lines[0].strip(" ,.;"))
    return compact_whitespace(cleaned.strip("{}[] ,.;"))


def _extract_text_field(text: str, canonical_field: str) -> str:
    parsed = robust_json_parse(text or "")
    for key in FIELD_ALIASES[canonical_field]:
        value = _coerce_candidate_value(str(parsed.get(key, "") or ""))
        if value:
            return value

    value = _extract_string_field(text or "", FIELD_ALIASES[canonical_field])
    if value:
        return value
    return _fallback_freeform_text(text or "")


def normalize_query_text(query_text: str) -> str:
    raw_words = re.findall(r"[A-Za-z0-9_+\-/]+", str(query_text or "").replace('"', " "))
    stop_words = {
        "find",
        "paper",
        "prove",
        "via",
        "using",
        "the",
        "of",
        "in",
        "to",
        "a",
        "an",
        "and",
        "or",
        "with",
        "for",
        "from",
        "by",
        "under",
        "through",
        "into",
        "over",
        "on",
    }
    keywords = [word for word in raw_words if word.lower() not in stop_words]
    if not keywords:
        keywords = raw_words
    keywords = keywords[:5]
    if not keywords:
        raise ValueError(f"Query too short: {query_text!r}")
    return " ".join(keywords)


def _message_content_to_text(message: Any) -> str:
    content = getattr(message, "content", None)
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        chunks = []
        for item in content:
            if isinstance(item, str):
                chunks.append(item)
                continue
            if isinstance(item, dict):
                text = item.get("text") or item.get("content")
            else:
                text = getattr(item, "text", None) or getattr(item, "content", None)
            if text:
                chunks.append(str(text))
        return "\n".join(chunks)
    return ""


def _looks_incomplete_generation(text: str) -> bool:
    cleaned = compact_whitespace(strip_code_fences(text or ""))
    if not cleaned:
        return True
    stripped = cleaned.strip(" \t\r\n\"'`{}[]:,;")
    if len(stripped) < 8:
        return True
    if stripped.lower() in {
        "anchor",
        "hint",
        "planning_anchor",
        "planning hint",
        "query",
        "search_query",
        "retrieval_query",
        "search terms",
    }:
        return True
    if cleaned.count("{") > cleaned.count("}"):
        return True
    if cleaned.count('"') % 2 == 1:
        return True
    return False


def _request_text_with_fallback(
    client: OpenAI,
    request_kwargs: Dict[str, object],
    *,
    fallback_max_tokens: tuple[int, ...] = (256, 512, 1024),
) -> str:
    base_max_tokens = int(request_kwargs.get("max_tokens") or 0)
    budgets = []
    for budget in (base_max_tokens, *fallback_max_tokens):
        if budget > 0 and budget not in budgets:
            budgets.append(budget)

    last_text = ""
    for budget in budgets:
        current_kwargs = dict(request_kwargs)
        current_kwargs["max_tokens"] = budget
        response = client.chat.completions.create(**current_kwargs)
        text = _message_content_to_text(response.choices[0].message)
        last_text = text or ""
        if not _looks_incomplete_generation(text):
            return text
    return last_text


@retry(stop=stop_after_attempt(3), wait=wait_fixed(2))
def predict_planning_anchor(
    client: OpenAI,
    model: str,
    setup: str,
    gap: str,
) -> tuple[str, str]:
    prompt = f"""
Role: Mathematical Research Assistant.
Task: Infer the immediate next proof intention before any retrieval happens.

[Context]
{setup}

[Gap]
{gap}

[Output constraints]
- Return one short leakage-safe planning hint.
- Describe the immediate proof objective or the kind of result needed next.
- Do NOT mention theorem names, author names, citation keys, source titles, DOI, or arXiv IDs.

Return JSON ONLY:
{{
  "planning_anchor": "<one short sentence>"
}}
"""
    request_kwargs: Dict[str, object] = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.0,
        "max_tokens": 1024,
    }
    extra_body = build_reasoning_extra_body()
    if extra_body:
        request_kwargs["extra_body"] = extra_body
    raw_content = _request_text_with_fallback(client, request_kwargs, fallback_max_tokens=(256, 512, 1024))
    planning_anchor = _extract_text_field(raw_content, "planning_anchor")
    planning_anchor = compact_whitespace(planning_anchor.strip(" ,.;"))
    if not planning_anchor:
        raise ValueError("Missing planning anchor")
    return planning_anchor, prompt


@retry(stop=stop_after_attempt(3), wait=wait_fixed(2))
def generate_search_query(
    client: OpenAI,
    model: str,
    setup: str,
    gap: str,
    *,
    planning_anchor: Optional[str] = None,
) -> tuple[str, str]:
    planning_section = planning_anchor.strip() if planning_anchor else "(none)"
    prompt = f"""
Role: Mathematical Research Assistant.
Task: Construct a retrieval query for the proof gap.

[Context]
{setup}

[Gap]
{gap}

[Planning Anchor]
{planning_section}

[Constraint: 5-WORD LIMIT]
- Focus on mathematical objects + properties.
- Avoid question-style phrasing.
- Use the planning anchor only as an intermediate guide for query formulation.

Return JSON ONLY:
{{
  "search_query": "<3-5 keywords>"
}}
"""
    request_kwargs: Dict[str, object] = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.0,
        "max_tokens": 1024,
    }
    extra_body = build_reasoning_extra_body()
    if extra_body:
        request_kwargs["extra_body"] = extra_body
    raw_content = _request_text_with_fallback(client, request_kwargs, fallback_max_tokens=(256, 512, 1024))
    raw_query = _extract_text_field(raw_content, "search_query")
    query = normalize_query_text(raw_query)
    return query, prompt


def generate_query_package(
    client: OpenAI,
    model: str,
    setup: str,
    gap: str,
    *,
    track: str,
    assist_anchor: str = "",
) -> Dict[str, Optional[str]]:
    planning_anchor = None
    planning_prompt = ""
    query_anchor = None
    query_anchor_source = "none"

    if track == "raw":
        planning_anchor, planning_prompt = predict_planning_anchor(client, model, setup, gap)
        query_anchor = planning_anchor
        query_anchor_source = "predicted_planning_anchor"
    elif track == "assist" and assist_anchor.strip():
        query_anchor = assist_anchor.strip()
        query_anchor_source = "gold_anchor_hint"

    query, query_prompt = generate_search_query(
        client,
        model,
        setup,
        gap,
        planning_anchor=query_anchor,
    )

    return {
        "query": query,
        "planning_anchor": planning_anchor,
        "planning_prompt": planning_prompt,
        "query_prompt": query_prompt,
        "query_anchor": query_anchor,
        "query_anchor_source": query_anchor_source,
    }
