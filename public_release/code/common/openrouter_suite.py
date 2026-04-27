from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional

from common.dataset_format import DEFAULT_DATASET_FILE


ROOT = Path(__file__).resolve().parents[1]
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_REASONING_EFFORT = "low"


@dataclass(frozen=True)
class ModelSpec:
    alias: str
    model_slug: str
    family: str


DEFAULT_OPENROUTER_MODELS: List[ModelSpec] = [
    ModelSpec(alias="gpt", model_slug="openai/gpt-5.2", family="openai"),
    ModelSpec(alias="gemini", model_slug="google/gemini-3.1-pro-preview", family="google"),
    ModelSpec(alias="grok", model_slug="x-ai/grok-4", family="xai"),
    ModelSpec(alias="claude", model_slug="anthropic/claude-opus-4.5", family="anthropic"),
    ModelSpec(alias="deepseek", model_slug="deepseek/deepseek-v3.2", family="deepseek"),
    ModelSpec(alias="qwen", model_slug="qwen/qwen3-235b-a22b-thinking-2507", family="qwen"),
    ModelSpec(alias="kimi", model_slug="moonshotai/kimi-k2-thinking", family="moonshot"),
]


DATASET_CANDIDATES: List[str] = [
    "construction/outputs/benchmark_dataset_eval200_balanced_20260422.jsonl",
    "construction/outputs/benchmark_dataset_top200_high_quality_20260422.jsonl",
    "construction/outputs/benchmark_dataset_current.jsonl",
    "construction/outputs/benchmark_dataset_coretask_curated_20260422.jsonl",
    "construction/outputs/benchmark_dataset_all_merged_high_quality_cleaned_20260422.jsonl",
    "construction/outputs/benchmark_dataset_usable_live_20260422.jsonl",
    DEFAULT_DATASET_FILE,
]


def parse_model_specs(items: Optional[Iterable[str]]) -> List[ModelSpec]:
    if not items:
        return list(DEFAULT_OPENROUTER_MODELS)

    specs: List[ModelSpec] = []
    seen = set()
    for item in items:
        raw = str(item or "").strip()
        if not raw:
            continue
        if "=" in raw:
            alias, slug = raw.split("=", 1)
            alias = alias.strip()
            slug = slug.strip()
        else:
            slug = raw
            alias = slug.rsplit("/", 1)[-1].replace(".", "_").replace("-", "_")
        if not alias or not slug:
            raise ValueError(f"Invalid model spec: {item!r}")
        if alias in seen:
            raise ValueError(f"Duplicate model alias: {alias}")
        seen.add(alias)
        specs.append(ModelSpec(alias=alias, model_slug=slug, family=slug.split("/", 1)[0]))
    if not specs:
        raise ValueError("No valid OpenRouter model specs were provided.")
    return specs


def model_spec_map(specs: Iterable[ModelSpec]) -> Dict[str, ModelSpec]:
    return {spec.alias: spec for spec in specs}


def resolve_dataset_path(dataset_arg: str = "") -> Path:
    if dataset_arg:
        path = Path(dataset_arg)
        if path.exists():
            return path
        raise FileNotFoundError(f"Dataset file not found: {path}")

    for rel in DATASET_CANDIDATES:
        candidate = ROOT / rel
        if candidate.exists():
            return candidate

    searched = "\n".join(str(ROOT / rel) for rel in DATASET_CANDIDATES)
    raise FileNotFoundError(
        "Could not resolve a default dataset file. Checked:\n"
        f"{searched}"
    )



def _truthy(value: str) -> bool:
    raw = str(value or "").strip().lower()
    return raw not in {"", "0", "false", "no", "off"}


def is_openrouter_base_url(base_url: str = "") -> bool:
    raw = str(base_url or "").strip().lower()
    if not raw:
        raw = str(os.getenv("OPENAI_BASE_URL") or os.getenv("SOLVER_BASE_URL") or "").strip().lower()
    return "openrouter.ai" in raw


def build_reasoning_extra_body(base_url: str = "") -> Dict[str, object]:
    if not is_openrouter_base_url(base_url):
        return {}
    if not _truthy(os.getenv("OPENROUTER_ENABLE_REASONING", "1")):
        return {}

    max_tokens_raw = str(os.getenv("OPENROUTER_REASONING_MAX_TOKENS", "") or "").strip()
    effort = str(
        os.getenv("OPENROUTER_REASONING_EFFORT")
        or os.getenv("LLM_REASONING_EFFORT")
        or DEFAULT_REASONING_EFFORT
    ).strip()
    reasoning: Dict[str, object] = {}
    if max_tokens_raw:
        try:
            reasoning["max_tokens"] = int(max_tokens_raw)
        except Exception:
            reasoning["effort"] = effort or DEFAULT_REASONING_EFFORT
    elif effort:
        reasoning["effort"] = effort
    else:
        reasoning["enabled"] = True

    exclude_raw = str(os.getenv("OPENROUTER_REASONING_EXCLUDE", "") or "").strip()
    if exclude_raw:
        reasoning["exclude"] = _truthy(exclude_raw)

    return {"reasoning": reasoning} if reasoning else {}
