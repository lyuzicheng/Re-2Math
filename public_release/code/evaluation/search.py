import os
import re
import json
import logging
import argparse
from datetime import datetime
from typing import List, Dict, Optional, Tuple, Any
from pathlib import Path
import sys

import httpx
from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_fixed

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from common.dataset_format import (
    CANONICAL_LOCAL_WINDOW,
    DEFAULT_DATASET_FILE,
    SUPPORTED_LOCAL_WINDOWS,
    build_generation_inputs,
    get_anchor_hint,
    get_citation_content,
    get_domain,
    get_global_context,
    get_instance_id,
    get_local_context_blocks,
    get_paper,
    get_reference_tool_latex,
    get_source_type,
    get_tool_family,
    normalize_official_local_window,
)
from common.citation_matching import (
    build_citation_title_oracle_query,
    check_title_match,
    extract_citation_title,
)
from common.openrouter_suite import build_reasoning_extra_body
from common.retrieval_backends import build_retriever
from common.retrieval_query_generation import generate_query_package, generate_search_query, predict_planning_anchor


# ============================================================
# Logging
# ============================================================

def setup_logging(level: str) -> logging.Logger:
    logging.basicConfig(
        level=level.upper(),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    return logging.getLogger("ScholarSearchEval")


logger = setup_logging(os.getenv("LOG_LEVEL", "INFO"))


# ============================================================
# Config helpers (no secrets in source)
# ============================================================


def robust_json_parse(text: str) -> Dict[str, Any]:
    text = str(text or "").strip()
    if not text:
        return {}
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S).strip()
    text = re.sub(r"^```json\s*", "", text, flags=re.MULTILINE)
    text = re.sub(r"^```\s*", "", text, flags=re.MULTILINE)
    text = re.sub(r"```$", "", text, flags=re.MULTILINE).strip()
    try:
        parsed = json.loads(text)
        return parsed if isinstance(parsed, dict) else {}
    except Exception:
        pass

    match = re.search(r"\{.*\}", text, flags=re.S)
    if not match:
        return {}
    try:
        parsed = json.loads(match.group(0))
        return parsed if isinstance(parsed, dict) else {}
    except Exception:
        return {}

def getenv_required(name: str, alt_names: Optional[List[str]] = None) -> str:
    """
    Read a required environment variable. Optionally try alternative names.
    Raises RuntimeError if not found.
    """
    alt_names = alt_names or []
    val = (os.getenv(name) or "").strip()
    if val:
        return val

    for alt in alt_names:
        val = (os.getenv(alt) or "").strip()
        if val:
            return val

    raise RuntimeError(
        f"Missing required environment variable: {name}\n"
        f"Set it before running. Example:\n"
        f"  export {name}='...'\n"
        f"Do NOT hardcode API keys in source code or commit them to GitHub."
    )


def getenv_optional(*names: str) -> str:
    for name in names:
        val = (os.getenv(name) or "").strip()
        if val:
            return val
    return ""


def build_openai_client(api_key: str, base_url: str) -> OpenAI:
    timeout = httpx.Timeout(60.0, connect=20.0)
    http_client = httpx.Client(timeout=timeout)
    return OpenAI(api_key=api_key, base_url=base_url, http_client=http_client, max_retries=2)


def test_llm_connection(client: OpenAI, model: str) -> bool:
    try:
        request_kwargs = {
            "model": model,
            "messages": [{"role": "user", "content": "ping"}],
            "max_tokens": 16,
            "temperature": 0.0,
        }
        extra_body = build_reasoning_extra_body()
        if extra_body:
            request_kwargs["extra_body"] = extra_body
        _ = client.chat.completions.create(**request_kwargs)
        return True
    except Exception as e:
        logger.error("LLM connection test failed: %s", e)
        return False


# ============================================================
# Query generation (LLM)
# ============================================================

@retry(stop=stop_after_attempt(3), wait=wait_fixed(2))
def judge_planning_match(client: OpenAI, model: str, predicted_anchor: str, gold_anchor: str) -> Tuple[bool, str]:
    prompt = f"""
Gold planning hint:
{gold_anchor}

Predicted planning hint:
{predicted_anchor}

Task:
Decide whether the predicted planning hint captures the same immediate proof intention as the gold planning hint.
Ignore wording and notation differences.
A more specific or slightly stronger planning hint still counts if it is aligned with the same next-step objective.

Return JSON ONLY:
{{ "is_match": <bool>, "reason": "<short reason>" }}
"""
    request_kwargs = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.0,
        "max_tokens": 128,
    }
    extra_body = build_reasoning_extra_body()
    if extra_body:
        request_kwargs["extra_body"] = extra_body
    response = client.chat.completions.create(**request_kwargs)
    parsed = robust_json_parse(response.choices[0].message.content or "")
    return bool(parsed.get("is_match", False)), str(parsed.get("reason", "") or "")


@retry(stop=stop_after_attempt(3), wait=wait_fixed(2))
def judge_query_match(
    client: OpenAI,
    model: str,
    predicted_query: str,
    reference_tool_latex: str,
    gt_citation: str,
) -> Tuple[bool, str]:
    prompt = f"""
Gold cited-source metadata:
{gt_citation}

Reference tool statement K*:
{reference_tool_latex}

Predicted retrieval query:
{predicted_query}

Task:
Decide whether the predicted retrieval query is well-targeted for retrieving a source that contains
the reference tool statement K* or a theorem-like statement that is essentially the same tool.
Ignore notation differences and bibliography wording.
A shorter query still counts if it still points to the right mathematical object, condition, inequality,
or theorem family.
Queries that are too generic, off-topic, or clearly point to the wrong tool do not count.

Return JSON ONLY:
{{ "is_match": <bool>, "reason": "<short reason>" }}
"""
    request_kwargs = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.0,
        "max_tokens": 128,
    }
    extra_body = build_reasoning_extra_body()
    if extra_body:
        request_kwargs["extra_body"] = extra_body
    response = client.chat.completions.create(**request_kwargs)
    parsed = robust_json_parse(response.choices[0].message.content or "")
    return bool(parsed.get("is_match", False)), str(parsed.get("reason", "") or "")


def extract_year(date_str: str) -> Optional[int]:
    if not date_str:
        return None
    m = re.search(r"(\d{4})", str(date_str))
    return int(m.group(1)) if m else None


# ============================================================
# Main
# ============================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run benchmark diagnostics for planning, query formation, or retrieval.")
    p.add_argument("--dataset", default=DEFAULT_DATASET_FILE, help="Path to JSONL dataset.")
    p.add_argument("--out", default="", help="Output JSON file path. If empty, auto-generate with timestamp.")
    p.add_argument("--track", choices=["raw", "assist"], default="raw")
    p.add_argument("--context-variant", choices=["local_only", "global_local"], default="global_local")
    p.add_argument(
        "--local-window",
        type=int,
        default=CANONICAL_LOCAL_WINDOW,
        help=(
            "Official local-context window m. "
            f"Recommended values: {SUPPORTED_LOCAL_WINDOWS}; default={CANONICAL_LOCAL_WINDOW}. "
            "Passing 0 also maps to the canonical window."
        ),
    )
    p.add_argument(
        "--diagnostic-mode",
        choices=["planning", "query", "retrieval"],
        default="planning",
        help=(
            "Diagnostic mode. planning measures anchor quality on the raw track; "
            "query measures query fidelity on the assist track; retrieval runs the search backend."
        ),
    )
    p.add_argument(
        "--query-mode",
        default="model",
        help="Query generation mode: model, citation_title_oracle. Deprecated alias: oracle_terminology.",
    )
    p.add_argument("--backend", choices=["scholar", "offline_metadata_bm25", "offline_fulltext_bm25"], default="scholar")
    p.add_argument("--snapshot", default="", help="Path to local retrieval snapshot JSONL for offline BM25 backends.")
    p.add_argument("--max-results", type=int, default=int(os.getenv("SCHOLAR_MAX_RESULTS", "100")))
    p.add_argument("--batch-size", type=int, default=int(os.getenv("SCHOLAR_BATCH_SIZE", "20")))
    p.add_argument("--sleep-sec", type=float, default=float(os.getenv("SCHOLAR_SLEEP_SEC", "0.5")))
    p.add_argument("--threshold", type=float, default=float(os.getenv("TITLE_MATCH_THRESHOLD", "0.99")))
    p.add_argument("--log-level", default=os.getenv("LOG_LEVEL", "INFO"))
    args = p.parse_args()
    if args.query_mode == "oracle_terminology":
        args.query_mode = "citation_title_oracle"
    if args.query_mode not in {"model", "citation_title_oracle"}:
        raise ValueError(f"Unsupported query mode: {args.query_mode}")
    args.local_window = normalize_official_local_window(args.local_window)
    if args.diagnostic_mode == "planning" and args.track != "raw":
        raise ValueError("Planning-only diagnostic is only defined for track=raw.")
    if args.diagnostic_mode == "planning" and args.query_mode != "model":
        raise ValueError("Planning-only diagnostic requires --query-mode model.")
    if args.diagnostic_mode == "query" and args.track != "assist":
        raise ValueError("Query-only diagnostic is only defined for track=assist.")
    return args


def load_jsonl(path: str) -> List[Dict[str, Any]]:
    data: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data.append(json.loads(line))
    return data


def main() -> None:
    args = parse_args()
    global logger
    logger = setup_logging(args.log_level)

    serpapi_key = None
    if args.diagnostic_mode == "retrieval" and args.backend == "scholar":
        serpapi_key = getenv_required("SERPAPI_API_KEY", alt_names=["SERPAPI_KEY"])

    client = None
    judge_client = None
    model = ""
    judge_model = ""
    base_url = ""
    judge_base_url = ""
    needs_generator = args.query_mode == "model"
    needs_judge = args.diagnostic_mode in {"planning", "query"}
    if needs_generator or needs_judge:
        openai_key = getenv_optional("OPENAI_API_KEY")
        if needs_generator and not openai_key:
            openai_key = getenv_required("OPENAI_API_KEY")
        base_url = getenv_optional("OPENAI_BASE_URL") or "https://api.openai.com/v1"
        model = getenv_optional("OPENAI_MODEL", "LLM_MODEL") or "gpt-4o-mini"
        if needs_generator:
            client = build_openai_client(openai_key, base_url)
        else:
            client = None

        judge_key = getenv_optional("JUDGE_API_KEY") or openai_key
        if needs_judge and not judge_key:
            judge_key = getenv_required("JUDGE_API_KEY", alt_names=["OPENAI_API_KEY"])
        judge_base_url = getenv_optional("JUDGE_BASE_URL") or base_url
        judge_model = getenv_optional("JUDGE_MODEL_NAME", "JUDGE_MODEL", "OPENAI_JUDGE_MODEL") or model
        if needs_judge:
            judge_client = build_openai_client(judge_key, judge_base_url)

    if needs_generator:
        if not test_llm_connection(client, model):
            raise RuntimeError("LLM connection failed. Check OPENAI_API_KEY / OPENAI_BASE_URL / OPENAI_MODEL.")
    if needs_judge and judge_client is not client:
        if not test_llm_connection(judge_client, judge_model):
            raise RuntimeError("Judge LLM connection failed. Check JUDGE_API_KEY / JUDGE_BASE_URL / JUDGE_MODEL_NAME.")

    if not os.path.exists(args.dataset):
        raise FileNotFoundError(f"Dataset file not found: {args.dataset}")

    dataset = load_jsonl(args.dataset)
    logger.info("Loaded dataset items=%d from %s", len(dataset), args.dataset)
    retriever = None
    if args.diagnostic_mode == "retrieval":
        retriever = build_retriever(
            args.backend,
            serpapi_key=serpapi_key,
            snapshot_path=(args.snapshot or None),
            batch_size=args.batch_size,
            sleep_sec=args.sleep_sec,
        )

    records: List[Dict[str, Any]] = []
    hits = 0
    planning_total = 0
    planning_hits = 0
    query_total = 0
    query_hits = 0
    for i, item in enumerate(dataset):
        item_id = get_instance_id(item) or f"item_{i}"
        setup, gap, assist_anchor = build_generation_inputs(
            item,
            track=args.track,
            context_variant=args.context_variant,
            local_window=(args.local_window or None),
        )
        gt_citation = get_citation_content(item)
        gt_title = extract_citation_title(gt_citation)
        gold_anchor = get_anchor_hint(item)
        reference_tool_latex = get_reference_tool_latex(item)
        paper_meta = get_paper(item)
        domain = get_domain(item)
        tool_family = get_tool_family(item)
        cited_source_type = get_source_type(item)
        cutoff_year = extract_year((item.get("paper", {}) or {}).get("publication_date", "") or item.get("publication_date", "") or "")

        logger.info("[%d/%d] Processing id=%s", i + 1, len(dataset), item_id)

        # 1) Generate the diagnostic artifact needed for this mode.
        query = None
        used_prompt = ""
        planning_prompt = ""
        predicted_anchor = None
        query_anchor = None
        query_anchor_source = "none"
        planning_is_match = None
        planning_reason = ""
        query_is_match = None
        query_reason = ""
        diagnostic_error = ""
        found_rank = -1
        found_item: Optional[Dict[str, Any]] = None
        best_score = 0.0
        best_reason = ""
        results: List[Dict[str, Any]] = []

        planning_counts_this_item = args.diagnostic_mode == "planning" and bool(gold_anchor)
        if planning_counts_this_item:
            planning_total += 1
            planning_is_match = False

        try:
            if args.diagnostic_mode == "planning":
                predicted_anchor, planning_prompt = predict_planning_anchor(client, model, setup, gap)
                if gold_anchor:
                    planning_is_match, planning_reason = judge_planning_match(judge_client, judge_model, predicted_anchor, gold_anchor)
                    planning_hits += int(planning_is_match)
            else:
                if args.query_mode == "citation_title_oracle":
                    query = build_citation_title_oracle_query(gt_citation)
                    if not query:
                        raise ValueError("Empty citation-title oracle query")
                    used_prompt = "citation_title_oracle_from_citation_title"
                    if args.track == "assist" and assist_anchor:
                        query_anchor = assist_anchor
                        query_anchor_source = "gold_anchor_hint"
                else:
                    if args.track == "raw":
                        predicted_anchor, planning_prompt = predict_planning_anchor(client, model, setup, gap)
                        query_anchor = predicted_anchor
                        query_anchor_source = "predicted_planning_anchor"
                    elif args.track == "assist" and assist_anchor:
                        query_anchor = assist_anchor
                        query_anchor_source = "gold_anchor_hint"
                    query, used_prompt = generate_search_query(
                        client,
                        model,
                        setup,
                        gap,
                        planning_anchor=query_anchor,
                    )
                if args.diagnostic_mode == "query":
                    if query and reference_tool_latex:
                        query_is_match, query_reason = judge_query_match(
                            judge_client,
                            judge_model,
                            query,
                            reference_tool_latex,
                            gt_citation,
                        )
                    elif query and not reference_tool_latex:
                        query_reason = "missing_reference_tool_latex"
                    if query_is_match is not None:
                        query_total += 1
                        query_hits += int(query_is_match)

        except Exception as e:
            diagnostic_error = re.sub(r"\s+", " ", str(e)).strip()
            logger.warning("Diagnostic generation failed for id=%s: %s", item_id, e)
            if planning_counts_this_item and not planning_reason:
                planning_reason = f"generation_failed: {diagnostic_error or type(e).__name__}"

        # 2) Retrieval is only executed in retrieval mode.
        if args.diagnostic_mode == "retrieval" and query:
            try:
                results = retriever.search(
                    query=query,
                    top_k=args.max_results,
                    max_year=cutoff_year,
                )
            except Exception as e:
                diagnostic_error = re.sub(r"\s+", " ", str(e)).strip()
                logger.warning("Retrieval failed for id=%s: %s", item_id, e)
                results = []

            for idx, res in enumerate(results):
                is_match, score, reason = check_title_match(
                    res.get("title", ""),
                    gt_citation,
                    result_snippet=res.get("snippet", ""),
                    threshold=args.threshold,
                )

                if score > best_score:
                    best_score = score
                    best_reason = reason

                if is_match:
                    found_rank = idx + 1
                    found_item = res
                    break

            if found_rank != -1:
                hits += 1
                logger.info("MATCH found at rank=%d title=%r", found_rank, (found_item or {}).get("title", ""))
            else:
                top1 = results[0]["title"] if results else ""
                logger.info("NO MATCH best_score=%.3f best_reason=%s top1=%r", best_score, best_reason, top1)

        records.append(
            {
                "id": item_id,
                "diagnostic_mode": args.diagnostic_mode,
                "query": query,
                "prompt": used_prompt,
                "planning_prompt": planning_prompt,
                "predicted_anchor": predicted_anchor,
                "query_anchor": query_anchor,
                "query_anchor_source": query_anchor_source,
                "gold_anchor": gold_anchor if args.track == "raw" else None,
                "reference_tool_latex": reference_tool_latex,
                "planning_is_match": planning_is_match,
                "planning_reason": planning_reason,
                "query_is_match": query_is_match,
                "query_reason": query_reason,
                "found": (found_rank != -1) if args.diagnostic_mode == "retrieval" else None,
                "found_rank": found_rank if args.diagnostic_mode == "retrieval" else None,
                "paper_id": paper_meta.get("paper_id", ""),
                "paper_title": paper_meta.get("title", ""),
                "domain": domain,
                "tool_family": tool_family,
                "source_type": cited_source_type,
                "backend": args.backend,
                "gt_citation": gt_citation,
                "gt_title": gt_title,
                "matched_title": (found_item or {}).get("title") if args.diagnostic_mode == "retrieval" else None,
                "matched_link": (found_item or {}).get("link") if args.diagnostic_mode == "retrieval" else None,
                "all_search_results": results if args.diagnostic_mode == "retrieval" else [],
                "debug": {
                    "best_score": best_score,
                    "best_reason": best_reason,
                    "diagnostic_error": diagnostic_error or None,
                    "threshold": args.threshold,
                    "cutoff_year": cutoff_year,
                    "track": args.track,
                    "context_variant": args.context_variant,
                    "local_window": args.local_window,
                    "query_mode": args.query_mode,
                    "diagnostic_mode": args.diagnostic_mode,
                    "backend": args.backend,
                    "snapshot": args.snapshot or None,
                },
            }
        )

    recall = (hits / len(dataset)) if dataset and args.diagnostic_mode == "retrieval" else None
    if recall is not None:
        logger.info("Final recall=%.4f (%d/%d)", recall, hits, len(dataset))
    if planning_total > 0:
        logger.info("Planning accuracy=%.4f (%d/%d)", planning_hits / planning_total, planning_hits, planning_total)
    if query_total > 0:
        logger.info("Query accuracy=%.4f (%d/%d)", query_hits / query_total, query_hits, query_total)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = args.out.strip() or str(ROOT / "evaluation" / "outputs" / f"search_log_full_{ts}.json")

    final_data = {
        "meta": {
            "total": len(dataset),
            "timestamp": ts,
            "model": model,
            "base_url": base_url,
            "judge_model": judge_model or None,
            "judge_base_url": judge_base_url or None,
            "strategy": "PlanningOrQueryDiagnostic" if args.diagnostic_mode != "retrieval" else "StrictTitleMatch + FrozenQuery + RetrievalBackend",
            "diagnostic_mode": args.diagnostic_mode,
            "track": args.track,
            "context_variant": args.context_variant,
            "local_window": args.local_window,
            "query_mode": args.query_mode,
            "backend": args.backend,
            "snapshot": args.snapshot or None,
        },
        "metrics": {
            "recall": recall,
            "planning_accuracy": (planning_hits / planning_total) if planning_total else None,
            "planning_success": planning_hits,
            "planning_total": planning_total,
            "query_accuracy": (query_hits / query_total) if query_total else None,
            "query_success": query_hits,
            "query_total": query_total,
        },
        "details": records,
    }

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(final_data, f, indent=2, ensure_ascii=False)

    logger.info("Saved results to %s", out_path)


if __name__ == "__main__":
    main()
