import os
import re
import json
import time
import logging
import argparse
from datetime import datetime
from typing import List, Dict, Optional, Tuple, Any

import httpx
from serpapi import GoogleSearch
from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_fixed


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


def build_openai_client(api_key: str, base_url: str) -> OpenAI:
    timeout = httpx.Timeout(60.0, connect=20.0)
    http_client = httpx.Client(timeout=timeout)
    return OpenAI(api_key=api_key, base_url=base_url, http_client=http_client, max_retries=2)


def test_llm_connection(client: OpenAI, model: str) -> bool:
    try:
        _ = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": "ping"}],
            max_tokens=5,
            temperature=0.0,
        )
        return True
    except Exception as e:
        logger.error("LLM connection test failed: %s", e)
        return False


# ============================================================
# Matching logic
# ============================================================

def clean_latex_text(text: str) -> str:
    """
    Convert a LaTeX-ish citation snippet to a normalized plain-text form,
    suitable for strict title matching.
    """
    if not text:
        return ""

    # Keep the content inside \command{...}
    text = re.sub(r"\\[a-zA-Z]+\{([^}]+)\}", r"\1", text)
    # Remove standalone commands like \newblock
    text = re.sub(r"\\[a-zA-Z]+", " ", text)

    # Remove braces and LaTeX spacing
    text = text.replace("~", " ")
    text = re.sub(r"[{}]", "", text)

    # Remove punctuation and symbols; keep alphanumerics and spaces
    text = re.sub(r"[^a-zA-Z0-9\s]", " ", text)

    # Normalize whitespace + lowercase
    return " ".join(text.split()).lower()


def extract_gt_title_strict(gt_str: str) -> str:
    """
    Extract a title candidate from a BibTeX/LaTeX-like citation string.
    Strategy:
      A) Split by \\newblock (common in .bbl) -> second block is often title
      B) Try quoted segments
      C) Fallback: second sentence fragment
    """
    if not gt_str:
        return ""

    title_candidate = ""

    blocks = re.split(r"\\newblock", gt_str)
    if len(blocks) >= 2:
        raw_title_part = blocks[1]

        # Cut off common metadata markers (journal/publisher fields etc.)
        split_markers = [
            "volume", "vol.", "pages", "pp.", "doi", "journal",
            "in proceedings", "in collection", "arxiv", "isbn", "publisher"
        ]
        raw_lower = raw_title_part.lower()
        cut_idx = len(raw_title_part)

        for marker in split_markers:
            idx = raw_lower.find(marker)
            if idx != -1 and idx < cut_idx:
                cut_idx = idx

        title_candidate = raw_title_part[:cut_idx]
    else:
        m = re.search(r'``(.*?)´´|"(.*?)"', gt_str)
        if m:
            title_candidate = m.group(1) or m.group(2) or ""
        else:
            parts = [p.strip() for p in gt_str.split(".") if p.strip()]
            title_candidate = parts[1] if len(parts) >= 2 else gt_str

    return clean_latex_text(title_candidate)


def check_title_match(
    result_title: str,
    gt_citation: str,
    result_snippet: str = "",
    threshold: float = 0.99,
) -> Tuple[bool, float, str]:
    """
    Strict match:
      1) Exact normalized string match -> True
      2) Jaccard similarity on token sets with a high threshold (default 0.99)
    """
    if not result_title or not gt_citation:
        return False, 0.0, "empty_input"

    gt_title_clean = extract_gt_title_strict(gt_citation)
    res_title_clean = clean_latex_text(result_title)

    if not gt_title_clean or not res_title_clean:
        return False, 0.0, "empty_title_after_clean"

    if gt_title_clean == res_title_clean:
        return True, 1.0, "exact_normalized_match"

    gt_tokens = set(gt_title_clean.split())
    res_tokens = set(res_title_clean.split())
    if not gt_tokens or not res_tokens:
        return False, 0.0, "no_tokens"

    intersection = gt_tokens.intersection(res_tokens)
    union = gt_tokens.union(res_tokens)
    score = len(intersection) / len(union)

    is_match = score >= threshold
    if is_match:
        return True, score, f"jaccard={score:.3f} >= {threshold:.3f}"

    diff = list((union - intersection))[:5]
    return False, score, f"jaccard={score:.3f} < {threshold:.3f}; diff={diff}"


# ============================================================
# Query generation (LLM)
# ============================================================

def strip_think_blocks(text: str) -> str:
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


@retry(stop=stop_after_attempt(3), wait=wait_fixed(2))
def generate_search_query(client: OpenAI, model: str, setup: str, gap: str) -> Tuple[str, str]:
    """
    Generate 3-5 keywords to search for the missing lemma/theorem.
    Retries on failure and on weak output.
    """
    prompt = f"""
Role: Mathematical Research Assistant.
Task: The user is stuck at a proof step (Gap) and does not know which theorem applies.
Extract the mathematical objects and desired properties from the gap to construct a search query.

[Context]
{setup}

[Gap]
{gap}

[Constraint: 5-WORD LIMIT]
- Focus on objects + properties (e.g., "eigenvalue perturbation bound").
- Avoid "How to" questions. Output descriptive keywords only.
- Output ONLY 3-5 keywords.
"""

    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
        max_tokens=64,
    )

    content = strip_think_blocks(response.choices[0].message.content or "")
    content = content.replace('"', "").strip()

    words = content.split()
    stop_words = {"find", "paper", "prove", "via", "using", "the", "of", "in", "to", "a", "an"}
    keywords = [w for w in words if w.lower() not in stop_words]

    if len(keywords) > 5:
        keywords = keywords[:5]

    # Force retry if the output is too weak (tenacity will retry)
    if len(keywords) < 2:
        raise ValueError(f"Query too short: {content!r}")

    return " ".join(keywords), prompt


# ============================================================
# SerpApi fetching (Google Scholar)
# ============================================================

@retry(stop=stop_after_attempt(3), wait=wait_fixed(2))
def fetch_scholar_page(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Fetch one page from SerpApi.
    Retries on errors and transient failures.
    """
    results = GoogleSearch(params).get_dict()
    if "error" in results:
        raise RuntimeError(f"SerpApi error: {results['error']}")
    return results


def fetch_top_results(
    serpapi_key: str,
    query: str,
    max_year: Optional[int] = None,
    target_count: int = 100,
    batch_size: int = 20,
    sleep_sec: float = 0.5,
) -> List[Dict[str, Any]]:
    """
    Page through Google Scholar results via SerpApi and return up to target_count results.
    """
    logger.info("Searching query=%r target=%d", query, target_count)

    all_results: List[Dict[str, Any]] = []

    for start in range(0, target_count, batch_size):
        params: Dict[str, Any] = {
            "q": query,
            "api_key": serpapi_key,
            "engine": "google_scholar",
            "num": batch_size,
            "start": start,
            "hl": "en",
            "gl": "us",
        }
        if max_year:
            params["as_yhi"] = max_year

        try:
            page = fetch_scholar_page(params)
        except Exception as e:
            logger.error("Failed fetching page start=%d: %s", start, e)
            break

        organic = page.get("organic_results", []) or []
        if not organic:
            break

        for item in organic:
            position = int(item.get("position", 0) or 0)
            all_results.append(
                {
                    "title": item.get("title", "") or "",
                    "link": item.get("link", "") or "",
                    "snippet": item.get("snippet", "") or "",
                    "rank": start + position + 1,
                }
            )

        if sleep_sec > 0:
            time.sleep(sleep_sec)

    logger.info("Fetched %d results.", len(all_results))
    return all_results


def extract_year(date_str: str) -> Optional[int]:
    if not date_str:
        return None
    m = re.search(r"(\d{4})", str(date_str))
    return int(m.group(1)) if m else None


# ============================================================
# Main
# ============================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate strict title match retrieval via SerpApi + LLM query generation.")
    p.add_argument("--dataset", default="benchmark_dataset.jsonl", help="Path to JSONL dataset.")
    p.add_argument("--out", default="", help="Output JSON file path. If empty, auto-generate with timestamp.")
    p.add_argument("--max-results", type=int, default=int(os.getenv("SCHOLAR_MAX_RESULTS", "100")))
    p.add_argument("--batch-size", type=int, default=int(os.getenv("SCHOLAR_BATCH_SIZE", "20")))
    p.add_argument("--sleep-sec", type=float, default=float(os.getenv("SCHOLAR_SLEEP_SEC", "0.5")))
    p.add_argument("--threshold", type=float, default=float(os.getenv("TITLE_MATCH_THRESHOLD", "0.99")))
    p.add_argument("--log-level", default=os.getenv("LOG_LEVEL", "INFO"))
    return p.parse_args()


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

    # Required secrets (env-only)
    serpapi_key = getenv_required("SERPAPI_API_KEY", alt_names=["SERPAPI_KEY"])
    openai_key = getenv_required("OPENAI_API_KEY")
    base_url = (os.getenv("OPENAI_BASE_URL") or "").strip() or "https://api.openai.com/v1"
    model = (os.getenv("OPENAI_MODEL") or os.getenv("LLM_MODEL") or "").strip() or "gpt-4o-mini"

    client = build_openai_client(openai_key, base_url)
    if not test_llm_connection(client, model):
        raise RuntimeError("LLM connection failed. Check OPENAI_API_KEY / OPENAI_BASE_URL / OPENAI_MODEL.")

    if not os.path.exists(args.dataset):
        raise FileNotFoundError(f"Dataset file not found: {args.dataset}")

    dataset = load_jsonl(args.dataset)
    logger.info("Loaded dataset items=%d from %s", len(dataset), args.dataset)

    records: List[Dict[str, Any]] = []
    hits = 0

    for i, item in enumerate(dataset):
        item_id = item.get("id", f"item_{i}")
        setup = item.get("problem_input", {}).get("global_context", {}).get("setup", "") or ""
        gap = item.get("problem_input", {}).get("local_context", {}).get("gap_objective", "") or ""
        gt_citation = item.get("ground_truth", {}).get("target_citation_content", "") or ""
        cutoff_year = extract_year(item.get("publication_date", "") or "")

        logger.info("[%d/%d] Processing id=%s", i + 1, len(dataset), item_id)

        # 1) Generate query
        try:
            query, used_prompt = generate_search_query(client, model, setup, gap)
        except Exception as e:
            logger.warning("Query generation failed for id=%s: %s", item_id, e)
            # Fallback query (deterministic)
            fallback = " ".join((gap.split()[:5] or setup.split()[:5]))
            query, used_prompt = fallback, "fallback_due_to_error"

        # 2) Fetch results
        results = fetch_top_results(
            serpapi_key=serpapi_key,
            query=query,
            max_year=cutoff_year,
            target_count=args.max_results,
            batch_size=args.batch_size,
            sleep_sec=args.sleep_sec,
        )

        # 3) Strict matching
        found_rank = -1
        found_item: Optional[Dict[str, Any]] = None
        best_score = 0.0
        best_reason = ""

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
                "query": query,
                "prompt": used_prompt,
                "found": found_rank != -1,
                "found_rank": found_rank,
                "gt_citation": gt_citation,
                "matched_title": (found_item or {}).get("title"),
                "matched_link": (found_item or {}).get("link"),
                "all_search_results": results,
                "debug": {
                    "best_score": best_score,
                    "best_reason": best_reason,
                    "threshold": args.threshold,
                    "cutoff_year": cutoff_year,
                },
            }
        )

    recall = (hits / len(dataset)) if dataset else 0.0
    logger.info("Final recall=%.4f (%d/%d)", recall, hits, len(dataset))

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = args.out.strip() or f"search_log_full_{ts}.json"

    final_data = {
        "meta": {
            "total": len(dataset),
            "timestamp": ts,
            "model": model,
            "base_url": base_url,
            "strategy": "StrictTitleMatch + LLMQuery + SerpApiScholar",
        },
        "metrics": {"recall": recall},
        "details": records,
    }

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(final_data, f, indent=2, ensure_ascii=False)

    logger.info("Saved results to %s", out_path)


if __name__ == "__main__":
    main()
