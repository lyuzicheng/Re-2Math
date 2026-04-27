from __future__ import annotations

import os
import json
import re
import logging
import argparse
import shutil
from dataclasses import dataclass
from typing import Dict, Optional, Any, List, Tuple
from pathlib import Path
import sys

import arxiv
import fitz  # PyMuPDF
import requests
from difflib import SequenceMatcher
from tqdm import tqdm
from openai import OpenAI

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from common.citation_matching import (
    check_title_match,
    compare_titles,
    extract_citation_title,
    is_strict_title_match,
)
from common.openrouter_suite import build_reasoning_extra_body
from common.dataset_format import (
    CANONICAL_LOCAL_WINDOW,
    DEFAULT_DATASET_FILE,
    SUPPORTED_LOCAL_WINDOWS,
    build_query_context,
    build_generation_inputs,
    format_local_context,
    get_anchor_hint,
    get_cited_arxiv_id,
    get_citation_content,
    get_domain,
    get_global_context,
    get_local_context_blocks,
    get_paper,
    get_reference_tool_latex,
    get_source_type,
    get_tool_family,
    load_dataset_as_dict,
    normalize_official_local_window,
)
from common.retrieval_backends import build_retriever
from common.retrieval_query_generation import generate_query_package
from common import fulltext_resolver


# ============================================================
# Logging
# ============================================================

def setup_logging(level: str) -> logging.Logger:
    logging.basicConfig(
        level=level.upper(),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    return logging.getLogger("SemanticEval")


logger = setup_logging(os.getenv("LOG_LEVEL", "INFO"))


# ============================================================
# Config / Secrets (env-only, safe for GitHub)
# ============================================================

@dataclass(frozen=True)
class LLMConfig:
    api_key: str
    base_url: str
    model_name: str
    timeout_sec: float = 120.0


DOWNLOAD_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"
    ),
    "Accept": "application/pdf,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
}


def getenv_required(name: str) -> str:
    val = (os.getenv(name) or "").strip()
    if not val:
        raise RuntimeError(
            f"Missing required environment variable: {name}\n"
            f"Do NOT hardcode API keys in source. Set it via env vars."
        )
    return val


def build_llm_config_from_env(prefix: str, fallback: Optional[LLMConfig] = None) -> LLMConfig:
    """
    Build an LLM config from environment variables.

    Required if fallback is None:
        {prefix}_API_KEY, {prefix}_BASE_URL, {prefix}_MODEL_NAME

    If fallback is provided:
        any missing field falls back to fallback's value.
    """
    api_key = (os.getenv(f"{prefix}_API_KEY") or "").strip()
    base_url = (os.getenv(f"{prefix}_BASE_URL") or "").strip()
    model_name = (os.getenv(f"{prefix}_MODEL_NAME") or "").strip()
    timeout = (os.getenv(f"{prefix}_TIMEOUT_SEC") or "").strip()

    if fallback is None:
        if not api_key:
            api_key = getenv_required(f"{prefix}_API_KEY")
        if not base_url:
            base_url = getenv_required(f"{prefix}_BASE_URL")
        if not model_name:
            model_name = getenv_required(f"{prefix}_MODEL_NAME")
    else:
        api_key = api_key or fallback.api_key
        base_url = base_url or fallback.base_url
        model_name = model_name or fallback.model_name

    timeout_sec = float(timeout) if timeout else (fallback.timeout_sec if fallback else 120.0)

    return LLMConfig(
        api_key=api_key,
        base_url=base_url,
        model_name=model_name,
        timeout_sec=timeout_sec,
    )


class LLMJsonCaller:
    """
    A small wrapper so we can use different models/clients for:
      - solver (full-text scanning / extraction)
      - judge (title extraction + math equivalence)
    """
    def __init__(self, cfg: LLMConfig):
        self.cfg = cfg
        self.client = OpenAI(api_key=cfg.api_key, base_url=cfg.base_url, timeout=cfg.timeout_sec)

    @staticmethod
    def robust_json_parse(llm_output: str) -> Dict[str, Any]:
        if not llm_output:
            return {}

        text = re.sub(r"<think>.*?</think>", "", llm_output, flags=re.S).strip()
        text = re.sub(r"^```json\s*", "", text, flags=re.MULTILINE)
        text = re.sub(r"^```\s*", "", text, flags=re.MULTILINE)
        text = re.sub(r"```$", "", text, flags=re.MULTILINE).strip()

        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        # Try Python literal dict (some models do this)
        try:
            import ast
            obj = ast.literal_eval(text)
            if isinstance(obj, dict):
                return obj
            return {}
        except Exception:
            pass

        # Last resort: try to salvage key fields
        try:
            score_match = re.search(r'"relevance_score"\s*:\s*(\d+)', text)
            score = int(score_match.group(1)) if score_match else 0

            theorem_match = re.search(r'"extracted_theorem"\s*:\s*"(.*)"', text, re.DOTALL)
            theorem = theorem_match.group(1) if theorem_match else None

            return {
                "relevance_score": score,
                "reasoning": "JSON parse failed (regex recovered).",
                "extracted_theorem": theorem,
            }
        except Exception:
            return {}

    def call_json(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.0,
        max_tokens: int = 4000,
    ) -> Dict[str, Any]:
        try:
            request_kwargs = {
                "model": self.cfg.model_name,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                "temperature": temperature,
                "max_tokens": max_tokens,
            }
            extra_body = build_reasoning_extra_body(self.cfg.base_url)
            if extra_body:
                request_kwargs["extra_body"] = extra_body
            resp = self.client.chat.completions.create(**request_kwargs)
            content = (resp.choices[0].message.content or "").strip()
            return self.robust_json_parse(content)
        except Exception as e:
            logger.error("LLM call failed (model=%s): %s", self.cfg.model_name, e)
            return {}


# ============================================================
# Retrieval / Download helpers
# ============================================================

def sanitize_filename(text: str, fallback: str) -> str:
    cleaned = re.sub(r"\$.*?\$", "", text or "")
    cleaned = re.sub(r'[\\/*?:"<>|]', "", cleaned).strip()
    return (cleaned or fallback)[:120]


def is_probable_html(text: str) -> bool:
    lowered = (text or "").lower()
    anti_bot_markers = [
        "<html",
        "<!doctype html",
        "captcha",
        "cloudflare",
        "access denied",
        "enable javascript",
        "robot check",
    ]
    return any(marker in lowered for marker in anti_bot_markers)


def validate_pdf(path: str) -> bool:
    try:
        doc = fitz.open(path)
        valid = len(doc) > 0
        doc.close()
        return valid
    except Exception:
        return False


def fetch_scholar_top_results(
    serpapi_key: str,
    query: str,
    max_results: int = 20,
    max_year: Optional[int] = None,
) -> List[Dict[str, Any]]:
    params: Dict[str, Any] = {
        "engine": "google_scholar",
        "q": query,
        "api_key": serpapi_key,
        "num": min(max_results, 20),
        "hl": "en",
        "gl": "us",
    }
    if max_year:
        params["as_yhi"] = max_year
    response = requests.get("https://serpapi.com/search.json", params=params, timeout=60)
    response.raise_for_status()
    payload = response.json()
    if "error" in payload:
        raise RuntimeError(payload["error"])
    return payload.get("organic_results", []) or []


def extract_candidate_urls(result: Dict[str, Any]) -> List[str]:
    urls: List[str] = []
    for resource in result.get("resources", []) or []:
        link = str(resource.get("link", "") or "")
        if link:
            urls.append(link)
    link = str(result.get("link", "") or "")
    if link:
        urls.append(link)

    expanded: List[str] = []
    for url in urls:
        expanded.append(url)
        if "arxiv.org/abs/" in url:
            expanded.append(url.replace("/abs/", "/pdf/") + ".pdf")
    deduped: List[str] = []
    seen = set()
    for url in expanded:
        if url and url not in seen:
            seen.add(url)
            deduped.append(url)
    return deduped


def download_pdf_from_url(url: str, save_path: str) -> Tuple[bool, str]:
    try:
        response = requests.get(url, headers=DOWNLOAD_HEADERS, stream=True, timeout=30)
        content_type = (response.headers.get("Content-Type") or "").lower()
        if response.status_code != 200:
            return False, f"http_{response.status_code}"

        head = response.raw.read(4096, decode_content=True)
        if not head:
            return False, "empty_response"
        if b"%PDF" not in head[:1024]:
            decoded_head = head.decode("utf-8", errors="ignore")
            if "pdf" not in content_type and is_probable_html(decoded_head):
                return False, "anti_bot_or_html"

        with open(save_path, "wb") as f:
            f.write(head)
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)

        if not validate_pdf(save_path):
            return False, "invalid_pdf"
        return True, "downloaded"
    except Exception as exc:
        return False, f"error:{exc}"


def find_equivalent_arxiv_paper(title: str, max_results: int = 5) -> Optional[Dict[str, Any]]:
    title = str(title or "").strip()
    query_tokens = re.findall(r"[A-Za-z0-9]+", title.lower())[:12]
    if len(query_tokens) < 3:
        return None

    query = " AND ".join(f"all:{token}" for token in query_tokens)
    client = arxiv.Client(page_size=max_results, delay_seconds=3.0, num_retries=3)
    search = arxiv.Search(
        query=query,
        max_results=max_results,
        sort_by=arxiv.SortCriterion.Relevance,
        sort_order=arxiv.SortOrder.Descending,
    )

    best_match: Optional[Dict[str, Any]] = None
    best_score = -1.0
    try:
        for result in client.results(search):
            candidate_title = str(result.title or "").strip()
            is_match, score, reason = is_strict_title_match(candidate_title, title)
            if not is_match:
                continue
            if score > best_score:
                best_score = score
                best_match = {
                    "title": candidate_title,
                    "pdf_url": getattr(result, "pdf_url", None) or result.entry_id.replace("/abs/", "/pdf/") + ".pdf",
                    "entry_id": result.entry_id,
                    "score": score,
                    "match_reason": reason,
                }
    except Exception as exc:
        logger.warning("arXiv title lookup failed for %r: %s", title[:120], exc)
    return best_match


def _link_or_copy_file(src: Path, dst: Path) -> None:
    src = src.resolve()
    dst.parent.mkdir(parents=True, exist_ok=True)
    if os.path.lexists(dst):
        dst.unlink()
    try:
        os.symlink(str(src), str(dst))
    except Exception:
        shutil.copy2(src, dst)


def _oracle_cache_root() -> Path:
    configured = str(os.getenv("ORACLE_CITED_SOURCE_CACHE_ROOT", "") or "").strip()
    if configured:
        return Path(configured).expanduser()
    return ROOT / "evaluation" / "outputs" / "_oracle_cited_source_cache"


def _shared_resolver_cache_root() -> Path:
    configured = str(os.getenv("FULLTEXT_CACHE_ROOT", "") or "").strip()
    if configured:
        return Path(configured).expanduser()
    return ROOT / "evaluation" / "outputs" / "_resolver_cache"


def _oracle_cache_key_candidates(target_title: str, doi: str, arxiv_id: str) -> List[str]:
    keys: List[str] = []
    normalized_arxiv = fulltext_resolver.clean_arxiv_id(arxiv_id)
    normalized_doi = str(doi or "").strip().lower()
    normalized_title = str(target_title or "").strip()
    if normalized_arxiv:
        keys.append(f"arxiv_{re.sub(r'[^a-z0-9]+', '_', normalized_arxiv.lower()).strip('_')}")
    if normalized_doi:
        keys.append(f"doi_{re.sub(r'[^a-z0-9]+', '_', normalized_doi).strip('_')[:160]}")
    if normalized_title:
        keys.append(f"title_{fulltext_resolver.title_hash(normalized_title)}")
    deduped: List[str] = []
    seen = set()
    for key in keys:
        if key and key not in seen:
            seen.add(key)
            deduped.append(key)
    return deduped


def _oracle_cache_meta_path(cache_root: Path, key: str) -> Path:
    return cache_root / "meta" / f"{key}.json"


def _oracle_cache_load(keys: List[str]) -> Optional[Tuple[Path, Dict[str, Any]]]:
    cache_root = _oracle_cache_root()
    for key in keys:
        meta_path = _oracle_cache_meta_path(cache_root, key)
        if not meta_path.exists():
            continue
        try:
            metadata = json.loads(meta_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        pdf_name = str(metadata.get("cache_pdf", "") or "").strip()
        if not pdf_name:
            continue
        pdf_path = cache_root / "pdfs" / pdf_name
        if pdf_path.exists() and validate_pdf(str(pdf_path)):
            return pdf_path, metadata
    return None


def _oracle_cache_store(src_pdf: str | Path, metadata: Dict[str, Any], keys: List[str]) -> None:
    if not keys:
        return
    src_path = Path(src_pdf)
    if not src_path.exists() or not validate_pdf(str(src_path)):
        return
    cache_root = _oracle_cache_root()
    primary_key = keys[0]
    pdf_dir = cache_root / "pdfs"
    meta_dir = cache_root / "meta"
    pdf_dir.mkdir(parents=True, exist_ok=True)
    meta_dir.mkdir(parents=True, exist_ok=True)
    pdf_name = f"{primary_key}.pdf"
    cached_pdf_path = pdf_dir / pdf_name
    if not (cached_pdf_path.exists() and validate_pdf(str(cached_pdf_path))):
        _link_or_copy_file(src_path, cached_pdf_path)
    cache_record = dict(metadata)
    cache_record["cache_pdf"] = pdf_name
    cache_record["cache_primary_key"] = primary_key
    for key in keys:
        _oracle_cache_meta_path(cache_root, key).write_text(
            json.dumps(cache_record, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )


def _materialize_oracle_cache_hit(
    case_dir_path: Path,
    *,
    cached_pdf: Path,
    cached_meta: Dict[str, Any],
    title_query: str,
    gt_citation: str,
    doi: str,
    arxiv_id: str,
) -> Dict[str, Any]:
    filename = str(cached_meta.get("filename") or "").strip() or "oracle_cited_source_cached.pdf"
    target_path = case_dir_path / filename
    if not (target_path.exists() and validate_pdf(str(target_path))):
        _link_or_copy_file(cached_pdf, target_path)
    return {
        "filename": filename,
        "title": str(cached_meta.get("title") or title_query or gt_citation).strip(),
        "status": "cached_global",
        "origin": f"shared_oracle_cache:{str(cached_meta.get('origin') or 'unknown')}",
        "source_url": str(cached_meta.get("source_url") or ""),
        "target_title": title_query,
        "doi": doi,
        "arxiv_id": arxiv_id,
        "is_oracle_cited_source": True,
        "title_match_reason": str(cached_meta.get("title_match_reason") or ""),
        "resolution_trace": list(cached_meta.get("resolution_trace") or []),
    }


def ensure_candidate_pdf(
    case_dir: str,
    rank: int,
    result: Dict[str, Any],
) -> Dict[str, Any]:
    title = str(result.get("title", "") or f"rank_{rank:02d}")
    safe_title = sanitize_filename(title, f"rank_{rank:02d}")
    filename = f"rank_{rank:02d}_{safe_title}.pdf"
    save_path = os.path.join(case_dir, filename)

    if os.path.exists(save_path) and validate_pdf(save_path):
        return {
            "filename": filename,
            "title": title,
            "status": "cached",
            "origin": "scholar",
            "source_url": str(result.get("link", "") or ""),
        }

    for url in extract_candidate_urls(result):
        success, status = download_pdf_from_url(url, save_path)
        if success:
            return {
                "filename": filename,
                "title": title,
                "status": status,
                "origin": "scholar",
                "source_url": url,
            }

    arxiv_match = find_equivalent_arxiv_paper(title)
    if arxiv_match:
        fallback_name = f"rank_{rank:02d}_{safe_title}_arxiv.pdf"
        fallback_path = os.path.join(case_dir, fallback_name)
        success, status = download_pdf_from_url(arxiv_match["pdf_url"], fallback_path)
        if success:
            return {
                "filename": fallback_name,
                "title": arxiv_match["title"],
                "status": status,
                "origin": "arxiv_fallback",
                "source_url": arxiv_match["pdf_url"],
                "title_match_reason": arxiv_match.get("match_reason", ""),
            }

    return {
        "filename": None,
        "title": title,
        "status": "unavailable",
        "origin": "none",
        "source_url": str(result.get("link", "") or ""),
    }


def ensure_oracle_cited_source_pdf(
    case_dir: str,
    *,
    gt_citation: str,
    cited_doi: str = "",
    cited_arxiv_id: str = "",
) -> Optional[Dict[str, Any]]:
    case_dir_path = Path(case_dir)
    title_query = extract_citation_title(gt_citation)
    doi = fulltext_resolver.extract_doi_from_text(cited_doi, gt_citation)
    arxiv_id = (
        fulltext_resolver.clean_arxiv_id(cited_arxiv_id)
        or fulltext_resolver.extract_arxiv_id_from_text(gt_citation)
    )
    cache_keys = _oracle_cache_key_candidates(title_query or str(gt_citation or "").strip(), doi, arxiv_id)
    preferred_targets = [
        ("oracle_cited_source.pdf", arxiv_id, "cited_arxiv_id"),
        ("oracle_cited_source_resolved.pdf", "", "cited_source_resolver"),
        ("oracle_cited_source_arxiv_search.pdf", "", "citation_title_arxiv_search"),
    ]

    for filename, candidate_id, origin in preferred_targets:
        save_path = os.path.join(case_dir, filename)
        if os.path.exists(save_path) and validate_pdf(save_path):
            item = {
                "filename": filename,
                "title": title_query or str(gt_citation or "").strip(),
                "status": "cached",
                "origin": origin,
                "source_url": f"https://arxiv.org/pdf/{candidate_id}.pdf" if candidate_id else "",
                "target_title": title_query,
                "is_oracle_cited_source": True,
            }
            _oracle_cache_store(save_path, item, cache_keys)
            return item

    cached = _oracle_cache_load(cache_keys)
    if cached:
        cached_pdf, cached_meta = cached
        return _materialize_oracle_cache_hit(
            case_dir_path,
            cached_pdf=cached_pdf,
            cached_meta=cached_meta,
            title_query=title_query,
            gt_citation=gt_citation,
            doi=doi,
            arxiv_id=arxiv_id,
        )

    if arxiv_id:
        save_path = os.path.join(case_dir, "oracle_cited_source.pdf")
        pdf_url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"
        success, status = download_pdf_from_url(pdf_url, save_path)
        if success:
            item = {
                "filename": "oracle_cited_source.pdf",
                "title": title_query or str(gt_citation or "").strip(),
                "status": status,
                "origin": "cited_arxiv_id",
                "source_url": pdf_url,
                "target_title": title_query,
                "is_oracle_cited_source": True,
            }
            _oracle_cache_store(save_path, item, cache_keys)
            return item

    if doi or title_query:
        candidate = {
            "title": title_query or str(gt_citation or "").strip(),
            "doi": doi,
            "arxiv_id": arxiv_id,
            "link": f"https://doi.org/{doi}" if doi else "",
            "snippet": str(gt_citation or ""),
            "publication_info": {"summary": str(gt_citation or "")},
            "resources": [],
        }
        try:
            cache_root = _shared_resolver_cache_root()
            resolved = fulltext_resolver.resolve_candidate_document(
                case_dir=case_dir_path,
                cache_root=cache_root,
                rank=0,
                result=candidate,
            )
            resolved_filename = str((resolved or {}).get("filename") or "")
            resolved_path = case_dir_path / resolved_filename if resolved_filename else None
            if resolved_path and resolved_path.exists() and validate_pdf(str(resolved_path)):
                alias_name = "oracle_cited_source_resolved.pdf"
                alias_path = case_dir_path / alias_name
                if not (alias_path.exists() and validate_pdf(str(alias_path))):
                    _link_or_copy_file(resolved_path, alias_path)
                item = {
                    "filename": alias_name,
                    "title": str((resolved or {}).get("title") or title_query or gt_citation).strip(),
                    "status": str((resolved or {}).get("status") or "resolved"),
                    "origin": f"cited_source_resolver:{(resolved or {}).get('origin') or 'resolver'}",
                    "source_url": str((resolved or {}).get("source_url") or ""),
                    "target_title": title_query,
                    "doi": doi,
                    "arxiv_id": arxiv_id,
                    "is_oracle_cited_source": True,
                    "resolution_trace": (resolved or {}).get("resolution_trace", []),
                }
                _oracle_cache_store(alias_path, item, cache_keys)
                return item
        except Exception as exc:
            logger.warning("Oracle cited-source resolver failed for title=%r doi=%r: %s", title_query[:120], doi, exc)

    if title_query:
        arxiv_match = find_equivalent_arxiv_paper(title_query)
        if arxiv_match:
            save_path = os.path.join(case_dir, "oracle_cited_source_arxiv_search.pdf")
            success, status = download_pdf_from_url(arxiv_match["pdf_url"], save_path)
            if success:
                item = {
                    "filename": "oracle_cited_source_arxiv_search.pdf",
                    "title": arxiv_match["title"],
                    "status": status,
                    "origin": "citation_title_arxiv_search",
                    "source_url": arxiv_match["pdf_url"],
                    "title_match_reason": arxiv_match.get("match_reason", ""),
                    "target_title": title_query,
                    "is_oracle_cited_source": True,
                }
                _oracle_cache_store(save_path, item, cache_keys)
                return item

    return {
        "filename": None,
        "title": title_query or str(gt_citation or "").strip(),
        "status": "unavailable",
        "origin": "oracle_cited_source",
        "source_url": "",
        "target_title": title_query,
        "is_oracle_cited_source": True,
    }


def prepare_task_pdfs(
    case_id: str,
    candidates: List[Dict[str, Any]],
    download_root: str,
) -> Tuple[str, List[Dict[str, Any]]]:
    case_dir = os.path.join(download_root, case_id)
    os.makedirs(case_dir, exist_ok=True)
    manifest: List[Dict[str, Any]] = []
    for idx, result in enumerate(candidates, start=1):
        search_rank = int(result.get("rank", idx) or idx)
        local_pdf_path = str(result.get("local_pdf_path", "") or "")
        if local_pdf_path and os.path.exists(local_pdf_path) and validate_pdf(local_pdf_path):
            title = str(result.get("title", "") or f"rank_{search_rank:02d}")
            safe_title = sanitize_filename(title, f"rank_{search_rank:02d}")
            filename = f"rank_{search_rank:02d}_{safe_title}.pdf"
            target_path = os.path.join(case_dir, filename)
            if not (os.path.exists(target_path) and validate_pdf(target_path)):
                try:
                    if os.path.lexists(target_path):
                        os.remove(target_path)
                    os.symlink(local_pdf_path, target_path)
                except Exception:
                    shutil.copy2(local_pdf_path, target_path)
            item = {
                "filename": filename,
                "title": title,
                "status": "linked_local",
                "origin": str(result.get("backend", "offline")) or "offline",
                "source_url": str(result.get("link", "") or local_pdf_path),
                "local_pdf_path": local_pdf_path,
            }
        else:
            item = ensure_candidate_pdf(case_dir, search_rank, result)
        item["shortlist_rank"] = idx
        item["rank"] = search_rank
        item["scholar_link"] = str(result.get("link", "") or "")
        item["backend_score"] = result.get("backend_score")
        item["backend_doc_id"] = result.get("backend_doc_id")
        manifest.append(item)
    return case_dir, manifest


def build_retrieval_generation_inputs(
    task_data: Dict[str, Any],
    track: str,
    context_variant: str = "global_local",
    local_window: Optional[int] = None,
) -> Tuple[str, str]:
    setup, gap, anchor_hint = build_generation_inputs(
        task_data,
        track=track,
        context_variant=context_variant,
        local_window=local_window,
    )
    if track == "assist" and anchor_hint:
        setup = "\n".join(part for part in [setup, f"Anchor hint: {anchor_hint}"] if part)
    return setup, gap


def compact_query_context(query_context: str, max_chars: int = 6000) -> str:
    text = re.sub(r"\s+", " ", str(query_context or "")).strip()
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rstrip() + " ..."


def compact_candidate_snippet(result: Dict[str, Any], max_chars: int = 360) -> str:
    snippet = re.sub(r"\s+", " ", str(result.get("snippet", "") or "")).strip()
    if len(snippet) <= max_chars:
        return snippet
    return snippet[:max_chars].rstrip() + " ..."


def normalize_judge_label(label: Any) -> str:
    raw = str(label or "").strip().lower()
    if raw in {"yes", "true", "match", "matched", "supported", "grounded", "sufficient"}:
        return "yes"
    if raw in {"no", "false", "not_match", "unsupported", "ungrounded", "insufficient"}:
        return "no"
    return "uncertain"


def judge_label_to_bool(label: Any) -> Optional[bool]:
    normalized = normalize_judge_label(label)
    if normalized == "yes":
        return True
    if normalized == "no":
        return False
    return None


def normalize_semantic_text(text: str) -> str:
    cleaned = re.sub(r"\\[a-zA-Z]+\*?(?:\[[^\]]*\])?\{([^}]*)\}", r" \1 ", text or "")
    cleaned = re.sub(r"\\[a-zA-Z]+\*?", " ", cleaned)
    cleaned = cleaned.replace("~", " ")
    cleaned = re.sub(r"[^a-zA-Z0-9]+", " ", cleaned)
    return " ".join(cleaned.lower().split())


def extract_theorem_like_blocks(text: str, block_chars: int = 2500, max_blocks: int = 36) -> List[str]:
    patterns = list(
        re.finditer(
            r"\b(theorem|lemma|proposition|corollary|claim|remark|definition)\b",
            text,
            flags=re.IGNORECASE,
        )
    )
    blocks: List[str] = []
    for match in patterns[:max_blocks]:
        start = max(0, match.start() - 300)
        end = min(len(text), match.start() + block_chars)
        blocks.append(text[start:end])
    if not blocks and text:
        chunks = [text[i : i + block_chars] for i in range(0, min(len(text), block_chars * 8), block_chars)]
        blocks.extend(chunks[:max_blocks])
    return blocks


def select_grounding_blocks(extracted_latex: str, source_text: str, max_blocks: int = 6) -> List[str]:
    candidate_blocks = extract_theorem_like_blocks(source_text)
    if not candidate_blocks:
        return []

    normalized_target = normalize_semantic_text(extracted_latex)
    if not normalized_target:
        return candidate_blocks[:max_blocks]

    target_tokens = set(normalized_target.split())
    scored: List[Tuple[float, str]] = []
    for block in candidate_blocks:
        normalized_block = normalize_semantic_text(block)
        block_tokens = set(normalized_block.split())
        overlap = len(target_tokens & block_tokens) / max(1, len(target_tokens))
        similarity = SequenceMatcher(None, normalized_target[:1500], normalized_block[:1500]).ratio()
        score = max(overlap, similarity)
        scored.append((score, block))

    scored.sort(key=lambda item: item[0], reverse=True)
    top_blocks = [block for score, block in scored[:max_blocks] if score > 0.0]
    return top_blocks or candidate_blocks[:max_blocks]


def shortlist_candidates_with_metadata(
    solver_llm: "LLMJsonCaller",
    query_context: str,
    search_query: str,
    query_anchor: Optional[str],
    candidates: List[Dict[str, Any]],
    shortlist_size: int,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    if shortlist_size <= 0 or len(candidates) <= shortlist_size:
        return list(candidates), {
            "mode": "bypass",
            "reasoning": "Shortlist bypassed because candidate count is already small.",
            "total_candidates": len(candidates),
            "selected_ranks": [int(item.get("rank", idx + 1) or idx + 1) for idx, item in enumerate(candidates)],
        }

    candidate_lines: List[str] = []
    for idx, result in enumerate(candidates, start=1):
        rank = int(result.get("rank", idx) or idx)
        title = str(result.get("title", "") or "")
        snippet = compact_candidate_snippet(result)
        backend = str(result.get("backend", "") or "")
        year = str(result.get("year", "") or "")
        candidate_lines.append(
            f"[{idx}] search_rank={rank} | title={title}\n"
            f"snippet={snippet}\n"
            f"backend={backend} | year={year}"
        )

    anchor_text = str(query_anchor or "").strip() or "None"
    prompt = f"""
You are doing lightweight candidate shortlisting for a mathematical retrieval benchmark.

Goal:
Select up to {shortlist_size} candidate papers that are most worth opening in full text.
Do NOT solve the problem. Use only the search metadata below.

Problem context:
{compact_query_context(query_context)}

Search query:
{search_query}

Planning anchor:
{anchor_text}

Candidates:
{chr(10).join(candidate_lines)}

Selection policy:
1) Prefer papers whose title/snippet strongly suggest they may contain the needed theorem, lemma, proposition, or technique.
2) Prefer research papers over obviously irrelevant or generic results.
3) Keep the shortlist small. Only include candidates genuinely worth full-text inspection.

Return JSON ONLY:
{{
  "selected_indices": [<candidate indices from the list above>],
  "reasoning": "<brief explanation>"
}}
"""
    try:
        data = solver_llm.call_json(
            system_prompt="You are a careful mathematical retrieval shortlister.",
            user_prompt=prompt,
            temperature=0.0,
            max_tokens=512,
        )
    except Exception as exc:
        data = {"selected_indices": [], "reasoning": f"shortlist_error:{exc}"}

    raw_indices = data.get("selected_indices", []) if isinstance(data, dict) else []
    selected_positions: List[int] = []
    seen = set()
    for raw in raw_indices if isinstance(raw_indices, list) else []:
        try:
            pos = int(raw)
        except Exception:
            continue
        if 1 <= pos <= len(candidates) and pos not in seen:
            seen.add(pos)
            selected_positions.append(pos)
        if len(selected_positions) >= shortlist_size:
            break

    fallback_used = False
    if not selected_positions:
        fallback_used = True
        selected_positions = list(range(1, min(shortlist_size, len(candidates)) + 1))

    shortlisted = [candidates[pos - 1] for pos in selected_positions]
    return shortlisted, {
        "mode": "metadata_llm",
        "reasoning": str(data.get("reasoning", "") or "") if isinstance(data, dict) else "",
        "total_candidates": len(candidates),
        "selected_indices": selected_positions,
        "selected_ranks": [int(candidates[pos - 1].get("rank", pos) or pos) for pos in selected_positions],
        "fallback_used": fallback_used,
        "prompt": prompt,
    }


def load_search_log_index(path: str) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    details = payload.get("details", payload if isinstance(payload, list) else []) or []
    index: Dict[str, Dict[str, Any]] = {}
    for row in details:
        case_id = str(row.get("id", "") or "").strip()
        if case_id:
            index[case_id] = row
    meta = payload.get("meta", {}) if isinstance(payload, dict) else {}
    return index, meta


# ============================================================
# Utils: PDF reading, text cleaning
# ============================================================

class Utils:
    @staticmethod
    def clean_text(text: str) -> str:
        if not text:
            return ""
        return re.sub(r"\s+", " ", text).strip()

    @staticmethod
    def read_pdf_header(pdf_path: str, char_limit: int = 4000) -> str:
        try:
            doc = fitz.open(pdf_path)
            text = ""
            for i in range(min(2, len(doc))):
                text += doc[i].get_text()
            doc.close()
            return Utils.clean_text(text)[:char_limit]
        except Exception:
            return ""

    @staticmethod
    def read_full_pdf(pdf_path: str, char_limit: int = 2_000_000) -> str:
        try:
            doc = fitz.open(pdf_path)
            text = ""
            for page in doc:
                text += page.get_text()
            doc.close()
            return Utils.clean_text(text)[:char_limit]
        except Exception:
            return ""


# ============================================================
# Solver: scan full texts and select a candidate paper
# ============================================================

class FullTextSolver:
    def __init__(
        self,
        task_id: str,
        query_context: str,
        folder_path: str,
        solver_llm: LLMJsonCaller,
        per_paper_prompt_chars: int = 150_000,
        early_stop_score: int = 9,
    ):
        self.task_id = task_id
        self.query_context = query_context
        self.folder_path = folder_path
        self.solver_llm = solver_llm
        self.per_paper_prompt_chars = per_paper_prompt_chars
        self.early_stop_score = early_stop_score

        # Main end-to-end evaluation must not inspect oracle-source cache files.
        self.pdf_files = sorted(
            [
                f
                for f in os.listdir(folder_path)
                if f.lower().endswith(".pdf") and not f.startswith("oracle_cited_source")
            ]
        )

    def evaluate_single_paper(self, filename: str) -> Dict[str, Any]:
        path = os.path.join(self.folder_path, filename)
        content = Utils.read_full_pdf(path)

        if not content:
            return {
                "filename": filename,
                "relevance_score": 0,
                "reasoning": "Empty content",
                "extracted_theorem": None,
            }

        prompt = f"""
You are a mathematical assistant for a "Math Gap-Filling" task.

=== PROBLEM CONTEXT ===
{self.query_context}

=== CANDIDATE PAPER ===
Filename: "{filename}"
Content (truncated):
{content[:self.per_paper_prompt_chars]}

=== TASK ===
1) Decide whether this paper contains the specific theorem/lemma needed to bridge the gap described in the provided benchmark context.
2) Output a relevance score (0-10):
   - 10 = high confidence this is the exact source containing the needed lemma.
   - 0 = irrelevant.
3) If score > 5, extract the relevant theorem/lemma LaTeX (as faithfully as possible).

Return JSON ONLY:
{{
  "filename": "{filename}",
  "relevance_score": <int 0-10>,
  "reasoning": "<short explanation>",
  "extracted_theorem": "<latex string or null>"
}}
"""

        data = self.solver_llm.call_json(
            system_prompt="You are a math expert assistant.",
            user_prompt=prompt,
            temperature=0.0,
            max_tokens=4000,
        )

        if not isinstance(data, dict):
            return {
                "filename": filename,
                "relevance_score": 0,
                "reasoning": "Invalid LLM output format",
                "extracted_theorem": None,
            }

        return {
            "filename": filename,
            "relevance_score": int(data.get("relevance_score", 0) or 0),
            "reasoning": str(data.get("reasoning", "") or ""),
            "extracted_theorem": data.get("extracted_theorem", None),
        }

    def run(self) -> Dict[str, Any]:
        if not self.pdf_files:
            return {"error": "no_pdfs"}

        logger.info("Task %s: scanning %d PDFs...", self.task_id, len(self.pdf_files))

        candidates: List[Dict[str, Any]] = []
        for f in self.pdf_files:
            result = self.evaluate_single_paper(f)
            candidates.append(result)

            if result.get("relevance_score", 0) >= self.early_stop_score:
                break

        if not candidates:
            return {"error": "no_candidates"}

        best = max(candidates, key=lambda x: int(x.get("relevance_score", 0) or 0))

        return {
            "selected_filename": best.get("filename"),
            "score": best.get("relevance_score"),
            "reasoning": best.get("reasoning"),
            "extracted_theorem": best.get("extracted_theorem"),
            "all_candidates": candidates,
        }


# ============================================================
# Evaluator: title matching + math equivalence (use judge model)
# ============================================================

class SemanticEvaluator:
    def __init__(
        self,
        folder_path: str,
        gt_data: Dict[str, str],
        judge_llm: LLMJsonCaller,
        download_manifest: Optional[List[Dict[str, Any]]] = None,
        query_context: str = "",
    ):
        self.folder_path = folder_path
        self.gt_latex = gt_data.get("target_lemma_latex", "") or ""
        self.gt_citation_str = gt_data.get("target_citation_content", "") or ""
        self.gt_anchor_hint = gt_data.get("target_anchor_hint", "") or ""
        self.gt_title = extract_citation_title(self.gt_citation_str)
        self.judge_llm = judge_llm
        self.query_context = query_context or ""
        self.download_manifest_index = {
            str(item.get("filename", "") or ""): item
            for item in (download_manifest or [])
            if item.get("filename")
        }

    def extract_title_from_string(self, text: str, is_citation: bool = False) -> str:
        context_desc = "citation string" if is_citation else "first page of a PDF"
        prompt = f"""
Extract the academic paper/book TITLE from the following text.

Text ({context_desc}):
{text[:2000]}

Return JSON ONLY:
{{ "extracted_title": "<string>" }}
"""
        res = self.judge_llm.call_json(
            system_prompt="You extract paper titles from messy text.",
            user_prompt=prompt,
            temperature=0.0,
            max_tokens=256,
        )
        if isinstance(res, dict):
            return (res.get("extracted_title", "") or "").strip()
        return ""

    def resolve_selected_title(self, selected_filename: Optional[str]) -> Tuple[str, str]:
        if not selected_filename:
            return "", "none"

        manifest_item = self.download_manifest_index.get(selected_filename, {})
        manifest_title = str(manifest_item.get("title", "") or "").strip()
        if manifest_title:
            return manifest_title, "download_manifest"

        path = os.path.join(self.folder_path, selected_filename)
        pdf_header_text = Utils.read_pdf_header(path)
        pdf_title = self.extract_title_from_string(pdf_header_text, is_citation=False)
        if pdf_title:
            return pdf_title, "pdf_header_llm"
        return "", "none"

    def check_retrieval_correctness(self, selected_filename: Optional[str]) -> Dict[str, Any]:
        gt_title = self.gt_title or (self.gt_citation_str[:160] or "").strip()
        if not selected_filename:
            return {
                "is_retrieved": False,
                "reason": "No file selected",
                "gt_title": gt_title,
                "selected_title": "",
                "selected_title_source": "none",
                "score": 0.0,
            }

        selected_title, selected_title_source = self.resolve_selected_title(selected_filename)
        if not gt_title or not selected_title:
            return {
                "is_retrieved": False,
                "reason": "Title extraction failed",
                "gt_title": gt_title,
                "selected_title": selected_title,
                "selected_title_source": selected_title_source,
                "score": 0.0,
            }

        is_match, score, match_reason = compare_titles(selected_title, gt_title, threshold=0.99)
        reason = (
            f"GT title: '{gt_title}' vs selected title: '{selected_title}' "
            f"(source={selected_title_source}; {match_reason})"
        )
        return {
            "is_retrieved": is_match,
            "reason": reason,
            "gt_title": gt_title,
            "selected_title": selected_title,
            "selected_title_source": selected_title_source,
            "score": score,
        }


    def judge_document_grounding(self, selected_filename: Optional[str], extracted_latex: Optional[str]) -> Dict[str, Any]:
        if not selected_filename:
            return {
                "label": "no",
                "is_supported": False,
                "reason": "No file selected",
                "confidence": 0.0,
                "evidence_span": "",
            }
        if not extracted_latex:
            return {
                "label": "no",
                "is_supported": False,
                "reason": "No extracted theorem",
                "confidence": 0.0,
                "evidence_span": "",
            }

        path = os.path.join(self.folder_path, selected_filename)
        source_text = Utils.read_full_pdf(path, char_limit=600_000)
        evidence_blocks = select_grounding_blocks(extracted_latex, source_text, max_blocks=6)
        evidence_text = "\n\n".join(f"[BLOCK {idx + 1}]\n{block}" for idx, block in enumerate(evidence_blocks))
        prompt = f"""
Selected source filename:
{selected_filename}

Candidate theorem statement:
{extracted_latex}

Source excerpts from the selected source document:
{evidence_text[:12000]}

Task:
Decide whether the candidate theorem statement is grounded in the selected source document.
Return "yes" if the statement is either verbatim or a mathematically faithful restatement of a theorem-like statement supported by the source excerpts.
Return "no" if the statement is unsupported, materially changes the assumptions or conclusion, or appears hallucinated.
Return "uncertain" only if the provided excerpts are genuinely insufficient to decide.

Return JSON ONLY:
{{
  "label": "yes|no|uncertain",
  "reason": "<short reason>",
  "confidence": <number 0 to 1>,
  "evidence_span": "<short quoted source excerpt or pointer>"
}}
"""
        res = self.judge_llm.call_json(
            system_prompt="You are a strict judge for document grounding of mathematical theorem statements.",
            user_prompt=prompt,
            temperature=0.0,
            max_tokens=700,
        )
        if not isinstance(res, dict):
            return {
                "label": "uncertain",
                "is_supported": None,
                "reason": "Judge JSON parse failed",
                "confidence": 0.0,
                "evidence_span": "",
            }
        label = normalize_judge_label(res.get("label"))
        return {
            "label": label,
            "is_supported": judge_label_to_bool(label),
            "reason": str(res.get("reason", "") or ""),
            "confidence": float(res.get("confidence", 0.0) or 0.0),
            "evidence_span": str(res.get("evidence_span", "") or ""),
        }

    def judge_tool_sufficiency(self, extracted_latex: Optional[str]) -> Dict[str, Any]:
        if not extracted_latex or not self.gt_latex:
            return {
                "label": "no",
                "is_sufficient": False,
                "reason": "Missing content",
                "confidence": 0.0,
            }

        prompt = f"""
Proof context:
{compact_query_context(self.query_context, max_chars=7000)}

Reference tool witness:
{self.gt_latex}

Predicted theorem statement:
{extracted_latex}

Task:
Decide whether the predicted theorem statement is sufficient to justify the target proof gap under the given proof context.
Return "yes" if it is mathematically equivalent to the reference witness up to benign notation changes, or if it is a strictly stronger statement whose assumptions are satisfied by the given proof context.
Return "no" if it is weaker, mismatched, or requires assumptions not supported by the context.
Return "uncertain" only if the proof context is genuinely insufficient to decide.

Return JSON ONLY:
{{
  "label": "yes|no|uncertain",
  "reason": "<short reason>",
  "confidence": <number 0 to 1>
}}
"""
        res = self.judge_llm.call_json(
            system_prompt="You are a strict judge for tool sufficiency in mathematical proofs.",
            user_prompt=prompt,
            temperature=0.0,
            max_tokens=700,
        )
        if not isinstance(res, dict):
            return {
                "label": "uncertain",
                "is_sufficient": None,
                "reason": "Judge JSON parse failed",
                "confidence": 0.0,
            }
        label = normalize_judge_label(res.get("label"))
        return {
            "label": label,
            "is_sufficient": judge_label_to_bool(label),
            "reason": str(res.get("reason", "") or ""),
            "confidence": float(res.get("confidence", 0.0) or 0.0),
        }

    def judge_math_equivalence(self, extracted_latex: Optional[str]) -> Tuple[bool, str]:
        decision = self.judge_tool_sufficiency(extracted_latex)
        return bool(decision.get("is_sufficient")), str(decision.get("reason", "") or "")

    def judge_planning_equivalence(self, predicted_anchor: Optional[str]) -> Tuple[bool, str]:
        if not predicted_anchor or not self.gt_anchor_hint:
            return False, "Missing content"

        prompt = f"""
Gold planning hint:
{self.gt_anchor_hint}

Predicted planning hint:
{predicted_anchor}

Task:
Decide whether the predicted planning hint captures the same immediate proof intention as the gold planning hint.
Ignore wording and notation differences.
A more specific or slightly stronger planning hint still counts if it is aligned with the same next-step objective.

Return JSON ONLY:
{{ "is_match": <bool>, "reason": "<short reason>" }}
"""
        res = self.judge_llm.call_json(
            system_prompt="You are a strict judge for mathematical planning-hint equivalence.",
            user_prompt=prompt,
            temperature=0.0,
            max_tokens=256,
        )
        if isinstance(res, dict):
            return bool(res.get("is_match", False)), str(res.get("reason", "") or "")
        return False, "Judge JSON parse failed"


# ============================================================
# Main
# ============================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Full-text semantic evaluation with separate solver and judge models.")
    p.add_argument(
        "--download-root",
        default="scholar_all_top20_downloads_gemini",
        help="Root folder used to cache per-task downloaded candidate PDFs.",
    )
    p.add_argument("--pdf-root", dest="download_root_legacy", default=None, help=argparse.SUPPRESS)
    p.add_argument("--dataset-file", default=DEFAULT_DATASET_FILE, help="Path to dataset JSONL.")
    p.add_argument(
        "--result-file",
        default=str(ROOT / "evaluation" / "outputs" / "evaluation_results_full_scan.json"),
        help="Where to save results JSON.",
    )
    p.add_argument("--search-log", default="", help="Optional search.py output JSON. If provided, reuse cached query/results instead of searching again.")
    p.add_argument("--backend", choices=["scholar", "offline_metadata_bm25", "offline_fulltext_bm25"], default="scholar")
    p.add_argument("--snapshot", default="", help="Path to local retrieval snapshot JSONL for offline BM25 backends.")
    p.add_argument("--per-paper-prompt-chars", type=int, default=80_000, help="Max chars per shortlisted paper passed to solver prompt.")
    p.add_argument("--early-stop-score", type=int, default=9, help="Early stop if solver finds a candidate with >= this score.")
    p.add_argument("--top-k", type=int, default=20, help="Number of Google Scholar results to try downloading per case (max 20).")
    p.add_argument("--shortlist-k", type=int, default=20, help="How many top retrieval results to inspect at the metadata-shortlisting stage.")
    p.add_argument(
        "--shortlist-size",
        type=int,
        default=1,
        help="How many metadata-shortlisted candidates to open in full text. Default=1 for the canonical top-20-to-1 pipeline.",
    )
    p.add_argument("--track", choices=["raw", "assist"], default="raw", help="Which official input track to expose.")
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
        "--skip-planning-judge",
        action="store_true",
        help="Skip planning-hint equivalence inside end-to-end and reuse standalone planning diagnostics externally.",
    )
    p.add_argument(
        "--shard-index",
        type=int,
        default=0,
        help="0-based shard index for dataset slicing. Used only when --num-shards > 1.",
    )
    p.add_argument(
        "--num-shards",
        type=int,
        default=1,
        help="Split the dataset into this many contiguous shards and evaluate only one shard.",
    )
    p.add_argument("--log-level", default=os.getenv("LOG_LEVEL", "INFO"))
    args = p.parse_args()
    args.local_window = normalize_official_local_window(args.local_window)
    if args.num_shards < 1:
        raise ValueError("--num-shards must be >= 1")
    if not 0 <= args.shard_index < args.num_shards:
        raise ValueError("--shard-index must satisfy 0 <= shard-index < num-shards")
    return args


def slice_case_ids_for_shard(case_ids: List[str], shard_index: int, num_shards: int) -> List[str]:
    if num_shards <= 1:
        return case_ids
    start = len(case_ids) * shard_index // num_shards
    end = len(case_ids) * (shard_index + 1) // num_shards
    return case_ids[start:end]


def build_empty_stats() -> Dict[str, int]:
    return {
        "total": 0,
        "downloaded_candidate_pdfs": 0,
        "cases_with_downloads": 0,
        "cached_search_cases": 0,
        "fresh_search_cases": 0,
        "shortlisted_candidates_total": 0,
        "cited_doc_matched": 0,
        "ground_positive": 0,
        "ground_evaluable": 0,
        "ground_abstained": 0,
        "suff_positive": 0,
        "suff_evaluable": 0,
        "suff_abstained": 0,
        "toolacc_positive": 0,
        "toolacc_evaluable": 0,
        "toolacc_abstained": 0,
        "suff_given_ground_positive": 0,
        "suff_given_ground_total": 0,
        "tool_success_proxy": 0,
        "cited_tool_success_proxy": 0,
        "alt_source_tool_success_proxy": 0,
        "retrieved_correctly": 0,
        "latex_matched": 0,
        "planning_total": 0,
        "planning_matched": 0,
    }


def compute_summary_from_stats(stats: Dict[str, Any], results: List[Dict[str, Any]]) -> Dict[str, Optional[float]]:
    total = int(stats.get("total", len(results)) or 0)
    cite_recall_hits = sum(1 for row in results if row.get("cite_recall_at_20") is True)
    return {
        "AnchorAcc(x_raw)": (stats["planning_matched"] / total) if total and stats["planning_total"] > 0 else None,
        "CiteRecall@20": (cite_recall_hits / total) if total else 0.0,
        "GroundRate": (stats["ground_positive"] / total) if total else 0.0,
        "ToolAcc": (stats["toolacc_positive"] / total) if total else 0.0,
        "AltSourceToolAcc": (stats["alt_source_tool_success_proxy"] / total) if total else 0.0,
    }


def write_result_payload(
    result_file: str,
    *,
    stats: Dict[str, Any],
    results: List[Dict[str, Any]],
    meta: Dict[str, Any],
    completed: bool,
) -> None:
    payload = {
        "stats": stats,
        "summary": compute_summary_from_stats(stats, results),
        "results": results,
        "meta": {
            **meta,
            "resume_checkpoint": {
                "completed": completed,
                "completed_cases": len(results),
            },
        },
    }
    with open(result_file, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def load_result_checkpoint(result_file: str) -> Optional[Dict[str, Any]]:
    path = Path(result_file)
    if not path.exists():
        return None
    try:
        with path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
    except Exception as exc:
        logger.warning("Failed to read existing checkpoint %s: %s", result_file, exc)
        return None
    if not isinstance(payload, dict):
        return None
    payload.setdefault("stats", {})
    payload.setdefault("results", [])
    payload.setdefault("meta", {})
    return payload


def main() -> None:
    args = parse_args()
    global logger
    logger = setup_logging(args.log_level)
    download_root = args.download_root_legacy or args.download_root
    os.makedirs(download_root, exist_ok=True)

    if not os.path.exists(args.dataset_file):
        raise FileNotFoundError(f"Dataset file not found: {args.dataset_file}")

    # Build separate model configs:
    # - solver: required
    # - judge: optional (fallback to solver if not provided)
    solver_cfg = build_llm_config_from_env("SOLVER", fallback=None)
    judge_cfg = build_llm_config_from_env("JUDGE", fallback=solver_cfg)
    search_cache: Dict[str, Dict[str, Any]] = {}
    search_cache_meta: Dict[str, Any] = {}
    if args.search_log:
        if not os.path.exists(args.search_log):
            raise FileNotFoundError(f"Search log not found: {args.search_log}")
        search_cache, search_cache_meta = load_search_log_index(args.search_log)
        logger.info("Loaded cached search entries=%d from %s", len(search_cache), args.search_log)

    retriever = None
    if not search_cache:
        serpapi_key = (os.getenv("SERPAPI_API_KEY") or os.getenv("SERPAPI_KEY") or "").strip() if args.backend == "scholar" else None
        if args.backend == "scholar" and not serpapi_key:
            raise RuntimeError("Missing required environment variable: SERPAPI_API_KEY or SERPAPI_KEY")
        retriever = build_retriever(
            args.backend,
            serpapi_key=serpapi_key,
            snapshot_path=(args.snapshot or None),
        )

    logger.info("Solver model: %s", solver_cfg.model_name)
    logger.info("Judge  model: %s", judge_cfg.model_name)
    if search_cache_meta:
        logger.info(
            "Search cache meta: track=%s context_variant=%s local_window=%s backend=%s query_mode=%s",
            search_cache_meta.get("track"),
            search_cache_meta.get("context_variant"),
            search_cache_meta.get("local_window"),
            search_cache_meta.get("backend"),
            search_cache_meta.get("query_mode"),
        )

    solver_llm = LLMJsonCaller(solver_cfg)
    judge_llm = LLMJsonCaller(judge_cfg)

    dataset = load_dataset_as_dict(args.dataset_file)

    all_case_ids = sorted(dataset.keys())
    shard_case_ids = slice_case_ids_for_shard(all_case_ids, args.shard_index, args.num_shards)
    result_meta = {
        "download_root": download_root,
        "dataset_file": args.dataset_file,
        "dataset_total_instances": len(all_case_ids),
        "evaluated_shard_instances": len(shard_case_ids),
        "solver_model": solver_cfg.model_name,
        "judge_model": judge_cfg.model_name,
        "primary_metric": "ToolAcc",
        "track": args.track,
        "top_k": min(args.top_k, 20),
        "shortlist_k": args.shortlist_k,
        "shortlist_size": args.shortlist_size,
        "backend": args.backend,
        "snapshot": args.snapshot or None,
        "search_log": args.search_log or None,
        "context_variant": args.context_variant,
        "local_window": args.local_window,
        "skip_planning_judge": args.skip_planning_judge,
        "shard_index": args.shard_index,
        "num_shards": args.num_shards,
    }
    checkpoint = load_result_checkpoint(args.result_file)
    if checkpoint:
        results = list(checkpoint.get("results", []) or [])
        stats = build_empty_stats()
        stats.update(checkpoint.get("stats", {}) or {})
        completed_case_ids = {str(row.get("id") or "") for row in results if str(row.get("id") or "")}
    else:
        results = []
        stats = build_empty_stats()
        completed_case_ids = set()
    case_ids = [case_id for case_id in shard_case_ids if case_id not in completed_case_ids]
    logger.info(
        "Starting evaluation on %d pending dataset instances (shard %d/%d of %d total; %d already checkpointed).",
        len(case_ids),
        args.shard_index + 1,
        args.num_shards,
        len(all_case_ids),
        len(completed_case_ids),
    )

    for case_id in tqdm(case_ids, desc="Tasks"):
        stats["total"] += 1
        task_data = dataset[case_id]
        paper_meta = get_paper(task_data)
        domain = get_domain(task_data)
        tool_family = get_tool_family(task_data)
        cited_source_type = get_source_type(task_data)
        query_context = build_query_context(
            task_data,
            track=args.track,
            context_variant=args.context_variant,
            local_window=(args.local_window or None),
        )
        gt_data = {
            "title": paper_meta.get("title", "") or "",
            "target_lemma_latex": get_reference_tool_latex(task_data),
            "target_citation_content": get_citation_content(task_data),
            "target_anchor_hint": get_anchor_hint(task_data),
        }

        predicted_anchor = None
        planning_generation_reason = ""
        is_planning_matched = None
        planning_match_reason = ""
        search_query = None
        query_prompt = ""
        planning_prompt = ""
        query_anchor = None
        query_anchor_source = "none"
        folder_path = os.path.join(download_root, case_id)
        download_manifest: List[Dict[str, Any]] = []
        shortlist_manifest: Dict[str, Any] = {}
        solver_out: Dict[str, Any] = {}
        error_message = None
        cite_recall_at_20 = None
        cite_found_rank = None
        cached_search_entry = search_cache.get(case_id)
        used_cached_search = False

        try:
            candidates: List[Dict[str, Any]]
            if cached_search_entry:
                search_query = str(cached_search_entry.get("query", "") or "").strip() or None
                predicted_anchor = cached_search_entry.get("predicted_anchor")
                query_prompt = str(cached_search_entry.get("prompt", "") or "")
                planning_prompt = str(cached_search_entry.get("planning_prompt", "") or "")
                query_anchor = cached_search_entry.get("query_anchor")
                query_anchor_source = str(cached_search_entry.get("query_anchor_source", "") or "none")
                cite_found_rank = cached_search_entry.get("found_rank")
                found_flag = cached_search_entry.get("found")
                if found_flag is not None:
                    cite_recall_at_20 = bool(found_flag and (not cite_found_rank or int(cite_found_rank) <= 20))
                candidates = list(cached_search_entry.get("all_search_results", []) or [])
                used_cached_search = bool(search_query) and bool(candidates)
                if used_cached_search:
                    stats["cached_search_cases"] += 1
                    planning_generation_reason = "Reused from cached search log."

            if not used_cached_search:
                stats["fresh_search_cases"] += 1
                setup_text, gap_text = build_retrieval_generation_inputs(
                    task_data,
                    track=args.track,
                    context_variant=args.context_variant,
                    local_window=(args.local_window or None),
                )
                package = generate_query_package(
                    solver_llm.client,
                    solver_cfg.model_name,
                    setup_text,
                    gap_text,
                    track=args.track,
                    assist_anchor=get_anchor_hint(task_data),
                )
                search_query = str(package.get("query", "") or "")
                predicted_anchor = package.get("planning_anchor")
                query_prompt = str(package.get("query_prompt", "") or "")
                planning_prompt = str(package.get("planning_prompt", "") or "")
                query_anchor = package.get("query_anchor")
                query_anchor_source = str(package.get("query_anchor_source", "") or "none")

                cutoff_year = None
                pub_date = str((paper_meta.get("publication_date", "") or "")).strip()
                year_match = re.search(r"(\d{4})", pub_date)
                if year_match:
                    cutoff_year = int(year_match.group(1))

                if retriever is None:
                    serpapi_key = (os.getenv("SERPAPI_API_KEY") or os.getenv("SERPAPI_KEY") or "").strip() if args.backend == "scholar" else None
                    if args.backend == "scholar" and not serpapi_key:
                        raise RuntimeError("Missing required environment variable: SERPAPI_API_KEY or SERPAPI_KEY")
                    retriever = build_retriever(
                        args.backend,
                        serpapi_key=serpapi_key,
                        snapshot_path=(args.snapshot or None),
                    )

                candidates = retriever.search(
                    query=search_query,
                    top_k=min(args.top_k, 20),
                    max_year=cutoff_year,
                )
                best_score = 0.0
                for idx, candidate in enumerate(candidates, start=1):
                    is_match, score, _match_reason = check_title_match(
                        str(candidate.get("title", "") or ""),
                        gt_data["target_citation_content"],
                    )
                    if score > best_score:
                        best_score = score
                    if is_match:
                        cite_found_rank = idx
                        cite_recall_at_20 = idx <= 20
                        break
                if cite_recall_at_20 is None:
                    cite_recall_at_20 = False if candidates else None

            shortlist_input = list(candidates[: max(1, min(args.shortlist_k, len(candidates)))])
            shortlisted_candidates, shortlist_manifest = shortlist_candidates_with_metadata(
                solver_llm,
                query_context=query_context,
                search_query=search_query or "",
                query_anchor=query_anchor,
                candidates=shortlist_input,
                shortlist_size=max(1, min(args.shortlist_size, len(shortlist_input) or 1)),
            )
            stats["shortlisted_candidates_total"] += len(shortlisted_candidates)

            folder_path, download_manifest = fulltext_resolver.prepare_task_documents(
                case_id,
                shortlisted_candidates,
                download_root,
            )
            successful_downloads = sum(1 for item in download_manifest if item.get("filename"))
            stats["downloaded_candidate_pdfs"] += successful_downloads
            if successful_downloads > 0:
                stats["cases_with_downloads"] += 1

            evaluator = SemanticEvaluator(
                folder_path,
                gt_data,
                judge_llm=judge_llm,
                download_manifest=download_manifest,
                query_context=query_context,
            )
            if args.track == "raw" and gt_data["target_anchor_hint"]:
                if args.skip_planning_judge:
                    if not planning_generation_reason:
                        planning_generation_reason = (
                            "Planning equivalence skipped in end-to-end; reuse standalone planning diagnostic output."
                        )
                    planning_match_reason = "skipped_planning_judge"
                else:
                    if not planning_generation_reason:
                        planning_generation_reason = "Generated before retrieval query and then used to condition query generation."
                    is_planning_matched, planning_match_reason = evaluator.judge_planning_equivalence(predicted_anchor)
                    stats["planning_total"] += 1
                    if is_planning_matched:
                        stats["planning_matched"] += 1

            solver = FullTextSolver(
                task_id=case_id,
                query_context=query_context,
                folder_path=folder_path,
                solver_llm=solver_llm,
                per_paper_prompt_chars=args.per_paper_prompt_chars,
                early_stop_score=args.early_stop_score,
            )
            solver_out = solver.run()
            if "error" in solver_out:
                error_message = str(solver_out["error"])
                raise RuntimeError(error_message)

            sel_file = solver_out.get("selected_filename")
            retrieval_decision = evaluator.check_retrieval_correctness(sel_file)
            is_retrieved = bool(retrieval_decision.get("is_retrieved", False))
            retr_reason = str(retrieval_decision.get("reason", "") or "")
            gt_title = str(retrieval_decision.get("gt_title", "") or "")
            pdf_title = str(retrieval_decision.get("selected_title", "") or "")
            selected_title_source = str(retrieval_decision.get("selected_title_source", "none") or "none")

            ext_latex = solver_out.get("extracted_theorem")
            ground_decision = evaluator.judge_document_grounding(sel_file, ext_latex)
            suff_decision = evaluator.judge_tool_sufficiency(ext_latex)

            ground_value = ground_decision.get("is_supported")
            suff_value = suff_decision.get("is_sufficient")
            toolacc_value: Optional[bool]
            if ground_value is None or suff_value is None:
                toolacc_value = None
            else:
                toolacc_value = bool(ground_value and suff_value)

            if ground_value is None:
                stats["ground_abstained"] += 1
            else:
                stats["ground_evaluable"] += 1
                stats["ground_positive"] += int(bool(ground_value))

            if suff_value is None:
                stats["suff_abstained"] += 1
            else:
                stats["suff_evaluable"] += 1
                stats["suff_positive"] += int(bool(suff_value))

            if toolacc_value is None:
                stats["toolacc_abstained"] += 1
            else:
                stats["toolacc_evaluable"] += 1
                stats["toolacc_positive"] += int(bool(toolacc_value))

            if ground_value is True and suff_value is not None:
                stats["suff_given_ground_total"] += 1
                stats["suff_given_ground_positive"] += int(bool(suff_value))

            is_matched = bool(suff_value is True)
            match_reason = str(suff_decision.get("reason", "") or "")
            tool_success_proxy = bool(toolacc_value is True)
            cited_tool_success_proxy = bool(is_retrieved and toolacc_value is True)
            alt_source_tool_success_proxy = bool((not is_retrieved) and toolacc_value is True)

            if is_retrieved:
                stats["cited_doc_matched"] += 1
                stats["retrieved_correctly"] += 1
            if tool_success_proxy:
                stats["tool_success_proxy"] += 1
            if is_matched:
                stats["latex_matched"] += 1
            if cited_tool_success_proxy:
                stats["cited_tool_success_proxy"] += 1
            if alt_source_tool_success_proxy:
                stats["alt_source_tool_success_proxy"] += 1

            logger.info(
                "Case=%s file=%s cited_doc=%s Ground=%s Suff=%s ToolAcc=%s alt_source=%s downloads=%d",
                case_id,
                sel_file,
                is_retrieved,
                ground_value,
                suff_value,
                toolacc_value,
                alt_source_tool_success_proxy,
                successful_downloads,
            )

            results.append(
                {
                    "id": case_id,
                    "selected_file": sel_file,
                    "search_query": search_query,
                    "query_prompt": query_prompt,
                    "planning_prompt": planning_prompt,
                    "query_anchor": query_anchor,
                    "query_anchor_source": query_anchor_source,
                    "shortlist_manifest": shortlist_manifest,
                    "download_manifest": download_manifest,
                    "cite_recall_at_20": cite_recall_at_20,
                    "cite_found_rank": cite_found_rank,
                    "is_retrieved": is_retrieved,
                    "cited_doc_match": is_retrieved,
                    "retrieval_reason": retr_reason,
                    "gt_title": gt_title,
                    "pdf_title": pdf_title,
                    "selected_title_source": selected_title_source,
                    "gt_citation": gt_data["target_citation_content"],
                    "Ground": ground_value,
                    "ground_label": ground_decision.get("label"),
                    "ground_reason": ground_decision.get("reason", ""),
                    "ground_confidence": ground_decision.get("confidence"),
                    "ground_evidence_span": ground_decision.get("evidence_span", ""),
                    "Suff": suff_value,
                    "suff_label": suff_decision.get("label"),
                    "suff_reason": suff_decision.get("reason", ""),
                    "suff_confidence": suff_decision.get("confidence"),
                    "ToolAcc": toolacc_value,
                    "is_matched": is_matched,
                    "tool_success_proxy": tool_success_proxy,
                    "cited_tool_success_proxy": cited_tool_success_proxy,
                    "alt_source_tool_success_proxy": alt_source_tool_success_proxy,
                    "match_reason": match_reason,
                    "predicted_anchor": predicted_anchor,
                    "gold_anchor": gt_data["target_anchor_hint"] if args.track == "raw" else None,
                    "planning_generation_reason": planning_generation_reason,
                    "is_planning_matched": is_planning_matched,
                    "planning_match_reason": planning_match_reason,
                    "gt_latex": gt_data["target_lemma_latex"],
                    "extracted_latex": ext_latex,
                    "paper_id": paper_meta.get("paper_id", ""),
                    "paper_title": paper_meta.get("title", ""),
                    "domain": domain,
                    "tool_family": tool_family,
                    "source_type": cited_source_type,
                    "solver_output": solver_out,
                    "judge": {
                        "ground": ground_decision,
                        "suff": suff_decision,
                    },
                    "meta": {
                        "solver_model": solver_cfg.model_name,
                        "judge_model": judge_cfg.model_name,
                        "track": args.track,
                        "download_root": download_root,
                        "backend": args.backend,
                        "snapshot": args.snapshot or None,
                        "search_log": args.search_log or None,
                        "used_cached_search": used_cached_search,
                        "shortlist_k": args.shortlist_k,
                        "shortlist_size": args.shortlist_size,
                        "context_variant": args.context_variant,
                        "local_window": args.local_window,
                        "skip_planning_judge": args.skip_planning_judge,
                        "shard_index": args.shard_index,
                        "num_shards": args.num_shards,
                    },
                }
                )
            write_result_payload(
                args.result_file,
                stats=stats,
                results=results,
                meta=result_meta,
                completed=False,
            )
        except Exception as exc:
            error_message = error_message or str(exc)
            logger.warning("Evaluation failed for %s: %s", case_id, error_message)
            results.append(
                {
                    "id": case_id,
                    "selected_file": None,
                    "search_query": search_query,
                    "query_prompt": query_prompt,
                    "planning_prompt": planning_prompt,
                    "query_anchor": query_anchor,
                    "query_anchor_source": query_anchor_source,
                    "shortlist_manifest": shortlist_manifest,
                    "download_manifest": download_manifest,
                    "cite_recall_at_20": cite_recall_at_20,
                    "cite_found_rank": cite_found_rank,
                    "is_retrieved": False,
                    "cited_doc_match": False,
                    "retrieval_reason": error_message,
                    "gt_title": extract_citation_title(gt_data["target_citation_content"]),
                    "pdf_title": "",
                    "selected_title_source": "none",
                    "gt_citation": gt_data["target_citation_content"],
                    "Ground": None,
                    "ground_label": None,
                    "ground_reason": error_message,
                    "ground_confidence": None,
                    "ground_evidence_span": "",
                    "Suff": None,
                    "suff_label": None,
                    "suff_reason": error_message,
                    "suff_confidence": None,
                    "ToolAcc": None,
                    "is_matched": False,
                    "tool_success_proxy": False,
                    "cited_tool_success_proxy": False,
                    "alt_source_tool_success_proxy": False,
                    "match_reason": error_message,
                    "predicted_anchor": predicted_anchor,
                    "gold_anchor": gt_data["target_anchor_hint"] if args.track == "raw" else None,
                    "planning_generation_reason": planning_generation_reason,
                    "is_planning_matched": is_planning_matched,
                    "planning_match_reason": planning_match_reason,
                    "gt_latex": gt_data["target_lemma_latex"],
                    "extracted_latex": None,
                    "paper_id": paper_meta.get("paper_id", ""),
                    "paper_title": paper_meta.get("title", ""),
                    "domain": domain,
                    "tool_family": tool_family,
                    "source_type": cited_source_type,
                    "solver_output": solver_out,
                    "error": error_message,
                    "judge": {
                        "ground": None,
                        "suff": None,
                    },
                    "meta": {
                        "solver_model": solver_cfg.model_name,
                        "judge_model": judge_cfg.model_name,
                        "track": args.track,
                        "download_root": download_root,
                        "backend": args.backend,
                        "snapshot": args.snapshot or None,
                        "search_log": args.search_log or None,
                        "used_cached_search": used_cached_search,
                        "shortlist_k": args.shortlist_k,
                        "shortlist_size": args.shortlist_size,
                        "context_variant": args.context_variant,
                        "local_window": args.local_window,
                        "skip_planning_judge": args.skip_planning_judge,
                        "shard_index": args.shard_index,
                        "num_shards": args.num_shards,
                    },
                }
                )
            write_result_payload(
                args.result_file,
                stats=stats,
                results=results,
                meta=result_meta,
                completed=False,
            )

    total = stats["total"]
    summary = compute_summary_from_stats(stats, results)
    if total > 0:
        if summary["AnchorAcc(x_raw)"] is not None:
            logger.info("AnchorAcc(x_raw): %.2f%%", 100.0 * summary["AnchorAcc(x_raw)"])
        logger.info("CiteRecall@20: %.2f%%", 100.0 * summary["CiteRecall@20"])
        logger.info("GroundRate: %.2f%%", 100.0 * summary["GroundRate"])
        logger.info("ToolAcc: %.2f%%", 100.0 * summary["ToolAcc"])
        logger.info("AltSourceToolAcc: %.2f%%", 100.0 * summary["AltSourceToolAcc"])
        logger.info(
            "Average shortlisted candidates per case: %.2f",
            stats["shortlisted_candidates_total"] / total,
        )
    if stats["planning_total"] > 0:
        logger.info("Planning accuracy: %.2f%%", 100.0 * stats["planning_matched"] / stats["planning_total"])
    write_result_payload(
        args.result_file,
        stats=stats,
        results=results,
        meta=result_meta,
        completed=True,
    )

    logger.info("Saved to %s", args.result_file)


if __name__ == "__main__":
    main()
