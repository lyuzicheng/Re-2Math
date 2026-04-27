from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import shutil
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple
from urllib.parse import quote

import arxiv
import fitz
import requests

from common.citation_matching import compare_titles, is_strict_title_match


logger = logging.getLogger("FullTextResolver")

REQUEST_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"
    ),
    "Accept": "application/pdf,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
}
OPENALEX_API_URL = "https://api.openalex.org"
DOI_RE = re.compile(r"10\.\d{4,9}/[-._;()/:A-Z0-9]+", flags=re.IGNORECASE)
ARXIV_URL_RE = re.compile(r"arxiv\.org/(?:abs|pdf)/([^/?#]+)", flags=re.IGNORECASE)
ARXIV_ID_RE = re.compile(r"\b\d{4}\.\d{4,5}(?:v\d+)?\b", flags=re.IGNORECASE)


def sanitize_filename(text: str, fallback: str) -> str:
    cleaned = re.sub(r"\$.*?\$", "", text or "")
    cleaned = re.sub(r'[\/*?:"<>|]', "", cleaned).strip()
    return (cleaned or fallback)[:120]


def normalize_title(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(text or "").lower())


def title_hash(text: str) -> str:
    return hashlib.sha1(normalize_title(text).encode("utf-8")).hexdigest()[:16]


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


def validate_pdf(path: str | Path) -> bool:
    try:
        doc = fitz.open(str(path))
        valid = len(doc) > 0
        doc.close()
        return valid
    except Exception:
        return False


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def safe_write_json(path: str | Path, payload: Any) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def load_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def clean_arxiv_id(raw: str) -> str:
    value = str(raw or "").strip()
    value = re.sub(r"\.pdf$", "", value, flags=re.IGNORECASE)
    return value


def extract_doi_from_text(*texts: Any) -> str:
    for text in texts:
        raw = str(text or "")
        match = DOI_RE.search(raw)
        if match:
            return match.group(0).rstrip(".,);]")
    return ""


def extract_arxiv_id_from_text(*texts: Any) -> str:
    for text in texts:
        raw = str(text or "")
        url_match = ARXIV_URL_RE.search(raw)
        if url_match:
            return clean_arxiv_id(url_match.group(1))
        id_match = ARXIV_ID_RE.search(raw)
        if id_match:
            return clean_arxiv_id(id_match.group(0))
    return ""


def infer_year(value: Any) -> Optional[int]:
    match = re.search(r"(19|20)\d{2}", str(value or ""))
    return int(match.group(0)) if match else None


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
    seen = set()
    for url in urls:
        if not url or url in seen:
            continue
        seen.add(url)
        expanded.append(url)
        if "arxiv.org/abs/" in url:
            expanded.append(url.replace("/abs/", "/pdf/") + ".pdf")
    return expanded


def looks_like_pdf_url(url: str) -> bool:
    lowered = str(url or "").lower()
    return lowered.endswith(".pdf") or "/pdf/" in lowered or "type=pdf" in lowered or "chapterpdf" in lowered


def canonical_work_id(candidate: Dict[str, Any]) -> str:
    if candidate.get("openalex_id"):
        return f"openalex_{str(candidate['openalex_id']).rstrip('/').split('/')[-1]}"
    if candidate.get("doi"):
        return f"doi_{re.sub(r'[^a-z0-9]+', '_', str(candidate['doi']).lower()).strip('_')[:120]}"
    if candidate.get("arxiv_id"):
        return f"arxiv_{re.sub(r'[^a-z0-9]+', '_', str(candidate['arxiv_id']).lower())}"
    return f"title_{title_hash(str(candidate.get('title', '') or 'untitled'))}"


def canonicalize_search_candidate(result: Dict[str, Any], rank: int) -> Dict[str, Any]:
    publication_info = result.get("publication_info", {}) or {}
    publication_summary = str(result.get("publication_summary", "") or publication_info.get("summary", "") or "")
    title = str(result.get("title", "") or f"rank_{rank:02d}")
    link = str(result.get("link", "") or "")
    snippet = str(result.get("snippet", "") or "")
    doi = extract_doi_from_text(link, snippet, publication_summary, result.get("doi", ""))
    arxiv_id = extract_arxiv_id_from_text(link, snippet, publication_summary, result.get("arxiv_id", ""))
    year = result.get("year") or infer_year(publication_summary) or infer_year(snippet)
    authors: List[str] = []
    for author in publication_info.get("authors", []) or []:
        name = str((author or {}).get("name", "") or "").strip()
        if name:
            authors.append(name)
    candidate = {
        "title": title,
        "search_rank": int(result.get("rank", rank) or rank),
        "backend": str(result.get("backend", "") or ""),
        "link": link,
        "snippet": snippet,
        "resources": list(result.get("resources", []) or []),
        "publication_info": publication_info,
        "publication_summary": publication_summary,
        "authors": authors,
        "year": year,
        "doi": doi,
        "arxiv_id": arxiv_id,
        "openalex_id": str(result.get("openalex_id", "") or ""),
        "candidate_urls": extract_candidate_urls(result),
        "local_pdf_path": str(result.get("local_pdf_path", "") or ""),
        "backend_score": result.get("backend_score"),
        "backend_doc_id": result.get("backend_doc_id"),
    }
    candidate["canonical_work_id"] = canonical_work_id(candidate)
    return candidate


def _openalex_params() -> Dict[str, str]:
    params: Dict[str, str] = {}
    mailto = str(os.getenv("OPENALEX_MAILTO", "") or "").strip()
    if mailto:
        params["mailto"] = mailto
    return params


def _openalex_get(path_or_url: str, params: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
    try:
        base = path_or_url if path_or_url.startswith("http") else f"{OPENALEX_API_URL}{path_or_url}"
        merged = dict(_openalex_params())
        if params:
            merged.update({k: v for k, v in params.items() if v not in {None, ""}})
        resp = requests.get(base, params=merged, headers={"User-Agent": REQUEST_HEADERS["User-Agent"]}, timeout=30)
        if resp.status_code != 200:
            return None
        return resp.json()
    except Exception as exc:
        logger.debug("OpenAlex request failed for %s: %s", path_or_url, exc)
        return None


def _extract_ids_from_openalex(work: Dict[str, Any]) -> Dict[str, str]:
    ids = work.get("ids", {}) or {}
    doi = str(ids.get("doi", "") or "")
    doi = doi.replace("https://doi.org/", "").replace("http://doi.org/", "")
    primary_location = work.get("primary_location") or {}
    arxiv_id = extract_arxiv_id_from_text(ids.get("arxiv", ""), primary_location.get("pdf_url", ""))
    return {
        "openalex_id": str(work.get("id", "") or ""),
        "doi": doi,
        "arxiv_id": arxiv_id,
    }


def _iter_openalex_locations(work: Dict[str, Any]) -> Iterable[Dict[str, Any]]:
    primary = work.get("primary_location")
    if isinstance(primary, dict) and primary:
        yield primary
    best = work.get("best_oa_location")
    if isinstance(best, dict) and best:
        yield best
    for location in work.get("locations", []) or []:
        if isinstance(location, dict) and location:
            yield location


def _collect_openalex_urls(work: Dict[str, Any]) -> List[Dict[str, str]]:
    urls: List[Dict[str, str]] = []
    seen = set()
    for location in _iter_openalex_locations(work):
        pdf_url = str(location.get("pdf_url", "") or "").strip()
        landing_url = str(location.get("landing_page_url", "") or "").strip()
        source_name = str(((location.get("source") or {}).get("display_name", "") or "")).strip()
        if pdf_url and pdf_url not in seen:
            seen.add(pdf_url)
            urls.append({"url": pdf_url, "origin": f"openalex_pdf:{source_name}" if source_name else "openalex_pdf"})
        if landing_url and landing_url not in seen:
            seen.add(landing_url)
            urls.append({"url": landing_url, "origin": f"openalex_landing:{source_name}" if source_name else "openalex_landing"})
    return urls


def lookup_openalex_work(candidate: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    doi = str(candidate.get("doi", "") or "").strip()
    if doi:
        encoded = quote(f"https://doi.org/{doi}", safe="")
        work = _openalex_get(f"/works/{encoded}")
        if isinstance(work, dict) and work.get("id"):
            return work

    title = str(candidate.get("title", "") or "").strip()
    if not title:
        return None
    payload = _openalex_get("/works", params={"search": title, "per-page": 5})
    results = (payload or {}).get("results", []) or []
    target_year = infer_year(candidate.get("year"))
    best_work: Optional[Dict[str, Any]] = None
    best_score = -1.0
    for work in results:
        display_title = str(work.get("display_name", "") or "").strip()
        is_match, score, _reason = compare_titles(display_title, title, threshold=0.93)
        if not is_match:
            continue
        score_value = float(score)
        work_year = infer_year(work.get("publication_year"))
        if target_year and work_year == target_year:
            score_value += 0.05
        if score_value > best_score:
            best_score = score_value
            best_work = work
    return best_work


def enrich_candidate_with_openalex(candidate: Dict[str, Any]) -> Dict[str, Any]:
    work = lookup_openalex_work(candidate)
    if not work:
        return candidate
    enriched = dict(candidate)
    ids = _extract_ids_from_openalex(work)
    enriched.update({k: v or enriched.get(k, "") for k, v in ids.items()})
    if work.get("display_name"):
        enriched["resolved_title"] = str(work.get("display_name") or "")
    if work.get("publication_year"):
        enriched["year"] = work.get("publication_year")
    openalex_urls = _collect_openalex_urls(work)
    enriched["openalex_urls"] = openalex_urls
    enriched["openalex_work"] = {
        "id": work.get("id", ""),
        "display_name": work.get("display_name", ""),
        "publication_year": work.get("publication_year"),
    }
    enriched["canonical_work_id"] = canonical_work_id(enriched)
    return enriched


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
                entry_id = str(result.entry_id or "")
                best_match = {
                    "title": candidate_title,
                    "pdf_url": getattr(result, "pdf_url", None) or entry_id.replace("/abs/", "/pdf/") + ".pdf",
                    "entry_id": entry_id,
                    "arxiv_id": extract_arxiv_id_from_text(entry_id),
                    "score": score,
                    "match_reason": reason,
                }
    except Exception as exc:
        logger.warning("arXiv title lookup failed for %r: %s", title[:120], exc)
    return best_match


def candidate_resolution_urls(candidate: Dict[str, Any]) -> List[Dict[str, str]]:
    urls: List[Dict[str, str]] = []
    seen = set()

    def add(url: str, origin: str) -> None:
        if not url:
            return
        key = str(url).strip()
        if not key or key in seen:
            return
        seen.add(key)
        urls.append({"url": key, "origin": origin})

    arxiv_id = str(candidate.get("arxiv_id", "") or "").strip()
    if arxiv_id:
        add(f"https://arxiv.org/pdf/{arxiv_id}.pdf", "arxiv_id")

    for item in candidate.get("openalex_urls", []) or []:
        add(str(item.get("url", "") or ""), str(item.get("origin", "openalex") or "openalex"))

    direct_urls = [url for url in candidate.get("candidate_urls", []) if looks_like_pdf_url(url)]
    indirect_urls = [url for url in candidate.get("candidate_urls", []) if url not in direct_urls]
    for url in direct_urls:
        add(url, "scholar_direct")
    for url in indirect_urls:
        add(url, "scholar_link")
    return urls


def download_pdf_from_url(url: str, save_path: str | Path) -> Tuple[bool, str]:
    try:
        response = requests.get(url, headers=REQUEST_HEADERS, stream=True, timeout=30)
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


def _link_or_copy(src: str | Path, dst: str | Path) -> None:
    src_path = Path(src)
    dst_path = Path(dst)
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    if dst_path.exists() or dst_path.is_symlink():
        dst_path.unlink()
    try:
        os.symlink(str(src_path), str(dst_path))
    except Exception:
        shutil.copy2(src_path, dst_path)


def _cache_root_for(download_root: str | Path) -> Path:
    explicit = str(os.getenv("FULLTEXT_CACHE_ROOT", "") or "").strip()
    if explicit:
        return ensure_dir(explicit)
    return ensure_dir(Path(download_root).resolve().parent / "_resolver_cache")


def _cache_paths(cache_root: Path, work_id: str) -> Tuple[Path, Path]:
    folder = ensure_dir(cache_root / work_id[:2] / work_id)
    return folder / "document.pdf", folder / "metadata.json"


def _materialize_manifest_item(
    *,
    case_dir: Path,
    rank: int,
    title: str,
    pdf_path: Path,
    status: str,
    origin: str,
    source_url: str,
    metadata: Dict[str, Any],
) -> Dict[str, Any]:
    safe_title = sanitize_filename(title, f"rank_{rank:02d}")
    filename = f"rank_{rank:02d}_{safe_title}.pdf"
    target_path = case_dir / filename
    _link_or_copy(pdf_path, target_path)
    return {
        "filename": filename,
        "title": title,
        "status": status,
        "acquisition_status": status,
        "origin": origin,
        "acquisition_source": origin,
        "source_url": source_url,
        "resolved_pdf_path": str(target_path),
        "canonical_work_id": metadata.get("canonical_work_id", ""),
        "doi": metadata.get("doi", ""),
        "arxiv_id": metadata.get("arxiv_id", ""),
        "openalex_id": metadata.get("openalex_id", ""),
        "local_pdf_path": str(metadata.get("local_pdf_path", "") or ""),
        "resolution_trace": metadata.get("resolution_trace", []),
    }


def resolve_candidate_document(
    *,
    case_dir: str | Path,
    cache_root: str | Path,
    rank: int,
    result: Dict[str, Any],
) -> Dict[str, Any]:
    case_dir_path = ensure_dir(case_dir)
    candidate = canonicalize_search_candidate(result, rank)
    resolution_trace: List[Dict[str, Any]] = []

    local_pdf_path = str(candidate.get("local_pdf_path", "") or "")
    if local_pdf_path and Path(local_pdf_path).exists() and validate_pdf(local_pdf_path):
        candidate["resolution_trace"] = [{"step": "local_pdf", "status": "linked_local", "path": local_pdf_path}]
        return _materialize_manifest_item(
            case_dir=case_dir_path,
            rank=rank,
            title=str(candidate.get("title", "") or f"rank_{rank:02d}"),
            pdf_path=Path(local_pdf_path),
            status="linked_local",
            origin=str(candidate.get("backend", "offline") or "offline"),
            source_url=str(candidate.get("link", "") or local_pdf_path),
            metadata=candidate,
        )

    candidate = enrich_candidate_with_openalex(candidate)
    candidate.setdefault("resolution_trace", resolution_trace)
    work_id = str(candidate.get("canonical_work_id") or canonical_work_id(candidate))
    candidate["canonical_work_id"] = work_id
    cache_pdf, cache_meta = _cache_paths(Path(cache_root), work_id)

    if cache_pdf.exists() and validate_pdf(cache_pdf):
        if cache_meta.exists():
            try:
                cached_meta = load_json(cache_meta)
                candidate.update({k: v for k, v in cached_meta.items() if v not in {None, ""}})
            except Exception:
                pass
        candidate["resolution_trace"] = [{"step": "cache", "status": "cached_global", "path": str(cache_pdf)}]
        return _materialize_manifest_item(
            case_dir=case_dir_path,
            rank=rank,
            title=str(candidate.get("resolved_title", "") or candidate.get("title", "") or f"rank_{rank:02d}"),
            pdf_path=cache_pdf,
            status="cached_global",
            origin="resolver_cache",
            source_url=str(candidate.get("source_url", "") or candidate.get("link", "") or ""),
            metadata=candidate,
        )

    urls_to_try = candidate_resolution_urls(candidate)
    if not urls_to_try:
        arxiv_match = find_equivalent_arxiv_paper(str(candidate.get("title", "") or ""))
        if arxiv_match:
            candidate["arxiv_id"] = candidate.get("arxiv_id") or arxiv_match.get("arxiv_id", "")
            urls_to_try.append({"url": str(arxiv_match.get("pdf_url", "") or ""), "origin": "title_arxiv_search"})
            if arxiv_match.get("title"):
                candidate["resolved_title"] = arxiv_match["title"]

    for item in urls_to_try:
        url = str(item.get("url", "") or "")
        origin = str(item.get("origin", "resolver") or "resolver")
        success, status = download_pdf_from_url(url, cache_pdf)
        resolution_trace.append({"step": "download", "origin": origin, "url": url, "status": status})
        if success:
            candidate["resolution_trace"] = resolution_trace
            candidate["source_url"] = url
            safe_write_json(cache_meta, candidate)
            return _materialize_manifest_item(
                case_dir=case_dir_path,
                rank=rank,
                title=str(candidate.get("resolved_title", "") or candidate.get("title", "") or f"rank_{rank:02d}"),
                pdf_path=cache_pdf,
                status=status,
                origin=origin,
                source_url=url,
                metadata=candidate,
            )

    candidate["resolution_trace"] = resolution_trace
    candidate["source_url"] = str(candidate.get("link", "") or "")
    safe_write_json(cache_meta, candidate)
    return {
        "filename": None,
        "title": str(candidate.get("resolved_title", "") or candidate.get("title", "") or f"rank_{rank:02d}"),
        "status": "unavailable",
        "acquisition_status": "unavailable",
        "origin": "none",
        "acquisition_source": "none",
        "source_url": str(candidate.get("source_url", "") or candidate.get("link", "") or ""),
        "resolved_pdf_path": "",
        "canonical_work_id": candidate.get("canonical_work_id", ""),
        "doi": candidate.get("doi", ""),
        "arxiv_id": candidate.get("arxiv_id", ""),
        "openalex_id": candidate.get("openalex_id", ""),
        "local_pdf_path": local_pdf_path,
        "resolution_trace": resolution_trace,
    }


def prepare_task_documents(
    case_id: str,
    candidates: List[Dict[str, Any]],
    download_root: str | Path,
) -> Tuple[str, List[Dict[str, Any]]]:
    case_dir = ensure_dir(Path(download_root) / case_id)
    cache_root = _cache_root_for(download_root)
    manifest: List[Dict[str, Any]] = []
    for idx, result in enumerate(candidates, start=1):
        search_rank = int(result.get("rank", idx) or idx)
        item = resolve_candidate_document(
            case_dir=case_dir,
            cache_root=cache_root,
            rank=search_rank,
            result=result,
        )
        item["shortlist_rank"] = idx
        item["rank"] = search_rank
        item["scholar_link"] = str(result.get("link", "") or "")
        item["backend_score"] = result.get("backend_score")
        item["backend_doc_id"] = result.get("backend_doc_id")
        manifest.append(item)
    return str(case_dir), manifest
