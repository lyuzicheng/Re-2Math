from __future__ import annotations

import argparse
import csv
import html
import json
import re
import time
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Dict, List, Optional

import arxiv
import requests


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs" / "published_paper_config.json"
DEFAULT_INPUT = ROOT / "construction" / "outputs" / "published_manifest_bootstrap.json"
DEFAULT_OUT = ROOT / "construction" / "outputs" / "published_manifest_bootstrap_matched.json"
DEFAULT_REVIEW_OUT = ROOT / "construction" / "outputs" / "published_manifest_bootstrap_review.json"
DEFAULT_LOCAL_INDEX = ROOT / "construction" / "outputs" / "published_manifest_openalex_arxiv_linked.json"
OPENALEX_BASE = "https://api.openalex.org"


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def normalize_space(text: str) -> str:
    cleaned = html.unescape(text or "")
    cleaned = re.sub(r"<[^>]+>", " ", cleaned)
    cleaned = re.sub(r"(?<=[a-z])(?=[A-Z])", " ", cleaned)
    return re.sub(r"\s+", " ", cleaned.strip())


def normalize_title(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", (text or "").lower())


def title_keyword_tokens(text: str, limit: int = 6) -> List[str]:
    stopwords = {
        "a",
        "an",
        "and",
        "as",
        "at",
        "by",
        "for",
        "from",
        "in",
        "into",
        "of",
        "on",
        "or",
        "the",
        "to",
        "via",
        "with",
    }
    tokens = re.findall(r"[A-Za-z0-9]+", normalize_space(text).lower())
    keywords: List[str] = []
    for token in tokens:
        if len(token) <= 2 or token in stopwords:
            continue
        keywords.append(token)
        if len(keywords) >= limit:
            break
    return keywords


def normalize_doi(text: str) -> str:
    return normalize_space(text).rstrip(".;,").lower()


def normalize_arxiv_id(text: str) -> str:
    clean = normalize_space(text)
    if not clean:
        return ""
    clean = re.sub(r"^(https?://)?(www\.)?arxiv\.org/(abs|pdf|e-print)/", "", clean, flags=re.IGNORECASE)
    clean = clean.replace(".pdf", "")
    clean = clean.split("?")[0].strip("/")
    if re.fullmatch(r"\d{4}\.\d{4,5}(v\d+)?", clean):
        return clean
    if re.fullmatch(r"[a-z\-]+(?:\.[A-Z]{2})?/\d{7}(v\d+)?", clean, flags=re.IGNORECASE):
        return clean
    return ""


def surname(author_name: str) -> str:
    parts = [part for part in re.split(r"\s+", normalize_space(author_name)) if part]
    return re.sub(r"[^a-z]", "", parts[-1].lower()) if parts else ""


def extract_openalex_id(text: str) -> str:
    raw = normalize_space(text)
    if not raw:
        return ""
    return raw.rstrip("/").split("/")[-1]


@dataclass
class CandidateScore:
    result: Dict[str, Any]
    title_similarity: float
    first_author_match: bool
    year_gap: Optional[int]
    doi_exact: bool

    @property
    def composite(self) -> float:
        score = self.title_similarity
        if self.first_author_match:
            score += 0.03
        if self.doi_exact:
            score += 0.5
        if self.year_gap is not None:
            score += max(0.0, 0.03 - 0.01 * min(self.year_gap, 3))
        return score


def build_search_queries(title: str, first_author: str, doi: str) -> List[str]:
    queries: List[str] = []
    clean_title = normalize_space(title)
    if doi:
        queries.append(f'all:"{doi}"')
    if clean_title:
        if first_author:
            queries.append(f'ti:"{clean_title}" AND au:"{first_author}"')
        queries.append(f'ti:"{clean_title}"')
        keywords = title_keyword_tokens(clean_title)
        if len(keywords) >= 3:
            keyword_query = " AND ".join(f'all:{token}' for token in keywords)
            if first_author:
                queries.append(f'{keyword_query} AND au:"{first_author}"')
            queries.append(keyword_query)

    deduped: List[str] = []
    seen = set()
    for query in queries:
        if query not in seen:
            seen.add(query)
            deduped.append(query)
    return deduped


def candidate_record_from_arxiv(result: arxiv.Result) -> Dict[str, Any]:
    return {
        "arxiv_id": result.get_short_id(),
        "arxiv_title": result.title or "",
        "doi": normalize_doi(getattr(result, "doi", "") or ""),
        "authors": [getattr(author, "name", "") for author in (result.authors or [])],
        "published_year": result.published.year if getattr(result, "published", None) else None,
        "entry_id": result.entry_id,
        "latex_link": f"https://arxiv.org/e-print/{result.get_short_id()}",
    }


def candidate_record_from_openalex_location(work: Dict[str, Any], location: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    source = location.get("source", {}) or {}
    location_id = str(location.get("id", "") or "")
    landing = str(location.get("landing_page_url", "") or "")
    pdf_url = str(location.get("pdf_url", "") or "")
    raw_source_name = str(location.get("raw_source_name", "") or "")
    source_name = str(source.get("display_name", "") or "")
    ids = work.get("ids", {}) or {}

    arxiv_raw = ""
    for candidate in [ids.get("arxiv", ""), landing, pdf_url, location_id, raw_source_name, source_name]:
        candidate_text = str(candidate or "")
        if "arxiv" in candidate_text.lower():
            arxiv_raw = candidate_text
            break
    arxiv_id = normalize_arxiv_id(arxiv_raw)
    if not arxiv_id:
        return None

    authorships = work.get("authorships", []) or []
    authors = [
        getattr((item.get("author", {}) or {}), "get", lambda *_: "")("display_name", "")
        if isinstance(item, dict)
        else ""
        for item in authorships
    ]
    authors = [str(author).strip() for author in authors if str(author).strip()]

    return {
        "arxiv_id": arxiv_id,
        "arxiv_title": str(work.get("display_name", "") or ""),
        "doi": normalize_doi(str(work.get("doi", "") or "")),
        "authors": authors,
        "published_year": work.get("publication_year"),
        "entry_id": f"https://arxiv.org/abs/{arxiv_id}",
        "latex_link": f"https://arxiv.org/e-print/{arxiv_id}",
    }


def candidate_record_from_row_hint(row: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    arxiv_id = normalize_arxiv_id(str(row.get("openalex_arxiv_id", "") or ""))
    if not arxiv_id:
        return None
    return {
        "arxiv_id": arxiv_id,
        "arxiv_title": str(row.get("published_title", "") or row.get("title", "") or ""),
        "doi": normalize_doi(str(row.get("doi", "") or "")),
        "authors": list(row.get("published_authors", []) or row.get("authors", []) or []),
        "published_year": row.get("year"),
        "entry_id": f"https://arxiv.org/abs/{arxiv_id}",
        "latex_link": f"https://arxiv.org/e-print/{arxiv_id}",
    }


def fetch_openalex_work(session: requests.Session, work_id: str) -> Optional[Dict[str, Any]]:
    clean_id = extract_openalex_id(work_id)
    if not clean_id:
        return None
    response = session.get(
        f"{OPENALEX_BASE}/works/{clean_id}",
        params={"select": "id,ids,doi,display_name,publication_year,authorships,locations,best_oa_location,primary_location"},
        timeout=30,
    )
    response.raise_for_status()
    return response.json()


def fetch_openalex_candidates(session: requests.Session, row: Dict[str, Any]) -> List[Dict[str, Any]]:
    work_id = str(row.get("openalex_work_id", "") or "")
    if not work_id:
        return []
    try:
        work = fetch_openalex_work(session, work_id)
    except Exception:
        return []
    if not work:
        return []

    candidates: List[Dict[str, Any]] = []
    seen = set()
    locations = []
    for key in ["best_oa_location", "primary_location"]:
        loc = work.get(key)
        if isinstance(loc, dict):
            locations.append(loc)
    locations.extend([loc for loc in (work.get("locations", []) or []) if isinstance(loc, dict)])
    for location in locations:
        record = candidate_record_from_openalex_location(work, location)
        if not record:
            continue
        key = record.get("arxiv_id") or record.get("entry_id")
        if key and key not in seen:
            seen.add(key)
            candidates.append(record)
    return candidates


def load_local_index(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []

    if path.suffix.lower() == ".csv":
        with path.open("r", encoding="utf-8", newline="") as f:
            rows = list(csv.DictReader(f))
    else:
        payload = load_json(path)
        if isinstance(payload, list):
            rows = payload
        elif isinstance(payload, dict):
            rows = payload.get("selected_papers") or payload.get("papers") or []
        else:
            rows = []

    records: List[Dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        arxiv_id = str(row.get("arxiv_id", "") or row.get("id", "") or "").strip()
        title = str(row.get("arxiv_title", "") or row.get("title", "") or row.get("published_title", "") or "").strip()
        if not arxiv_id and not title:
            continue
        authors = row.get("authors") or row.get("published_authors") or []
        if isinstance(authors, str):
            authors = [part.strip() for part in authors.split(";") if part.strip()]
        published_raw = str(row.get("published", "") or row.get("publication_date", "") or "").strip()
        year_match = re.match(r"(\d{4})", published_raw)
        records.append(
            {
                "arxiv_id": arxiv_id,
                "arxiv_title": title,
                "doi": normalize_doi(str(row.get("doi", "") or "")),
                "authors": list(authors),
                "published_year": int(year_match.group(1)) if year_match else None,
                "entry_id": str(row.get("paper_link", "") or row.get("arxiv_url", "") or f"https://arxiv.org/abs/{arxiv_id}"),
                "latex_link": str(row.get("latex_link", "") or (f"https://arxiv.org/e-print/{arxiv_id}" if arxiv_id else "")),
            }
        )
    return records


def fetch_local_candidates(index_rows: List[Dict[str, Any]], title: str, first_author: str, doi: str) -> List[Dict[str, Any]]:
    target_title = normalize_title(title)
    matches: List[Dict[str, Any]] = []
    for row in index_rows:
        cand_title = normalize_title(str(row.get("arxiv_title", "") or ""))
        cand_doi = normalize_doi(str(row.get("doi", "") or ""))
        cand_first_author = surname((row.get("authors") or [""])[0] if row.get("authors") else "")
        title_score = SequenceMatcher(None, target_title, cand_title).ratio() if target_title and cand_title else 0.0
        if doi and cand_doi and doi == cand_doi:
            matches.append(row)
            continue
        if target_title and cand_title and title_score >= 0.6:
            matches.append(row)
            continue
        if first_author and cand_first_author and first_author == cand_first_author and title_score >= 0.4:
            matches.append(row)
    deduped: List[Dict[str, Any]] = []
    seen = set()
    for row in matches:
        key = row.get("arxiv_id") or row.get("entry_id")
        if key and key not in seen:
            seen.add(key)
            deduped.append(row)
    return deduped


def fetch_remote_candidates(client: arxiv.Client, title: str, first_author: str, doi: str, max_results: int) -> List[Dict[str, Any]]:
    all_results: List[Dict[str, Any]] = []
    seen_ids = set()
    for query in build_search_queries(title, first_author, doi):
        search = arxiv.Search(
            query=query,
            max_results=max_results,
            sort_by=arxiv.SortCriterion.Relevance,
            sort_order=arxiv.SortOrder.Descending,
        )
        for attempt in range(3):
            try:
                results = list(client.results(search))
                for result in results:
                    short_id = result.get_short_id()
                    if short_id not in seen_ids:
                        seen_ids.add(short_id)
                        all_results.append(candidate_record_from_arxiv(result))
                break
            except Exception as exc:
                if "HTTP 429" not in str(exc) or attempt == 2:
                    results = []
                    break
                time.sleep(6.0 * (attempt + 1))
        if all_results:
            break
    return all_results


def fetch_candidates(
    client: Optional[arxiv.Client],
    local_index: List[Dict[str, Any]],
    title: str,
    first_author: str,
    doi: str,
    max_results: int,
) -> List[Dict[str, Any]]:
    local_candidates = fetch_local_candidates(local_index, title=title, first_author=first_author, doi=doi)
    if local_candidates:
        return local_candidates
    if client is None:
        return []
    return fetch_remote_candidates(client, title=title, first_author=first_author, doi=doi, max_results=max_results)


def score_candidate(row: Dict[str, Any], result: Dict[str, Any]) -> CandidateScore:
    target_title = str(row.get("published_title", "") or row.get("title", "") or "")
    target_authors = list(row.get("published_authors", []) or row.get("authors", []) or [])
    target_first_author = surname(target_authors[0] if target_authors else "")
    target_year = row.get("year")
    target_doi = normalize_doi(str(row.get("doi", "") or ""))

    arxiv_title = str(result.get("arxiv_title", "") or "")
    title_similarity = SequenceMatcher(None, normalize_title(target_title), normalize_title(arxiv_title)).ratio()
    if normalize_title(target_title) and normalize_title(arxiv_title) and (
        normalize_title(target_title) in normalize_title(arxiv_title)
        or normalize_title(arxiv_title) in normalize_title(target_title)
    ):
        title_similarity = max(title_similarity, 0.99)

    arxiv_authors = list(result.get("authors", []) or [])
    arxiv_first_author = surname(arxiv_authors[0] if arxiv_authors else "")
    first_author_match = bool(target_first_author and arxiv_first_author and target_first_author == arxiv_first_author)

    arxiv_year = result.get("published_year")
    year_gap = abs(int(target_year) - arxiv_year) if target_year and arxiv_year else None
    doi_exact = bool(target_doi and normalize_doi(str(result.get("doi", "") or "")) == target_doi)

    return CandidateScore(
        result=result,
        title_similarity=title_similarity,
        first_author_match=first_author_match,
        year_gap=year_gap,
        doi_exact=doi_exact,
    )


def classify_match(row: Dict[str, Any], scores: List[CandidateScore], config: dict) -> tuple[Optional[CandidateScore], str, float, str]:
    rules = config.get("matching_rules", {}) or {}
    threshold = float(rules.get("title_similarity_threshold", 0.94))
    require_author = bool(rules.get("require_first_author_match", True))
    year_tolerance = int(rules.get("year_tolerance", 2))
    require_unique = bool(rules.get("require_unique_best_candidate", True))

    if not scores:
        return None, "no_arxiv_twin", 0.0, "No arXiv candidates found."

    scores = sorted(scores, key=lambda item: item.composite, reverse=True)
    best = scores[0]
    runner_up = scores[1] if len(scores) > 1 else None

    if best.doi_exact:
        return best, "doi_exact", 1.0, "Published DOI matches arXiv candidate DOI."

    year_ok = best.year_gap is None or best.year_gap <= year_tolerance
    author_ok = best.first_author_match or not require_author
    unique_ok = not require_unique or runner_up is None or (best.composite - runner_up.composite) >= 0.03

    if best.title_similarity >= threshold and author_ok and year_ok and unique_ok:
        return best, "title_author_year", best.title_similarity, "High title similarity with compatible author and year."

    if runner_up is not None and abs(best.composite - runner_up.composite) < 0.03:
        return None, "needs_manual_review", best.title_similarity, "Multiple arXiv candidates are too close."

    if not author_ok:
        return None, "author_mismatch", best.title_similarity, "Best title match failed first-author check."

    if not year_ok:
        return None, "weak_title_alignment", best.title_similarity, "Best title match failed year compatibility."

    return None, "weak_title_alignment", best.title_similarity, "No candidate met the automatic matching threshold."


def apply_match(row: Dict[str, Any], winner: Optional[CandidateScore], status: str, confidence: float, notes: str) -> Dict[str, Any]:
    updated = dict(row)
    updated["match_notes"] = notes
    updated["match_confidence"] = confidence

    if winner is None:
        updated["status"] = status
        return updated

    result = winner.result
    short_id = str(result.get("arxiv_id", "") or "")
    updated["arxiv_id"] = short_id
    updated["arxiv_title"] = str(result.get("arxiv_title", "") or "")
    updated["arxiv_url"] = str(result.get("entry_id", "") or f"https://arxiv.org/abs/{short_id}")
    updated["latex_link"] = str(result.get("latex_link", "") or (f"https://arxiv.org/e-print/{short_id}" if short_id else ""))
    updated["match_type"] = status
    updated["status"] = "matched_auto" if status in {"doi_exact", "title_author_year"} else "matched_manual"
    return updated


def write_payloads(out_path: Path, review_path: Path, config_name: str, input_name: str, matched_rows: List[dict], review_rows: List[dict], checkpoint_label: Optional[str] = None) -> None:
    meta = {
        "config": config_name,
        "input": input_name,
        "row_count": len(matched_rows),
        "matched_count": sum(1 for row in matched_rows if row.get("status") in {"matched_auto", "matched_manual"}),
        "review_count": len(review_rows),
    }
    if checkpoint_label:
        meta["checkpoint"] = checkpoint_label
    out_payload = {"meta": meta, "papers": matched_rows}
    review_payload = {"meta": meta, "papers": review_rows}
    out_path.parent.mkdir(parents=True, exist_ok=True)
    review_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out_payload, indent=2, ensure_ascii=False), encoding="utf-8")
    review_path.write_text(json.dumps(review_payload, indent=2, ensure_ascii=False), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Match published-paper manifest rows to arXiv twins.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--input", default=str(DEFAULT_INPUT))
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument("--review-out", default=str(DEFAULT_REVIEW_OUT))
    parser.add_argument("--max-results", type=int, default=8)
    parser.add_argument("--local-index", default=str(DEFAULT_LOCAL_INDEX), help="Optional local arXiv metadata cache (JSON/CSV) used before remote API search.")
    parser.add_argument("--disable-remote", action="store_true", help="Only use the local metadata index; do not query the arXiv API.")
    parser.add_argument("--checkpoint-every", type=int, default=10, help="Write partial matched/review outputs every N rows.")
    parser.add_argument("--arxiv-delay-seconds", type=float, default=0.5)
    parser.add_argument("--arxiv-retries", type=int, default=1)
    args = parser.parse_args()

    config = load_json(Path(args.config))
    payload = load_json(Path(args.input))
    if isinstance(payload, list):
        rows = list(payload)
    elif isinstance(payload, dict):
        rows = list(payload.get("papers", []) or [])
    else:
        rows = []
    local_index = load_local_index(Path(args.local_index)) if args.local_index else []
    client = None if args.disable_remote else arxiv.Client(
        page_size=min(args.max_results, 20),
        delay_seconds=max(args.arxiv_delay_seconds, 0.0),
        num_retries=max(args.arxiv_retries, 0),
    )
    openalex_session = requests.Session()
    openalex_session.headers.update({"User-Agent": "AI4MathBenchmark/1.0 (published-arxiv-match)"})

    matched_rows: List[dict] = []
    review_rows: List[dict] = []
    out_path = Path(args.out)
    review_path = Path(args.review_out)
    config_name = Path(args.config).name
    input_name = Path(args.input).name
    for idx, row in enumerate(rows, start=1):
        title = str(row.get("published_title", "") or row.get("title", "") or "")
        authors = list(row.get("published_authors", []) or row.get("authors", []) or [])
        first_author = surname(authors[0] if authors else "")
        doi = str(row.get("doi", "") or "")

        candidates: List[Dict[str, Any]] = []
        row_hint = candidate_record_from_row_hint(row)
        if row_hint:
            candidates = [row_hint]
        if not candidates:
            candidates = fetch_openalex_candidates(openalex_session, row)
        if not candidates:
            candidates = fetch_candidates(
                client=client,
                local_index=local_index,
                title=title,
                first_author=first_author,
                doi=doi,
                max_results=args.max_results,
            )
        scores = [score_candidate(row, candidate) for candidate in candidates]
        winner, match_type, confidence, notes = classify_match(row, scores, config)
        updated = apply_match(row, winner, match_type, confidence, notes)
        diagnostics = [
            {
                "arxiv_id": score.result.get("arxiv_id", ""),
                "arxiv_title": score.result.get("arxiv_title", ""),
                "title_similarity": round(score.title_similarity, 4),
                "first_author_match": score.first_author_match,
                "year_gap": score.year_gap,
                "doi_exact": score.doi_exact,
                "composite": round(score.composite, 4),
            }
            for score in sorted(scores, key=lambda item: item.composite, reverse=True)[:5]
        ]
        updated["candidate_diagnostics"] = diagnostics

        matched_rows.append(updated)
        if updated["status"] not in {"matched_auto", "matched_manual"}:
            review_rows.append(updated)

        if args.checkpoint_every > 0 and idx % args.checkpoint_every == 0:
            write_payloads(
                out_path=out_path,
                review_path=review_path,
                config_name=config_name,
                input_name=input_name,
                matched_rows=matched_rows,
                review_rows=review_rows,
                checkpoint_label=f"{idx}/{len(rows)}",
            )
            print(f"Checkpoint: processed {idx}/{len(rows)} rows", flush=True)

    write_payloads(
        out_path=out_path,
        review_path=review_path,
        config_name=config_name,
        input_name=input_name,
        matched_rows=matched_rows,
        review_rows=review_rows,
    )
    print(json.dumps(load_json(out_path)["meta"], indent=2, ensure_ascii=False), flush=True)
    print(f"Saved matched manifest to {out_path}", flush=True)
    print(f"Saved review queue to {review_path}", flush=True)


if __name__ == "__main__":
    main()
