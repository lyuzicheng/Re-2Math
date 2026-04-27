from __future__ import annotations

import argparse
import json
import os
import re
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple
import sys

import requests

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from construction.merge_paper_manifests import dedup_key, load_rows
from construction.prepare_published_manifest import build_manifest_rows, select_balanced_rows


DEFAULT_CONFIG = ROOT / "configs" / "published_paper_config.json"
DEFAULT_OUT = ROOT / "construction" / "outputs" / "published_manifest_bootstrap.json"
DEFAULT_RAW_OUT = ROOT / "construction" / "outputs" / "published_candidates_bootstrap_raw.json"
OPENALEX_BASE = "https://api.openalex.org"


def normalize_space(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip())


def normalize_name(text: str) -> str:
    lowered = (text or "").lower().replace("&", " and ")
    lowered = re.sub(r"\bthe\b", " ", lowered)
    lowered = re.sub(r"\band\b", " ", lowered)
    return re.sub(r"[^a-z0-9]+", "", lowered)


def reconstruct_abstract(index: Dict[str, List[int]] | None) -> str:
    if not index:
        return ""
    positions: Dict[int, str] = {}
    for token, token_positions in index.items():
        for pos in token_positions:
            positions[int(pos)] = token
    return " ".join(token for _, token in sorted(positions.items()))


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


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_exclude_keys(paths: Iterable[str]) -> tuple[set[str], Dict[str, int]]:
    keys: set[str] = set()
    counts: Dict[str, int] = {}
    for raw_path in paths or []:
        path = Path(str(raw_path))
        if not path.is_file():
            continue
        rows = load_rows(path)
        counts[path.name] = len(rows)
        for row in rows:
            keys.add(dedup_key(row))
    return keys, counts


def source_aliases(venue_name: str) -> List[str]:
    aliases = [venue_name]
    upper = venue_name.upper()
    if venue_name == "Journal of Machine Learning Research":
        aliases.extend(["JMLR", upper])
    return aliases


def score_source_match(venue_name: str, source: Dict[str, Any]) -> float:
    target = normalize_name(venue_name)
    display_name = str(source.get("display_name", "") or "")
    alt_names = [str(item) for item in source.get("alternate_titles", []) or []]
    candidates = [display_name, *alt_names]
    scores: List[float] = []
    for name in candidates:
        norm = normalize_name(name)
        if not norm:
            continue
        if norm == target:
            scores.append(1.0)
        elif norm in target or target in norm:
            scores.append(0.96)
        else:
            overlap = len(set(re.findall(r"[a-z0-9]+", norm)) & set(re.findall(r"[a-z0-9]+", target)))
            scores.append(0.5 + 0.01 * overlap)
    for alias in source_aliases(venue_name):
        if normalize_name(alias) == normalize_name(display_name):
            scores.append(0.99)
    return max(scores or [0.0])


def openalex_get(path: str, params: Dict[str, Any], max_retries: int = 5, retry_delay_seconds: float = 2.0) -> Dict[str, Any]:
    headers = {
        "User-Agent": "AI4MathBenchmark/1.0 (published-first bootstrap)"
    }
    mailto = os.environ.get("OPENALEX_MAILTO", "").strip()
    query = dict(params)
    if mailto:
        query["mailto"] = mailto
    last_error: Optional[Exception] = None
    for attempt in range(max(1, max_retries)):
        try:
            response = requests.get(f"{OPENALEX_BASE}{path}", params=query, headers=headers, timeout=30)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as exc:
            last_error = exc
            if attempt + 1 >= max(1, max_retries):
                break
            time.sleep(retry_delay_seconds * (attempt + 1))
    if last_error is not None:
        raise last_error
    raise RuntimeError(f"OpenAlex request failed without an exception for {path}")


def resolve_openalex_source(venue_name: str) -> Tuple[Optional[Dict[str, Any]], List[Dict[str, Any]]]:
    payload = openalex_get("/sources", {"search": venue_name, "per-page": 10})
    results = payload.get("results", []) or []
    ranked = sorted(results, key=lambda item: score_source_match(venue_name, item), reverse=True)
    best = ranked[0] if ranked and score_source_match(venue_name, ranked[0]) >= 0.9 else None
    return best, ranked[:5]


def iter_openalex_works(source_id: str, year_start: int, year_end: int, fetch_limit: int, per_page: int, max_pages: int) -> List[Dict[str, Any]]:
    works: List[Dict[str, Any]] = []
    pages = 0
    cursor = "*"
    while len(works) < fetch_limit and pages < max_pages:
        payload = openalex_get(
            "/works",
            {
                "filter": ",".join(
                    [
                        f"primary_location.source.id:{source_id}",
                        f"from_publication_date:{year_start}-01-01",
                        f"to_publication_date:{year_end}-12-31",
                        "type:article",
                        "is_paratext:false",
                    ]
                ),
                "sort": "publication_date:desc",
                "per-page": min(per_page, fetch_limit - len(works)),
                "cursor": cursor,
            },
        )
        batch = payload.get("results", []) or []
        if not batch:
            break
        works.extend(batch)
        pages += 1
        cursor = str((payload.get("meta", {}) or {}).get("next_cursor") or "")
        if not cursor:
            break
    return works[:fetch_limit]


def work_to_seed_row(domain: str, venue_name: str, work: Dict[str, Any]) -> Dict[str, Any]:
    doi = str(work.get("doi", "") or "")
    doi = doi.replace("https://doi.org/", "").replace("http://doi.org/", "")
    authorships = work.get("authorships", []) or []
    authors = [
        normalize_space(str((item.get("author", {}) or {}).get("display_name", "") or ""))
        for item in authorships
        if normalize_space(str((item.get("author", {}) or {}).get("display_name", "") or ""))
    ]
    primary_location = work.get("primary_location", {}) or {}
    source = primary_location.get("source", {}) or {}
    ids = work.get("ids", {}) or {}
    biblio = work.get("biblio", {}) or {}
    first_page = str(biblio.get("first_page", "") or "").strip()
    last_page = str(biblio.get("last_page", "") or "").strip()
    notes = ""
    if first_page and last_page and first_page.isdigit() and last_page.isdigit():
        page_count = int(last_page) - int(first_page) + 1
        if page_count > 0:
            notes = f"{page_count} pages"

    openalex_arxiv_id = normalize_arxiv_id(str(ids.get("arxiv", "") or ""))
    if not openalex_arxiv_id:
        for location in work.get("locations", []) or []:
            if not isinstance(location, dict):
                continue
            candidates = [
                str(location.get("landing_page_url", "") or ""),
                str(location.get("pdf_url", "") or ""),
                str(location.get("id", "") or ""),
                str(location.get("raw_source_name", "") or ""),
            ]
            source_name = str((location.get("source", {}) or {}).get("display_name", "") or "")
            candidates.append(source_name)
            for candidate in candidates:
                if "arxiv" in candidate.lower():
                    openalex_arxiv_id = normalize_arxiv_id(candidate)
                    break
            if openalex_arxiv_id:
                break

    return {
        "domain": domain,
        "title": normalize_space(str(work.get("display_name", "") or "")),
        "authors": authors,
        "venue": normalize_space(str(source.get("display_name", "") or venue_name)),
        "year": work.get("publication_year"),
        "publication_date": normalize_space(str(work.get("publication_date", "") or "")),
        "doi": normalize_space(doi).lower(),
        "published_url": normalize_space(str(primary_location.get("landing_page_url", "") or work.get("id", "") or "")),
        "published_source_route": "metadata_aggregator:openalex",
        "abstract": reconstruct_abstract(work.get("abstract_inverted_index")),
        "notes": notes,
        "openalex_work_id": str(work.get("id", "") or ""),
        "openalex_arxiv_id": openalex_arxiv_id,
        "openalex_has_arxiv_link": bool(openalex_arxiv_id),
    }


def fetch_candidates_from_openalex(config: Dict[str, Any], exclude_keys: Optional[set[str]] = None) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    selection_defaults = config.get("selection_defaults", {}) or {}
    bootstrap = (config.get("bootstrap_sources", {}) or {}).get("openalex", {}) or {}
    year_start = int(selection_defaults.get("publication_year_start") or 2020)
    year_end = int(selection_defaults.get("publication_year_end") or 2026)
    max_candidates_per_venue = int(selection_defaults.get("max_candidates_per_venue") or 40)
    fetch_multiplier = int(bootstrap.get("fetch_multiplier") or 2)
    fetch_limit = max_candidates_per_venue * max(1, fetch_multiplier)
    per_page = int(bootstrap.get("per_page") or 50)
    max_pages = int(bootstrap.get("max_pages_per_venue") or 4)

    raw_rows: List[Dict[str, Any]] = []
    source_diagnostics: List[Dict[str, Any]] = []
    unresolved_venues: List[Dict[str, Any]] = []
    excluded_existing = 0

    for domain_item in config.get("domains", []) or []:
        domain = str(domain_item.get("domain", "") or "")
        for venue_name in domain_item.get("seed_venues", []) or []:
            print(f"[openalex-bootstrap] resolving {domain} | {venue_name}", flush=True)
            source, ranked = resolve_openalex_source(str(venue_name))
            diagnostic = {
                "domain": domain,
                "venue": venue_name,
                "resolved": bool(source),
                "source_id": str((source or {}).get("id", "") or ""),
                "source_display_name": str((source or {}).get("display_name", "") or ""),
                "candidate_sources": [
                    {
                        "id": str(item.get("id", "") or ""),
                        "display_name": str(item.get("display_name", "") or ""),
                        "score": round(score_source_match(str(venue_name), item), 4),
                    }
                    for item in ranked
                ],
            }
            source_diagnostics.append(diagnostic)
            if not source:
                unresolved_venues.append(diagnostic)
                continue

            source_id = str(source.get("id", "") or "")
            works = iter_openalex_works(
                source_id=source_id,
                year_start=year_start,
                year_end=year_end,
                fetch_limit=fetch_limit,
                per_page=per_page,
                max_pages=max_pages,
            )
            print(
                f"[openalex-bootstrap] fetched {len(works)} works for {domain} | {venue_name}",
                flush=True,
            )
            for work in works:
                seed_row = work_to_seed_row(domain, str(venue_name), work)
                if exclude_keys and dedup_key(seed_row) in exclude_keys:
                    excluded_existing += 1
                    continue
                raw_rows.append(seed_row)

    return raw_rows, {
        "source_diagnostics": source_diagnostics,
        "unresolved_venues": unresolved_venues,
        "raw_row_count": len(raw_rows),
        "excluded_existing_count": excluded_existing,
    }


def build_payload(config: Dict[str, Any], raw_rows: List[Dict[str, Any]], diagnostics: Dict[str, Any]) -> Dict[str, Any]:
    manifest_rows = build_manifest_rows(raw_rows, config)
    selected_rows, selection_meta = select_balanced_rows(manifest_rows, config)
    return {
        "meta": {
            "config": Path(DEFAULT_CONFIG).name,
            "source_route": "metadata_aggregator:openalex",
            "input_row_count": len(raw_rows),
            "row_count": len(selected_rows),
            **selection_meta,
            **diagnostics,
        },
        "papers": selected_rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Bootstrap a published-paper candidate manifest from metadata aggregators before arXiv matching.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument("--raw-out", default=str(DEFAULT_RAW_OUT))
    parser.add_argument("--exclude-manifest", action="append", default=[], help="Manifest file whose rows should be excluded from the bootstrap results.")
    args = parser.parse_args()

    config = load_json(Path(args.config))
    exclude_keys, exclude_counts = load_exclude_keys(args.exclude_manifest)
    raw_rows, diagnostics = fetch_candidates_from_openalex(config, exclude_keys=exclude_keys)
    payload = build_payload(config, raw_rows, diagnostics)
    payload["meta"]["exclude_manifests"] = [Path(item).name for item in args.exclude_manifest]
    payload["meta"]["exclude_manifest_counts"] = exclude_counts

    out_path = Path(args.out)
    raw_out_path = Path(args.raw_out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    raw_out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    raw_out_path.write_text(json.dumps({"meta": payload["meta"], "papers": raw_rows}, indent=2, ensure_ascii=False), encoding="utf-8")

    domain_counts = Counter(str(row.get("domain", "") or "") for row in payload.get("papers", []))
    summary = {
        "row_count": payload["meta"]["row_count"],
        "input_row_count": payload["meta"]["input_row_count"],
        "unresolved_venue_count": len(diagnostics.get("unresolved_venues", []) or []),
        "domain_counts": dict(domain_counts),
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"Saved manifest to {out_path}")
    print(f"Saved raw candidate rows to {raw_out_path}")


if __name__ == "__main__":
    main()
