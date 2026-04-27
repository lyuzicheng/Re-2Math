from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs" / "published_paper_config.json"
DEFAULT_SEEDS = ROOT / "configs" / "published_seed_papers.sample.json"
DEFAULT_OUT = ROOT / "construction" / "outputs" / "published_manifest_sample.json"


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def normalize_space(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip())


def slugify(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", (text or "").lower()).strip("-")


def make_paper_key(row: Dict[str, Any]) -> str:
    doi = normalize_space(str(row.get("doi", "") or ""))
    if doi:
        return f"doi:{doi.lower()}"
    venue = slugify(str(row.get("venue", "") or "venue"))
    year = str(row.get("year", "") or "year")
    title = slugify(str(row.get("title", "") or "untitled"))[:80]
    return f"{venue}:{year}:{title}"


def load_seed_rows(path: Path) -> List[dict]:
    suffix = path.suffix.lower()
    if suffix == ".json":
        payload = load_json(path)
        if isinstance(payload, list):
            return payload
        if isinstance(payload, dict):
            papers = payload.get("papers", [])
            if isinstance(papers, list):
                return papers
        raise ValueError(f"Unsupported JSON seed structure in {path}")

    if suffix == ".jsonl":
        rows: List[dict] = []
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    rows.append(json.loads(line))
        return rows

    if suffix == ".csv":
        with path.open("r", encoding="utf-8", newline="") as f:
            return list(csv.DictReader(f))

    raise ValueError(f"Unsupported seed file format: {path}")


def domain_seed_venues(config: dict) -> Dict[str, List[str]]:
    mapping: Dict[str, List[str]] = {}
    for item in config.get("domains", []) or []:
        mapping[str(item.get("domain", "") or "")] = list(item.get("seed_venues", []) or [])
    return mapping


def contains_any(text: str, keywords: Iterable[str]) -> List[str]:
    lowered = (text or "").lower()
    hits: List[str] = []
    for kw in keywords:
        needle = kw.lower().strip()
        if not needle:
            continue
        pattern = r"\b" + re.escape(needle) + r"\b"
        if re.search(pattern, lowered):
            hits.append(kw)
    return hits


def parse_page_count(text: str) -> int:
    match = re.search(r"(\d+)\s*pages?", text or "", flags=re.IGNORECASE)
    return int(match.group(1)) if match else 0


def score_proof_richness(row: Dict[str, Any], config: dict, venues_by_domain: Dict[str, List[str]]) -> tuple[float, List[str], str]:
    score = 0.0
    signals: List[str] = []

    title = normalize_space(str(row.get("title", "") or ""))
    abstract = normalize_space(str(row.get("abstract", "") or row.get("summary", "") or ""))
    venue = normalize_space(str(row.get("venue", "") or ""))
    notes = normalize_space(str(row.get("notes", "") or row.get("comment", "") or ""))
    domain = str(row.get("domain", "") or "")

    haystack = " ".join([title, abstract, venue, notes])
    negatives = contains_any(haystack, config.get("proof_rich_filter", {}).get("exclude_keywords", []))
    if negatives:
        score -= 4.0
        signals.extend(f"exclude:{kw}" for kw in negatives)

    if venue and venue in venues_by_domain.get(domain, []):
        score += 2.0
        signals.append("seed_venue")

    doi = normalize_space(str(row.get("doi", "") or ""))
    if doi:
        score += 2.0
        signals.append("doi")

    if abstract:
        abstract_words = len(abstract.split())
        if abstract_words >= 80:
            score += 1.0
            signals.append("abstract_length_ok")
        elif abstract_words >= 40:
            score += 0.5
            signals.append("abstract_present")

    page_count = parse_page_count(notes)
    if page_count >= 30:
        score += 2.0
        signals.append(f"pages:{page_count}")
    elif page_count >= 20:
        score += 1.5
        signals.append(f"pages:{page_count}")
    elif page_count >= 10:
        score += 0.5
        signals.append(f"pages:{page_count}")

    positive_terms = ["theorem", "proof", "lemma", "proposition", "corollary", "appendix"]
    found_terms = contains_any(haystack, positive_terms)
    if found_terms:
        score += min(1.5, 0.5 * len(found_terms))
        signals.extend(f"signal:{term}" for term in found_terms[:3])

    status = "published_candidate"
    if negatives:
        status = "rejected"
    return score, signals, status


def compute_collection_plan(config: dict) -> Dict[str, Any]:
    domains = [str(item.get("domain", "") or "") for item in config.get("domains", []) or [] if str(item.get("domain", "") or "")]
    selection_defaults = config.get("selection_defaults", {}) or {}
    build_targets = config.get("build_targets", {}) or {}
    balanced = bool(build_targets.get("balanced_by_domain", True))
    domain_count = max(1, len(domains))

    total_instances = int(build_targets.get("target_total_instances") or 0)
    if not total_instances:
        total_instances = int(selection_defaults.get("target_instances_per_domain") or 0) * domain_count

    instances_per_domain = int(build_targets.get("target_instances_per_domain") or 0)
    if not instances_per_domain:
        instances_per_domain = math.ceil(total_instances / domain_count) if balanced else int(selection_defaults.get("target_instances_per_domain") or 0)

    expected_instances_per_matched_paper = float(build_targets.get("expected_instances_per_matched_paper") or 1.0)
    expected_match_rate = float(build_targets.get("expected_match_rate") or 1.0)
    matched_papers_per_domain = int(build_targets.get("target_matched_papers_per_domain") or 0)
    if not matched_papers_per_domain:
        matched_papers_per_domain = math.ceil(instances_per_domain / max(expected_instances_per_matched_paper, 0.1))

    candidate_papers_per_domain = int(build_targets.get("target_candidates_per_domain") or 0)
    if not candidate_papers_per_domain:
        candidate_papers_per_domain = math.ceil(matched_papers_per_domain / max(expected_match_rate, 0.1))

    return {
        "domain_count": domain_count,
        "balanced_by_domain": balanced,
        "target_total_instances": total_instances,
        "target_instances_per_domain": instances_per_domain,
        "expected_instances_per_matched_paper": expected_instances_per_matched_paper,
        "expected_match_rate": expected_match_rate,
        "target_matched_papers_per_domain": matched_papers_per_domain,
        "target_candidates_per_domain": candidate_papers_per_domain,
        "domains": domains,
    }


def row_sort_key(row: Dict[str, Any]) -> Tuple[float, int, int, str]:
    has_doi = 1 if str(row.get("doi", "") or "").strip() else 0
    has_openalex_arxiv = 1 if bool(row.get("openalex_has_arxiv_link")) else 0
    seed_venue = 1 if "seed_venue" in set(row.get("proof_rich_signals", []) or []) else 0
    return (
        float(row.get("proof_rich_score", 0.0) or 0.0),
        has_openalex_arxiv,
        has_doi,
        seed_venue,
        str(row.get("published_title", "") or ""),
    )


def select_balanced_rows(rows: List[dict], config: dict) -> Tuple[List[dict], Dict[str, Any]]:
    plan = compute_collection_plan(config)
    if not rows:
        return [], {"selection_plan": plan, "domain_summary": {}, "venue_summary": {}}

    max_candidates_per_venue = int(config.get("selection_defaults", {}).get("max_candidates_per_venue") or 0) or None
    target_candidates_per_domain = int(plan["target_candidates_per_domain"])

    rows_by_domain: Dict[str, List[dict]] = defaultdict(list)
    for row in rows:
        rows_by_domain[str(row.get("domain", "") or "")].append(row)

    selected: List[dict] = []
    venue_usage: Counter[Tuple[str, str]] = Counter()
    dropped_by_domain: Dict[str, int] = {}
    for domain in plan["domains"]:
        domain_rows = rows_by_domain.get(domain, [])
        domain_rows = sorted(domain_rows, key=row_sort_key, reverse=True)
        kept = 0
        dropped = 0
        for row in domain_rows:
            venue = str(row.get("venue", "") or "")
            venue_key = (domain, venue)
            if max_candidates_per_venue and venue and venue_usage[venue_key] >= max_candidates_per_venue:
                dropped += 1
                continue
            if kept >= target_candidates_per_domain:
                dropped += 1
                continue
            selected.append(row)
            kept += 1
            if venue:
                venue_usage[venue_key] += 1
        dropped_by_domain[domain] = dropped

    domain_summary: Dict[str, Any] = {}
    for domain in plan["domains"]:
        domain_rows = [row for row in selected if str(row.get("domain", "") or "") == domain]
        status_counts = Counter(str(row.get("status", "") or "") for row in domain_rows)
        venue_counts = Counter(str(row.get("venue", "") or "") for row in domain_rows)
        domain_summary[domain] = {
            "selected_candidates": len(domain_rows),
            "target_candidates": target_candidates_per_domain,
            "selected_venues": len(venue_counts),
            "max_candidates_per_venue": max_candidates_per_venue,
            "dropped_candidates": dropped_by_domain.get(domain, 0),
            "status_counts": dict(status_counts),
        }

    venue_summary = {
        domain: dict(sorted(Counter(str(row.get("venue", "") or "") for row in selected if str(row.get("domain", "") or "") == domain).items()))
        for domain in plan["domains"]
    }

    return selected, {
        "selection_plan": plan,
        "domain_summary": domain_summary,
        "venue_summary": venue_summary,
    }


def build_manifest_rows(seed_rows: List[dict], config: dict) -> List[dict]:
    venues_by_domain = domain_seed_venues(config)
    rows: List[dict] = []
    for raw in seed_rows:
        authors = raw.get("authors", [])
        if isinstance(authors, str):
            authors = [normalize_space(part) for part in authors.split(";") if normalize_space(part)]
        authors = [normalize_space(str(author)) for author in authors if normalize_space(str(author))]

        domain = str(raw.get("domain", "") or raw.get("field_label", "") or "").strip()
        score, signals, status = score_proof_richness(raw, config, venues_by_domain)
        row = {
            "paper_key": make_paper_key(raw),
            "domain": domain,
            "published_title": normalize_space(str(raw.get("title", "") or "")),
            "published_authors": authors,
            "abstract": normalize_space(str(raw.get("abstract", "") or raw.get("summary", "") or "")),
            "notes": normalize_space(str(raw.get("notes", "") or raw.get("comment", "") or "")),
            "venue": normalize_space(str(raw.get("venue", "") or "")),
            "year": int(raw.get("year")) if str(raw.get("year", "")).strip() else None,
            "publication_date": normalize_space(str(raw.get("publication_date", "") or "")),
            "doi": normalize_space(str(raw.get("doi", "") or "")).lower(),
            "published_url": normalize_space(str(raw.get("published_url", "") or raw.get("url", "") or "")),
            "published_source_route": normalize_space(str(raw.get("published_source_route", "") or "manual_seed")),
            "proof_rich_score": score,
            "proof_rich_signals": signals,
            "openalex_work_id": normalize_space(str(raw.get("openalex_work_id", "") or "")),
            "openalex_arxiv_id": normalize_space(str(raw.get("openalex_arxiv_id", "") or "")),
            "openalex_has_arxiv_link": bool(raw.get("openalex_has_arxiv_link")),
            "arxiv_id": "",
            "arxiv_title": "",
            "arxiv_url": "",
            "latex_link": "",
            "match_type": "",
            "match_confidence": 0.0,
            "match_notes": "",
            "status": status,
        }
        rows.append(row)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare a published-paper manifest for the published-first, arXiv-backed pipeline.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--seed-file", default=str(DEFAULT_SEEDS))
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    args = parser.parse_args()

    config = load_json(Path(args.config))
    seed_rows = load_seed_rows(Path(args.seed_file))
    manifest_rows = build_manifest_rows(seed_rows, config)
    selected_rows, selection_meta = select_balanced_rows(manifest_rows, config)

    payload = {
        "meta": {
            "config": Path(args.config).name,
            "seed_file": Path(args.seed_file).name,
            "input_row_count": len(manifest_rows),
            "row_count": len(selected_rows),
            **selection_meta,
        },
        "papers": selected_rows,
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(payload["meta"], indent=2, ensure_ascii=False))
    print(f"Saved manifest to {out_path}")


if __name__ == "__main__":
    main()
