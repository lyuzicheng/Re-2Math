from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / "construction" / "outputs" / "published_manifest_full_pool_merged.json"


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def normalize_space(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip())


def normalize_title(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", normalize_space(text).lower())


def normalize_name(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", normalize_space(text).lower())


def normalize_doi(text: str) -> str:
    clean = normalize_space(text)
    clean = clean.replace("https://doi.org/", "").replace("http://doi.org/", "")
    return clean.rstrip(".;,").lower()


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


def source_name(path: Path) -> str:
    return path.name


def listify_authors(value: Any) -> List[str]:
    if isinstance(value, list):
        return [normalize_space(str(item)) for item in value if normalize_space(str(item))]
    if isinstance(value, str):
        return [part for part in (normalize_space(x) for x in re.split(r";|, and | and ", value)) if part]
    return []


def normalized_row(row: Dict[str, Any], src: str) -> Dict[str, Any]:
    item = dict(row)
    if not item.get("published_title") and item.get("title"):
        item["published_title"] = item["title"]
    if not item.get("published_authors") and item.get("authors"):
        item["published_authors"] = item["authors"]
    arxiv_id = normalize_arxiv_id(str(item.get("arxiv_id", "") or item.get("openalex_arxiv_id", "") or ""))
    if arxiv_id:
        item["arxiv_id"] = arxiv_id
        item["openalex_arxiv_id"] = arxiv_id
        if not item.get("arxiv_url"):
            item["arxiv_url"] = f"https://arxiv.org/abs/{arxiv_id}"
        if not item.get("latex_link"):
            item["latex_link"] = f"https://arxiv.org/e-print/{arxiv_id}"
    item["doi"] = normalize_doi(str(item.get("doi", "") or ""))
    item["published_title"] = normalize_space(str(item.get("published_title", "") or ""))
    item["published_authors"] = listify_authors(item.get("published_authors"))
    item["venue"] = normalize_space(str(item.get("venue", "") or ""))
    item["domain"] = normalize_space(str(item.get("domain", "") or ""))
    item["source_manifests"] = sorted({*list(item.get("source_manifests", []) or []), src})
    return item


def dedup_key(row: Dict[str, Any]) -> str:
    arxiv_id = normalize_arxiv_id(str(row.get("arxiv_id", "") or row.get("openalex_arxiv_id", "") or ""))
    if arxiv_id:
        return f"arxiv:{arxiv_id}"
    doi = normalize_doi(str(row.get("doi", "") or ""))
    if doi:
        return f"doi:{doi}"
    paper_key = normalize_space(str(row.get("paper_key", "") or ""))
    if paper_key:
        return f"paper_key:{paper_key}"
    title = normalize_title(str(row.get("published_title", "") or row.get("title", "") or ""))
    authors = listify_authors(row.get("published_authors") or row.get("authors"))
    first_author = normalize_name(authors[0]) if authors else ""
    year = str(row.get("year", "") or "")
    return f"title:{title}|author:{first_author}|year:{year}"


def row_quality(row: Dict[str, Any]) -> Tuple[float, float, int, int]:
    nonempty = sum(
        1
        for value in row.values()
        if value not in ("", None, [], {})
    )
    return (
        float(row.get("proof_rich_score", 0.0) or 0.0),
        float(row.get("match_confidence", 0.0) or 0.0),
        1 if str(row.get("arxiv_id", "") or row.get("openalex_arxiv_id", "")).strip() else 0,
        nonempty,
    )


def merge_lists(left: Iterable[Any], right: Iterable[Any]) -> List[Any]:
    merged: List[Any] = []
    seen = set()
    for item in [*(left or []), *(right or [])]:
        marker = json.dumps(item, ensure_ascii=False, sort_keys=True) if isinstance(item, (dict, list)) else str(item)
        if marker in seen:
            continue
        seen.add(marker)
        merged.append(item)
    return merged


def merge_rows(existing: Dict[str, Any], new_row: Dict[str, Any]) -> Dict[str, Any]:
    primary, secondary = (new_row, existing) if row_quality(new_row) > row_quality(existing) else (existing, new_row)
    merged = dict(primary)
    for key, value in secondary.items():
        if key not in merged or merged[key] in ("", None, [], {}):
            merged[key] = value
            continue
        if key in {"proof_rich_signals", "source_manifests", "selection_reasons"}:
            merged[key] = merge_lists(merged.get(key, []), value if isinstance(value, list) else [value])
        elif key == "published_authors":
            merged[key] = merge_lists(merged.get(key, []), listify_authors(value))
    merged["source_manifests"] = sorted(set(merged.get("source_manifests", []) or []))
    return merged


def load_rows(path: Path) -> List[Dict[str, Any]]:
    payload = load_json(path)
    if isinstance(payload, list):
        rows = payload
    elif isinstance(payload, dict):
        rows = payload.get("papers", []) or payload.get("selected_papers", []) or []
    else:
        rows = []
    src = source_name(path)
    return [normalized_row(row, src) for row in rows if isinstance(row, dict)]


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge paper manifests into a single deduplicated pool.")
    parser.add_argument("--inputs", nargs="+", required=True, help="Input manifest paths.")
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    args = parser.parse_args()

    merged_by_key: Dict[str, Dict[str, Any]] = {}
    input_counts: Dict[str, int] = {}
    duplicate_count = 0
    for raw_path in args.inputs:
        path = Path(raw_path)
        rows = load_rows(path)
        input_counts[path.name] = len(rows)
        for row in rows:
            key = dedup_key(row)
            if key in merged_by_key:
                duplicate_count += 1
                merged_by_key[key] = merge_rows(merged_by_key[key], row)
            else:
                merged_by_key[key] = row

    papers = sorted(
        merged_by_key.values(),
        key=lambda row: (
            str(row.get("domain", "") or ""),
            -float(row.get("proof_rich_score", 0.0) or 0.0),
            str(row.get("published_title", "") or ""),
        ),
    )
    domain_counts = Counter(str(row.get("domain", "") or "") for row in papers)
    meta = {
        "sources": [Path(item).name for item in args.inputs],
        "input_counts": input_counts,
        "row_count": len(papers),
        "duplicate_rows_collapsed": duplicate_count,
        "domain_counts": dict(domain_counts),
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({"meta": meta, "papers": papers}, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(meta, indent=2, ensure_ascii=False))
    print(f"Saved merged manifest to {out_path}")


if __name__ == "__main__":
    main()
