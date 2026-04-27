from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from common.dataset_format import (
    DEFAULT_DATASET_FILE,
    get_citation_content,
    get_citation_doi,
    get_cited_arxiv_id,
    load_dataset_as_dict,
)
from evaluation.end_to_end_eval import ensure_oracle_cited_source_pdf, validate_pdf


PREFERRED_FILENAMES = (
    "oracle_cited_source.pdf",
    "oracle_cited_source_resolved.pdf",
    "oracle_cited_source_arxiv_search.pdf",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Backfill shared oracle cited-source cache from existing local PDFs.")
    parser.add_argument("--dataset-file", default=str(ROOT / DEFAULT_DATASET_FILE))
    parser.add_argument("--download-root", required=True)
    parser.add_argument("--out", default="")
    return parser.parse_args()


def has_local_oracle_pdf(case_dir: Path) -> bool:
    for filename in PREFERRED_FILENAMES:
        path = case_dir / filename
        if path.exists() and validate_pdf(str(path)):
            return True
    return False


def main() -> None:
    args = parse_args()
    dataset = load_dataset_as_dict(args.dataset_file)
    download_root = Path(args.download_root)
    results: List[Dict[str, Any]] = []
    status_counts: Counter[str] = Counter()
    origin_counts: Counter[str] = Counter()

    for case_dir in sorted(path for path in download_root.iterdir() if path.is_dir()):
        case_id = case_dir.name
        row = dataset.get(case_id)
        if not row or not has_local_oracle_pdf(case_dir):
            continue
        materialized = ensure_oracle_cited_source_pdf(
            str(case_dir),
            gt_citation=get_citation_content(row),
            cited_doi=get_citation_doi(row),
            cited_arxiv_id=get_cited_arxiv_id(row),
        )
        item = {
            "id": case_id,
            "status": str((materialized or {}).get("status") or "missing"),
            "origin": str((materialized or {}).get("origin") or "none"),
            "filename": (materialized or {}).get("filename"),
            "source_url": str((materialized or {}).get("source_url") or ""),
        }
        status_counts[item["status"]] += 1
        origin_counts[item["origin"]] += 1
        results.append(item)

    payload = {
        "meta": {
            "dataset_file": args.dataset_file,
            "download_root": str(download_root),
            "total_backfilled": len(results),
        },
        "counts": {
            "status": dict(status_counts),
            "origin": dict(origin_counts),
        },
        "results": results,
    }

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
