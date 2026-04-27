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
from evaluation.end_to_end_eval import ensure_oracle_cited_source_pdf


OUT_DIR = ROOT / "evaluation" / "outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def resolve_case_ids(dataset: Dict[str, dict], eval_file: str = "", limit: int | None = None) -> List[str]:
    if eval_file:
        payload = load_json(Path(eval_file))
        rows = payload.get("results", payload if isinstance(payload, list) else []) or []
        case_ids = [str(row.get("id") or "").strip() for row in rows]
    else:
        case_ids = sorted(dataset.keys())

    deduped: List[str] = []
    seen = set()
    for case_id in case_ids:
        if case_id and case_id in dataset and case_id not in seen:
            seen.add(case_id)
            deduped.append(case_id)
    if limit is not None:
        deduped = deduped[:limit]
    return deduped


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Materialize oracle cited-source PDFs for oracle-source evaluation.")
    parser.add_argument("--dataset-file", default=str(ROOT / DEFAULT_DATASET_FILE))
    parser.add_argument("--eval-file", default="", help="Optional end-to-end eval JSON used to select case ids.")
    parser.add_argument("--download-root", required=True, help="Per-model download root shared with end-to-end evaluation.")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--out", default="")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset = load_dataset_as_dict(args.dataset_file)
    download_root = Path(args.download_root)
    download_root.mkdir(parents=True, exist_ok=True)

    case_ids = resolve_case_ids(dataset, eval_file=args.eval_file, limit=args.limit)
    results: List[dict] = []
    status_counts: Counter[str] = Counter()
    origin_counts: Counter[str] = Counter()

    for case_id in case_ids:
        row = dataset[case_id]
        case_dir = download_root / case_id
        case_dir.mkdir(parents=True, exist_ok=True)
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
            "eval_file": args.eval_file or None,
            "download_root": str(download_root),
            "total_cases": len(case_ids),
        },
        "counts": {
            "status": dict(status_counts),
            "origin": dict(origin_counts),
        },
        "results": results,
    }

    out_path = Path(args.out) if args.out else OUT_DIR / "oracle_materialization.json"
    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding='utf-8')
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
