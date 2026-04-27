from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "evaluation" / "outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def dump_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def parse_label_path(items: List[str], *, value_name: str) -> Dict[str, Path]:
    mapping: Dict[str, Path] = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"{value_name} must use label=path format: {item}")
        label, raw_path = item.split("=", 1)
        label = label.strip()
        raw_path = raw_path.strip()
        if not label or not raw_path:
            raise ValueError(f"Invalid {value_name}: {item}")
        mapping[label] = Path(raw_path)
    return mapping


def normalize_bool(value: Any) -> Optional[bool]:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    raw = str(value).strip().lower()
    if raw in {"true", "1", "yes"}:
        return True
    if raw in {"false", "0", "no"}:
        return False
    return None


def load_search_index(path: Path) -> Dict[str, dict]:
    payload = load_json(path)
    details = payload.get("details", payload if isinstance(payload, list) else []) or []
    return {str(row.get("id") or ""): row for row in details if str(row.get("id") or "")}


def enrich_cite_recall(row: dict, search_index: Dict[str, dict]) -> Optional[bool]:
    existing = row.get("cite_recall_at_20")
    normalized_existing = normalize_bool(existing)
    if normalized_existing is not None:
        return normalized_existing

    search_row = search_index.get(str(row.get("id") or ""))
    if not search_row:
        return None
    found = normalize_bool(search_row.get("found"))
    found_rank = search_row.get("found_rank")
    if found is None:
        return None
    if found_rank is None:
        return bool(found)
    try:
        rank = int(found_rank)
    except Exception:
        return bool(found)
    return bool(found and 0 < rank <= 20)


def assign_bucket(row: dict, cite_recall_at_20: Optional[bool]) -> str:
    ground = normalize_bool(row.get("Ground"))
    suff = normalize_bool(row.get("Suff"))
    toolacc = normalize_bool(row.get("ToolAcc"))

    if toolacc is True:
        return "success"
    if cite_recall_at_20 is False:
        return "retrieval_miss"
    if ground is None or suff is None:
        return "judge_abstained"
    if ground is False:
        return "ungrounded_extraction"
    if ground is True and suff is False:
        return "grounded_but_insufficient"
    return "other_or_unobserved"


def summarize_run(label: str, eval_path: Path, search_path: Optional[Path]) -> Tuple[dict, List[dict]]:
    payload = load_json(eval_path)
    rows = payload.get("results", payload if isinstance(payload, list) else []) or []
    search_index = load_search_index(search_path) if search_path else {}

    bucket_counts = Counter()
    detailed_rows: List[dict] = []
    for row in rows:
        cite_recall_at_20 = enrich_cite_recall(row, search_index)
        bucket = assign_bucket(row, cite_recall_at_20)
        bucket_counts[bucket] += 1
        detailed_rows.append(
            {
                "id": row.get("id", ""),
                "paper_title": row.get("paper_title", ""),
                "domain": row.get("domain", ""),
                "tool_family": row.get("tool_family", ""),
                "selected_file": row.get("selected_file", ""),
                "cite_recall_at_20": cite_recall_at_20,
                "Ground": row.get("Ground"),
                "Suff": row.get("Suff"),
                "ToolAcc": row.get("ToolAcc"),
                "bucket": bucket,
            }
        )

    total = len(rows)
    summary = {
        "model": label,
        "eval_file": str(eval_path),
        "search_log": str(search_path) if search_path else None,
        "total": total,
        "judge_abstained": bucket_counts["judge_abstained"],
        "retrieval_miss": bucket_counts["retrieval_miss"],
        "ungrounded_extraction": bucket_counts["ungrounded_extraction"],
        "grounded_but_insufficient": bucket_counts["grounded_but_insufficient"],
        "success": bucket_counts["success"],
        "other_or_unobserved": bucket_counts["other_or_unobserved"],
    }
    for key in [
        "judge_abstained",
        "retrieval_miss",
        "ungrounded_extraction",
        "grounded_but_insufficient",
        "success",
        "other_or_unobserved",
    ]:
        summary[f"{key}_rate"] = (summary[key] / total) if total else 0.0
    return summary, detailed_rows


def pct(value: float) -> str:
    return f"{100.0 * value:.1f}%"


def build_markdown(summaries: List[dict]) -> str:
    include_other = any(summary.get("other_or_unobserved", 0) > 0 for summary in summaries)
    lines = [
        "# Failure Decomposition",
        "",
        "| Model | Judge abstained | Retrieval miss | Ungrounded extraction | Grounded but insufficient | Success |"
        + (" Other / unobserved |" if include_other else ""),
        "| --- | --- | --- | --- | --- | --- |" + (" --- |" if include_other else ""),
    ]
    for summary in summaries:
        row = (
            f"| {summary['model']} | "
            f"{summary['judge_abstained']}/{summary['total']} ({pct(summary['judge_abstained_rate'])}) | "
            f"{summary['retrieval_miss']}/{summary['total']} ({pct(summary['retrieval_miss_rate'])}) | "
            f"{summary['ungrounded_extraction']}/{summary['total']} ({pct(summary['ungrounded_extraction_rate'])}) | "
            f"{summary['grounded_but_insufficient']}/{summary['total']} ({pct(summary['grounded_but_insufficient_rate'])}) | "
            f"{summary['success']}/{summary['total']} ({pct(summary['success_rate'])}) |"
        )
        if include_other:
            row += f" {summary['other_or_unobserved']}/{summary['total']} ({pct(summary['other_or_unobserved_rate'])}) |"
        lines.append(row)
    lines.append("")
    if include_other:
        lines.append("`Other / unobserved` means the row could not be placed into the main buckets, usually because retrieval-side citation recall was unavailable.")
        lines.append("")
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize end-to-end results into failure-decomposition buckets.")
    parser.add_argument(
        "--run",
        action="append",
        default=[],
        help="Required label=eval_file pair. May be passed multiple times.",
    )
    parser.add_argument(
        "--search-log",
        action="append",
        default=[],
        help="Optional label=search_log pair. Used to fill CiteRecall@20 when missing from the eval file.",
    )
    parser.add_argument("--out-json", default="")
    parser.add_argument("--out-md", default="")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    eval_runs = parse_label_path(args.run, value_name="run")
    if not eval_runs:
        raise ValueError("At least one --run label=eval_file is required.")
    search_logs = parse_label_path(args.search_log, value_name="search-log") if args.search_log else {}

    summaries: List[dict] = []
    details: Dict[str, List[dict]] = {}
    for label, eval_path in eval_runs.items():
        summary, rows = summarize_run(label, eval_path, search_logs.get(label))
        summaries.append(summary)
        details[label] = rows

    payload = {
        "runs": summaries,
        "details": details,
    }

    out_json = Path(args.out_json) if args.out_json else OUT_DIR / "failure_decomposition.json"
    out_md = Path(args.out_md) if args.out_md else OUT_DIR / "failure_decomposition.md"
    dump_json(out_json, payload)
    out_md.write_text(build_markdown(summaries), encoding="utf-8")
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    print(f"Saved to {out_json} and {out_md}")


if __name__ == "__main__":
    main()
