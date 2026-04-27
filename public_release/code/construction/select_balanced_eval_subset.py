from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from construction.finalize_coretask_snapshot import MANUAL_FIXES, suspicious_anchor, suspicious_reference_tool
from construction.select_top_quality_subset import local_context_text, locator_text, quality_score, ranking_key
from common.dataset_format import get_domain, get_instance_id, get_paper, load_jsonl

DEFAULT_QUOTAS = {
    "analysis_pde": 40,
    "geometry_topology": 40,
    "algebra_number_theory": 40,
    "probability_statistics_control": 40,
    "combinatorics_discrete": 40,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Select a balanced Eval-200 subset for 8-model formal evaluation.")
    parser.add_argument("--input", default=str(ROOT / "construction/outputs/benchmark_dataset_current.jsonl"))
    parser.add_argument("--output", default=str(ROOT / "construction/outputs/benchmark_dataset_eval200_balanced_20260422.jsonl"))
    parser.add_argument("--summary", default=str(ROOT / "construction/outputs/benchmark_dataset_eval200_balanced_20260422.summary.json"))
    parser.add_argument("--per-paper-cap", type=int, default=2)
    return parser.parse_args()


def hard_filter_reason(row: dict[str, Any]) -> str | None:
    instance_id = get_instance_id(row)
    if instance_id in MANUAL_FIXES:
        return "manual_fix_case"
    if (row.get("z") or {}).get("source_type") != "journal_paper":
        return "source_type_not_journal_paper"
    if not locator_text(row):
        return "missing_locator"
    anchor = ((row.get("x") or {}).get("anchor_hint") or "").strip()
    tool = ((row.get("y") or {}).get("reference_tool_latex") or "").strip()
    if not anchor:
        return "missing_anchor_hint"
    if suspicious_anchor(anchor):
        return "anchor_suspicious_or_corrupted"
    if not tool:
        return "missing_reference_tool"
    if suspicious_reference_tool(tool):
        return "reference_tool_suspicious_or_corrupted"
    if not local_context_text(row).strip():
        return "missing_local_context"
    return None


def strict_quality_pass(row: dict[str, Any]) -> bool:
    anchor = ((row.get("x") or {}).get("anchor_hint") or "").strip()
    tool = ((row.get("y") or {}).get("reference_tool_latex") or "").strip()
    local_text = local_context_text(row)
    return len(anchor) >= 50 and len(tool) >= 80 and len(local_text) >= 120 and (not suspicious_anchor(anchor)) and (not suspicious_reference_tool(tool))


def select_balanced(rows: list[dict[str, Any]], per_paper_cap: int) -> list[dict[str, Any]]:
    filtered: list[dict[str, Any]] = []
    for row in rows:
        if hard_filter_reason(row) is None:
            filtered.append(row)

    by_domain: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in filtered:
        by_domain[get_domain(row)].append(row)

    selected: list[dict[str, Any]] = []
    for domain, quota in DEFAULT_QUOTAS.items():
        candidates = sorted(by_domain.get(domain, []), key=ranking_key, reverse=True)
        per_paper = Counter()
        taken = []
        for row in candidates:
            paper_id = str(get_paper(row).get("paper_id", "") or "")
            if per_paper[paper_id] >= per_paper_cap:
                continue
            taken.append(row)
            per_paper[paper_id] += 1
            if len(taken) >= quota:
                break
        if len(taken) < quota:
            raise RuntimeError(f"Domain {domain} could only supply {len(taken)} rows under the current constraints; needed {quota}.")
        selected.extend(taken)
    return selected


def summarize(rows: list[dict[str, Any]], input_file: str, output_file: str, per_paper_cap: int) -> dict[str, Any]:
    rows_per_paper = Counter(str(get_paper(row).get("paper_id", "") or "") for row in rows)
    strict_count = sum(int(strict_quality_pass(row)) for row in rows)
    return {
        "input_file": input_file,
        "output_file": output_file,
        "selection_policy": {
            "goal": "balanced formal Eval-200 for 8-model suite",
            "quotas": DEFAULT_QUOTAS,
            "hard_filter": {
                "source_type": "journal_paper",
                "locator_required": True,
                "exclude_manual_fix_cases": True,
                "anchor_hint_nonempty": True,
                "exclude_suspicious_anchor": True,
                "reference_tool_nonempty": True,
                "exclude_suspicious_reference_tool": True,
                "local_context_nonempty": True,
            },
            "ranking": "same quality_score / ranking_key as select_top_quality_subset.py",
            "per_paper_cap": per_paper_cap,
        },
        "selected_summary": {
            "rows": len(rows),
            "unique_papers": len(rows_per_paper),
            "domain_counts": dict(Counter(get_domain(row) for row in rows)),
            "domain_paper_counts": {
                domain: len({str(get_paper(row).get("paper_id", "") or "") for row in rows if get_domain(row) == domain})
                for domain in DEFAULT_QUOTAS
            },
            "rows_per_paper_histogram": dict(Counter(rows_per_paper.values())),
            "max_rows_single_paper": max(rows_per_paper.values()) if rows_per_paper else 0,
            "strict_quality_pass_count": strict_count,
            "strict_quality_pass_rate": strict_count / len(rows) if rows else 0.0,
            "quality_score_histogram": dict(sorted(Counter(quality_score(row) for row in rows).items(), reverse=True)),
            "restated_true": sum(int(bool((row.get("y") or {}).get("restated_in_citing_paper"))) for row in rows),
            "strict_locator": sum(int(bool(((row.get("z") or {}).get("locator") or "").strip())) for row in rows),
            "snippet_only": sum(int((not ((row.get("z") or {}).get("locator") or "").strip()) and bool(((row.get("z") or {}).get("locator_snippet") or "").strip())) for row in rows),
        },
        "top10_ids": [get_instance_id(row) for row in rows[:10]],
    }


def main() -> None:
    args = parse_args()
    rows = load_jsonl(Path(args.input))
    selected = select_balanced(rows, per_paper_cap=args.per_paper_cap)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for row in selected:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    summary = summarize(selected, args.input, args.output, args.per_paper_cap)
    Path(args.summary).write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
