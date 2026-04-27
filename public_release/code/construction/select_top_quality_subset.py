from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from construction.finalize_coretask_snapshot import MANUAL_FIXES, suspicious_anchor, suspicious_reference_tool
from common.dataset_format import (
    DEFAULT_DATASET_FILE,
    get_anchor_hint,
    get_citation_doi,
    get_citation_locator,
    get_citation_locator_snippet,
    get_cited_arxiv_id,
    get_domain,
    get_instance_id,
    get_local_context_blocks,
    get_reference_tool_latex,
    get_reference_tool_type,
    get_restated_in_citing_paper,
    get_source_type,
    load_jsonl,
)

THEOREM_LIKE_TOOL_TYPES = {"theorem", "lemma", "proposition", "corollary", "criterion"}
GENERIC_ANCHOR_PREFIXES = (
    "to state our main results",
    "the following result",
    "under the current proof hypotheses",
    "the idea is to",
    "in fact, locally",
    "which slightly rephrased shows",
    "this complex is also denoted",
    "if is trivial",
    "to show that",
    "we follow",
    "recall that",
    "in fact,",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Select a high-quality top-N subset from the curated benchmark dataset.")
    parser.add_argument("--input", default=str(ROOT / "construction/outputs/benchmark_dataset_current.jsonl"))
    parser.add_argument("--output", default=str(ROOT / "construction/outputs/benchmark_dataset_top200_high_quality_20260422.jsonl"))
    parser.add_argument("--summary", default=str(ROOT / "construction/outputs/benchmark_dataset_top200_high_quality_20260422.summary.json"))
    parser.add_argument("--top-n", type=int, default=200)
    return parser.parse_args()


def local_context_text(row: dict[str, Any]) -> str:
    return " ".join(block.strip() for block in get_local_context_blocks(row) if block.strip())


def locator_text(row: dict[str, Any]) -> str:
    return (get_citation_locator(row) or get_citation_locator_snippet(row) or "").strip()


def hard_filter_reason(row: dict[str, Any]) -> str | None:
    instance_id = get_instance_id(row)
    anchor = get_anchor_hint(row)
    tool = get_reference_tool_latex(row)
    source_type = get_source_type(row)
    locator = locator_text(row)
    local_text = local_context_text(row)

    if source_type != "journal_paper":
        return "source_type_not_journal_paper"
    if not locator:
        return "missing_locator"
    if instance_id in MANUAL_FIXES:
        return "manual_fix_case"
    if len(anchor) < 50:
        return "anchor_too_short"
    if suspicious_anchor(anchor):
        return "anchor_suspicious_or_corrupted"
    if len(tool) < 80:
        return "reference_tool_too_short"
    if suspicious_reference_tool(tool):
        return "reference_tool_suspicious_or_corrupted"
    if len(local_text) < 120:
        return "local_context_too_short"
    return None


def anchor_penalty(anchor: str) -> int:
    lowered = (anchor or "").strip().lower()
    return 1 if any(lowered.startswith(prefix) for prefix in GENERIC_ANCHOR_PREFIXES) else 0


def quality_score(row: dict[str, Any]) -> int:
    score = 0
    anchor = get_anchor_hint(row)
    tool = get_reference_tool_latex(row)
    local_text = local_context_text(row)
    locator = locator_text(row)
    tool_type = get_reference_tool_type(row)

    if get_restated_in_citing_paper(row):
        score += 3
    if get_cited_arxiv_id(row):
        score += 2
    if get_citation_doi(row):
        score += 1
    if tool_type in THEOREM_LIKE_TOOL_TYPES:
        score += 1
    if len(tool) >= 140:
        score += 1
    if len(locator) >= 120:
        score += 1
    if 70 <= len(anchor) <= 220:
        score += 1
    if 180 <= len(local_text) <= 2600:
        score += 1
    score -= anchor_penalty(anchor)
    return score


def ranking_key(row: dict[str, Any]) -> tuple[Any, ...]:
    locator = locator_text(row)
    local_text = local_context_text(row)
    tool = get_reference_tool_latex(row)
    return (
        quality_score(row),
        int(bool(get_restated_in_citing_paper(row))),
        int(bool(get_citation_doi(row))),
        int(bool(get_cited_arxiv_id(row))),
        int(get_reference_tool_type(row) in THEOREM_LIKE_TOOL_TYPES),
        len(locator),
        len(tool),
        len(local_text),
        get_instance_id(row),
    )


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    domain_counts = Counter(get_domain(row) for row in rows)
    source_type_counts = Counter(get_source_type(row) for row in rows)
    tool_type_counts = Counter(get_reference_tool_type(row) for row in rows)
    score_counts = Counter(quality_score(row) for row in rows)
    return {
        "rows": len(rows),
        "domain_counts": dict(domain_counts),
        "source_type_counts": dict(source_type_counts),
        "reference_tool_type_counts": dict(tool_type_counts),
        "quality_score_histogram": dict(sorted(score_counts.items(), reverse=True)),
        "restated_true": sum(int(get_restated_in_citing_paper(row)) for row in rows),
        "with_locator": sum(int(bool(locator_text(row))) for row in rows),
        "with_doi": sum(int(bool(get_citation_doi(row))) for row in rows),
        "with_arxiv_id": sum(int(bool(get_cited_arxiv_id(row))) for row in rows),
    }


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)
    summary_path = Path(args.summary)

    rows = load_jsonl(input_path)
    kept: list[dict[str, Any]] = []
    rejected_reasons = Counter()
    for row in rows:
        reason = hard_filter_reason(row)
        if reason is None:
            kept.append(row)
        else:
            rejected_reasons[reason] += 1

    kept_sorted = sorted(kept, key=ranking_key, reverse=True)
    selected = kept_sorted[: args.top_n]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for row in selected:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    summary = {
        "input_file": str(input_path),
        "output_file": str(output_path),
        "top_n": args.top_n,
        "rows_seen": len(rows),
        "hard_filter_kept": len(kept),
        "hard_filter_rejected": sum(rejected_reasons.values()),
        "hard_filter_rejected_by_reason": dict(rejected_reasons),
        "hard_filter_definition": {
            "source_type": "journal_paper",
            "locator_required": True,
            "exclude_manual_fix_cases": True,
            "min_anchor_chars": 50,
            "exclude_suspicious_anchor": True,
            "min_reference_tool_chars": 80,
            "exclude_suspicious_reference_tool": True,
            "min_local_context_chars": 120,
        },
        "ranking_features": {
            "restated_in_citing_paper": "+3",
            "cited_arxiv_id_present": "+2",
            "doi_present": "+1",
            "theorem_like_tool_type": "+1",
            "reference_tool_len_ge_140": "+1",
            "locator_len_ge_120": "+1",
            "anchor_len_between_70_and_220": "+1",
            "local_context_len_between_180_and_2600": "+1",
            "generic_anchor_prefix_penalty": "-1",
        },
        "selected_summary": summarize(selected),
        "top10_ids": [get_instance_id(row) for row in selected[:10]],
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
