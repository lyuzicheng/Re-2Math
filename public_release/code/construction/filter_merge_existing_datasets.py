import argparse
import glob
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List
import re

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from construction.mine_dataset import (
    build_instance_signature,
    empty_local_context_row_is_acceptable,
    is_near_duplicate_instance,
    local_context_has_structural_artifacts,
    normalize_local_context_blocks,
    normalize_signature_text,
    normalize_target_theorem_text,
    restated_statement_supported_by_evidence,
    sanitize_anchor_hint_text,
    setup_text_rejection_reason,
    target_theorem_rejection_reason,
    tool_statement_rejection_reason,
)


DEFAULT_GLOBS = [
    "construction/outputs/benchmark_dataset_large_deepseek*.jsonl",
]

ANCHOR_SEMANTIC_REJECT_PATTERNS = [
    re.compile(r"(?:\b(?:in|see|appears\s+in|proved\s+in|shown\s+in|given\s+in|cf)\.?)$", re.IGNORECASE),
    re.compile(
        r"\b(?:has\s+been\s+well[\-\s]studied|is\s+somewhat\s+simpler|there\s+is\s+no\s+need\s+to\s+invoke|"
        r"space\s+bound\s+appears\s+in|numerical\s+studies\s+in|used\s+optimal\s+transportation\s+tools|"
        r"we\s+also\s+refer\s+to|is\s+asserted\s+to\s+be|based\s+on\s+numerical\s+evidence|"
        r"the\s+second\s+author\s+studied|main\s+result\s+in)\b",
        re.IGNORECASE,
    ),
]

TOOL_SEMANTIC_REJECT_PATTERNS = [
    re.compile(r"^If\s+(?:For\s+|is\s+|with\s+|on\s+|the\s+|simomentum\b|Nimal\b|ty\b|e\s+case\b)", re.IGNORECASE),
    re.compile(
        r"(?:there\s+is\s+no\s+need\s+to\s+invoke|numerical\s+studies\s+in|has\s+been\s+well[\-\s]studied|"
        r"space\s+bound\s+appears\s+in|recently,\s*a|our\s+nu\.|see\.?$|however|we\s+also\s+refer\s+to|"
        r"is\s+asserted\s+to\s+be|used\s+optimal\s+transportation\s+tools|is\s+very\s+fast[\-\s]growing)",
        re.IGNORECASE,
    ),
    re.compile(r"(?:then\s+Since|then\s+indicate\s+that|then\s+there\s+is\s+no\s+need|cf,\s*then|then\s*\))", re.IGNORECASE),
]


def discover_input_paths(patterns: List[str], output_path: Path) -> List[Path]:
    paths: List[Path] = []
    seen: set[Path] = set()
    for pattern in patterns:
        for match in sorted(glob.glob(pattern)):
            path = Path(match)
            name = path.name
            if not path.is_file():
                continue
            if name.endswith(".progress.jsonl"):
                continue
            if name == output_path.name:
                continue
            if "tmp_smoke" in name:
                continue
            if path in seen:
                continue
            seen.add(path)
            paths.append(path)
    return paths


def salvage_row(row: Dict[str, Any]) -> Dict[str, Any] | None:
    x = row.get("x", {}) or {}
    y = row.get("y", {}) or {}
    z = row.get("z", {}) or {}
    global_context = x.get("global_context", {}) or {}

    target_theorem = normalize_target_theorem_text(global_context.get("target_theorem"))
    if not target_theorem:
        return None
    if target_theorem_rejection_reason(target_theorem, strict_mode=True):
        return None

    setup = normalize_signature_text(global_context.get("setup"))
    if setup_text_rejection_reason(setup):
        return None

    anchor = sanitize_anchor_hint_text(x.get("anchor_hint"))
    if not anchor:
        return None

    tool = normalize_signature_text(y.get("reference_tool_latex"))
    if tool_statement_rejection_reason(tool):
        return None
    if any(pattern.search(anchor) for pattern in ANCHOR_SEMANTIC_REJECT_PATTERNS):
        return None
    if any(pattern.search(tool) for pattern in TOOL_SEMANTIC_REJECT_PATTERNS):
        return None

    local_context = normalize_local_context_blocks(x.get("local_context", []))
    if local_context_has_structural_artifacts(local_context):
        return None
    restated = bool(y.get("restated_in_citing_paper", False))
    locator = str(z.get("locator_snippet", "") or z.get("locator", "") or "")
    if not local_context:
        if not empty_local_context_row_is_acceptable(row):
            return None
        if not (
            restated
            and restated_statement_supported_by_evidence(
                tool,
                locator,
                local_context,
                "",
            )
        ):
            return None

    salvaged = {
        **row,
        "x": {
            **x,
            "global_context": {
                "setup": setup,
                "target_theorem": target_theorem,
            },
            "local_context": local_context,
            "anchor_hint": anchor,
        },
        "y": {
            **y,
            "reference_tool_latex": tool,
            "reference_tool_type": normalize_signature_text(y.get("reference_tool_type")),
            "restated_in_citing_paper": restated,
        },
    }
    return salvaged


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--glob",
        action="append",
        dest="patterns",
        default=[],
        help="Input glob pattern; can be passed multiple times.",
    )
    parser.add_argument(
        "--output",
        default="construction/outputs/benchmark_dataset_large_deepseek_salvaged_merge_20260418.jsonl",
    )
    parser.add_argument(
        "--summary",
        default="construction/outputs/benchmark_dataset_large_deepseek_salvaged_merge_20260418.summary.json",
    )
    args = parser.parse_args()

    output_path = Path(args.output)
    summary_path = Path(args.summary)
    patterns = args.patterns or list(DEFAULT_GLOBS)
    input_paths = discover_input_paths(patterns, output_path)

    seen_signatures: List[Dict[str, str]] = []
    stats = Counter()
    kept_by_file = Counter()
    seen_instance_ids: set[str] = set()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f_out:
        for path in input_paths:
            with path.open("r", encoding="utf-8") as f_in:
                for line in f_in:
                    if not line.strip():
                        continue
                    stats["rows_seen"] += 1
                    try:
                        row = json.loads(line)
                    except json.JSONDecodeError:
                        stats["bad_json"] += 1
                        continue

                    instance_id = str(row.get("instance_id", "") or "").strip()
                    if not instance_id:
                        stats["missing_instance_id"] += 1
                        continue

                    salvaged = salvage_row(row)
                    if not salvaged:
                        stats["rejected_qc"] += 1
                        continue

                    if instance_id in seen_instance_ids:
                        stats["duplicate_instance_id"] += 1
                        continue

                    signature = build_instance_signature(salvaged["x"] | salvaged["y"])
                    if is_near_duplicate_instance(signature, seen_signatures):
                        stats["duplicate_signature"] += 1
                        continue

                    seen_instance_ids.add(instance_id)
                    seen_signatures.append(signature)
                    kept_by_file[path.name] += 1
                    stats["kept"] += 1
                    f_out.write(json.dumps(salvaged, ensure_ascii=False) + "\n")

    summary = {
        "output_file": str(output_path),
        "input_patterns": patterns,
        "input_files": [str(path) for path in input_paths],
        "stats": dict(stats),
        "kept_by_file": dict(kept_by_file),
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
