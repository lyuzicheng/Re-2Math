from __future__ import annotations

import argparse
import json
import math
import random
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


ROOT = Path(__file__).resolve().parents[1]

import sys

if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from common.dataset_format import (
    DEFAULT_DATASET_FILE,
    get_domain,
    get_paper,
    get_source_type,
    get_tool_family,
    load_dataset_as_dict,
)


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def wilson_interval(successes: int, total: int, z: float = 1.96) -> tuple[float, float]:
    if total == 0:
        return 0.0, 0.0
    phat = successes / total
    denom = 1 + (z * z) / total
    center = (phat + (z * z) / (2 * total)) / denom
    margin = z * math.sqrt((phat * (1 - phat) / total) + (z * z) / (4 * total * total)) / denom
    return max(0.0, center - margin), min(1.0, center + margin)


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


def positive_indicator(row: dict, key: str) -> bool:
    return row.get(key) is True


def distinct_nonempty_count(rows: Iterable[dict], key: str) -> int:
    values = {
        str(row.get(key) or "").strip()
        for row in rows
        if str(row.get(key) or "").strip()
    }
    return len(values)


def macro_rate_total(rows: List[dict], key: str, group_key: str) -> Optional[float]:
    grouped: Dict[str, List[dict]] = defaultdict(list)
    for row in rows:
        group = str(row.get(group_key) or "").strip()
        if not group:
            continue
        grouped[group].append(row)
    if not grouped:
        return None
    per_group = [sum(1 for row in group_rows if positive_indicator(row, key)) / len(group_rows) for group_rows in grouped.values()]
    return sum(per_group) / len(per_group)


def clustered_bootstrap_rate_total(
    rows: List[dict],
    key: str,
    cluster_key: str,
    *,
    n_boot: int = 1000,
    seed: int = 0,
) -> Optional[tuple[float, float]]:
    grouped: Dict[str, List[dict]] = defaultdict(list)
    for row in rows:
        cluster = str(row.get(cluster_key) or "").strip()
        if not cluster:
            continue
        grouped[cluster].append(row)
    if not grouped:
        return None

    rng = random.Random(seed)
    clusters = list(grouped.keys())
    estimates: List[float] = []
    for _ in range(n_boot):
        sampled_clusters = [rng.choice(clusters) for _ in range(len(clusters))]
        sampled_rows = [row for cluster in sampled_clusters for row in grouped[cluster]]
        if sampled_rows:
            estimates.append(sum(1 for row in sampled_rows if positive_indicator(row, key)) / len(sampled_rows))
    if not estimates:
        return None
    estimates.sort()
    lo_idx = int(0.025 * (len(estimates) - 1))
    hi_idx = int(0.975 * (len(estimates) - 1))
    return estimates[lo_idx], estimates[hi_idx]


def summarize_total_metric(rows: List[dict], key: str, *, n_boot: int, seed: int) -> Optional[dict]:
    if not rows:
        return None
    success = sum(1 for row in rows if positive_indicator(row, key))
    total = len(rows)
    wilson = wilson_interval(success, total)
    cluster_ci = clustered_bootstrap_rate_total(rows, key, "paper_id", n_boot=n_boot, seed=seed)
    return {
        "success": success,
        "total": total,
        "rate": success / total if total else 0.0,
        "wilson95": list(wilson),
        "paper_cluster_bootstrap95": list(cluster_ci) if cluster_ci else None,
        "paper_macro": macro_rate_total(rows, key, "paper_id"),
        "domain_macro": macro_rate_total(rows, key, "domain"),
    }


def summarize_conditional_total_metric(
    rows: List[dict],
    key: str,
    *,
    condition_key: str,
    n_boot: int,
    seed: int,
) -> Optional[dict]:
    conditioned = [row for row in rows if positive_indicator(row, condition_key)]
    if not conditioned:
        return None
    return summarize_total_metric(conditioned, key, n_boot=n_boot, seed=seed)


def attach_dataset_metadata(rows: Iterable[dict], dataset: Dict[str, dict]) -> List[dict]:
    enriched = []
    for row in rows:
        case_id = str(row.get("id") or "")
        ds = dataset.get(case_id, {})
        paper = get_paper(ds) if ds else {}
        enriched_row = dict(row)
        enriched_row.setdefault("paper_id", paper.get("paper_id", ""))
        enriched_row.setdefault("paper_title", paper.get("title", ""))
        enriched_row.setdefault("domain", get_domain(ds) if ds else "")
        enriched_row.setdefault("tool_family", get_tool_family(ds) if ds else "")
        enriched_row.setdefault("source_type", get_source_type(ds) if ds else "")
        enriched.append(enriched_row)
    return enriched


def load_eval_rows(path: Path, dataset: Dict[str, dict]) -> List[dict]:
    payload = load_json(path)
    raw_rows = payload.get("results", payload if isinstance(payload, list) else []) or []
    rows = attach_dataset_metadata(raw_rows, dataset)
    for row in rows:
        ground = normalize_bool(row.get("Ground"))
        if ground is None and "grounded" in row:
            ground = normalize_bool(row.get("grounded"))

        suff = normalize_bool(row.get("Suff"))
        if suff is None and "is_matched" in row:
            suff = normalize_bool(row.get("is_matched"))

        toolacc = normalize_bool(row.get("ToolAcc"))
        if toolacc is None and ground is not None and suff is not None:
            toolacc = bool(ground and suff)
        if toolacc is None and "tool_success_proxy" in row:
            toolacc = normalize_bool(row.get("tool_success_proxy"))

        cited_doc_match = normalize_bool(row.get("cited_doc_match", row.get("is_retrieved")))
        cite_recall = normalize_bool(row.get("cite_recall_at_20"))
        planning_match = normalize_bool(row.get("is_planning_matched", row.get("planning_is_match")))
        alt_source_toolacc = None if toolacc is None else bool((cited_doc_match is False) and toolacc)

        row["Ground"] = ground
        row["Suff"] = suff
        row["ToolAcc"] = toolacc
        row["cited_doc_match"] = cited_doc_match
        row["cite_recall_at_20"] = cite_recall
        row["is_planning_matched"] = planning_match
        row["alt_source_tool_success_proxy"] = alt_source_toolacc
    return rows


def load_oracle_rows(path: Path, dataset: Dict[str, dict]) -> List[dict]:
    payload = load_json(path)
    raw_rows = payload.get("results", payload if isinstance(payload, list) else []) or []
    rows = attach_dataset_metadata(raw_rows, dataset)
    for row in rows:
        status = str(row.get("status") or "").strip().lower()
        unresolved = status in {"missing_cited_source_pdf", "missing_pdf", "unavailable", "not_found"}
        toolacc = normalize_bool(row.get("ToolAcc"))
        if toolacc is None and not unresolved and "tool_success_proxy" in row:
            toolacc = normalize_bool(row.get("tool_success_proxy"))
        if unresolved:
            toolacc = None
        row["ToolAcc"] = toolacc
        row["oracle_resolved"] = toolacc is not None
    return rows


def load_planning_rows(path: Path, dataset: Dict[str, dict]) -> List[dict]:
    payload = load_json(path)
    raw_rows = payload.get("details", payload if isinstance(payload, list) else []) or []
    rows = attach_dataset_metadata(raw_rows, dataset)
    for row in rows:
        row["planning_is_match"] = normalize_bool(row.get("planning_is_match"))
    return rows


def rate_over_total(rows: List[dict], key: str) -> Optional[float]:
    if not rows:
        return None
    return sum(1 for row in rows if positive_indicator(row, key)) / len(rows)


def build_stratified_breakdown(
    eval_rows: List[dict],
    oracle_rows: List[dict],
    *,
    group_key: str,
) -> Dict[str, dict]:
    if not eval_rows:
        return {}
    oracle_by_id = {
        str(row.get("id") or ""): row
        for row in oracle_rows
        if str(row.get("id") or "") and row.get("ToolAcc") is not None
    }
    grouped: Dict[str, List[dict]] = defaultdict(list)
    for row in eval_rows:
        group = str(row.get(group_key) or "").strip() or "unknown"
        grouped[group].append(row)

    summary: Dict[str, dict] = {}
    for group, group_rows in sorted(grouped.items()):
        oracle_group_rows = [oracle_by_id[str(row.get("id") or "")] for row in group_rows if str(row.get("id") or "") in oracle_by_id]
        summary[group] = {
            "papers": distinct_nonempty_count(group_rows, "paper_id"),
            "gaps": len(group_rows),
            "GroundRate": rate_over_total(group_rows, "Ground"),
            "ToolAcc": rate_over_total(group_rows, "ToolAcc"),
            "Oracle ToolAcc": rate_over_total(oracle_group_rows, "ToolAcc") if oracle_group_rows else None,
        }
    return summary


def pct(x: Optional[float]) -> str:
    if x is None:
        return "NA"
    return f"{100.0 * x:.1f}%"


def summarize_oracle_coverage(eval_rows: List[dict], oracle_rows: List[dict]) -> Optional[dict]:
    if not eval_rows:
        return None
    resolved_ids = {
        str(row.get("id") or "")
        for row in oracle_rows
        if str(row.get("id") or "") and row.get("ToolAcc") is not None
    }
    success = sum(1 for row in eval_rows if str(row.get("id") or "") in resolved_ids)
    total = len(eval_rows)
    wilson = wilson_interval(success, total)
    return {
        "success": success,
        "total": total,
        "rate": success / total if total else 0.0,
        "wilson95": list(wilson),
        "paper_cluster_bootstrap95": None,
        "paper_macro": None,
        "domain_macro": None,
    }


def interval_text(metric: Optional[dict], key: str) -> str:
    if not metric:
        return "NA"
    value = metric.get(key)
    if not value:
        return "NA"
    return f"[{pct(value[0])}, {pct(value[1])}]"


def metric_row(label: str, metric: Optional[dict]) -> str:
    if not metric:
        return f"| {label} | NA | NA | NA | NA | NA |"
    return (
        f"| {label} | {metric['success']}/{metric['total']} | {pct(metric['rate'])} | "
        f"{interval_text(metric, 'wilson95')} | {interval_text(metric, 'paper_cluster_bootstrap95')} | "
        f"{pct(metric.get('paper_macro'))} |"
    )


def build_markdown(payload: dict) -> str:
    lines = [
        "# Final Paper Evaluation Summary",
        "",
        f"- Dataset: `{payload['meta']['dataset_file']}`",
        f"- Dataset rows: `{payload['meta']['dataset_rows']}`",
        f"- Evaluated rows: `{payload['meta']['evaluated_rows']}`",
        f"- Oracle-evaluated rows: `{payload['meta']['oracle_rows']}`",
        f"- End-to-end eval: `{payload['meta'].get('eval_file') or 'NA'}`",
        f"- Planning eval: `{payload['meta'].get('planning_file') or 'NA'}`",
        f"- Oracle eval: `{payload['meta'].get('oracle_file') or 'NA'}`",
        f"- Bootstrap samples: `{payload['meta']['bootstrap_samples']}`",
        "",
        "## Main Table 1: Core End-to-End",
        "",
        "| Metric | Count | Rate | Wilson 95% CI | Paper-Cluster 95% CI | Paper Macro |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for label in ["AnchorAcc(x_raw)", "CiteRecall@20", "GroundRate", "ToolAcc"]:
        lines.append(metric_row(label, payload["main_table_1"].get(label)))

    lines.extend(
        [
            "",
            "## Main Table 2: Oracle Gap and Source-Invariant Success",
            "",
            "| Metric | Count | Rate | Wilson 95% CI | Paper-Cluster 95% CI | Paper Macro |",
            "| --- | --- | --- | --- | --- | --- |",
        ]
    )
    for label in ["ToolAcc", "OracleCoverage", "Oracle ToolAcc", "AltSourceToolAcc", "AltSourceSuccessRate"]:
        lines.append(metric_row(label, payload["main_table_2"].get(label)))

    delta_to_oracle = payload["main_table_2"].get("DeltaToOracle")
    lines.append("")
    lines.append(f"- `Δ to Oracle`: {pct(delta_to_oracle)}")
    lines.append("")
    lines.append("## Appendix Table A3: Stratified Breakdown")
    lines.append("")

    for section_name, section_rows in payload["appendix_stratified"].items():
        lines.append(f"### {section_name}")
        lines.append("")
        lines.append("| Stratum | #papers | #gaps | GroundRate | ToolAcc | Oracle ToolAcc |")
        lines.append("| --- | --- | --- | --- | --- | --- |")
        for stratum, stats in section_rows.items():
            lines.append(
                f"| {stratum} | {stats['papers']} | {stats['gaps']} | "
                f"{pct(stats.get('GroundRate'))} | {pct(stats.get('ToolAcc'))} | {pct(stats.get('Oracle ToolAcc'))} |"
            )
        lines.append("")

    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate the final non-redundant paper metrics.")
    parser.add_argument("--dataset", default=str(ROOT / DEFAULT_DATASET_FILE))
    parser.add_argument("--eval-file", required=True, help="End-to-end evaluation JSON.")
    parser.add_argument("--oracle-file", default="", help="Optional oracle-source evaluation JSON.")
    parser.add_argument("--planning-file", default="", help="Optional dedicated planning diagnostic JSON.")
    parser.add_argument("--out-json", default="", help="Output JSON path.")
    parser.add_argument("--out-md", default="", help="Output markdown path.")
    parser.add_argument("--bootstrap-samples", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset = load_dataset_as_dict(args.dataset)
    eval_rows = load_eval_rows(Path(args.eval_file), dataset)
    oracle_rows = load_oracle_rows(Path(args.oracle_file), dataset) if args.oracle_file else []
    planning_rows = load_planning_rows(Path(args.planning_file), dataset) if args.planning_file else []
    anchor_rows = planning_rows or eval_rows

    main_table_1 = {
        "AnchorAcc(x_raw)": summarize_total_metric(anchor_rows, "planning_is_match", n_boot=args.bootstrap_samples, seed=args.seed),
        "CiteRecall@20": summarize_total_metric(eval_rows, "cite_recall_at_20", n_boot=args.bootstrap_samples, seed=args.seed),
        "GroundRate": summarize_total_metric(eval_rows, "Ground", n_boot=args.bootstrap_samples, seed=args.seed),
        "ToolAcc": summarize_total_metric(eval_rows, "ToolAcc", n_boot=args.bootstrap_samples, seed=args.seed),
    }

    toolacc_metric = main_table_1["ToolAcc"]
    oracle_evaluable_rows = [row for row in oracle_rows if row.get("ToolAcc") is not None]
    oracle_coverage = summarize_oracle_coverage(eval_rows, oracle_rows) if oracle_rows else None
    oracle_metric = (
        summarize_total_metric(oracle_evaluable_rows, "ToolAcc", n_boot=args.bootstrap_samples, seed=args.seed)
        if oracle_evaluable_rows
        else None
    )
    alt_source_toolacc = summarize_total_metric(
        eval_rows,
        "alt_source_tool_success_proxy",
        n_boot=args.bootstrap_samples,
        seed=args.seed,
    )
    alt_source_success = summarize_conditional_total_metric(
        eval_rows,
        "alt_source_tool_success_proxy",
        condition_key="ToolAcc",
        n_boot=args.bootstrap_samples,
        seed=args.seed,
    )
    delta_to_oracle = None
    if toolacc_metric and oracle_metric:
        delta_to_oracle = oracle_metric["rate"] - toolacc_metric["rate"]

    main_table_2 = {
        "ToolAcc": toolacc_metric,
        "OracleCoverage": oracle_coverage,
        "Oracle ToolAcc": oracle_metric,
        "DeltaToOracle": delta_to_oracle,
        "AltSourceToolAcc": alt_source_toolacc,
        "AltSourceSuccessRate": alt_source_success,
    }

    appendix_stratified = {
        "By domain": build_stratified_breakdown(eval_rows, oracle_rows, group_key="domain"),
        "By tool family": build_stratified_breakdown(eval_rows, oracle_rows, group_key="tool_family"),
        "By source type": build_stratified_breakdown(eval_rows, oracle_rows, group_key="source_type"),
    }

    payload = {
        "meta": {
            "dataset_file": args.dataset,
            "dataset_rows": len(dataset),
            "evaluated_rows": len(eval_rows),
            "oracle_rows": len(oracle_rows),
            "oracle_evaluable_rows": len(oracle_evaluable_rows),
            "eval_file": args.eval_file,
            "planning_file": args.planning_file or None,
            "oracle_file": args.oracle_file or None,
            "bootstrap_samples": args.bootstrap_samples,
            "seed": args.seed,
        },
        "main_table_1": main_table_1,
        "main_table_2": main_table_2,
        "appendix_stratified": appendix_stratified,
    }

    out_json = Path(args.out_json) if args.out_json else ROOT / "evaluation" / "outputs" / "final_paper_eval_summary.json"
    out_md = Path(args.out_md) if args.out_md else ROOT / "evaluation" / "outputs" / "final_paper_eval_summary.md"
    out_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    out_md.write_text(build_markdown(payload), encoding="utf-8")
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    print("Saved to", out_json, "and", out_md)


if __name__ == "__main__":
    main()
