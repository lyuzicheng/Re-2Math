from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "evaluation" / "outputs"
SUITES_DIR = OUT_DIR / "openrouter_suites"
UNIFIED_ORACLE_PATH = OUT_DIR / "revision_oracle_materialization_20260425.json"


@dataclass(frozen=True)
class ModelRun:
    alias: str
    display_name: str
    suite_name: str

    @property
    def root(self) -> Path:
        return SUITES_DIR / self.suite_name / self.alias


MODEL_RUNS: List[ModelRun] = [
    ModelRun("gpt", "GPT-5.2", "eval200_openrouter_8models_parallel_20260422"),
    ModelRun("gemini", "Gemini 3.1 Pro", "eval200_openrouter_gemini31_rerun_full_20260423_v2"),
    ModelRun("claude", "Claude Opus 4.5", "eval200_openrouter_8models_parallel_20260422"),
    ModelRun("deepseek", "DeepSeek V3.2", "eval200_openrouter_deepseek_v32_rerun_20260423"),
    ModelRun("qwen", "Qwen3-235B Thinking", "eval200_openrouter_8models_parallel_20260422"),
    ModelRun("kimi", "Kimi K2 Thinking", "eval200_openrouter_8models_parallel_20260422"),
    ModelRun("grok", "Grok 4", "eval200_openrouter_8models_parallel_20260422"),
]


PLANNING_FILES = {
    "Raw local_only m=5": "search_planning_raw_local_only_m5.json",
    "Raw global_local m=1": "search_planning_raw_global_local_m1.json",
    "Raw global_local m=3": "search_planning_raw_global_local_m3.json",
    "Raw global_local m=5": "search_planning_raw_global_local_m5.json",
}

QUERY_FILES = {
    "Assist global_local m=5": "search_query_assist_global_local_m5.json",
}

MAIN1_KEYS = ["AnchorAcc(x_raw)", "CiteRecall@20", "GroundRate", "ToolAcc"]
MAIN2_KEYS = [
    "ToolAcc",
    "OracleCoverage",
    "Oracle ToolAcc",
    "DeltaToOracle",
    "AltSourceToolAcc",
    "AltSourceSuccessRate",
]
UNCERTAINTY_KEYS = [
    "AnchorAcc(x_raw)",
    "CiteRecall@20",
    "GroundRate",
    "ToolAcc",
    "OracleCoverage",
    "Oracle ToolAcc",
    "AltSourceToolAcc",
    "AltSourceSuccessRate",
]


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def pct(value: Optional[float]) -> str:
    if value is None:
        return ""
    return f"{100.0 * value:.1f}%"


def format_count_rate(metric: Optional[Dict[str, Any]]) -> str:
    if not isinstance(metric, dict):
        return ""
    success = metric.get("success")
    total = metric.get("total")
    rate = metric.get("rate")
    if success is None or total is None or rate is None:
        return ""
    return f"{success}/{total} ({pct(float(rate))})"


def format_ci(ci: Optional[List[float]]) -> str:
    if not ci or len(ci) != 2:
        return ""
    return f"[{100.0 * ci[0]:.1f}%, {100.0 * ci[1]:.1f}%]"


def load_unified_oracle_coverage() -> Optional[Dict[str, Any]]:
    if not UNIFIED_ORACLE_PATH.exists():
        return None
    payload = load_json(UNIFIED_ORACLE_PATH)
    total = payload.get("dataset_cases")
    success = payload.get("oracle_evaluable_union_count")
    if not isinstance(total, int) or not isinstance(success, int) or total <= 0:
        return None
    return {
        "success": success,
        "total": total,
        "rate": success / total,
        "wilson95": [0.306, 0.439] if (success, total) == (74, 200) else None,
    }


def latest_log(model_root: Path) -> Optional[Path]:
    logs_dir = model_root / "logs"
    if not logs_dir.exists():
        return None
    logs = sorted(logs_dir.glob("*.log"), key=lambda path: path.stat().st_mtime, reverse=True)
    return logs[0] if logs else None


def get_progress(model_root: Path) -> str:
    log_path = latest_log(model_root)
    if not log_path:
        return "not_started"
    return f"{log_path.stem}: running"


def load_summary(model_root: Path) -> Optional[Dict[str, Any]]:
    path = model_root / "results" / "minimal_publishable_summary.json"
    return load_json(path) if path.exists() else None


def load_metric_file(model_root: Path, filename: str) -> Optional[Dict[str, Any]]:
    path = model_root / "results" / filename
    return load_json(path) if path.exists() else None


def collect_model_payload(run: ModelRun) -> Dict[str, Any]:
    model_root = run.root
    summary = load_summary(model_root)
    completed = summary is not None
    payload: Dict[str, Any] = {
        "model": run.display_name,
        "alias": run.alias,
        "suite_name": run.suite_name,
        "status": "completed" if completed else "running",
        "progress": "completed" if completed else get_progress(model_root),
        "summary": summary,
        "planning_ablation": {},
        "query_ablation": {},
    }

    for label, filename in PLANNING_FILES.items():
        data = load_metric_file(model_root, filename)
        metrics = data.get("metrics", {}) if isinstance(data, dict) else {}
        payload["planning_ablation"][label] = {
            "success": metrics.get("planning_success"),
            "total": metrics.get("planning_total"),
            "rate": metrics.get("planning_accuracy"),
        }

    for label, filename in QUERY_FILES.items():
        data = load_metric_file(model_root, filename)
        metrics = data.get("metrics", {}) if isinstance(data, dict) else {}
        payload["query_ablation"][label] = {
            "success": metrics.get("query_success"),
            "total": metrics.get("query_total"),
            "rate": metrics.get("query_accuracy"),
        }
    return payload


def build_table(headers: List[str], rows: List[List[str]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def build_main_tables(model_payloads: List[Dict[str, Any]]) -> List[str]:
    oracle_override = load_unified_oracle_coverage()
    rows1 = []
    rows2 = []
    for payload in model_payloads:
        summary = payload["summary"] or {}
        main1 = summary.get("main_table_1", {})
        main2 = summary.get("main_table_2", {})
        oracle_metric = oracle_override or main2.get("OracleCoverage")
        rows1.append([
            payload["model"],
            payload["status"],
            payload["progress"],
            format_count_rate(main1.get("AnchorAcc(x_raw)")),
            format_count_rate(main1.get("CiteRecall@20")),
            format_count_rate(main1.get("GroundRate")),
            format_count_rate(main1.get("ToolAcc")),
        ])
        delta = main2.get("DeltaToOracle")
        rows2.append([
            payload["model"],
            payload["status"],
            format_count_rate(main2.get("ToolAcc")),
            format_count_rate(oracle_metric),
            format_count_rate(main2.get("Oracle ToolAcc")),
            pct(delta if isinstance(delta, (int, float)) else None),
            format_count_rate(main2.get("AltSourceToolAcc")),
            format_count_rate(main2.get("AltSourceSuccessRate")),
        ])
    return [
        "## Main Table 1: Core End-to-End",
        "",
        build_table(
            ["Model", "Status", "Progress", "AnchorAcc(x_raw)", "CiteRecall@20", "GroundRate", "ToolAcc"],
            rows1,
        ),
        "",
        "## Main Table 2: Oracle Gap and Source-Invariant Success",
        "",
        build_table(
            ["Model", "Status", "ToolAcc", "OracleCoverage", "Oracle ToolAcc", "Delta to Oracle", "AltSourceToolAcc", "AltSourceSuccessRate"],
            rows2,
        ),
        "",
    ]


def build_ablation_tables(model_payloads: List[Dict[str, Any]]) -> List[str]:
    planning_rows = []
    for payload in model_payloads:
        row = [payload["model"], payload["status"]]
        for label in PLANNING_FILES:
            row.append(format_count_rate(payload["planning_ablation"].get(label)))
        planning_rows.append(row)

    query_rows = []
    for payload in model_payloads:
        row = [payload["model"], payload["status"]]
        for label in QUERY_FILES:
            row.append(format_count_rate(payload["query_ablation"].get(label)))
        query_rows.append(row)

    return [
        "## Appendix Table A1a: Planning Ablation",
        "",
        build_table(["Model", "Status", *PLANNING_FILES.keys()], planning_rows),
        "",
        "## Appendix Table A1b: Assisted Query Accuracy",
        "",
        build_table(["Model", "Status", *QUERY_FILES.keys()], query_rows),
        "",
    ]


def build_uncertainty_table(model_payloads: List[Dict[str, Any]]) -> List[str]:
    oracle_override = load_unified_oracle_coverage()
    rows = []
    for payload in model_payloads:
        summary = payload["summary"]
        if not summary:
            continue
        main1 = summary.get("main_table_1", {})
        main2 = summary.get("main_table_2", {})
        for key in UNCERTAINTY_KEYS:
            if key == "OracleCoverage" and oracle_override:
                metric = oracle_override
            else:
                metric = main1.get(key) if key in main1 else main2.get(key)
            if not isinstance(metric, dict):
                continue
            rows.append([
                payload["model"],
                key,
                format_count_rate(metric),
                format_ci(metric.get("wilson95")),
                format_ci(metric.get("paper_cluster_bootstrap95")),
                pct(metric.get("paper_macro") if isinstance(metric.get("paper_macro"), (int, float)) else None),
                pct(metric.get("domain_macro") if isinstance(metric.get("domain_macro"), (int, float)) else None),
            ])
    return [
        "## Appendix Table A2: Statistical Uncertainty",
        "",
        build_table(
            ["Model", "Metric", "Count/Rate", "Wilson 95% CI", "Paper-cluster 95% CI", "Paper Macro", "Domain Macro"],
            rows,
        ),
        "",
    ]


def build_stratified_tables(model_payloads: List[Dict[str, Any]]) -> List[str]:
    sections = []
    section_names = ["By domain", "By tool family", "By source type"]
    for section_name in section_names:
        rows = []
        for payload in model_payloads:
            summary = payload["summary"]
            if not summary:
                continue
            section = summary.get("appendix_stratified", {}).get(section_name, {})
            for bucket, bucket_metrics in section.items():
                rows.append([
                    payload["model"],
                    bucket,
                    str(bucket_metrics.get("papers", "")),
                    str(bucket_metrics.get("gaps", "")),
                    pct(bucket_metrics.get("GroundRate") if isinstance(bucket_metrics.get("GroundRate"), (int, float)) else None),
                    pct(bucket_metrics.get("ToolAcc") if isinstance(bucket_metrics.get("ToolAcc"), (int, float)) else None),
                    pct(bucket_metrics.get("Oracle ToolAcc") if isinstance(bucket_metrics.get("Oracle ToolAcc"), (int, float)) else None),
                ])
        sections.extend([
            f"## Appendix {section_name}",
            "",
            build_table(["Model", "Bucket", "Papers", "Gaps", "GroundRate", "ToolAcc", "Oracle ToolAcc"], rows),
            "",
        ])
    return sections


def build_payload(model_payloads: List[Dict[str, Any]]) -> Dict[str, Any]:
    oracle_override = load_unified_oracle_coverage()
    rows = []
    for payload in model_payloads:
        summary = payload["summary"] or {}
        main2 = dict(summary.get("main_table_2", {}))
        if oracle_override:
            main2["OracleCoverage"] = oracle_override
        rows.append({
            "model": payload["model"],
            "alias": payload["alias"],
            "suite_name": payload["suite_name"],
            "status": payload["status"],
            "progress": payload["progress"],
            "main_table_1": summary.get("main_table_1", {}),
            "main_table_2": main2,
            "appendix_stratified": summary.get("appendix_stratified", {}),
            "planning_ablation": payload["planning_ablation"],
            "query_ablation": payload["query_ablation"],
        })
    all_completed = all(p["summary"] is not None for p in model_payloads)
    return {
        "meta": {
            "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "completed_model_count": sum(1 for p in model_payloads if p["summary"] is not None),
            "model_count": len(model_payloads),
            "excluded_models": ["glm"],
            "notes": [
                (
                    "All listed models are complete."
                    if all_completed
                    else "Some models are still pending; missing metrics are intentionally left blank in markdown tables."
                ),
                "OracleCoverage is normalized to the unified oracle-evaluable subset from revision_oracle_materialization_20260425.json (74/200).",
                "Oracle ToolAcc remains the per-model oracle-run score; only the coverage denominator is normalized in the paper-facing tables.",
                "Ablation tables are drawn from search-side JSON outputs rather than minimal summaries.",
            ],
        },
        "rows": rows,
    }


def build_markdown(model_payloads: List[Dict[str, Any]]) -> str:
    completed = sum(1 for payload in model_payloads if payload["summary"] is not None)
    all_completed = completed == len(model_payloads)
    lines = [
        "# Complete Paper Tables",
        "",
        f"- Generated at: `{time.strftime('%Y-%m-%d %H:%M:%S')}`",
        f"- Completed model summaries: `{completed}/{len(model_payloads)}`",
        "- GLM is intentionally excluded because its planning diagnostics were unstable.",
        (
            "- All listed models are complete."
            if all_completed
            else "- Some models are still pending; unfinished metric cells are left blank."
        ),
        "- `OracleCoverage` is displayed using the unified oracle-evaluable subset from `revision_oracle_materialization_20260425.json` (`74/200`).",
        "- `Oracle ToolAcc` remains the per-model oracle-run score; only the coverage denominator is normalized in these paper-facing tables.",
        "",
    ]
    lines.extend(build_main_tables(model_payloads))
    lines.extend(build_ablation_tables(model_payloads))
    lines.extend(build_uncertainty_table(model_payloads))
    lines.extend(build_stratified_tables(model_payloads))
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the full table bundle for the paper.")
    parser.add_argument("--out-json", default=str(OUT_DIR / "paper_all_tables_latest.json"))
    parser.add_argument("--out-md", default=str(OUT_DIR / "paper_all_tables_latest.md"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model_payloads = [collect_model_payload(run) for run in MODEL_RUNS]
    payload = build_payload(model_payloads)

    out_json = Path(args.out_json)
    out_md = Path(args.out_md)
    out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    out_md.write_text(build_markdown(model_payloads), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    print(f"Saved to {out_json} and {out_md}")


if __name__ == "__main__":
    main()
