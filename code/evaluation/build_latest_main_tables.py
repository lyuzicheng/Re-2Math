from __future__ import annotations

import argparse
import json
import re
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
    model_slug: str
    suite_name: str

    @property
    def root(self) -> Path:
        return SUITES_DIR / self.suite_name / self.alias


DEFAULT_MODEL_RUNS: List[ModelRun] = [
    ModelRun(
        alias="gpt",
        display_name="GPT-5.2",
        model_slug="openai/gpt-5.2",
        suite_name="eval200_openrouter_8models_parallel_20260422",
    ),
    ModelRun(
        alias="gemini",
        display_name="Gemini 3.1 Pro",
        model_slug="google/gemini-3.1-pro-preview",
        suite_name="eval200_openrouter_gemini31_rerun_full_20260423_v2",
    ),
    ModelRun(
        alias="claude",
        display_name="Claude Opus 4.5",
        model_slug="anthropic/claude-opus-4.5",
        suite_name="eval200_openrouter_8models_parallel_20260422",
    ),
    ModelRun(
        alias="deepseek",
        display_name="DeepSeek V3.2",
        model_slug="deepseek/deepseek-v3.2",
        suite_name="eval200_openrouter_deepseek_v32_rerun_20260423",
    ),
    ModelRun(
        alias="qwen",
        display_name="Qwen3-235B Thinking",
        model_slug="qwen/qwen3-235b-a22b-thinking-2507",
        suite_name="eval200_openrouter_8models_parallel_20260422",
    ),
    ModelRun(
        alias="kimi",
        display_name="Kimi K2 Thinking",
        model_slug="moonshotai/kimi-k2-thinking",
        suite_name="eval200_openrouter_8models_parallel_20260422",
    ),
    ModelRun(
        alias="grok",
        display_name="Grok 4",
        model_slug="x-ai/grok-4",
        suite_name="eval200_openrouter_8models_parallel_20260422",
    ),
]


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def pct(value: Optional[float]) -> str:
    if value is None:
        return ""
    return f"{100.0 * value:.1f}%"


def metric_rate(table: Dict[str, Any], key: str) -> Optional[float]:
    value = table.get(key)
    if isinstance(value, dict):
        rate = value.get("rate")
        return float(rate) if rate is not None else None
    return None


def metric_count(table: Dict[str, Any], key: str) -> str:
    value = table.get(key)
    if not isinstance(value, dict):
        return ""
    success = value.get("success")
    total = value.get("total")
    if success is None or total is None:
        return ""
    return f"{success}/{total}"


def metric_cell(count: str, rate: Optional[float]) -> str:
    if not count and rate is None:
        return ""
    return f"{count} ({pct(rate)})" if count else pct(rate)


def load_unified_oracle_coverage() -> Optional[dict]:
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
    }


def latest_log(model_root: Path) -> Optional[Path]:
    logs_dir = model_root / "logs"
    if not logs_dir.exists():
        return None
    logs = sorted(logs_dir.glob("*.log"), key=lambda path: path.stat().st_mtime, reverse=True)
    return logs[0] if logs else None


def step_name_from_log(log_path: Optional[Path]) -> str:
    if not log_path:
        return "not_started"
    return log_path.stem


def infer_progress(log_path: Optional[Path]) -> str:
    if not log_path or not log_path.exists():
        return "not_started"
    text = log_path.read_text(errors="ignore")
    if "Traceback" in text:
        return "failed"
    if "Saved to" in text or "Saved " in text:
        return "step_completed"

    processing = re.findall(r"\[(\d+)/(\d+)\] Processing id=([^\s]+)", text)
    if processing:
        current, total, _ = processing[-1]
        return f"{current}/{total}"

    tasks = re.findall(r"Tasks:\s+\d+%[^\r\n]*?\|\s*(\d+)/(\d+)", text)
    if tasks:
        current, total = tasks[-1]
        return f"{current}/{total}"
    return "running"


def summarize_model(run: ModelRun) -> dict:
    model_root = run.root
    summary_path = model_root / "results" / "minimal_publishable_summary.json"
    log_path = latest_log(model_root)
    status = "completed" if summary_path.exists() else "running"
    if not model_root.exists():
        status = "missing"
    if summary_path.exists():
        progress = "completed"
    else:
        progress = f"{step_name_from_log(log_path)}: {infer_progress(log_path)}"

    row: Dict[str, Any] = {
        "model": run.display_name,
        "alias": run.alias,
        "model_slug": run.model_slug,
        "suite_name": run.suite_name,
        "status": status,
        "progress": progress,
        "summary_path": str(summary_path) if summary_path.exists() else "",
        "latest_log": str(log_path) if log_path else "",
        "evaluated_rows": None,
        "oracle_evaluable_rows": None,
        "AnchorAcc(x_raw)": None,
        "AnchorAcc(x_raw)_count": "",
        "CiteRecall@20": None,
        "CiteRecall@20_count": "",
        "GroundRate": None,
        "GroundRate_count": "",
        "ToolAcc": None,
        "ToolAcc_count": "",
        "OracleCoverage": None,
        "OracleCoverage_count": "",
        "Oracle ToolAcc": None,
        "Oracle ToolAcc_count": "",
        "DeltaToOracle": None,
        "AltSourceToolAcc": None,
        "AltSourceToolAcc_count": "",
        "AltSourceSuccessRate": None,
        "AltSourceSuccessRate_count": "",
    }

    if not summary_path.exists():
        return row

    payload = load_json(summary_path)
    meta = payload.get("meta", {})
    main1 = payload.get("main_table_1", {})
    main2 = payload.get("main_table_2", {})
    row["evaluated_rows"] = meta.get("evaluated_rows")
    row["oracle_evaluable_rows"] = meta.get("oracle_evaluable_rows")
    for key in ["AnchorAcc(x_raw)", "CiteRecall@20", "GroundRate", "ToolAcc"]:
        row[key] = metric_rate(main1, key)
        row[f"{key}_count"] = metric_count(main1, key)
    for key in ["OracleCoverage", "Oracle ToolAcc", "AltSourceToolAcc", "AltSourceSuccessRate"]:
        row[key] = metric_rate(main2, key)
        row[f"{key}_count"] = metric_count(main2, key)
    row["DeltaToOracle"] = main2.get("DeltaToOracle")
    return row


def oracle_coverage_for_row(row: dict, override: Optional[dict]) -> tuple[str, Optional[float]]:
    if override:
        return metric_count({"OracleCoverage": override}, "OracleCoverage"), override["rate"]
    return row["OracleCoverage_count"], row["OracleCoverage"]


def build_main_table_1(rows: List[dict]) -> str:
    lines = [
        "## Main Table 1: Core End-to-End",
        "",
        "| Model | Status | Progress | AnchorAcc(x_raw) | CiteRecall@20 | GroundRate | ToolAcc |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in rows:
        lines.append(
            f"| {row['model']} | {row['status']} | {row['progress']} | "
            f"{metric_cell(row['AnchorAcc(x_raw)_count'], row['AnchorAcc(x_raw)'])} | "
            f"{metric_cell(row['CiteRecall@20_count'], row['CiteRecall@20'])} | "
            f"{metric_cell(row['GroundRate_count'], row['GroundRate'])} | "
            f"{metric_cell(row['ToolAcc_count'], row['ToolAcc'])} |"
        )
    return "\n".join(lines)


def build_main_table_2(rows: List[dict]) -> str:
    override = load_unified_oracle_coverage()
    lines = [
        "## Main Table 2: Oracle Gap and Source-Invariant Success",
        "",
        "| Model | Status | ToolAcc | OracleCoverage | Oracle ToolAcc | Delta to Oracle | AltSourceToolAcc | AltSourceSuccessRate |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in rows:
        oracle_count, oracle_rate = oracle_coverage_for_row(row, override)
        lines.append(
            f"| {row['model']} | {row['status']} | "
            f"{metric_cell(row['ToolAcc_count'], row['ToolAcc'])} | "
            f"{metric_cell(oracle_count, oracle_rate)} | "
            f"{metric_cell(row['Oracle ToolAcc_count'], row['Oracle ToolAcc'])} | "
            f"{pct(row['DeltaToOracle'])} | "
            f"{metric_cell(row['AltSourceToolAcc_count'], row['AltSourceToolAcc'])} | "
            f"{metric_cell(row['AltSourceSuccessRate_count'], row['AltSourceSuccessRate'])} |"
        )
    return "\n".join(lines)


def build_markdown(rows: List[dict]) -> str:
    completed = sum(1 for row in rows if row["status"] == "completed")
    lines = [
        "# Latest Paper Main Tables",
        "",
        f"- Generated at: `{time.strftime('%Y-%m-%d %H:%M:%S')}`",
        f"- Completed model summaries: `{completed}/{len(rows)}`",
        "- GLM is intentionally excluded because its planning diagnostics were unstable.",
        "- All listed models are complete.",
        "- `OracleCoverage` is displayed using the unified oracle-evaluable subset from `revision_oracle_materialization_20260425.json` (`74/200`), i.e. the union of cited-source materializations under the release protocol.",
        "- `Oracle ToolAcc` still reflects each model's completed oracle run; only the coverage denominator is normalized in this paper-facing table.",
        "",
        build_main_table_1(rows),
        "",
        build_main_table_2(rows),
        "",
    ]
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build latest paper-facing main tables across model suites.")
    parser.add_argument("--out-json", default=str(OUT_DIR / "paper_main_tables_latest.json"))
    parser.add_argument("--out-md", default=str(OUT_DIR / "paper_main_tables_latest.md"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = [summarize_model(run) for run in DEFAULT_MODEL_RUNS]
    payload = {
        "meta": {
            "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "excluded_models": ["glm"],
            "model_count": len(rows),
            "completed_model_count": sum(1 for row in rows if row["status"] == "completed"),
        },
        "rows": rows,
    }
    out_json = Path(args.out_json)
    out_md = Path(args.out_md)
    write_json(out_json, payload)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text(build_markdown(rows), encoding="utf-8")
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    print(f"Saved to {out_json} and {out_md}")


if __name__ == "__main__":
    main()
