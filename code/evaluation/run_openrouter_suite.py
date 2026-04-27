from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from common.openrouter_suite import OPENROUTER_BASE_URL, ModelSpec, parse_model_specs, resolve_dataset_path


OUT_DIR = ROOT / "evaluation" / "outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def getenv_any(*names: str) -> str:
    for name in names:
        value = str(os.getenv(name, "") or "").strip()
        if value:
            return value
    return ""


def pct(value: Optional[float]) -> str:
    if value is None:
        return "NA"
    return f"{100.0 * value:.1f}%"


def ensure_api_key(explicit_key: str = "") -> str:
    api_key = str(explicit_key or "").strip() or getenv_any("OPENROUTER_API_KEY", "OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("Missing OpenRouter API key. Set OPENROUTER_API_KEY (recommended) or OPENAI_API_KEY.")
    return api_key


def ensure_serpapi_key() -> None:
    if getenv_any("SERPAPI_API_KEY", "SERPAPI_KEY"):
        return
    raise RuntimeError("Missing SERPAPI_API_KEY / SERPAPI_KEY. The retrieval run needs Scholar search.")


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def build_subset_dataset(dataset_path: Path, suite_root: Path, limit: Optional[int]) -> Path:
    if limit is None:
        return dataset_path
    subset_path = suite_root / f"dataset_subset_top{limit}.jsonl"
    with dataset_path.open("r", encoding="utf-8") as src, subset_path.open("w", encoding="utf-8") as dst:
        for idx, line in enumerate(src):
            if idx >= limit:
                break
            dst.write(line)
    return subset_path


def env_for_search(
    base_env: Dict[str, str],
    api_key: str,
    model_slug: str,
    reasoning_effort: str,
    judge_model: str = "",
) -> Dict[str, str]:
    env = dict(base_env)
    env["OPENAI_API_KEY"] = api_key
    env["OPENAI_BASE_URL"] = OPENROUTER_BASE_URL
    env["OPENAI_MODEL"] = model_slug
    env["JUDGE_API_KEY"] = str(base_env.get("JUDGE_API_KEY") or api_key)
    env["JUDGE_BASE_URL"] = str(base_env.get("JUDGE_BASE_URL") or OPENROUTER_BASE_URL)
    env["JUDGE_MODEL_NAME"] = str(base_env.get("JUDGE_MODEL_NAME") or judge_model or model_slug)
    env["OPENROUTER_ENABLE_REASONING"] = str(base_env.get("OPENROUTER_ENABLE_REASONING") or "1")
    env["OPENROUTER_REASONING_EFFORT"] = str(base_env.get("OPENROUTER_REASONING_EFFORT") or reasoning_effort)
    return env


def env_for_eval(base_env: Dict[str, str], api_key: str, solver_model: str, judge_model: str, reasoning_effort: str) -> Dict[str, str]:
    env = dict(base_env)
    env["SOLVER_API_KEY"] = api_key
    env["SOLVER_BASE_URL"] = OPENROUTER_BASE_URL
    env["SOLVER_MODEL_NAME"] = solver_model
    env["JUDGE_API_KEY"] = api_key
    env["JUDGE_BASE_URL"] = OPENROUTER_BASE_URL
    env["JUDGE_MODEL_NAME"] = judge_model
    env["OPENROUTER_ENABLE_REASONING"] = str(base_env.get("OPENROUTER_ENABLE_REASONING") or "1")
    env["OPENROUTER_REASONING_EFFORT"] = str(base_env.get("OPENROUTER_REASONING_EFFORT") or reasoning_effort)
    return env


def format_cmd(cmd: Iterable[str]) -> str:
    return shlex.join([str(part) for part in cmd])


def run_command(*, cmd: List[str], env: Dict[str, str], log_path: Path, dry_run: bool) -> Dict[str, Any]:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    if dry_run:
        log_path.write_text(f"[dry-run] {format_cmd(cmd)}\n", encoding="utf-8")
        return {"status": "dry_run", "log": str(log_path), "command": cmd, "returncode": None}

    with log_path.open("w", encoding="utf-8") as logf:
        logf.write(f"$ {format_cmd(cmd)}\n")
        logf.flush()
        proc = subprocess.run(
            cmd,
            cwd=str(ROOT),
            env=env,
            stdout=logf,
            stderr=subprocess.STDOUT,
            check=False,
            text=True,
        )
    return {
        "status": "ok" if proc.returncode == 0 else "failed",
        "log": str(log_path),
        "command": cmd,
        "returncode": proc.returncode,
    }


def should_skip(outputs: Iterable[Path], resume: bool) -> bool:
    if not resume:
        return False
    paths = list(outputs)
    return bool(paths) and all(path.exists() for path in paths)


def path_exists_for_suite(path: Path, dry_run: bool) -> bool:
    return dry_run or path.exists()


def append_snapshot_args(cmd: List[str], snapshot: str) -> None:
    if snapshot:
        cmd.extend(["--snapshot", snapshot])


def make_search_cmd(
    dataset_path: Path,
    out_path: Path,
    *,
    track: str,
    context_variant: str,
    local_window: int,
    diagnostic_mode: str,
    backend: Optional[str],
    top_k: Optional[int],
    snapshot: str,
    log_level: str,
) -> List[str]:
    cmd = [
        sys.executable,
        str(ROOT / "evaluation" / "search.py"),
        "--dataset",
        str(dataset_path),
        "--out",
        str(out_path),
        "--track",
        track,
        "--context-variant",
        context_variant,
        "--local-window",
        str(local_window),
        "--diagnostic-mode",
        diagnostic_mode,
        "--query-mode",
        "model",
        "--log-level",
        log_level,
    ]
    if backend:
        cmd.extend(["--backend", backend])
    if top_k is not None:
        cmd.extend(["--max-results", str(top_k)])
    append_snapshot_args(cmd, snapshot)
    return cmd


def make_end_to_end_cmd(
    dataset_path: Path,
    out_path: Path,
    *,
    search_log: Path,
    download_root: Path,
    backend: str,
    snapshot: str,
    top_k: int,
    shortlist_k: int,
    shortlist_size: int,
    log_level: str,
    skip_planning_judge: bool = False,
) -> List[str]:
    cmd = [
        sys.executable,
        str(ROOT / "evaluation" / "end_to_end_eval.py"),
        "--dataset-file",
        str(dataset_path),
        "--result-file",
        str(out_path),
        "--search-log",
        str(search_log),
        "--download-root",
        str(download_root),
        "--track",
        "raw",
        "--context-variant",
        "global_local",
        "--local-window",
        "5",
        "--backend",
        backend,
        "--top-k",
        str(top_k),
        "--shortlist-k",
        str(shortlist_k),
        "--shortlist-size",
        str(shortlist_size),
        "--log-level",
        log_level,
    ]
    append_snapshot_args(cmd, snapshot)
    if skip_planning_judge:
        cmd.append("--skip-planning-judge")
    return cmd


def make_end_to_end_sharded_cmd(
    dataset_path: Path,
    out_path: Path,
    *,
    search_log: Path,
    download_root: Path,
    log_dir: Path,
    backend: str,
    snapshot: str,
    top_k: int,
    shortlist_k: int,
    shortlist_size: int,
    log_level: str,
    num_shards: int,
    resume: bool,
    skip_planning_judge: bool = False,
) -> List[str]:
    cmd = [
        sys.executable,
        str(ROOT / "evaluation" / "run_end_to_end_sharded.py"),
        "--dataset-file",
        str(dataset_path),
        "--result-file",
        str(out_path),
        "--search-log",
        str(search_log),
        "--download-root",
        str(download_root),
        "--log-dir",
        str(log_dir),
        "--track",
        "raw",
        "--context-variant",
        "global_local",
        "--local-window",
        "5",
        "--backend",
        backend,
        "--top-k",
        str(top_k),
        "--shortlist-k",
        str(shortlist_k),
        "--shortlist-size",
        str(shortlist_size),
        "--num-shards",
        str(num_shards),
        "--log-level",
        log_level,
    ]
    append_snapshot_args(cmd, snapshot)
    if resume:
        cmd.append("--resume")
    if skip_planning_judge:
        cmd.append("--skip-planning-judge")
    return cmd


def make_oracle_materialization_cmd(dataset_path: Path, eval_path: Path, download_root: Path, out_path: Path) -> List[str]:
    return [
        sys.executable,
        str(ROOT / "evaluation" / "materialize_oracle_sources.py"),
        "--dataset-file",
        str(dataset_path),
        "--eval-file",
        str(eval_path),
        "--download-root",
        str(download_root),
        "--out",
        str(out_path),
    ]


def make_oracle_eval_cmd(dataset_path: Path, eval_path: Path, download_root: Path, out_path: Path, oracle_mode: str) -> List[str]:
    return [
        sys.executable,
        str(ROOT / "evaluation" / "oracle_source_eval.py"),
        "--mode",
        oracle_mode,
        "--dataset-file",
        str(dataset_path),
        "--eval-file",
        str(eval_path),
        "--download-root",
        str(download_root),
        "--track",
        "raw",
        "--context-variant",
        "global_local",
        "--local-window",
        "5",
        "--out",
        str(out_path),
    ]


def make_minimal_summary_cmd(
    dataset_path: Path,
    eval_path: Path,
    summary_json: Path,
    summary_md: Path,
    oracle_path: Optional[Path],
    planning_path: Optional[Path],
) -> List[str]:
    cmd = [
        sys.executable,
        str(ROOT / "evaluation" / "minimal_publishable_eval.py"),
        "--dataset",
        str(dataset_path),
        "--eval-file",
        str(eval_path),
        "--out-json",
        str(summary_json),
        "--out-md",
        str(summary_md),
    ]
    if oracle_path is not None:
        cmd.extend(["--oracle-file", str(oracle_path)])
    if planning_path is not None:
        cmd.extend(["--planning-file", str(planning_path)])
    return cmd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the full OpenRouter evaluation suite for multiple models in one command.")
    parser.add_argument("--dataset", default="", help="Dataset JSONL path. If empty, auto-resolve the current usable dataset.")
    parser.add_argument("--suite-name", default="", help="Optional output folder name. Default: openrouter_suite_<timestamp>.")
    parser.add_argument("--out-root", default=str(OUT_DIR / "openrouter_suites"))
    parser.add_argument("--model", action="append", default=[], help="Optional alias=slug override. If omitted, uses the default 8-model suite.")
    parser.add_argument("--api-key", default="", help="Optional OpenRouter API key override.")
    parser.add_argument("--judge-model", default="", help="Optional fixed judge model slug. Default: same as the solver model.")
    parser.add_argument("--limit", type=int, default=None, help="Optional dataset cap for smoke tests.")
    parser.add_argument("--backend", choices=["scholar", "offline_metadata_bm25", "offline_fulltext_bm25"], default="scholar")
    parser.add_argument("--snapshot", default="")
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--shortlist-k", type=int, default=20)
    parser.add_argument("--shortlist-size", type=int, default=1)
    parser.add_argument("--oracle-mode", choices=["fulltext", "blocks"], default="fulltext")
    parser.add_argument("--reasoning-effort", choices=["minimal", "low", "medium", "high", "xhigh"], default="low")
    parser.add_argument("--parallel-models", type=int, default=1, help="Number of model pipelines to run concurrently.")
    parser.add_argument("--end-to-end-shards", type=int, default=1, help="Parallel shards for the end-to-end stage only.")
    parser.add_argument("--resume", action="store_true", help="Skip steps whose output files already exist.")
    parser.add_argument("--skip-oracle", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--log-level", default=os.getenv("LOG_LEVEL", "INFO"))
    return parser.parse_args()


def run_model_pipeline(
    spec: ModelSpec,
    *,
    suite_root: Path,
    dataset_path: Path,
    api_key: str,
    judge_model: str,
    args: argparse.Namespace,
    base_env: Dict[str, str],
) -> Dict[str, Any]:
    model_root = suite_root / spec.alias
    results_dir = model_root / "results"
    logs_dir = model_root / "logs"
    downloads_dir = model_root / "downloads"
    results_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)
    downloads_dir.mkdir(parents=True, exist_ok=True)
    if args.end_to_end_shards < 1:
        raise ValueError("--end-to-end-shards must be >= 1")

    outputs = {
        "planning_local_only_m5": results_dir / "search_planning_raw_local_only_m5.json",
        "planning_global_local_m1": results_dir / "search_planning_raw_global_local_m1.json",
        "planning_global_local_m3": results_dir / "search_planning_raw_global_local_m3.json",
        "planning_global_local_m5": results_dir / "search_planning_raw_global_local_m5.json",
        "query_assist_global_local_m5": results_dir / "search_query_assist_global_local_m5.json",
        "retrieval_raw_global_local_m5": results_dir / "search_retrieval_raw_global_local_m5.json",
        "end_to_end_raw_global_local_m5": results_dir / "end_to_end_raw_global_local_m5.json",
        "oracle_materialization": results_dir / "oracle_materialization.json",
        "oracle_fulltext_raw_global_local_m5": results_dir / f"oracle_{args.oracle_mode}_raw_global_local_m5.json",
        "minimal_summary_json": results_dir / "minimal_publishable_summary.json",
        "minimal_summary_md": results_dir / "minimal_publishable_summary.md",
    }

    plan = [
        {
            "name": "planning_local_only_m5",
            "outputs": [outputs["planning_local_only_m5"]],
            "env": env_for_search(base_env, api_key, spec.model_slug, args.reasoning_effort, judge_model or spec.model_slug),
            "cmd": make_search_cmd(
                dataset_path,
                outputs["planning_local_only_m5"],
                track="raw",
                context_variant="local_only",
                local_window=5,
                diagnostic_mode="planning",
                backend=None,
                top_k=None,
                snapshot="",
                log_level=args.log_level,
            ),
        },
        {
            "name": "planning_global_local_m1",
            "outputs": [outputs["planning_global_local_m1"]],
            "env": env_for_search(base_env, api_key, spec.model_slug, args.reasoning_effort, judge_model or spec.model_slug),
            "cmd": make_search_cmd(
                dataset_path,
                outputs["planning_global_local_m1"],
                track="raw",
                context_variant="global_local",
                local_window=1,
                diagnostic_mode="planning",
                backend=None,
                top_k=None,
                snapshot="",
                log_level=args.log_level,
            ),
        },
        {
            "name": "planning_global_local_m3",
            "outputs": [outputs["planning_global_local_m3"]],
            "env": env_for_search(base_env, api_key, spec.model_slug, args.reasoning_effort, judge_model or spec.model_slug),
            "cmd": make_search_cmd(
                dataset_path,
                outputs["planning_global_local_m3"],
                track="raw",
                context_variant="global_local",
                local_window=3,
                diagnostic_mode="planning",
                backend=None,
                top_k=None,
                snapshot="",
                log_level=args.log_level,
            ),
        },
        {
            "name": "planning_global_local_m5",
            "outputs": [outputs["planning_global_local_m5"]],
            "env": env_for_search(base_env, api_key, spec.model_slug, args.reasoning_effort, judge_model or spec.model_slug),
            "cmd": make_search_cmd(
                dataset_path,
                outputs["planning_global_local_m5"],
                track="raw",
                context_variant="global_local",
                local_window=5,
                diagnostic_mode="planning",
                backend=None,
                top_k=None,
                snapshot="",
                log_level=args.log_level,
            ),
        },
        {
            "name": "query_assist_global_local_m5",
            "outputs": [outputs["query_assist_global_local_m5"]],
            "env": env_for_search(base_env, api_key, spec.model_slug, args.reasoning_effort, judge_model or spec.model_slug),
            "cmd": make_search_cmd(
                dataset_path,
                outputs["query_assist_global_local_m5"],
                track="assist",
                context_variant="global_local",
                local_window=5,
                diagnostic_mode="query",
                backend=None,
                top_k=None,
                snapshot="",
                log_level=args.log_level,
            ),
        },
        {
            "name": "retrieval_raw_global_local_m5",
            "outputs": [outputs["retrieval_raw_global_local_m5"]],
            "env": env_for_search(base_env, api_key, spec.model_slug, args.reasoning_effort, judge_model or spec.model_slug),
            "cmd": make_search_cmd(
                dataset_path,
                outputs["retrieval_raw_global_local_m5"],
                track="raw",
                context_variant="global_local",
                local_window=5,
                diagnostic_mode="retrieval",
                backend=args.backend,
                top_k=min(args.top_k, 20),
                snapshot=args.snapshot,
                log_level=args.log_level,
            ),
        },
        {
            "name": "end_to_end_raw_global_local_m5",
            "outputs": [outputs["end_to_end_raw_global_local_m5"]],
            "env": env_for_eval(base_env, api_key, spec.model_slug, judge_model or spec.model_slug, args.reasoning_effort),
            "cmd": (
                make_end_to_end_sharded_cmd(
                    dataset_path,
                    outputs["end_to_end_raw_global_local_m5"],
                    search_log=outputs["retrieval_raw_global_local_m5"],
                    download_root=downloads_dir,
                    log_dir=logs_dir,
                    backend=args.backend,
                    snapshot=args.snapshot,
                    top_k=min(args.top_k, 20),
                    shortlist_k=args.shortlist_k,
                    shortlist_size=args.shortlist_size,
                    log_level=args.log_level,
                    num_shards=args.end_to_end_shards,
                    resume=args.resume,
                    skip_planning_judge=True,
                )
                if args.end_to_end_shards > 1
                else make_end_to_end_cmd(
                    dataset_path,
                    outputs["end_to_end_raw_global_local_m5"],
                    search_log=outputs["retrieval_raw_global_local_m5"],
                    download_root=downloads_dir,
                    backend=args.backend,
                    snapshot=args.snapshot,
                    top_k=min(args.top_k, 20),
                    shortlist_k=args.shortlist_k,
                    shortlist_size=args.shortlist_size,
                    log_level=args.log_level,
                    skip_planning_judge=True,
                )
            ),
        },
    ]

    oracle_output: Optional[Path] = None
    if not args.skip_oracle:
        oracle_output = outputs["oracle_fulltext_raw_global_local_m5"]
        plan.extend(
            [
                {
                    "name": "oracle_materialization",
                    "outputs": [outputs["oracle_materialization"]],
                    "env": dict(base_env),
                    "cmd": make_oracle_materialization_cmd(
                        dataset_path,
                        outputs["end_to_end_raw_global_local_m5"],
                        downloads_dir,
                        outputs["oracle_materialization"],
                    ),
                },
                {
                    "name": "oracle_fulltext_raw_global_local_m5",
                    "outputs": [outputs["oracle_fulltext_raw_global_local_m5"]],
                    "env": env_for_eval(base_env, api_key, spec.model_slug, judge_model or spec.model_slug, args.reasoning_effort),
                    "cmd": make_oracle_eval_cmd(
                        dataset_path,
                        outputs["end_to_end_raw_global_local_m5"],
                        downloads_dir,
                        outputs["oracle_fulltext_raw_global_local_m5"],
                        args.oracle_mode,
                    ),
                },
            ]
        )

    plan.append(
        {
            "name": "minimal_summary",
            "outputs": [outputs["minimal_summary_json"], outputs["minimal_summary_md"]],
            "env": dict(base_env),
            "cmd": make_minimal_summary_cmd(
                dataset_path,
                outputs["end_to_end_raw_global_local_m5"],
                outputs["minimal_summary_json"],
                outputs["minimal_summary_md"],
                oracle_output,
                outputs["planning_global_local_m5"],
            ),
        }
    )

    steps: List[Dict[str, Any]] = []
    status = "ok"
    failed_step = None

    for step in plan:
        log_path = logs_dir / f"{step['name']}.log"
        if should_skip(step["outputs"], args.resume):
            if not log_path.exists():
                log_path.write_text("[resume] skipped because outputs already exist\n", encoding="utf-8")
            step_result = {
                "name": step["name"],
                "status": "skipped_existing",
                "log": str(log_path),
                "outputs": [str(path) for path in step["outputs"]],
                "command": step["cmd"],
                "returncode": None,
            }
        else:
            result = run_command(cmd=step["cmd"], env=step["env"], log_path=log_path, dry_run=args.dry_run)
            step_result = {
                "name": step["name"],
                "status": result["status"],
                "log": result["log"],
                "outputs": [str(path) for path in step["outputs"]],
                "command": result["command"],
                "returncode": result["returncode"],
            }
        steps.append(step_result)
        if step_result["status"] == "failed":
            status = "failed"
            failed_step = step["name"]
            if args.fail_fast:
                break

    return {
        "alias": spec.alias,
        "model_slug": spec.model_slug,
        "judge_model": judge_model or spec.model_slug,
        "status": status,
        "failed_step": failed_step,
        "download_root": str(downloads_dir),
        "outputs": {name: str(path) for name, path in outputs.items()},
        "steps": steps,
    }


def build_suite_table(model_runs: List[Dict[str, Any]], suite_root: Path) -> Dict[str, str]:
    rows: List[dict] = []
    for run in model_runs:
        summary_path = Path(run["outputs"].get("minimal_summary_json", ""))
        if run.get("status") == "failed" or not summary_path.exists():
            rows.append(
                {
                    "model": run["alias"],
                    "model_slug": run["model_slug"],
                    "status": run.get("status"),
                    "AnchorAcc(x_raw)": None,
                    "CiteRecall@20": None,
                    "GroundRate": None,
                    "ToolAcc": None,
                    "OracleCoverage": None,
                    "Oracle ToolAcc": None,
                    "DeltaToOracle": None,
                    "AltSourceToolAcc": None,
                    "AltSourceSuccessRate": None,
                }
            )
            continue
        payload = load_json(summary_path)
        main1 = payload.get("main_table_1", {})
        main2 = payload.get("main_table_2", {})
        rows.append(
            {
                "model": run["alias"],
                "model_slug": run["model_slug"],
                "status": run.get("status"),
                "AnchorAcc(x_raw)": (main1.get("AnchorAcc(x_raw)") or {}).get("rate"),
                "CiteRecall@20": (main1.get("CiteRecall@20") or {}).get("rate"),
                "GroundRate": (main1.get("GroundRate") or {}).get("rate"),
                "ToolAcc": (main1.get("ToolAcc") or {}).get("rate"),
                "OracleCoverage": (main2.get("OracleCoverage") or {}).get("rate"),
                "Oracle ToolAcc": (main2.get("Oracle ToolAcc") or {}).get("rate"),
                "DeltaToOracle": main2.get("DeltaToOracle"),
                "AltSourceToolAcc": (main2.get("AltSourceToolAcc") or {}).get("rate"),
                "AltSourceSuccessRate": (main2.get("AltSourceSuccessRate") or {}).get("rate"),
            }
        )

    payload = {"rows": rows}
    json_path = suite_root / "suite_tables.json"
    md_path = suite_root / "suite_tables.md"
    write_json(json_path, payload)

    lines = [
        "# OpenRouter Suite Tables",
        "",
        "| Model | OpenRouter slug | AnchorAcc(x_raw) | CiteRecall@20 | GroundRate | ToolAcc | OracleCoverage | Oracle ToolAcc | Δ to Oracle | AltSourceToolAcc | AltSourceSuccessRate |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in rows:
        lines.append(
            f"| {row['model']} | {row['model_slug']} | {pct(row['AnchorAcc(x_raw)'])} | {pct(row['CiteRecall@20'])} | {pct(row['GroundRate'])} | {pct(row['ToolAcc'])} | {pct(row['OracleCoverage'])} | {pct(row['Oracle ToolAcc'])} | {pct(row['DeltaToOracle'])} | {pct(row['AltSourceToolAcc'])} | {pct(row['AltSourceSuccessRate'])} |"
        )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {"suite_tables_json": str(json_path), "suite_tables_md": str(md_path)}


def main() -> None:
    args = parse_args()
    api_key = ensure_api_key(args.api_key)
    if not args.dry_run and args.backend == "scholar":
        ensure_serpapi_key()

    model_specs = parse_model_specs(args.model)
    dataset_path = resolve_dataset_path(args.dataset)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    suite_name = args.suite_name or f"openrouter_suite_{timestamp}"
    suite_root = Path(args.out_root) / suite_name
    suite_root.mkdir(parents=True, exist_ok=True)
    dataset_for_run = build_subset_dataset(dataset_path, suite_root, args.limit)

    base_env = dict(os.environ)
    base_env["PYTHONUNBUFFERED"] = "1"

    if args.parallel_models < 1:
        raise ValueError("--parallel-models must be >= 1")

    model_runs: List[Dict[str, Any]] = []
    if args.parallel_models == 1 or len(model_specs) <= 1:
        for spec in model_specs:
            model_run = run_model_pipeline(
                spec,
                suite_root=suite_root,
                dataset_path=dataset_for_run,
                api_key=api_key,
                judge_model=args.judge_model,
                args=args,
                base_env=base_env,
            )
            model_runs.append(model_run)
            if model_run["status"] == "failed" and args.fail_fast:
                break
    else:
        future_map = {}
        run_by_alias: Dict[str, Dict[str, Any]] = {}
        max_workers = min(args.parallel_models, len(model_specs))
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            for spec in model_specs:
                future = executor.submit(
                    run_model_pipeline,
                    spec,
                    suite_root=suite_root,
                    dataset_path=dataset_for_run,
                    api_key=api_key,
                    judge_model=args.judge_model,
                    args=args,
                    base_env=base_env,
                )
                future_map[future] = spec
            for future in as_completed(future_map):
                spec = future_map[future]
                run_by_alias[spec.alias] = future.result()
        model_runs = [run_by_alias[spec.alias] for spec in model_specs if spec.alias in run_by_alias]

    suite_outputs: Dict[str, str] = {}

    ablation_runs: List[str] = []
    for run in model_runs:
        outputs = run.get("outputs", {})
        mapping = [
            (f"{run['alias']}:planning:local_only:m5", outputs.get("planning_local_only_m5")),
            (f"{run['alias']}:planning:global_local:m1", outputs.get("planning_global_local_m1")),
            (f"{run['alias']}:planning:global_local:m3", outputs.get("planning_global_local_m3")),
            (f"{run['alias']}:planning:global_local:m5", outputs.get("planning_global_local_m5")),
            (f"{run['alias']}:query:assist:m5", outputs.get("query_assist_global_local_m5")),
        ]
        for label, raw_path in mapping:
            if raw_path and path_exists_for_suite(Path(raw_path), args.dry_run):
                ablation_runs.append(f"{label}={raw_path}")

    if ablation_runs:
        cmd = [
            sys.executable,
            str(ROOT / "evaluation" / "ablation_summary.py"),
            "--out-json",
            str(suite_root / "ablation_summary.json"),
            "--out-md",
            str(suite_root / "ablation_summary.md"),
        ]
        for run_spec in ablation_runs:
            cmd.extend(["--run", run_spec])
        run_command(cmd=cmd, env=dict(base_env), log_path=suite_root / "ablation_summary.log", dry_run=args.dry_run)
        suite_outputs["ablation_summary_json"] = str(suite_root / "ablation_summary.json")
        suite_outputs["ablation_summary_md"] = str(suite_root / "ablation_summary.md")

    failure_runs: List[str] = []
    failure_search_logs: List[str] = []
    for run in model_runs:
        outputs = run.get("outputs", {})
        eval_path = outputs.get("end_to_end_raw_global_local_m5")
        search_path = outputs.get("retrieval_raw_global_local_m5")
        if eval_path and path_exists_for_suite(Path(eval_path), args.dry_run):
            failure_runs.append(f"{run['alias']}={eval_path}")
        if search_path and path_exists_for_suite(Path(search_path), args.dry_run):
            failure_search_logs.append(f"{run['alias']}={search_path}")

    if failure_runs:
        cmd = [
            sys.executable,
            str(ROOT / "evaluation" / "failure_decomposition.py"),
            "--out-json",
            str(suite_root / "failure_decomposition.json"),
            "--out-md",
            str(suite_root / "failure_decomposition.md"),
        ]
        for run_spec in failure_runs:
            cmd.extend(["--run", run_spec])
        for search_spec in failure_search_logs:
            cmd.extend(["--search-log", search_spec])
        run_command(cmd=cmd, env=dict(base_env), log_path=suite_root / "failure_decomposition.log", dry_run=args.dry_run)
        suite_outputs["failure_decomposition_json"] = str(suite_root / "failure_decomposition.json")
        suite_outputs["failure_decomposition_md"] = str(suite_root / "failure_decomposition.md")

    suite_outputs.update(build_suite_table(model_runs, suite_root))

    manifest = {
        "meta": {
            "generated_at": timestamp,
            "suite_name": suite_name,
            "suite_root": str(suite_root),
            "dataset": str(dataset_for_run),
            "dataset_source": str(dataset_path),
            "openrouter_base_url": OPENROUTER_BASE_URL,
            "judge_model": args.judge_model or None,
            "backend": args.backend,
            "snapshot": args.snapshot or None,
            "top_k": min(args.top_k, 20),
            "shortlist_k": args.shortlist_k,
            "shortlist_size": args.shortlist_size,
            "end_to_end_shards": args.end_to_end_shards,
            "oracle_mode": None if args.skip_oracle else args.oracle_mode,
            "reasoning_effort": args.reasoning_effort,
            "dry_run": args.dry_run,
            "resume": args.resume,
        },
        "models": model_runs,
        "suite_outputs": suite_outputs,
    }
    manifest_path = suite_root / "suite_manifest.json"
    write_json(manifest_path, manifest)
    print(json.dumps(manifest, indent=2, ensure_ascii=False))
    print(f"Saved to {manifest_path}")


if __name__ == "__main__":
    main()
