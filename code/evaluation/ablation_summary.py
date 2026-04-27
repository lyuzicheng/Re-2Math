from __future__ import annotations

import argparse
import json
import math
import random
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "evaluation" / "outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def dump_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def parse_label_path(items: List[str]) -> Dict[str, Path]:
    mapping: Dict[str, Path] = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"Run must use label=path format: {item}")
        label, raw_path = item.split("=", 1)
        label = label.strip()
        raw_path = raw_path.strip()
        if not label or not raw_path:
            raise ValueError(f"Invalid run spec: {item}")
        mapping[label] = Path(raw_path)
    return mapping


def wilson_interval(successes: int, total: int, z: float = 1.96) -> tuple[float, float]:
    if total == 0:
        return 0.0, 0.0
    phat = successes / total
    denom = 1 + (z * z) / total
    center = (phat + (z * z) / (2 * total)) / denom
    margin = z * math.sqrt((phat * (1 - phat) / total) + (z * z) / (4 * total * total)) / denom
    return max(0.0, center - margin), min(1.0, center + margin)


def clustered_bootstrap_rate(rows: List[dict], metric_key: str, *, n_boot: int, seed: int) -> Optional[tuple[float, float]]:
    grouped: Dict[str, List[dict]] = defaultdict(list)
    for row in rows:
        cluster = str(row.get("paper_id") or "").strip()
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
            estimates.append(sum(1 for row in sampled_rows if row.get(metric_key) is True) / len(sampled_rows))
    if not estimates:
        return None
    estimates.sort()
    lo_idx = int(0.025 * (len(estimates) - 1))
    hi_idx = int(0.975 * (len(estimates) - 1))
    return estimates[lo_idx], estimates[hi_idx]


def pct(value: Optional[float]) -> str:
    if value is None:
        return "NA"
    return f"{100.0 * value:.1f}%"


def summarize_run(label: str, path: Path, *, n_boot: int, seed: int) -> dict:
    payload = load_json(path)
    meta = payload.get("meta", {}) if isinstance(payload, dict) else {}
    rows = payload.get("details", payload if isinstance(payload, list) else []) or []

    diagnostic_mode = str(meta.get("diagnostic_mode", "") or "")
    track = str(meta.get("track", "") or "")
    if diagnostic_mode not in {"planning", "query"}:
        raise ValueError(f"{label}: ablation summary only supports planning/query logs, got diagnostic_mode={diagnostic_mode!r}")
    if diagnostic_mode == "planning" and track != "raw":
        raise ValueError(f"{label}: planning ablations must use the raw track, got track={track!r}")
    if diagnostic_mode == "query" and track != "assist":
        raise ValueError(f"{label}: query ablations must use the assist track, got track={track!r}")

    metric_key = "planning_is_match" if diagnostic_mode == "planning" else "query_is_match"
    metric_label = "AnchorAcc(x_raw)" if diagnostic_mode == "planning" else "QueryAcc(x_assist)"

    success = sum(1 for row in rows if row.get(metric_key) is True)
    total = len(rows)
    wilson = wilson_interval(success, total) if total else None
    cluster = clustered_bootstrap_rate(rows, metric_key, n_boot=n_boot, seed=seed)
    return {
        "label": label,
        "path": str(path),
        "metric": metric_label,
        "track": track,
        "context_variant": meta.get("context_variant"),
        "local_window": meta.get("local_window"),
        "success": success,
        "total": total,
        "rate": success / total if total else 0.0,
        "wilson95": list(wilson) if wilson else None,
        "paper_cluster_bootstrap95": list(cluster) if cluster else None,
    }


def build_markdown(runs: List[dict]) -> str:
    lines = [
        "# Appendix Table A1: Task-Design Ablation",
        "",
        "| Condition | Metric | Context | Local window m | Rate | Wilson 95% CI | Paper-Cluster 95% CI |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for run in runs:
        wilson = run.get("wilson95")
        cluster = run.get("paper_cluster_bootstrap95")
        wilson_text = "NA" if not wilson else f"[{pct(wilson[0])}, {pct(wilson[1])}]"
        cluster_text = "NA" if not cluster else f"[{pct(cluster[0])}, {pct(cluster[1])}]"
        lines.append(
            f"| {run['label']} | {run.get('metric') or 'NA'} | {run.get('context_variant') or 'NA'} | "
            f"{run.get('local_window') or 'NA'} | {run['success']}/{run['total']} ({pct(run['rate'])}) | "
            f"{wilson_text} | {cluster_text} |"
        )
    lines.append("")
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize the final non-redundant task-design ablations.")
    parser.add_argument(
        "--run",
        action="append",
        default=[],
        help="Required label=search_log pair. May be passed multiple times.",
    )
    parser.add_argument("--out-json", default="")
    parser.add_argument("--out-md", default="")
    parser.add_argument("--bootstrap-samples", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    runs = parse_label_path(args.run)
    if not runs:
        raise ValueError("At least one --run label=search_log is required.")

    summaries = [
        summarize_run(label, path, n_boot=args.bootstrap_samples, seed=args.seed)
        for label, path in runs.items()
    ]
    summaries.sort(
        key=lambda item: (
            str(item.get("metric") or ""),
            str(item.get("context_variant") or ""),
            int(item.get("local_window") or 0),
            item["label"],
        )
    )

    payload = {
        "meta": {
            "bootstrap_samples": args.bootstrap_samples,
            "seed": args.seed,
        },
        "runs": summaries,
    }

    out_json = Path(args.out_json) if args.out_json else OUT_DIR / "ablation_summary.json"
    out_md = Path(args.out_md) if args.out_md else OUT_DIR / "ablation_summary.md"
    dump_json(out_json, payload)
    out_md.write_text(build_markdown(summaries), encoding="utf-8")
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    print(f"Saved to {out_json} and {out_md}")


if __name__ == "__main__":
    main()
