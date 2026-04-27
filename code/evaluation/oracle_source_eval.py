from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import fitz  # PyMuPDF

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from common.dataset_format import (
    CANONICAL_LOCAL_WINDOW,
    DEFAULT_DATASET_FILE,
    SUPPORTED_LOCAL_WINDOWS,
    build_query_context,
    get_citation_content,
    get_reference_tool_latex,
    load_dataset_as_dict,
    normalize_official_local_window,
)
from evaluation.end_to_end_eval import LLMJsonCaller, SemanticEvaluator, build_llm_config_from_env


OUT_DIR = ROOT / "evaluation" / "outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_CONFIGS = {
    "gemini": {
        "eval_file": ROOT / "evaluation" / "outputs" / "evaluation_results_full_scan_gm_corrected.json",
        "search_log": ROOT / "evaluation" / "outputs" / "search_log_full_20251228_230756.json",
        "download_root": ROOT / "scholar_all_top20_downloads_gemini",
    },
    "deepseek": {
        "eval_file": ROOT / "evaluation" / "outputs" / "evaluation_results_full_scan_ds_corrected.json",
        "search_log": ROOT / "evaluation" / "outputs" / "search_log_full_20251229_193110.json",
        "download_root": ROOT / "scholar_all_top20_downloads_ds",
    },
    "qwen": {
        "eval_file": ROOT / "evaluation" / "outputs" / "evaluation_results_full_scan_qwen_corrected.json",
        "search_log": ROOT / "evaluation" / "outputs" / "search_log_full_20251229_202649.json",
        "download_root": ROOT / "scholar_all_top20_downloads_qwen",
    },
    "claude": {
        "eval_file": ROOT / "evaluation" / "outputs" / "evaluation_results_full_scan_claude_corrected.json",
        "search_log": ROOT / "evaluation" / "outputs" / "search_log_full_20251229_214942.json",
        "download_root": ROOT / "scholar_all_top20_downloads_claude",
    },
}


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def resolve_case_ids(dataset: Dict[str, dict], eval_file: Optional[Path], limit: Optional[int]) -> tuple[List[str], str]:
    if eval_file:
        payload = load_json(eval_file)
        raw_rows = payload.get("results", payload if isinstance(payload, list) else []) or []
        case_ids = [str(row.get("id", "")).strip() for row in raw_rows if str(row.get("id", "")).strip() in dataset]
        selection = "eval_subset"
    else:
        case_ids = sorted(dataset.keys())
        selection = "all_dataset_cases"

    deduped = []
    seen = set()
    for case_id in case_ids:
        if case_id and case_id not in seen:
            seen.add(case_id)
            deduped.append(case_id)

    if limit is not None:
        deduped = deduped[:limit]
    return deduped, selection


def clean_text(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def read_pdf_text(path: Path, char_limit: int = 2_000_000) -> str:
    doc = fitz.open(path)
    try:
        text = "".join(page.get_text() for page in doc)
    finally:
        doc.close()
    return clean_text(text)[:char_limit]


def extract_theorem_like_blocks(text: str, block_chars: int = 2500, max_blocks: int = 24) -> List[str]:
    patterns = list(re.finditer(r"\b(theorem|lemma|proposition|corollary|claim|remark)\b", text, flags=re.IGNORECASE))
    blocks = []
    for match in patterns[:max_blocks]:
        start = max(0, match.start() - 300)
        end = min(len(text), match.start() + block_chars)
        blocks.append(text[start:end])
    if not blocks and text:
        chunks = [text[i : i + block_chars] for i in range(0, min(len(text), block_chars * 8), block_chars)]
        blocks.extend(chunks[:max_blocks])
    return blocks


def normalize_filename_title(filename: str) -> str:
    stem = Path(filename).stem
    stem = re.sub(r"^rank_\d+_", "", stem)
    stem = stem.replace("_", " ")
    return re.sub(r"[^a-z0-9]+", " ", stem.lower()).strip()


def extract_title_from_citation(citation: str) -> str:
    if not citation:
        return ""
    blocks = re.split(r"\\newblock", citation)
    if len(blocks) >= 2:
        title_candidate = blocks[1]
        raw_lower = title_candidate.lower()
        cut_idx = len(title_candidate)
        for marker in [
            "volume",
            "vol.",
            "pages",
            "pp.",
            "doi",
            "journal",
            "in proceedings",
            "in collection",
            "arxiv",
            "isbn",
            "publisher",
        ]:
            idx = raw_lower.find(marker)
            if idx != -1 and idx < cut_idx:
                cut_idx = idx
        title_candidate = title_candidate[:cut_idx]
    else:
        parts = [part.strip() for part in citation.split(".") if part.strip()]
        title_candidate = parts[1] if len(parts) >= 2 else citation
    return normalize_filename_title(title_candidate)


def resolve_cited_source_filename(case_id: str, citation_content: str, download_root: Path) -> Optional[str]:
    folder = download_root / case_id
    pdf_names = sorted(path.name for path in folder.glob("*.pdf"))
    if not pdf_names:
        return None

    oracle_named = [
        name
        for name in pdf_names
        if name.startswith("oracle_cited_source") and (folder / name).is_file()
    ]
    if oracle_named:
        return sorted(oracle_named)[0]

    non_rank = [name for name in pdf_names if not re.match(r"^rank_\d+_", name)]
    if not non_rank:
        return None
    if len(non_rank) == 1:
        return non_rank[0]

    cited_title = extract_title_from_citation(citation_content)
    if cited_title:
        scored = sorted(
            (
                (
                    2.0 if cited_title == normalize_filename_title(name) else
                    1.0 if cited_title in normalize_filename_title(name) or normalize_filename_title(name) in cited_title else
                    0.0,
                    name,
                )
                for name in non_rank
            ),
            reverse=True,
        )
        if scored and scored[0][0] > 0.0:
            return scored[0][1]
    return None


def run_oracle_doc(
    llm: LLMJsonCaller,
    query_context: str,
    mode: str,
    content: str,
) -> Dict[str, Any]:
    if mode == "fulltext":
        source_section = content[:150_000]
        task_desc = "The cited source paper is guaranteed correct. Extract the exact theorem/lemma that solves the gap."
    else:
        blocks = extract_theorem_like_blocks(content)
        serialized = "\n\n".join(f"[BLOCK {idx + 1}]\n{block}" for idx, block in enumerate(blocks))
        source_section = serialized[:120_000]
        task_desc = "The cited source paper is guaranteed correct. Only theorem-like blocks are provided; select the relevant one and extract the exact theorem/lemma."

    prompt = f"""
You are a mathematical assistant running an oracle-source extraction experiment.

=== PROBLEM CONTEXT ===
{query_context}

=== ORACLE SOURCE CONTENT ===
{source_section}

=== TASK ===
{task_desc}

Return JSON ONLY:
{{
  "reasoning": "<short explanation>",
  "extracted_theorem": "<latex string or null>"
}}
"""
    result = llm.call_json(
        system_prompt="You are a math expert assistant.",
        user_prompt=prompt,
        temperature=0.0,
        max_tokens=4000,
    )
    if not isinstance(result, dict):
        return {"reasoning": "invalid_json", "extracted_theorem": None}
    return {
        "reasoning": str(result.get("reasoning", "") or ""),
        "extracted_theorem": result.get("extracted_theorem"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Oracle-source theorem extraction on cited source PDFs.")
    parser.add_argument("--model", choices=sorted(MODEL_CONFIGS.keys()), default=None, help="Optional legacy preset name.")
    parser.add_argument("--mode", choices=["fulltext", "blocks"], required=True)
    parser.add_argument("--limit", type=int, default=None, help="Optional cap for quick smoke tests.")
    parser.add_argument("--out", default=None, help="Optional output path.")
    parser.add_argument("--dataset-file", default=str(ROOT / DEFAULT_DATASET_FILE))
    parser.add_argument(
        "--eval-file",
        default="",
        help="Optional evaluation result JSON path used only to select a subset of case ids. If omitted, iterate over the dataset directly.",
    )
    parser.add_argument("--search-log", default="", help="Explicit search log JSON path.")
    parser.add_argument("--download-root", default="", help="Explicit candidate-PDF root path.")
    parser.add_argument("--track", choices=["raw", "assist"], default="raw")
    parser.add_argument("--context-variant", choices=["local_only", "global_local"], default="global_local")
    parser.add_argument(
        "--local-window",
        type=int,
        default=CANONICAL_LOCAL_WINDOW,
        help=(
            "Official local-context window m. "
            f"Recommended values: {SUPPORTED_LOCAL_WINDOWS}; default={CANONICAL_LOCAL_WINDOW}. "
            "Passing 0 also maps to the canonical window."
        ),
    )
    args = parser.parse_args()
    args.local_window = normalize_official_local_window(args.local_window)

    solver_cfg = build_llm_config_from_env("SOLVER", fallback=None)
    judge_cfg = build_llm_config_from_env("JUDGE", fallback=solver_cfg)
    solver_llm = LLMJsonCaller(solver_cfg)
    judge_llm = LLMJsonCaller(judge_cfg)

    dataset = load_dataset_as_dict(args.dataset_file)

    eval_file = Path(args.eval_file) if args.eval_file else None
    search_log = Path(args.search_log) if args.search_log else None

    if args.download_root:
        download_root = Path(args.download_root)
        model_label = args.model or "custom"
    elif args.model:
        cfg = MODEL_CONFIGS[args.model]
        eval_file = eval_file or cfg["eval_file"]
        search_log = search_log or cfg["search_log"]
        download_root = cfg["download_root"]
        model_label = args.model
    else:
        raise ValueError("Provide either --download-root or --model.")

    case_ids, case_selection = resolve_case_ids(dataset, eval_file, args.limit)
    results = []
    total = 0
    ground_hits = 0
    ground_evaluable = 0
    suff_hits = 0
    suff_evaluable = 0
    toolacc_hits = 0
    toolacc_evaluable = 0
    resolved = 0

    for case_id in case_ids:
        citation_content = get_citation_content(dataset[case_id])
        gt_filename = resolve_cited_source_filename(case_id, citation_content, download_root)
        if not gt_filename:
            results.append(
                {
                    "id": case_id,
                    "status": "missing_cited_source_pdf",
                    "gt_filename": None,
                    "Ground": None,
                    "Suff": None,
                    "ToolAcc": None,
                    "tool_success_proxy": False,
                    "is_matched": False,
                }
            )
            continue

        resolved += 1
        pdf_path = download_root / case_id / gt_filename
        content = read_pdf_text(pdf_path)
        query_context = build_query_context(
            dataset[case_id],
            track=args.track,
            context_variant=args.context_variant,
            local_window=(args.local_window or None),
        )
        solver_result = run_oracle_doc(solver_llm, query_context, args.mode, content)

        evaluator = SemanticEvaluator(
            folder_path=str(download_root / case_id),
            gt_data={
                "target_lemma_latex": get_reference_tool_latex(dataset[case_id]),
                "target_citation_content": get_citation_content(dataset[case_id]),
            },
            judge_llm=judge_llm,
            query_context=query_context,
        )
        ground_decision = evaluator.judge_document_grounding(gt_filename, solver_result.get("extracted_theorem"))
        suff_decision = evaluator.judge_tool_sufficiency(solver_result.get("extracted_theorem"))
        ground_value = ground_decision.get("is_supported")
        suff_value = suff_decision.get("is_sufficient")
        toolacc_value = None if ground_value is None or suff_value is None else bool(ground_value and suff_value)
        is_match = bool(suff_value is True)
        match_reason = str(suff_decision.get("reason", "") or "")
        tool_success_proxy = bool(toolacc_value is True)

        total += 1
        if ground_value is not None:
            ground_evaluable += 1
            ground_hits += int(bool(ground_value))
        if suff_value is not None:
            suff_evaluable += 1
            suff_hits += int(bool(suff_value))
        if toolacc_value is not None:
            toolacc_evaluable += 1
            toolacc_hits += int(bool(toolacc_value))
        results.append(
            {
                "id": case_id,
                "status": "ok",
                "oracle_mode": args.mode,
                "gt_filename": gt_filename,
                "Ground": ground_value,
                "ground_label": ground_decision.get("label"),
                "ground_reason": ground_decision.get("reason", ""),
                "ground_confidence": ground_decision.get("confidence"),
                "ground_evidence_span": ground_decision.get("evidence_span", ""),
                "Suff": suff_value,
                "suff_label": suff_decision.get("label"),
                "suff_reason": suff_decision.get("reason", ""),
                "suff_confidence": suff_decision.get("confidence"),
                "ToolAcc": toolacc_value,
                "tool_success_proxy": tool_success_proxy,
                "is_matched": is_match,
                "match_reason": match_reason,
                "solver_output": solver_result,
                "judge": {
                    "ground": ground_decision,
                    "suff": suff_decision,
                },
            }
        )

    requested_cases = len(case_ids)
    ground_abstained = requested_cases - ground_evaluable
    suff_abstained = requested_cases - suff_evaluable
    toolacc_abstained = requested_cases - toolacc_evaluable

    payload = {
        "model": model_label,
        "mode": args.mode,
        "primary_metric": "OracleToolAcc",
        "case_selection": case_selection,
        "track": args.track,
        "context_variant": args.context_variant,
        "local_window": args.local_window,
        "eval_file": str(eval_file) if eval_file else None,
        "search_log": str(search_log) if search_log else None,
        "download_root": str(download_root),
        "resolved_gt_filename": resolved,
        "requested_cases": requested_cases,
        "evaluated": total,
        "ground_hits": ground_hits,
        "ground_evaluable": ground_evaluable,
        "GroundRate": ground_hits / requested_cases if requested_cases else 0.0,
        "GroundAbstainRate": ground_abstained / requested_cases if requested_cases else 0.0,
        "ResolvedGroundRate": ground_hits / total if total else None,
        "suff_hits": suff_hits,
        "suff_evaluable": suff_evaluable,
        "SuffRate": suff_hits / requested_cases if requested_cases else 0.0,
        "SuffAbstainRate": suff_abstained / requested_cases if requested_cases else 0.0,
        "ResolvedSuffRate": suff_hits / total if total else None,
        "toolacc_hits": toolacc_hits,
        "toolacc_evaluable": toolacc_evaluable,
        "ToolAcc": toolacc_hits / requested_cases if requested_cases else 0.0,
        "ToolAccAbstainRate": toolacc_abstained / requested_cases if requested_cases else 0.0,
        "ResolvedToolAcc": toolacc_hits / total if total else None,
        "tool_success_proxy_hits": toolacc_hits,
        "tool_success_proxy": toolacc_hits / requested_cases if requested_cases else 0.0,
        "resolved_only_tool_success_proxy": toolacc_hits / total if total else None,
        "matched": toolacc_hits,
        "accuracy": toolacc_hits / requested_cases if requested_cases else 0.0,
        "results": results,
    }
    out_path = Path(args.out) if args.out else OUT_DIR / f"oracle_source_{model_label}_{args.mode}.json"
    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    print("Saved to", out_path)


if __name__ == "__main__":
    main()
