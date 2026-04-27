from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple


DEFAULT_DATASET_FILE = "construction/outputs/benchmark_dataset_current.jsonl"
CANONICAL_LOCAL_WINDOW = 5
SUPPORTED_LOCAL_WINDOWS = (1, 3, 5)
MAX_LOCAL_CONTEXT_BLOCKS = max(SUPPORTED_LOCAL_WINDOWS)


def load_jsonl(path: str | Path) -> List[dict]:
    rows: List[dict] = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def normalize_official_local_window(local_window: int | None) -> int:
    if local_window is None or local_window <= 0:
        return CANONICAL_LOCAL_WINDOW
    if local_window not in SUPPORTED_LOCAL_WINDOWS:
        supported = ",".join(str(value) for value in SUPPORTED_LOCAL_WINDOWS)
        raise ValueError(
            f"Unsupported local window m={local_window}. "
            f"Official benchmark windows are {{{supported}}} and canonical m={CANONICAL_LOCAL_WINDOW}."
        )
    return local_window


def get_instance_id(row: dict) -> str:
    return str(row.get("instance_id") or row.get("id") or "")


def load_dataset_as_dict(path: str | Path) -> Dict[str, dict]:
    dataset: Dict[str, dict] = {}
    for row in load_jsonl(path):
        instance_id = get_instance_id(row)
        if instance_id:
            dataset[instance_id] = row
    return dataset


def get_paper(row: dict) -> dict:
    if "paper" in row and isinstance(row["paper"], dict):
        return row["paper"]
    return {
        "paper_id": row.get("arxiv_id", ""),
        "title": row.get("title", ""),
        "publication_date": row.get("publication_date", ""),
    }


def get_x(row: dict) -> dict:
    if "x" in row and isinstance(row["x"], dict):
        return row["x"]
    return row.get("problem_input", {}) or {}


def get_y(row: dict) -> dict:
    if "y" in row and isinstance(row["y"], dict):
        return row["y"]
    gt = row.get("ground_truth", {}) or {}
    return {
        "reference_tool_latex": gt.get("target_lemma_latex", ""),
        "reference_tool_type": gt.get("target_lemma_type", ""),
        "restated_in_citing_paper": gt.get("restated_in_citing_paper", False),
    }


def get_z(row: dict) -> dict:
    if "z" in row and isinstance(row["z"], dict):
        return row["z"]
    gt = row.get("ground_truth", {}) or {}
    return {
        "citation_key": gt.get("target_citation_key", ""),
        "citation_content": gt.get("target_citation_content", ""),
        "source_type": "",
        "locator": "",
        "doi": "",
        "arxiv_id": "",
    }


def get_strata(row: dict) -> dict:
    if "strata" in row and isinstance(row["strata"], dict):
        return row["strata"]
    extension_meta = row.get("extension_meta", {}) or {}
    return {
        "domain": extension_meta.get("field_label", ""),
        "tool_family": extension_meta.get("tool_family", ""),
    }


def get_global_context(row: dict) -> dict:
    x = get_x(row)
    global_context = x.get("global_context", {}) or {}
    if "target_theorem" in global_context:
        return global_context
    return {
        "setup": global_context.get("setup", ""),
        "target_theorem": global_context.get("target_theorem", "") or global_context.get("goal", ""),
    }


def get_local_context_blocks(row: dict) -> List[str]:
    x = get_x(row)
    local_context = x.get("local_context", [])
    if isinstance(local_context, list):
        blocks = [str(block or "").strip() for block in local_context if str(block or "").strip()]
        return blocks[-MAX_LOCAL_CONTEXT_BLOCKS:]

    if isinstance(local_context, dict):
        blocks: List[str] = []
        anchor = str(local_context.get("anchor_latex", "") or "").strip()
        objective = str(local_context.get("gap_objective", "") or "").strip()
        if anchor:
            blocks.append(anchor)
        if objective:
            blocks.append(objective)
        return blocks

    return []


def slice_local_context_blocks(row: dict, local_window: int | None = None) -> List[str]:
    blocks = get_local_context_blocks(row)
    if local_window is None or local_window <= 0:
        local_window = CANONICAL_LOCAL_WINDOW
    return blocks[-local_window:]


def get_anchor_hint(row: dict) -> str:
    x = get_x(row)
    return str(x.get("anchor_hint", "") or "").strip()


def get_reference_tool_latex(row: dict) -> str:
    return str(get_y(row).get("reference_tool_latex", "") or "").strip()


def get_reference_tool_type(row: dict) -> str:
    return str(get_y(row).get("reference_tool_type", "") or "").strip()


def get_restated_in_citing_paper(row: dict) -> bool:
    return bool(get_y(row).get("restated_in_citing_paper", False))


def get_citation_key(row: dict) -> str:
    return str(get_z(row).get("citation_key", "") or "").strip()


def get_citation_content(row: dict) -> str:
    return str(get_z(row).get("citation_content", "") or "").strip()


def get_source_type(row: dict) -> str:
    return str(get_z(row).get("source_type", "") or "").strip()


def get_citation_locator(row: dict) -> str:
    z = get_z(row)
    return str(z.get("locator", "") or z.get("locator_snippet", "") or "").strip()


def get_citation_locator_snippet(row: dict) -> str:
    z = get_z(row)
    return str(z.get("locator_snippet", "") or z.get("locator", "") or "").strip()


def get_citation_doi(row: dict) -> str:
    return str(get_z(row).get("doi", "") or "").strip()


def get_cited_arxiv_id(row: dict) -> str:
    return str(get_z(row).get("arxiv_id", "") or "").strip()


def get_domain(row: dict) -> str:
    return str(get_strata(row).get("domain", "") or "").strip()


def get_tool_family(row: dict) -> str:
    return str(get_strata(row).get("tool_family", "") or "").strip()


def format_local_context(blocks: Iterable[str]) -> str:
    lines = []
    for idx, block in enumerate(blocks, start=1):
        clean = str(block or "").strip()
        if clean:
            lines.append(f"{idx}. {clean}")
    return "\n".join(lines)


def build_context_parts(
    row: dict,
    *,
    track: str = "assist",
    context_variant: str = "global_local",
    local_window: int | None = CANONICAL_LOCAL_WINDOW,
) -> Tuple[str, List[str], str]:
    gc = get_global_context(row)
    local_blocks = slice_local_context_blocks(row, local_window=local_window)
    anchor_hint = get_anchor_hint(row)
    setup = str(gc.get("setup", "") or "")
    target_theorem = str(gc.get("target_theorem", "") or "")

    if context_variant == "local_only":
        setup = ""
        target_theorem = ""
    elif context_variant != "global_local":
        raise ValueError(f"Unsupported context_variant: {context_variant}")

    if track != "assist":
        anchor_hint = ""

    global_text = "\n".join(
        part
        for part in [setup, f"Target Theorem: {target_theorem}" if target_theorem else ""]
        if part
    )
    return global_text, local_blocks, anchor_hint


def build_query_context(
    row: dict,
    track: str = "assist",
    *,
    context_variant: str = "global_local",
    local_window: int | None = CANONICAL_LOCAL_WINDOW,
) -> str:
    global_text, local_blocks, anchor_hint = build_context_parts(
        row,
        track=track,
        context_variant=context_variant,
        local_window=local_window,
    )

    parts = [
        "[Global Context]",
        global_text or "(empty)",
        "",
        "[Local Context]",
        format_local_context(local_blocks) or "(empty)",
    ]

    if track == "assist":
        parts.extend(["", "[Anchor Hint]", anchor_hint or "(empty)"])

    return "\n".join(parts)


def build_generation_inputs(
    row: dict,
    *,
    track: str = "assist",
    context_variant: str = "global_local",
    local_window: int | None = CANONICAL_LOCAL_WINDOW,
) -> Tuple[str, str, str]:
    global_text, local_blocks, anchor_hint = build_context_parts(
        row,
        track=track,
        context_variant=context_variant,
        local_window=local_window,
    )
    gap = format_local_context(local_blocks) or "(empty)"
    return global_text, gap, anchor_hint
