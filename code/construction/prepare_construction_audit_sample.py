from __future__ import annotations

import argparse
import csv
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List

import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from common.dataset_format import (  # noqa: E402
    get_anchor_hint,
    get_citation_content,
    get_citation_doi,
    get_citation_key,
    get_citation_locator_snippet,
    get_cited_arxiv_id,
    get_domain,
    get_global_context,
    get_instance_id,
    get_local_context_blocks,
    get_paper,
    get_reference_tool_latex,
    get_restated_in_citing_paper,
    get_source_type,
    get_tool_family,
    load_jsonl,
)


OUT_DIR = ROOT / "construction" / "outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def dump_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def stratified_by_domain(rows: List[dict], per_domain: int, seed: int) -> List[dict]:
    grouped: Dict[str, List[dict]] = defaultdict(list)
    for row in rows:
        grouped[get_domain(row)].append(row)

    rng = random.Random(seed)
    sampled: List[dict] = []
    for domain, bucket in sorted(grouped.items()):
        shuffled = list(bucket)
        rng.shuffle(shuffled)
        sampled.extend(shuffled[:per_domain])
    rng.shuffle(sampled)
    return sampled


def build_packet(row: dict) -> dict:
    paper = get_paper(row)
    global_context = get_global_context(row)
    return {
        "id": get_instance_id(row),
        "paper_id": paper.get("paper_id", ""),
        "paper_title": paper.get("title", ""),
        "domain": get_domain(row),
        "tool_family": get_tool_family(row),
        "global_setup": global_context.get("setup", ""),
        "target_theorem": global_context.get("target_theorem", ""),
        "local_context": "\n".join(get_local_context_blocks(row)),
        "anchor_hint": get_anchor_hint(row),
        "reference_tool_latex": get_reference_tool_latex(row),
        "restated_in_citing_paper": get_restated_in_citing_paper(row),
        "citation_key": get_citation_key(row),
        "citation_content": get_citation_content(row),
        "citation_locator_snippet": get_citation_locator_snippet(row),
        "source_type": get_source_type(row),
        "cited_doi": get_citation_doi(row),
        "cited_arxiv_id": get_cited_arxiv_id(row),
    }


def write_outputs(packets: List[dict], out_prefix: Path, *, dataset_file: str, per_domain: int, seed: int) -> Dict[str, str]:
    jsonl_path = out_prefix.with_suffix(".jsonl")
    csv_path = out_prefix.with_suffix(".csv")
    meta_path = out_prefix.with_suffix(".meta.json")
    instructions_path = out_prefix.with_suffix(".instructions.txt")

    with jsonl_path.open("w", encoding="utf-8") as f:
        for packet in packets:
            f.write(json.dumps(packet, ensure_ascii=False) + "\n")

    fieldnames = [
        "id",
        "paper_id",
        "paper_title",
        "domain",
        "tool_family",
        "global_setup",
        "target_theorem",
        "local_context",
        "anchor_hint",
        "reference_tool_latex",
        "restated_in_citing_paper",
        "citation_key",
        "citation_content",
        "citation_locator_snippet",
        "source_type",
        "cited_doi",
        "cited_arxiv_id",
        "human_instrumental_citation",
        "human_local_context_aligned",
        "human_reference_witness_sufficient",
        "human_anchor_leakage_free",
        "human_citation_metadata_correct",
        "human_external_source_requirement",
        "human_notes",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for packet in packets:
            row = dict(packet)
            row.update(
                {
                    "human_instrumental_citation": "",
                    "human_local_context_aligned": "",
                    "human_reference_witness_sufficient": "",
                    "human_anchor_leakage_free": "",
                    "human_citation_metadata_correct": "",
                    "human_external_source_requirement": "",
                    "human_notes": "",
                }
            )
            writer.writerow(row)

    instructions = """Construction audit annotation guide

Allowed labels:
- yes
- no
- uncertain

Columns to fill:
- human_instrumental_citation
- human_local_context_aligned
- human_reference_witness_sufficient
- human_anchor_leakage_free
- human_citation_metadata_correct
- human_external_source_requirement
- human_notes

Definitions:
- instrumental_citation: the cited source is genuinely needed for the proof transition rather than serving as background/history/definition only.
- local_context_aligned: the provided local context is the proof state immediately before the citation-triggered gap.
- reference_witness_sufficient: the stored reference tool is sufficient to close the gap under the supplied proof context.
- anchor_leakage_free: the anchor does not reveal theorem titles, author names, citation keys, DOI/arXiv identifiers, or near-verbatim bibliographic metadata.
- citation_metadata_correct: citation text/DOI/arXiv/source type match the cited source.
- external_source_requirement: the gap requires an external source rather than a result already established inside the citing paper.
"""
    instructions_path.write_text(instructions, encoding="utf-8")

    meta_payload = {
        "dataset_file": dataset_file,
        "sample_size": len(packets),
        "per_domain": per_domain,
        "seed": seed,
        "domain_counts": dict(Counter(packet["domain"] for packet in packets)),
        "outputs": {
            "jsonl": str(jsonl_path),
            "csv": str(csv_path),
            "instructions": str(instructions_path),
        },
    }
    dump_json(meta_path, meta_payload)
    return {
        "jsonl": str(jsonl_path),
        "csv": str(csv_path),
        "meta": str(meta_path),
        "instructions": str(instructions_path),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare a stratified construction-audit sample packet.")
    parser.add_argument(
        "--dataset",
        default=str(ROOT / "construction" / "outputs" / "benchmark_dataset_eval200_balanced_20260422.jsonl"),
    )
    parser.add_argument("--per-domain", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out-prefix", default="")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = load_jsonl(args.dataset)
    sampled = stratified_by_domain(rows, per_domain=args.per_domain, seed=args.seed)
    packets = [build_packet(row) for row in sampled]
    out_prefix = (
        Path(args.out_prefix)
        if args.out_prefix
        else OUT_DIR / f"construction_audit_eval{len(packets)}_{Path(args.dataset).stem}"
    )
    outputs = write_outputs(
        packets,
        out_prefix,
        dataset_file=args.dataset,
        per_domain=args.per_domain,
        seed=args.seed,
    )
    print(json.dumps({"sampled": len(packets), "outputs": outputs}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
