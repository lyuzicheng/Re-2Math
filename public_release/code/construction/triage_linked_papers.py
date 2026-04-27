from __future__ import annotations

import argparse
import datetime
import json
import os
import re
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from construction.extract_latex_text import ArxivLatexExtractor


DEFAULT_CONFIG = ROOT / "configs" / "published_paper_config.json"
DEFAULT_IN = ROOT / "construction" / "outputs" / "published_manifest_bootstrap.json"
DEFAULT_OUT = ROOT / "construction" / "outputs" / "published_manifest_triaged.json"


LATEX_CITE_PATTERN = re.compile(r"\\cite[a-zA-Z*]*\{([^}]+)\}")
TOOL_NAME_PATTERN = re.compile(
    r"\b(?:theorem|lemma|proposition|corollary|claim|criterion|inequality|estimate|bound)\b",
    re.IGNORECASE,
)
TOOL_CITATION_PATTERNS = [
    (
        "tool_then_cite",
        re.compile(
            r"\b(?:theorem|lemma|proposition|corollary|claim|criterion|inequality|estimate|bound)\b"
            r"[^\n]{0,120}?\\cite[a-zA-Z*]*\{[^}]+\}",
            re.IGNORECASE,
        ),
        2.5,
    ),
    (
        "cite_then_tool",
        re.compile(
            r"\\cite[a-zA-Z*]*\{[^}]+\}[^\n]{0,120}?"
            r"\b(?:theorem|lemma|proposition|corollary|claim|criterion|inequality|estimate|bound)\b",
            re.IGNORECASE,
        ),
        2.0,
    ),
    (
        "trigger_then_cite",
        re.compile(
            r"\b(?:by|using|apply(?:ing)?|via|from|invoke(?:d|s|ing)?)\b"
            r"[^\n]{0,120}?\\cite[a-zA-Z*]*\{[^}]+\}",
            re.IGNORECASE,
        ),
        2.0,
    ),
]
PROOF_HEADING_PATTERN = re.compile(
    r"\\begin\{proof\}|\\section\*?\{[^}]*proof[^}]*\}|\\subsection\*?\{[^}]*proof[^}]*\}",
    re.IGNORECASE,
)
THEOREM_ENV_PATTERN = re.compile(
    r"\\begin\{(?:theorem|thm|lemma|lem|proposition|prop|corollary|cor|claim)\}",
    re.IGNORECASE,
)
PROOF_ENV_PATTERN = re.compile(r"\\begin\{proof\}(.*?)\\end\{proof\}", re.IGNORECASE | re.DOTALL)
NEGATIVE_KEYWORDS = re.compile(
    r"\b(?:survey|lecture notes|thesis|editorial|erratum|correction|table of contents|numerical study|simulation)\b",
    re.IGNORECASE,
)


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def normalize_space(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip())


def normalize_manifest_row(row: Dict[str, Any]) -> Dict[str, Any]:
    normalized = dict(row)
    arxiv_id = normalize_space(
        str(
            normalized.get("arxiv_id")
            or normalized.get("openalex_arxiv_id")
            or ""
        )
    )
    normalized["arxiv_id"] = arxiv_id
    normalized["latex_link"] = normalize_space(
        str(normalized.get("latex_link") or (f"https://arxiv.org/e-print/{arxiv_id}" if arxiv_id else ""))
    )
    normalized["published_title"] = normalize_space(
        str(normalized.get("published_title") or normalized.get("title") or "")
    )
    normalized["venue"] = normalize_space(str(normalized.get("venue") or ""))
    normalized["domain"] = normalize_space(str(normalized.get("domain") or ""))
    normalized["abstract"] = normalize_space(str(normalized.get("abstract") or normalized.get("summary") or ""))
    normalized["notes"] = normalize_space(str(normalized.get("notes") or ""))
    normalized["proof_rich_score"] = float(normalized.get("proof_rich_score", 0.0) or 0.0)
    return normalized


def metadata_filter(rows: List[Dict[str, Any]], config: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    triage_cfg = (config.get("triage_defaults", {}) or {})
    require_arxiv = bool(triage_cfg.get("require_openalex_arxiv_link", True))
    min_score = float(triage_cfg.get("metadata_min_proof_rich_score", 0.0) or 0.0)
    source_fetch_per_domain = int(triage_cfg.get("source_fetch_per_domain", 0) or 0)
    max_per_venue = int(config.get("selection_defaults", {}).get("max_candidates_per_venue", 0) or 0)

    filtered: List[Dict[str, Any]] = []
    rejected_reasons = Counter()
    by_domain: Dict[str, List[Dict[str, Any]]] = defaultdict(list)

    for raw in rows:
        row = normalize_manifest_row(raw)
        if require_arxiv and not row.get("arxiv_id"):
            rejected_reasons["missing_arxiv_link"] += 1
            continue
        if row.get("proof_rich_score", 0.0) < min_score:
            rejected_reasons["low_metadata_score"] += 1
            continue
        haystack = " ".join([row.get("published_title", ""), row.get("abstract", ""), row.get("notes", "")])
        if NEGATIVE_KEYWORDS.search(haystack):
            rejected_reasons["negative_keyword"] += 1
            continue
        by_domain[row.get("domain", "")].append(row)

    venue_usage: Counter[Tuple[str, str]] = Counter()
    for domain, domain_rows in by_domain.items():
        ranked = sorted(
            domain_rows,
            key=lambda item: (
                float(item.get("proof_rich_score", 0.0) or 0.0),
                1 if item.get("arxiv_id") else 0,
                str(item.get("published_title", "") or ""),
            ),
            reverse=True,
        )
        kept = 0
        for row in ranked:
            venue_key = (domain, str(row.get("venue", "") or ""))
            if max_per_venue and row.get("venue") and venue_usage[venue_key] >= max_per_venue:
                rejected_reasons["venue_cap"] += 1
                continue
            if source_fetch_per_domain and kept >= source_fetch_per_domain:
                rejected_reasons["domain_fetch_cap"] += 1
                continue
            venue_usage[venue_key] += 1
            kept += 1
            filtered.append(row)

    return filtered, {"rejected_reasons": dict(rejected_reasons)}


class CandidateTriageExtractor(ArxivLatexExtractor):
    def process_paper_with_bib(self, paper_id: str, latex_link: str) -> Tuple[bool, str, Dict[str, str]]:
        try:
            temp_root = os.path.join(tempfile.gettempdir(), "benchmark_builder_tmp")
            os.makedirs(temp_root, exist_ok=True)
            with tempfile.TemporaryDirectory(dir=temp_root) as temp_work_dir:
                download_dir = os.path.join(temp_work_dir, "download")
                extract_dir = os.path.join(temp_work_dir, "extract")
                os.makedirs(download_dir, exist_ok=True)
                os.makedirs(extract_dir, exist_ok=True)

                archive_path = self.download_latex_source(latex_link, download_dir)
                if not archive_path:
                    return False, "", {}
                if not self.extract_archive(archive_path, extract_dir):
                    return False, "", {}

                bib_mapping = self.extract_bib_mapping(extract_dir)
                ordered_tex_files = self.determine_tex_file_order(extract_dir)
                if not ordered_tex_files:
                    ordered_tex_files = self.find_tex_files(extract_dir)
                if not ordered_tex_files:
                    return False, "", {}

                parts: List[str] = []
                for tex_file in ordered_tex_files:
                    text = self.extract_text_from_tex(tex_file)
                    parts.append(f"\n% --- File: {os.path.basename(tex_file)} ---\n{text}\n")
                return True, "\n".join(parts), bib_mapping
        except Exception:
            return False, "", {}


def count_unique_cite_keys(text: str) -> int:
    keys = set()
    for raw_keys in LATEX_CITE_PATTERN.findall(text or ""):
        for item in raw_keys.split(","):
            clean = normalize_space(item).lower()
            if clean:
                keys.add(clean)
    return len(keys)


def source_triage_metrics(full_text: str, bib_mapping: Dict[str, str]) -> Dict[str, Any]:
    text = str(full_text or "")
    lowered = text.lower()

    pattern_hits: Dict[str, int] = {}
    tool_citation_score = 0.0
    examples: List[str] = []
    for label, pattern, weight in TOOL_CITATION_PATTERNS:
        matches = pattern.findall(text)
        if not matches:
            continue
        pattern_hits[label] = len(matches)
        tool_citation_score += min(3, len(matches)) * weight
        if len(examples) < 3:
            for match in matches[: 3 - len(examples)]:
                examples.append(normalize_space(str(match))[:180])

    proof_blocks = PROOF_ENV_PATTERN.findall(text)
    proof_cite_hits = sum(len(LATEX_CITE_PATTERN.findall(block)) for block in proof_blocks)
    proof_tool_hits = 0
    for block in proof_blocks:
        for _, pattern, _ in TOOL_CITATION_PATTERNS:
            proof_tool_hits += len(pattern.findall(block))

    proof_heading_hits = len(PROOF_HEADING_PATTERN.findall(text))
    proof_mentions = len(re.findall(r"\bproof\b", lowered))
    theorem_like_env_count = len(THEOREM_ENV_PATTERN.findall(text))
    unique_cite_count = count_unique_cite_keys(text)
    bib_entries = len(bib_mapping or {})

    score = 0.0
    if bib_entries >= 12:
        score += 1.0
    if bib_entries >= 24:
        score += 1.0
    if theorem_like_env_count >= 4:
        score += 1.0
    if theorem_like_env_count >= 10:
        score += 1.0
    if proof_heading_hits >= 1:
        score += 1.0
    if proof_heading_hits >= 3:
        score += 1.0
    if unique_cite_count >= 8:
        score += 1.0
    if unique_cite_count >= 15:
        score += 1.0
    if proof_cite_hits >= 2:
        score += 1.0
    if proof_tool_hits >= 1:
        score += 2.0
    if proof_tool_hits >= 3:
        score += 1.0
    score += min(4.0, tool_citation_score)
    if proof_heading_hits == 0:
        score -= 2.0
    if theorem_like_env_count == 0:
        score -= 1.0

    return {
        "source_triage_score": round(score, 2),
        "full_text_chars": len(text),
        "bib_entries": bib_entries,
        "unique_cite_count": unique_cite_count,
        "proof_heading_hits": proof_heading_hits,
        "proof_mentions": proof_mentions,
        "theorem_like_env_count": theorem_like_env_count,
        "proof_block_count": len(proof_blocks),
        "proof_cite_hits": proof_cite_hits,
        "proof_tool_hits": proof_tool_hits,
        "tool_citation_score": round(tool_citation_score, 2),
        "pattern_hits": pattern_hits,
        "examples": examples,
    }


def source_accept(metrics: Dict[str, Any], config: Dict[str, Any]) -> Tuple[bool, List[str]]:
    triage_cfg = (config.get("triage_defaults", {}) or {})
    reasons: List[str] = []
    if int(metrics.get("bib_entries", 0) or 0) < int(triage_cfg.get("min_bibliography_entries", 0) or 0):
        reasons.append("low_bibliography")
    if int(metrics.get("proof_heading_hits", 0) or 0) < int(triage_cfg.get("min_proof_headings", 0) or 0):
        reasons.append("missing_proof_headings")
    if int(metrics.get("theorem_like_env_count", 0) or 0) < int(triage_cfg.get("min_theorem_like_envs", 0) or 0):
        reasons.append("few_theorem_envs")
    if int(metrics.get("unique_cite_count", 0) or 0) < int(triage_cfg.get("min_unique_cite_keys", 0) or 0):
        reasons.append("few_citations")
    if int(metrics.get("proof_cite_hits", 0) or 0) < int(triage_cfg.get("min_proof_cite_keys", 0) or 0):
        reasons.append("few_proof_citations")
    if float(metrics.get("tool_citation_score", 0.0) or 0.0) < float(triage_cfg.get("min_tool_citation_score", 0.0) or 0.0):
        reasons.append("weak_tool_citation_patterns")
    if float(metrics.get("source_triage_score", 0.0) or 0.0) < float(triage_cfg.get("min_source_triage_score", 0.0) or 0.0):
        reasons.append("low_source_triage_score")
    return not reasons, reasons


def rank_for_shortlist(row: Dict[str, Any]) -> Tuple[float, float, int, str]:
    metrics = row.get("source_triage", {}) or {}
    return (
        float(metrics.get("source_triage_score", 0.0) or 0.0),
        float(row.get("proof_rich_score", 0.0) or 0.0),
        int(metrics.get("proof_tool_hits", 0) or 0),
        str(row.get("published_title", "") or ""),
    )


def shortlist_rows(rows: List[Dict[str, Any]], config: Dict[str, Any]) -> List[Dict[str, Any]]:
    triage_cfg = (config.get("triage_defaults", {}) or {})
    shortlist_per_domain = int(triage_cfg.get("shortlist_per_domain", 0) or 0)
    max_shortlist_per_venue = int(triage_cfg.get("max_shortlist_per_venue", 0) or 0)
    venue_usage: Counter[Tuple[str, str]] = Counter()
    accepted: List[Dict[str, Any]] = []
    by_domain: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_domain[str(row.get("domain", "") or "")].append(row)

    for domain, domain_rows in by_domain.items():
        ranked = sorted(domain_rows, key=rank_for_shortlist, reverse=True)
        kept = 0
        for row in ranked:
            venue_key = (domain, str(row.get("venue", "") or ""))
            if max_shortlist_per_venue and row.get("venue") and venue_usage[venue_key] >= max_shortlist_per_venue:
                continue
            if shortlist_per_domain and kept >= shortlist_per_domain:
                continue
            accepted.append(row)
            kept += 1
            venue_usage[venue_key] += 1
    return accepted


def build_output_payload(
    *,
    input_manifest: str,
    metadata_rows: List[Dict[str, Any]],
    evaluated: List[Dict[str, Any]],
    shortlisted: List[Dict[str, Any]],
    metadata_only: bool,
    meta_diag: Dict[str, Any],
    source_rejections: Counter,
    processed_source_candidates: int | None = None,
    checkpoint: bool = False,
) -> Dict[str, Any]:
    domain_counts = Counter(str(row.get("domain", "") or "") for row in shortlisted)
    venue_counts = Counter((str(row.get("domain", "") or ""), str(row.get("venue", "") or "")) for row in shortlisted)
    meta: Dict[str, Any] = {
        "input_manifest": input_manifest,
        "metadata_candidate_count": len(metadata_rows),
        "evaluated_candidate_count": len(evaluated),
        "shortlist_count": len(shortlisted),
        "metadata_only": bool(metadata_only),
        "metadata_rejections": meta_diag.get("rejected_reasons", {}),
        "source_rejections": dict(source_rejections),
        "domain_counts": dict(domain_counts),
        "venue_counts": {
            domain: {venue: count for (d, venue), count in venue_counts.items() if d == domain}
            for domain in sorted(domain_counts)
        },
    }
    if processed_source_candidates is not None:
        meta.update(
            {
                "checkpoint": bool(checkpoint),
                "source_candidates_processed": int(processed_source_candidates),
                "source_candidates_remaining": max(0, len(metadata_rows) - int(processed_source_candidates)),
                "updated_at": datetime.datetime.now().isoformat(timespec="seconds"),
            }
        )
    return {"meta": meta, "papers": shortlisted}


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def write_progress_json(
    path: Path,
    *,
    input_manifest: str,
    metadata_rows: List[Dict[str, Any]],
    evaluated: List[Dict[str, Any]],
    source_rejections: Counter,
    processed_source_candidates: int,
    checkpoint_path: Path | None,
    checkpoint_shortlist_count: int | None = None,
) -> None:
    payload: Dict[str, Any] = {
        "input_manifest": input_manifest,
        "metadata_candidate_count": len(metadata_rows),
        "source_candidates_processed": int(processed_source_candidates),
        "source_candidates_remaining": max(0, len(metadata_rows) - int(processed_source_candidates)),
        "evaluated_candidate_count": len(evaluated),
        "source_rejections": dict(source_rejections),
        "updated_at": datetime.datetime.now().isoformat(timespec="seconds"),
    }
    if checkpoint_path is not None:
        payload["checkpoint_manifest"] = str(checkpoint_path)
    if checkpoint_shortlist_count is not None:
        payload["checkpoint_shortlist_count"] = int(checkpoint_shortlist_count)
    write_json(path, payload)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a high-quality shortlist of arXiv-linked published papers using cheap source triage.")
    parser.add_argument("--input", default=str(DEFAULT_IN))
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument("--metadata-only", action="store_true")
    parser.add_argument("--checkpoint-out", default="")
    parser.add_argument("--progress-out", default="")
    parser.add_argument("--checkpoint-every", type=int, default=25)
    args = parser.parse_args()

    payload = load_json(Path(args.input))
    rows = list(payload.get("papers", payload if isinstance(payload, list) else []) or [])
    config = load_json(Path(args.config))
    out_path = Path(args.out)
    checkpoint_path = Path(args.checkpoint_out) if str(args.checkpoint_out or "").strip() else None
    progress_path = Path(args.progress_out) if str(args.progress_out or "").strip() else None
    checkpoint_every = max(0, int(args.checkpoint_every or 0))

    metadata_rows, meta_diag = metadata_filter(rows, config)
    extractor = CandidateTriageExtractor()

    evaluated: List[Dict[str, Any]] = []
    source_rejections = Counter()
    processed_source_candidates = 0

    if progress_path is not None:
        write_progress_json(
            progress_path,
            input_manifest=Path(args.input).name,
            metadata_rows=metadata_rows,
            evaluated=evaluated,
            source_rejections=source_rejections,
            processed_source_candidates=processed_source_candidates,
            checkpoint_path=checkpoint_path,
        )

    if args.metadata_only:
        for row in metadata_rows:
            normalized = dict(row)
            normalized["selection_status"] = "metadata_pass"
            evaluated.append(normalized)
    else:
        for idx, row in enumerate(metadata_rows, start=1):
            normalized = dict(row)
            success, full_text, bib_mapping = extractor.process_paper_with_bib(
                str(normalized.get("arxiv_id") or ""),
                str(normalized.get("latex_link") or ""),
            )
            if not success or len(full_text) < 1000:
                source_rejections["source_extract_failed"] += 1
                continue
            metrics = source_triage_metrics(full_text, bib_mapping)
            accepted, reasons = source_accept(metrics, config)
            normalized["source_triage"] = metrics
            normalized["selection_status"] = "source_pass" if accepted else "source_reject"
            normalized["selection_reasons"] = reasons
            if accepted:
                evaluated.append(normalized)
            else:
                for reason in reasons:
                    source_rejections[reason] += 1
            processed_source_candidates = idx
            if progress_path is not None:
                write_progress_json(
                    progress_path,
                    input_manifest=Path(args.input).name,
                    metadata_rows=metadata_rows,
                    evaluated=evaluated,
                    source_rejections=source_rejections,
                    processed_source_candidates=processed_source_candidates,
                    checkpoint_path=checkpoint_path,
                )
            if checkpoint_path is not None and checkpoint_every and processed_source_candidates % checkpoint_every == 0:
                checkpoint_shortlisted = shortlist_rows(evaluated, config)
                checkpoint_payload = build_output_payload(
                    input_manifest=Path(args.input).name,
                    metadata_rows=metadata_rows,
                    evaluated=evaluated,
                    shortlisted=checkpoint_shortlisted,
                    metadata_only=bool(args.metadata_only),
                    meta_diag=meta_diag,
                    source_rejections=source_rejections,
                    processed_source_candidates=processed_source_candidates,
                    checkpoint=True,
                )
                write_json(checkpoint_path, checkpoint_payload)
                print(
                    f"Checkpoint saved to {checkpoint_path} "
                    f"(processed={processed_source_candidates}/{len(metadata_rows)} shortlist={len(checkpoint_shortlisted)})"
                )
                if progress_path is not None:
                    write_progress_json(
                        progress_path,
                        input_manifest=Path(args.input).name,
                        metadata_rows=metadata_rows,
                        evaluated=evaluated,
                        source_rejections=source_rejections,
                        processed_source_candidates=processed_source_candidates,
                        checkpoint_path=checkpoint_path,
                        checkpoint_shortlist_count=len(checkpoint_shortlisted),
                    )

    shortlisted = shortlist_rows(evaluated, config)
    processed_final = processed_source_candidates if not args.metadata_only else len(metadata_rows)
    output = build_output_payload(
        input_manifest=Path(args.input).name,
        metadata_rows=metadata_rows,
        evaluated=evaluated,
        shortlisted=shortlisted,
        metadata_only=bool(args.metadata_only),
        meta_diag=meta_diag,
        source_rejections=source_rejections,
        processed_source_candidates=processed_final,
        checkpoint=False,
    )
    write_json(out_path, output)
    if checkpoint_path is not None:
        checkpoint_payload = build_output_payload(
            input_manifest=Path(args.input).name,
            metadata_rows=metadata_rows,
            evaluated=evaluated,
            shortlisted=shortlisted,
            metadata_only=bool(args.metadata_only),
            meta_diag=meta_diag,
            source_rejections=source_rejections,
            processed_source_candidates=processed_final,
            checkpoint=False,
        )
        write_json(checkpoint_path, checkpoint_payload)
    if progress_path is not None:
        write_progress_json(
            progress_path,
            input_manifest=Path(args.input).name,
            metadata_rows=metadata_rows,
            evaluated=evaluated,
            source_rejections=source_rejections,
            processed_source_candidates=processed_final,
            checkpoint_path=checkpoint_path,
            checkpoint_shortlist_count=len(shortlisted),
        )
    print(json.dumps(output["meta"], indent=2, ensure_ascii=False))
    print(f"Saved shortlist to {out_path}")


if __name__ == "__main__":
    main()
