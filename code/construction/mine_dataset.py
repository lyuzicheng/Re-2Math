import os
import re
import json
import logging
import datetime
import tempfile
import sys
import time
import hashlib
from collections import Counter
from difflib import SequenceMatcher
from typing import Any, List, Dict, Optional, Tuple
from pathlib import Path

import requests
import httpx
from openai import OpenAI
try:
    from json_repair import repair_json
except ImportError:
    repair_json = None

# Local modules (expected to be in the same repo)
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

DEFAULT_ID_LIST_FILE = ROOT / "configs" / "arxiv_ids.txt"

from construction.arxiv_retriever import ArxivMathPaperRetriever
from construction.extract_latex_text import ArxivLatexExtractor


# ============================================================
# Configuration (Open-source friendly: no hardcoded secrets)
# ============================================================

CONFIG: Dict[str, object] = {
    # REQUIRED: set via environment variable
    "API_KEY": os.getenv("OPENAI_API_KEY", ""),

    # Optional: can be overridden via environment variables
    "BASE_URL": os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"),
    "MODEL_NAME": os.getenv("OPENAI_MODEL", "gpt-4o-mini"),

    # Dataset and retrieval settings
    "ID_LIST_FILE": os.getenv("ARXIV_ID_LIST_FILE", str(DEFAULT_ID_LIST_FILE)),
    "PAPER_MANIFEST_FILE": os.getenv("PAPER_MANIFEST_FILE", ""),
    "CATEGORY": os.getenv("ARXIV_CATEGORY", "math.PR"),
    "MAX_PAPERS": int(os.getenv("MAX_PAPERS", "50")),
    "TIME_WINDOW_DAYS": int(os.getenv("TIME_WINDOW_DAYS", "180")),
    "OUTPUT_FILE": os.getenv("OUTPUT_FILE", str(ROOT / "construction" / "outputs" / "benchmark_dataset_v2.jsonl")),
    "PROGRESS_FILE": os.getenv("PROGRESS_FILE", ""),
    "PAPER_SHARD_COUNT": max(1, int(os.getenv("PAPER_SHARD_COUNT", "1"))),
    "PAPER_SHARD_INDEX": int(os.getenv("PAPER_SHARD_INDEX", "0")),
    "PAPER_SORT_BY_PROOF_RICH": os.getenv("PAPER_SORT_BY_PROOF_RICH", "1").lower() not in {"0", "false", "no"},
    "THROUGHPUT_MODE": os.getenv("THROUGHPUT_MODE", "0").lower() not in {"0", "false", "no"},
    "STARTUP_API_SANITY_CHECK_ENABLED": os.getenv("STARTUP_API_SANITY_CHECK_ENABLED", "1").lower() not in {"0", "false", "no"},
    "TARGET_PROOF_RECOVERY_MAX_ANCHORS": int(os.getenv("TARGET_PROOF_RECOVERY_MAX_ANCHORS", "12")),
    "LLM_STRUCTURED_REPAIR_ENABLED": os.getenv("LLM_STRUCTURED_REPAIR_ENABLED", "1").lower() not in {"0", "false", "no"},
    "STAGE1_STRUCTURED_REPAIR_ENABLED": os.getenv("STAGE1_STRUCTURED_REPAIR_ENABLED", "1").lower() not in {"0", "false", "no"},
    "STAGE2_STRUCTURED_REPAIR_ENABLED": os.getenv("STAGE2_STRUCTURED_REPAIR_ENABLED", "1").lower() not in {"0", "false", "no"},
    "STAGE1_MAX_TOKENS": int(os.getenv("STAGE1_MAX_TOKENS", "3000")),
    "STAGE2_BATCH_MAX_TOKENS": int(os.getenv("STAGE2_BATCH_MAX_TOKENS", "4500")),
    "STAGE2_SINGLE_MAX_TOKENS": int(os.getenv("STAGE2_SINGLE_MAX_TOKENS", "1800")),
    "STAGE1_REASONER_MAX_TOKENS": int(os.getenv("STAGE1_REASONER_MAX_TOKENS", "6500")),
    "STAGE1_REASONER_RETRY_MAX_TOKENS": int(os.getenv("STAGE1_REASONER_RETRY_MAX_TOKENS", "8500")),
    "STAGE2_REASONER_MAX_TOKENS": int(os.getenv("STAGE2_REASONER_MAX_TOKENS", "2800")),
    "STAGE2_REASONER_RETRY_MAX_TOKENS": int(os.getenv("STAGE2_REASONER_RETRY_MAX_TOKENS", "3600")),

    # Safety / resource limits
    "STAGE1_MAX_CHARS": int(os.getenv("STAGE1_MAX_CHARS", "60000")),
    "STAGE1_COMPACT_VIEW_ENABLED": os.getenv("STAGE1_COMPACT_VIEW_ENABLED", "1").lower() not in {"0", "false", "no"},
    "STAGE1_FRONT_MATTER_CHARS": int(os.getenv("STAGE1_FRONT_MATTER_CHARS", "8000")),
    "STAGE1_EXCERPT_RADIUS_CHARS": int(os.getenv("STAGE1_EXCERPT_RADIUS_CHARS", "1600")),
    "STAGE1_MAX_EXCERPTS": int(os.getenv("STAGE1_MAX_EXCERPTS", "6")),
    "STAGE1_DOSSIER_MAX_CHARS": int(os.getenv("STAGE1_DOSSIER_MAX_CHARS", "24000")),
    "STAGE1_TARGET_HINT_MAX_HITS": int(os.getenv("STAGE1_TARGET_HINT_MAX_HITS", "4")),
    "STAGE1_RETRY_ENABLED": os.getenv("STAGE1_RETRY_ENABLED", "1").lower() not in {"0", "false", "no"},
    "STAGE1_TARGETED_RECALL_ENABLED": os.getenv("STAGE1_TARGETED_RECALL_ENABLED", "1").lower() not in {"0", "false", "no"},
    "STAGE1_TARGETED_RECALL_ON_EMPTY_CITATIONS": os.getenv("STAGE1_TARGETED_RECALL_ON_EMPTY_CITATIONS", "1").lower() not in {"0", "false", "no"},
    "STAGE1_PROOF_LOCAL_RECALL_ENABLED": os.getenv("STAGE1_PROOF_LOCAL_RECALL_ENABLED", "1").lower() not in {"0", "false", "no"},
    "STAGE1_PROOF_LOCAL_RECALL_MAX_CHARS": int(os.getenv("STAGE1_PROOF_LOCAL_RECALL_MAX_CHARS", "24000")),
    "STAGE2_MAX_CHARS": int(os.getenv("STAGE2_MAX_CHARS", "20000")),
    "MAX_STAGE2_CITATIONS_PER_PAPER": int(os.getenv("MAX_STAGE2_CITATIONS_PER_PAPER", "4")),
    "STAGE2_BATCH_ENABLED": os.getenv("STAGE2_BATCH_ENABLED", "0").lower() not in {"0", "false", "no"},
    "MAX_STAGE2_SINGLE_CALLS_PER_PAPER": int(os.getenv("MAX_STAGE2_SINGLE_CALLS_PER_PAPER", "12")),
    "STAGE2_WEAK_TOOL_RETRY_ENABLED": os.getenv("STAGE2_WEAK_TOOL_RETRY_ENABLED", "1").lower() not in {"0", "false", "no"},
    "LOCAL_CONTEXT_MAX_BLOCKS": int(os.getenv("LOCAL_CONTEXT_MAX_BLOCKS", "5")),
    "STAGE2_LOCAL_SLICE_BEFORE_CHARS": int(os.getenv("STAGE2_LOCAL_SLICE_BEFORE_CHARS", "3500")),
    "STAGE2_LOCAL_SLICE_AFTER_CHARS": int(os.getenv("STAGE2_LOCAL_SLICE_AFTER_CHARS", "1200")),
    "EXTERNAL_TOOL_SCOUT_ENABLED": os.getenv("EXTERNAL_TOOL_SCOUT_ENABLED", "1").lower() not in {"0", "false", "no"},
    "EXTERNAL_TOOL_SCOUT_MIN_SCORE": float(os.getenv("EXTERNAL_TOOL_SCOUT_MIN_SCORE", "3.0")),
    "EXTERNAL_TOOL_SCOUT_MAX_CHARS": int(os.getenv("EXTERNAL_TOOL_SCOUT_MAX_CHARS", "120000")),
    "STRICT_QC_MODE": os.getenv("STRICT_QC_MODE", "1").lower() not in {"0", "false", "no"},
    "STRICT_QC_REJECT_OTHER_TOOL_TYPE": os.getenv("STRICT_QC_REJECT_OTHER_TOOL_TYPE", "1").lower() not in {"0", "false", "no"},
    "STRICT_QC_REJECT_EMPTY_LOCAL_CONTEXT": os.getenv("STRICT_QC_REJECT_EMPTY_LOCAL_CONTEXT", "1").lower() not in {"0", "false", "no"},
    "STRICT_QC_REJECT_STRUCTURAL_LOCAL_CONTEXT": os.getenv("STRICT_QC_REJECT_STRUCTURAL_LOCAL_CONTEXT", "1").lower() not in {"0", "false", "no"},
    "STRICT_QC_REJECT_EMPTY_SETUP": os.getenv("STRICT_QC_REJECT_EMPTY_SETUP", "1").lower() not in {"0", "false", "no"},
    "STRICT_QC_REJECT_EMPTY_ANCHOR": os.getenv("STRICT_QC_REJECT_EMPTY_ANCHOR", "1").lower() not in {"0", "false", "no"},
    "STRICT_QC_REQUIRE_RESTATED_SUPPORT": os.getenv("STRICT_QC_REQUIRE_RESTATED_SUPPORT", "1").lower() not in {"0", "false", "no"},
    "FAST_PREFILTER_ENABLED": os.getenv("FAST_PREFILTER_ENABLED", "1").lower() not in {"0", "false", "no"},
    "FAST_PREFILTER_MIN_PROOF_CHARS": int(os.getenv("FAST_PREFILTER_MIN_PROOF_CHARS", "500")),
    "FAST_PREFILTER_REQUIRE_ZERO_PROOF_CITES": os.getenv("FAST_PREFILTER_REQUIRE_ZERO_PROOF_CITES", "1").lower() not in {"0", "false", "no"},
    "FAST_PREFILTER_MAX_SCOUT_SCORE": float(os.getenv("FAST_PREFILTER_MAX_SCOUT_SCORE", "5.0")),
    "FAST_PREFILTER_RESPECT_SCOUT_SCORE": os.getenv("FAST_PREFILTER_RESPECT_SCOUT_SCORE", "1").lower() not in {"0", "false", "no"},
    "FAST_PREFILTER_MIN_SOURCE_PRIORITY": float(os.getenv("FAST_PREFILTER_MIN_SOURCE_PRIORITY", "3.0")),
    "SETUP_RECOVERY_WINDOW_CHARS": int(os.getenv("SETUP_RECOVERY_WINDOW_CHARS", "12000")),
    "SETUP_MAX_CHARS": int(os.getenv("SETUP_MAX_CHARS", "2200")),
    "REQUEST_TIMEOUT_SECONDS": float(os.getenv("REQUEST_TIMEOUT_SECONDS", "420")),
    "REQUEST_CONNECT_TIMEOUT_SECONDS": float(os.getenv("REQUEST_CONNECT_TIMEOUT_SECONDS", "30")),
    "OPENAI_MAX_RETRIES": int(os.getenv("OPENAI_MAX_RETRIES", "1")),
}


# ============================================================
# Logging
# ============================================================

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("BenchmarkBuilder")

CONTROL_CHAR_PATTERN = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
ARCHIVE_HEADER_ARTIFACT_PATTERN = re.compile(
    r"\b[\w./-]+\.tex\s+\d{5,}(?:\s+\d+){3,}\s+ustar\b.*$",
    re.IGNORECASE,
)
STRUCTURED_ARTIFACT_PATTERN = re.compile(r"\bMALFORMED_OUTPUT\b\s*:?", re.IGNORECASE)
SOURCE_FILE_MARKER_PATTERN = re.compile(
    r"%\s*---\s*File:\s*[^%\n]+?---",
    re.IGNORECASE,
)
NON_TOOL_STATEMENT_PATTERN = re.compile(
    r"\b(?:"
    r"for\s+more\s+information"
    r"|for\s+background"
    r"|for\s+details"
    r"|for\s+notation"
    r"|for\s+definitions?"
    r"|we\s+refer\s+to"
    r"|see(?:\s+also)?\s+(?:e\.g\.\s+)?"
    r"|as\s+discussed\s+in"
    r"|is\s+discussed\s+in"
    r"|for\s+an?\s+overview"
    r"|for\s+a\s+survey"
    r"|for\s+the\s+proof\s+of"
    r")\b",
    re.IGNORECASE,
)
GENERIC_CONTEXT_PREFIX_PATTERN = re.compile(
    r"^under\s+the\s+current\s+proof\s+hypotheses,\s*",
    re.IGNORECASE,
)
BROKEN_TOOL_STATEMENT_PATTERN = re.compile(
    r"(?:"
    r"\\end\{(?:equation|align|align\*|eqnarray|gather|multline)\}"
    r"|\\begin\{(?:equation|align|align\*|eqnarray|gather|multline)\}"
    r"|\bit\s+was\s+(?:proved|shown|established)\s+in\.\s*$"
    r"|\bsee\s+[A-Z]?[a-z]+?\.\s*$"
    r")",
    re.IGNORECASE,
)
TOOL_NARRATION_PATTERN = re.compile(
    r"\b(?:"
    r"future\s+work"
    r"|natural\s+question\s+remains\s+unsettled"
    r"|the\s+treatment\s+for\s+this\s+case"
    r"|first\s+and\s+foremost"
    r"|closest\s+predecessor"
    r"|aligned\s+where"
    r"|as\s+a\s+consequence\s+of"
    r"|the\s+argument\s+in"
    r"|apply\s+the\s+involution\s+properties"
    r"|establish\s+the\s+baseline"
    r"|a\s+series\s+of\s+papers"
    r"|we\s+would\s+like\s+to"
    r"|we\s+have\s+been\s+concerned\s+with"
    r"|in\s+this\s+paper"
    r"|the\s+strategy\s+of\s+proof"
    r"|recent\s+work\s+of"
    r")\b",
    re.IGNORECASE,
)
TOOL_SOURCE_REPORT_PATTERN = re.compile(
    r"(?:"
    r"\b(?:theorem|lemma|proposition|corollary|claim)\s+"
    r"(?:\d+(?:\.\d+)*|\w+)"
    r"[^\n]{0,80}\b(?:states?|asserts?|says?)\s+that\b"
    r"|\b(?:theorem|lemma|proposition|corollary|claim)\s+"
    r"(?:\d+(?:\.\d+)*|\w+)\s*:\s*"
    r")",
    re.IGNORECASE,
)
TOOL_ENV_WRAPPER_PATTERN = re.compile(
    r"\\(?:begin|end)\{(?:proof|proposition|prop|lemma|lem|theorem|thm|corollary|cor|claim|"
    r"example|remark|rem|question|enumerate|itemize)\}",
    re.IGNORECASE,
)
TOOL_DISCOURSE_RESIDUE_PATTERN = re.compile(
    r"\b(?:"
    r"provided\s+a\s+linear[\-\s]time\s+algorithm"
    r"|recent\s+work\s+of"
    r"|as\s+discussed\s+in"
    r"|proof[\-\s]local\s+citation"
    r"|we\s+follow\s+the\s+argument\s+of"
    r"|our\s+solution\s+is\s+to\s+use"
    r"|all\s+we\s+can\s+say\s+is\s+that"
    r"|it\s+is\s+well\s+known\s+that"
    r")\b",
    re.IGNORECASE,
)
TOOL_PROMPT_FRAGMENT_PATTERN = re.compile(
    r"(?:"
    r"^if\s+(?:for\b|is\b|with\b|on\b|the\b|simomentum\b|nimal\b|our\s+solution\b|all\s+we\s+can\s+say\b|\\begin\{)"
    r"|\bthen\s+(?:since|indeed|there\s+is\s+no\s+need|we\s+know)\b"
    r")",
    re.IGNORECASE,
)
TOOL_SEMANTIC_FRAGMENT_PATTERN = re.compile(
    r"\b(?:"
    r"numerical\s+studies\s+in"
    r"|has\s+been\s+well[\-\s]studied"
    r"|space\s+bound\s+appears\s+in"
    r"|there\s+is\s+no\s+need\s+to\s+invoke"
    r"|the\s+second\s+author\s+studied"
    r"|used\s+optimal\s+transportation\s+tools"
    r"|is\s+very\s+fast[\-\s]growing"
    r"|based\s+on\s+numerical\s+evidence"
    r"|our\s+nu\."
    r"|have\s+shown,\s+using\s+a\s+different\s+set\s+of\s+tools"
    r"|we\s+follow\s+the\s+argument\s+of"
    r"|our\s+solution\s+is\s+to\s+use"
    r"|all\s+we\s+can\s+say\s+is\s+that"
    r"|it\s+is\s+well\s+known\s+that"
    r")\b",
    re.IGNORECASE,
)
TOOL_DANGLING_END_PATTERN = re.compile(
    r"(?:\b(?:in|see|appears\s+in|proved\s+in|shown\s+in|given\s+in)\.?)$",
    re.IGNORECASE,
)
TOOL_LOCATOR_ARTIFACT_PATTERN = re.compile(
    r"(?:"
    r"\b(?:theorem|lemma|corollary|proposition|section)\s*~"
    r"|\\(?:ref|cref|Cref|eqref)\{"
    r")",
    re.IGNORECASE,
)
TOOL_BAD_PREFIX_PATTERN = re.compile(
    r"^(?:"
    r"if\s+)?(?:"
    r"in\s+addition"
    r"|note\s+also\s+that"
    r"|recall\s+that"
    r"|indeed"
    r"|thus"
    r"|hence"
    r"|therefore"
    r"|consequently"
    r"|case\s+[A-Z]\b"
    r"|aligned\s+where"
    r"|a\s+general\s+lower\s+bound"
    r"|functoriality\s+of"
    r"|the\s+treatment\s+for\s+this\s+case"
    r"|the\s+closest\s+predecessor"
    r"|we\s+also\s+know\s+that"
    r"|apply\s+the"
    r"|establish\s+the\s+baseline"
    r"|proof[\-\s]local\s+citation"
    r")\b",
    re.IGNORECASE,
)
ANCHOR_NOISE_PATTERN = re.compile(
    r"(?:"
    r"\(\s*\)"
    r"|\b\^\s*_\b"
    r"|\b_\s*\^\b"
    r"|\$\s*\^\s*_\s*\$"
    r"|aligned\s+where"
    r"|future\s+work"
    r"|natural\s+question\s+remains\s+unsettled"
    r"|as\s+a\s+consequence\s+of"
    r"|the\s+argument\s+in"
    r"|proof[\-\s]local\s+citation"
    r"|theorem_like"
    r"|trigger_word"
    r"|explicit_locator"
    r"|result_transition"
    r"|as\s+discussed\s+in"
    r"|recent\s+work\s+of"
    r"|we\s+follow\s+the\s+argument\s+of"
    r"|all\s+we\s+can\s+say\s+is\s+that"
    r")",
    re.IGNORECASE,
)
ANCHOR_LOCATOR_ARTIFACT_PATTERN = re.compile(
    r"(?:"
    r"\bproof\s+of\b"
    r"|\b(?:lemma|theorem|corollary|proposition|remark|section)\s*~"
    r"|\\(?:ref|cref|Cref|eqref)\{"
    r")",
    re.IGNORECASE,
)
ANCHOR_DISCOURSE_FRAGMENT_PATTERN = re.compile(
    r"\b(?:"
    r"has\s+been\s+well[\-\s]studied"
    r"|is\s+somewhat\s+simpler"
    r"|there\s+is\s+no\s+need\s+to\s+invoke"
    r"|space\s+bound\s+appears\s+in"
    r"|numerical\s+studies\s+in"
    r"|used\s+optimal\s+transportation\s+tools"
    r"|we\s+also\s+refer\s+to"
    r"|is\s+asserted\s+to\s+be"
    r"|based\s+on\s+numerical\s+evidence"
    r"|the\s+second\s+author\s+studied"
    r"|main\s+result\s+in"
    r")\b",
    re.IGNORECASE,
)
ANCHOR_DANGLING_END_PATTERN = re.compile(
    r"(?:\b(?:in|see|appears\s+in|proved\s+in|shown\s+in|given\s+in)\.?)$",
    re.IGNORECASE,
)
ANCHOR_ACTION_VERB_PATTERN = re.compile(
    r"\b(?:"
    r"apply|use|invoke|extend|reduce|derive|deduce|show|prove|obtain|establish|construct|choose|"
    r"control|bound|estimate|compare|identify|pass|lift|exploit|conclude|introduce|verify|compute|"
    r"rule\s+out|relate|combine|decompose|factor|recall"
    r")\b",
    re.IGNORECASE,
)
SETUP_SIGNAL_PATTERN = re.compile(
    r"\b(?:"
    r"let|assume|suppose|consider|denote|define|fix|given|throughout|"
    r"under\s+the\s+assumption|in\s+this\s+section|we\s+work\s+with|"
    r"for\s+any|for\s+every|subject\s+to|problem|equation|system|operator|"
    r"space|manifold|group|graph|process|measure|distribution"
    r")\b",
    re.IGNORECASE,
)
SETUP_REJECT_PATTERN = re.compile(
    r"(?:"
    r"\\begin\{(?:abstract|proof|theorem|thm|lemma|lem|proposition|prop|corollary|cor|claim|remark|example)\}"
    r"|^\s*\\item\s*\["
    r"|(?:^|\s)(?:Proof|Theorem|Lemma|Proposition|Corollary|Claim)\b"
    r")",
    re.IGNORECASE,
)


# ============================================================
# Helper functions
# ============================================================

def load_ids_from_file(file_path: str) -> List[str]:
    """Load arXiv IDs from a text file (one id per line). Lines starting with # are ignored."""
    if not file_path or not os.path.exists(file_path):
        return []
    ids: List[str] = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            clean = line.strip()
            if clean and not clean.startswith("#"):
                ids.append(clean)
    return ids


def load_papers_from_manifest(file_path: str, target_ids: Optional[List[str]] = None) -> List[Dict[str, Any]]:
    """Load arXiv-backed paper metadata from a published-paper manifest."""
    if not file_path or not os.path.exists(file_path):
        return []

    with open(file_path, "r", encoding="utf-8") as f:
        payload = json.load(f)

    if isinstance(payload, list):
        papers = payload
    elif isinstance(payload, dict):
        papers = payload.get("papers", []) or []
    else:
        papers = []
    requested_ids = {paper_id.strip() for paper_id in (target_ids or []) if paper_id.strip()}
    rows: List[Dict[str, Any]] = []
    seen_arxiv_ids: set[str] = set()

    for paper in papers:
        arxiv_id = str(
            paper.get("openalex_arxiv_id")
            or paper.get("arxiv_id")
            or paper.get("id")
            or ""
        ).strip()
        if not arxiv_id:
            continue
        if requested_ids and arxiv_id not in requested_ids:
            continue
        if arxiv_id in seen_arxiv_ids:
            continue

        published_title = str(paper.get("published_title") or "").strip()
        arxiv_title = str(paper.get("arxiv_title") or "").strip()
        title = published_title or arxiv_title or arxiv_id

        rows.append(
            {
                "id": arxiv_id,
                "paper_link": str(paper.get("arxiv_url") or f"https://arxiv.org/abs/{arxiv_id}"),
                "latex_link": str(paper.get("latex_link") or f"https://arxiv.org/e-print/{arxiv_id}"),
                "title": title,
                "published": str(paper.get("publication_date") or paper.get("published") or "Unknown"),
                "summary": str(paper.get("summary") or ""),
                "authors": list(paper.get("published_authors") or paper.get("authors") or []),
                "domain": str(paper.get("domain") or ""),
                "venue": str(paper.get("venue") or ""),
                "abstract": str(paper.get("abstract") or ""),
                "notes": str(paper.get("notes") or ""),
                "doi": str(paper.get("doi") or ""),
                "published_title": published_title,
                "arxiv_title": arxiv_title,
                "proof_rich_score": paper.get("proof_rich_score"),
                "match_confidence": paper.get("match_confidence"),
            }
        )
        seen_arxiv_ids.add(arxiv_id)

    if requested_ids:
        present = {row["id"] for row in rows}
        missing = sorted(requested_ids - present)
        if missing:
            logger.warning("Manifest file is missing %d requested arXiv IDs.", len(missing))

    return rows


def parse_optional_float(value: Any, default: float = -1.0) -> float:
    try:
        if value is None or value == "":
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def stable_paper_shard(paper_id: str, shard_count: int) -> int:
    if shard_count <= 1:
        return 0
    digest = hashlib.md5(str(paper_id or "").encode("utf-8")).hexdigest()
    return int(digest[:8], 16) % shard_count


def apply_paper_order_and_shard(papers: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows = list(papers or [])
    if bool(CONFIG.get("PAPER_SORT_BY_PROOF_RICH", True)):
        rows.sort(
            key=lambda item: (
                parse_optional_float(item.get("proof_rich_score")),
                parse_optional_float(item.get("match_confidence")),
                str(item.get("published") or ""),
                str(item.get("id") or ""),
            ),
            reverse=True,
        )

    shard_count = int(CONFIG.get("PAPER_SHARD_COUNT", 1))
    shard_index = int(CONFIG.get("PAPER_SHARD_INDEX", 0))
    if shard_count <= 1:
        return rows
    if shard_index < 0 or shard_index >= shard_count:
        raise ValueError(f"Invalid shard index {shard_index} for shard count {shard_count}")

    filtered = [
        item for item in rows
        if stable_paper_shard(str(item.get("id") or ""), shard_count) == shard_index
    ]
    logger.info(
        "Shard selection kept %d/%d papers for shard %d/%d.",
        len(filtered),
        len(rows),
        shard_index + 1,
        shard_count,
    )
    return filtered


TERMINAL_PAPER_STATUSES = {
    "completed",
    "external_tool_scout_reject",
    "fast_prefilter_no_proof_citations",
    "target_theorem_not_main",
    "main_proof_not_found",
    "proof_span_not_aligned",
    "proof_span_too_short",
    "no_citations",
    "no_stage2_candidates",
    "no_locator_aligned_citations",
    "no_valid_stage2_instances",
}


def load_processed_paper_ids(progress_file: str) -> set[str]:
    """Load paper IDs that reached a terminal status in the progress JSONL file."""
    if not progress_file or not os.path.exists(progress_file):
        return set()

    processed: set[str] = set()
    with open(progress_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            paper_id = str(obj.get("paper_id") or "").strip()
            status = str(obj.get("status") or "").strip()
            if paper_id and status in TERMINAL_PAPER_STATUSES:
                processed.add(paper_id)
    return processed


def append_progress_record(
    handle,
    *,
    paper_id: str,
    title: str,
    status: str,
    valid_gaps: int,
    detail: str = "",
    elapsed_seconds: Optional[float] = None,
) -> None:
    record = {
        "paper_id": paper_id,
        "title": title,
        "status": status,
        "valid_gaps": int(valid_gaps),
        "detail": detail,
        "updated_at": datetime.datetime.now().isoformat(timespec="seconds"),
    }
    if elapsed_seconds is not None:
        record["elapsed_seconds"] = round(float(elapsed_seconds), 2)
    handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    handle.flush()


def normalize_key(key: str) -> str:
    """Normalize BibTeX keys to improve matching across files."""
    if not key:
        return ""
    return re.sub(r"[^a-z0-9]", "", key.lower())


DOI_PATTERN = re.compile(r"\b10\.\d{4,9}/[-._;()/:A-Z0-9]+\b", re.IGNORECASE)
ARXIV_PATTERN = re.compile(r"\b(?:arxiv:)?([0-9]{4}\.[0-9]{4,5}(?:v\d+)?)\b", re.IGNORECASE)
CITE_COMMAND_RE = r"\\cite[a-zA-Z*]*(?:\[[^\]]*\]){0,2}\{[^}]+\}"
GENERIC_TOOL_PATTERN = re.compile(
    r"^(?:theorem|lemma|proposition|corollary|claim|criterion|inequality|quantization theorem)"
    r"(?:\s+\d+(?:\.\d+)*)?$",
    re.IGNORECASE,
)
LATEX_CITE_PATTERN = re.compile(r"\\cite[a-zA-Z*]*(?:\[[^\]]*\]){0,2}\{([^}]+)\}")
TOOL_NAME_PATTERN = re.compile(
    r"\b(?:theorem|lemma|proposition|corollary|claim|criterion|inequality|estimate|bound)\b",
    re.IGNORECASE,
)
TOOL_CITATION_PATTERNS = [
    (
        "tool_then_cite",
        re.compile(
            r"\b(?:theorem|lemma|proposition|corollary|claim|criterion|inequality|estimate|bound)\b"
            rf"[^\n]{{0,120}}?{CITE_COMMAND_RE}",
            re.IGNORECASE,
        ),
        2.5,
    ),
    (
        "cite_then_tool",
        re.compile(
            rf"{CITE_COMMAND_RE}[^\n]{{0,120}}?"
            r"\b(?:theorem|lemma|proposition|corollary|claim|criterion|inequality|estimate|bound)\b",
            re.IGNORECASE,
        ),
        2.0,
    ),
    (
        "trigger_then_cite",
        re.compile(
            r"\b(?:by|using|apply(?:ing)?|via|from|invoke(?:d|s|ing)?)\b"
            rf"[^\n]{{0,120}}?{CITE_COMMAND_RE}",
            re.IGNORECASE,
        ),
        2.0,
    ),
]
PROOF_HEADING_PATTERN = re.compile(
    r"\\begin\{proof\}|\\section\*?\{[^}]*proof[^}]*\}|\\subsection\*?\{[^}]*proof[^}]*\}",
    re.IGNORECASE,
)
THEOREM_LIKE_PATTERN = re.compile(
    r"\\begin\{(?:theorem|lemma|proposition|corollary|claim)\}"
    r"|(?:^|\n)\s*(?:Theorem|Lemma|Proposition|Corollary|Claim)\b"
    r"|\\section\*?\{[^}]*main results?[^}]*\}"
    r"|\\section\*?\{[^}]*introduction[^}]*\}",
    re.IGNORECASE,
)
THEOREM_ENV_BEGIN_PATTERN = re.compile(r"\\begin\{(?P<env>[A-Za-z*]+)\}(?:\[[^\]]*\])?", re.IGNORECASE)
TEXTUAL_TARGET_HEADING_PATTERN = re.compile(
    r"(?:^|\n)\s*(?:Main\s+Theorem|Theorem|Proposition|Corollary)\b[^\n]{0,200}",
    re.IGNORECASE,
)
LATEX_LABEL_PATTERN = re.compile(r"\\label\{([^}]+)\}")
LATEX_REF_COMMAND_PATTERN = re.compile(r"\\[A-Za-z]*ref\{([^}]+)\}")
PROOF_START_PATTERN = re.compile(
    r"\\begin\{proof\}(?:\[[^\]]*\])?"
    r"|(?:^|\n)\s*Proof(?:\s+of\b[^\n]{0,160})?[\s.:]",
    re.IGNORECASE,
)
PROOF_END_PATTERN = re.compile(r"\\end\{proof\}|\\qed(?:here)?\b|\\qedsymbol\b", re.IGNORECASE)
PROOF_FALLBACK_BOUNDARY_PATTERN = re.compile(
    r"\\(?:sub)*section\*?\{"
    r"|\\begin\{(?:theorem|thm|proposition|prop|corollary|lemma|lem|claim|result)[A-Za-z*]*\}"
    r"|(?:^|\n)\s*(?:Theorem|Lemma|Proposition|Corollary|Claim)\b",
    re.IGNORECASE,
)
TARGET_PROOF_CUE_PATTERN = re.compile(
    r"\b(?:proof|prove|proves|proved|show|shows|showing|conclude|concludes|concluded|"
    r"complete|completes|completed|finish|finishes|finished)\b",
    re.IGNORECASE,
)
TARGET_PROOF_CONCLUSION_PATTERN = re.compile(
    r"\b(?:conclude(?:s|d)?|complete(?:s|d)?|finish(?:es|ed)?)\b[^\n]{0,120}\bproof\b",
    re.IGNORECASE,
)
LOCAL_CITATION_TRIGGER_PATTERN = re.compile(
    r"\b(?:by|using|apply(?:ing)?|via|from|invoke(?:d|s|ing)?|follow(?:s|ing)?|combine|combined with)\b",
    re.IGNORECASE,
)
LOCATOR_LABEL_PATTERN = re.compile(
    r"\b(?:theorem|lemma|proposition|corollary|claim|criterion|inequality|estimate|bound)\s*"
    r"(?:\d+(?:\.\d+)*)?",
    re.IGNORECASE,
)
RESULT_FLOW_PATTERN = re.compile(
    r"\b(?:therefore|hence|thus|consequently|it follows that|shows? that|ensures? that|implies? that|"
    r"yields? that|gives? that|by the results? in|according to)\b",
    re.IGNORECASE,
)
METHOD_REFERENCE_PATTERN = re.compile(
    r"\b(?:approach(?:es)?(?:\s+to|\s+taken in)?|in the notation of|notation of|introduced in|"
    r"following the proof of|using the argument in the proof of|same (?:notation|notational conventions) as|"
    r"coincide with those of|construction in|modified to adapt|strategy and techniques of)\b",
    re.IGNORECASE,
)
BACKGROUND_REFERENCE_PATTERN = re.compile(
    r"\b(?:we\s+refer\s+to|for\s+more\s+information|for\s+background|for\s+details|for\s+notation|"
    r"for\s+definitions?|for\s+a\s+survey|for\s+an?\s+overview|for\s+a\s+comprehensive\s+theory|"
    r"nice\s+survey|historical\s+remark|introduced\s+in|systematically\s+studied|studied\s+by|"
    r"inspired\s+by|extended\s+to|adaptation\s+of|exception\s+of|with\s+the\s+exception\s+of)\b",
    re.IGNORECASE,
)
COMPARATIVE_REFERENCE_PATTERN = re.compile(
    r"\b(?:weaker\s+than|stronger\s+than|similar\s+to|analogous\s+to|close\s+to|compare(?:d)?\s+with)\b",
    re.IGNORECASE,
)
REMARK_CONTEXT_PATTERN = re.compile(
    r"\\(?:begin|end)\{remark\}|\\(?:begin|end)\{itemize\}|\\(?:begin|end)\{enumerate\}",
    re.IGNORECASE,
)
METHOD_FALLBACK_REJECT_PATTERN = re.compile(
    r"\b(?:obtained by|solving|initial value problem|combining|introducing|simplified|notation|construction)\b",
    re.IGNORECASE,
)
DISCOURSE_PREFIX_PATTERN = re.compile(
    r"^(?:in particular|indeed|recall that|note that|now|thus|hence|therefore|consequently)\s*,\s*",
    re.IGNORECASE,
)
PREDICATE_HEAD_PATTERN = re.compile(
    r"\b(?:belongs to|lies in|is|are|has|have|satisfies|fulfills?|obeys|solve(?:s)?|holds?|contains?)\b",
    re.IGNORECASE,
)
IMPLICATION_SPLIT_PATTERN = re.compile(
    r"\b(?:and therefore|therefore|and hence|hence|thus|consequently)\b",
    re.IGNORECASE,
)

CATEGORY_TO_DOMAIN = {
    "math.AP": "analysis_pde",
    "math.CA": "analysis_pde",
    "math.DG": "geometry_topology",
    "math.GT": "geometry_topology",
    "math.AG": "algebra_number_theory",
    "math.NT": "algebra_number_theory",
    "math.CO": "combinatorics_discrete",
    "math.OC": "probability_statistics_control",
    "math.PR": "probability_statistics_control",
    "stat.TH": "probability_statistics_control",
    "stat.ML": "probability_statistics_control",
    "cs.LG": "probability_statistics_control",
    "cs.DS": "probability_statistics_control",
    "cs.CC": "probability_statistics_control",
    "cs.LO": "probability_statistics_control",
}

DOMAIN_HEURISTIC_PATTERNS: Dict[str, List[str]] = {
    "analysis_pde": [
        r"\bnavier\b",
        r"\beuler\b",
        r"\bschr(?:o|ö)dinger\b",
        r"\bwave equation\b",
        r"\belliptic\b",
        r"\bparabolic\b",
        r"\bhamilton[\-\s]jacobi\b",
        r"\bhomogenization\b",
        r"\bfractional diffusion\b",
        r"\bfree surface\b",
        r"\bincompressible\b",
        r"\bsobolev\b",
    ],
    "geometry_topology": [
        r"\bmanifold(?:s)?\b",
        r"\bcalabi[\-\s]yau\b",
        r"\bk[aä]hler\b",
        r"\briemannian\b",
        r"\bhyperbolic\b",
        r"\bcontact\b",
        r"\bsymplectic\b",
        r"\bgeodesic\b",
        r"\btopolog(?:y|ical)\b",
        r"\bg_?2\b",
        r"\bricci\b",
        r"\bmirror symmetry\b",
        r"\bhyperk[aä]hler\b",
    ],
    "algebra_number_theory": [
        r"\blanglands\b",
        r"\bgalois\b",
        r"\bautomorphic\b",
        r"\bshimura\b",
        r"\bmoduli\b",
        r"\balgebraic stack(?:s)?\b",
        r"\bl[\-\s]?function(?:s)?\b",
        r"\bhecke\b",
        r"\bp[\-\s]?adic\b",
        r"\bnumber field(?:s)?\b",
        r"\brepresentation(?:s)?\b",
        r"\bbernstein\b",
        r"\bsato[\-\s]tate\b",
    ],
    "combinatorics_discrete": [
        r"\bgraph(?:s)?\b",
        r"\bhypergraph(?:s)?\b",
        r"\bmatroid(?:s)?\b",
        r"\bramsey\b",
        r"\btur[aá]n\b",
        r"\bplabic\b",
        r"\bcombinator(?:ics|ial)\b",
        r"\bqueue[\-\s]number\b",
        r"\bcoloring\b",
        r"\bcayley graph(?:s)?\b",
        r"\bspanning tree\b",
        r"\btwin[\-\s]width\b",
        r"\bminor\b",
    ],
    "probability_statistics_control": [
        r"\bprobab(?:ility|ilistic)?\b",
        r"\bstochast(?:ic|ics)\b",
        r"\bmarkov\b",
        r"\bbayes(?:ian)?\b",
        r"\bestimation\b",
        r"\blearning\b",
        r"\boptimization\b",
        r"\bcontrol(?:s)?\b",
        r"\bmcmc\b",
        r"\brandom\b",
        r"\bmean field game(?:s)?\b",
        r"\bparticle system(?:s)?\b",
        r"\bdecision process(?:es)?\b",
        r"\bpercolation\b",
        r"\brisks?\b",
    ],
}

DISTINCTIVE_VENUE_TO_DOMAIN = {
    "Analysis & PDE": "analysis_pde",
    "Communications in Partial Differential Equations": "analysis_pde",
    "Journal of Functional Analysis": "analysis_pde",
    "Annals of PDE": "analysis_pde",
    "Geometry & Topology": "geometry_topology",
    "Journal of Differential Geometry": "geometry_topology",
    "Journal of Topology": "geometry_topology",
    "Algebra & Number Theory": "algebra_number_theory",
    "Combinatorica": "combinatorics_discrete",
    "Electronic Journal of Combinatorics": "combinatorics_discrete",
    "European Journal of Combinatorics": "combinatorics_discrete",
    "Journal of Combinatorial Theory, Series A": "combinatorics_discrete",
    "Journal of Combinatorial Theory, Series B": "combinatorics_discrete",
    "Random Structures & Algorithms": "combinatorics_discrete",
    "SIAM Journal on Discrete Mathematics": "combinatorics_discrete",
    "Annals of Probability": "probability_statistics_control",
    "Annals of Applied Probability": "probability_statistics_control",
    "Probability Theory and Related Fields": "probability_statistics_control",
    "Annals of Statistics": "probability_statistics_control",
    "Journal of Machine Learning Research": "probability_statistics_control",
    "Mathematical Programming": "probability_statistics_control",
    "Mathematics of Operations Research": "probability_statistics_control",
    "Bernoulli": "probability_statistics_control",
    "SIAM Journal on Control and Optimization": "probability_statistics_control",
    "SIAM Journal on Optimization": "probability_statistics_control",
}

ALLOWED_TOOL_FAMILIES = {
    "concentration_tail_bound",
    "compactness_embedding_existence",
    "spectral_perturbation_matrix_inequality",
    "convexity_duality_optimization",
    "counting_combinatorial_inequality",
    "algebraic_structure_lemma",
    "asymptotic_limit_theorem",
    "geometric_topological_criterion",
    "dynamics_ergodic_estimate",
    "other",
}


def extract_doi(text: str) -> str:
    match = DOI_PATTERN.search(text or "")
    return match.group(0).rstrip(".,;") if match else ""


def extract_arxiv_id(text: str) -> str:
    match = ARXIV_PATTERN.search(text or "")
    return match.group(1) if match else ""


def infer_source_type(citation_text: str) -> str:
    lowered = (citation_text or "").lower()
    if not lowered:
        return "unknown"
    if "lecture notes" in lowered:
        return "lecture_notes"
    if any(token in lowered for token in [
        "tata institute of fundamental research",
        "lecture notes in mathematics",
        "notes by",
    ]):
        return "lecture_notes"
    if any(token in lowered for token in [
        "monograph",
        "monographs",
        "mathematical surveys and monographs",
        "memoirs of the american mathematical society",
        "annals of mathematics studies",
        "graduate studies in mathematics",
        "cambridge studies in advanced mathematics",
    ]):
        return "book"
    if "survey" in lowered:
        return "survey"
    if "thesis" in lowered:
        return "thesis"
    if any(token in lowered for token in ["arxiv", "preprint", "working paper", "unpublished manuscript", "manuscript"]):
        return "preprint"
    if any(token in lowered for token in [
        "princeton university press",
        "cambridge university press",
        "oxford university press",
        "springer",
        "birkh",
        "wiley",
        "world scientific",
        "chelsea publishing",
        "ams chelsea publishing",
        "mc graw hill",
        "mcgraw hill",
        "american mathematical society, providence",
    ]):
        return "book"
    if re.search(r"\breprint of the \d{4} original\b", lowered):
        return "book"
    if "proceedings" in lowered or "conference" in lowered:
        return "conference_paper"
    if any(token in lowered for token in ["icml", "neurips", "iclr", "aistats", "colt", "stoc", "focs", "soda"]):
        return "conference_paper"
    if any(token in lowered for token in ["journal", "annals", "inventiones", "acta mathematica", "siam", "duke mathematical journal", "combinatorica"]):
        return "journal_paper"
    if any(token in lowered for token in [
        "proc. london math. soc.",
        "proc. lond. math. soc.",
        "j. amer. math. soc.",
        "j. eur. math. soc.",
        "j. combin. theory",
        "anal. \\& pde",
        "anal. & pde",
        "analysis & pde",
        "camb. j. math.",
        "illinois j. math.",
        "math. ann.",
        "math. z.",
        "adv. math.",
        "proc. amer. math. soc.",
    ]):
        return "journal_paper"
    if re.search(r"(?:^|[\s\\])j\.\s*[a-z]", lowered):
        return "journal_paper"
    if re.search(r"\b(?:journal|commun\.|ann\.|invent\.|duke|bernoulli|probab\.|topol\.|geom\.|math\.)", lowered):
        return "journal_paper"
    if re.search(r"\b\d+\s*\(\d{4}\)\s*,\s*no\.\s*\d+\s*,\s*\d{1,5}\s*[-–]{1,2}\s*\d{1,5}\b", lowered):
        return "journal_paper"
    if re.search(r"\bno\.\s*\d+\b", lowered) and re.search(r"\b\d{4}\b", lowered):
        return "journal_paper"
    if "press" in lowered or "publisher" in lowered:
        return "book"
    return "unknown"


def has_bibliography_metadata(citation_text: str) -> bool:
    return str(citation_text or "").strip() not in {"", "Citation content not found"}


def clause_has_mathematical_signal(text: str) -> bool:
    cleaned = normalize_proof_clause(text)
    if not cleaned:
        return False
    if TOOL_NARRATION_PATTERN.search(cleaned) or TOOL_BAD_PREFIX_PATTERN.match(cleaned):
        return False
    if any(token in cleaned for token in [r"\\", "$", "=", "<", ">", "≤", "≥", r"\le", r"\ge", "_", "^"]):
        return True
    if re.search(r"\b(?:for all|for every|there exists|belongs to|lies in|satisfies|obeys|implies?|yields?)\b", cleaned, re.IGNORECASE):
        return True
    if PREDICATE_HEAD_PATTERN.search(cleaned) and len(re.findall(r"\b\w+\b", cleaned)) >= 4:
        return True
    return False


def tool_statement_rejection_reason(statement: str) -> str:
    text = re.sub(r"\s+", " ", str(statement or "")).strip()
    if not text:
        return "empty"
    lowered = text.lower()
    if re.match(r"^if\s+(?:since|by|moreover|indeed)\b", lowered):
        return "prompt_fragment"
    if r"\cite" in text:
        return "contains_citation_command"
    if any(phrase in lowered for phrase in ["desired contradiction", "as claimed", "we then have"]):
        return "discourse_phrase"
    if TOOL_NARRATION_PATTERN.search(text):
        return "narrative_phrase"
    if TOOL_SOURCE_REPORT_PATTERN.search(text):
        return "source_report"
    if TOOL_ENV_WRAPPER_PATTERN.search(text):
        return "env_wrapper"
    if TOOL_DISCOURSE_RESIDUE_PATTERN.search(text):
        return "discourse_residue"
    if TOOL_PROMPT_FRAGMENT_PATTERN.search(text):
        return "prompt_fragment"
    if TOOL_SEMANTIC_FRAGMENT_PATTERN.search(text):
        return "semantic_fragment"
    if TOOL_DANGLING_END_PATTERN.search(text):
        return "dangling_locator"
    if TOOL_LOCATOR_ARTIFACT_PATTERN.search(text) and not clause_has_mathematical_signal(text):
        return "locator_artifact"
    if GENERIC_CONTEXT_PREFIX_PATTERN.match(text):
        return "context_prefix"
    if TOOL_BAD_PREFIX_PATTERN.match(text):
        return "bad_prefix"
    contextual_prefix_match = GENERIC_CONTEXT_PREFIX_PATTERN.match(text)
    if contextual_prefix_match:
        stripped = text[contextual_prefix_match.end() :].strip()
        if not stripped:
            return "context_only"
        stripped_lower = stripped.lower()
        if re.match(
            r"^(?:as|since|because|that|which|who|where|when|while|or\b|to\b|in the case of|in the related case of|"
            r"provided\s+(?:a|an|the)\b|recent\s+work\s+of\b|have\s+shown\b|the\s+second\s+author\s+studied\b|"
            r"is\s+very\s+fast[\-\s]growing\b|by\s+establishing\b|numerical\s+studies\s+in\b)\b",
            stripped_lower,
        ):
            return "contextual_fragment"
        if not clause_has_mathematical_signal(stripped):
            return "contextual_without_math_signal"
    if GENERIC_TOOL_PATTERN.fullmatch(text):
        return "generic_tool_name"
    if NON_TOOL_STATEMENT_PATTERN.search(text):
        return "non_tool_reference"
    if BROKEN_TOOL_STATEMENT_PATTERN.search(text):
        return "broken_latex"
    if len(re.findall(r"\\\[", text)) != len(re.findall(r"\\\]", text)):
        return "display_math_unbalanced"
    if len(re.findall(r"(?<!\\)\$", text)) % 2 != 0:
        return "inline_math_unbalanced"
    open_braces = text.count("{")
    close_braces = text.count("}")
    if open_braces != close_braces:
        return "brace_unbalanced"
    if re.search(r"[_^](?:[\s.,;:)\]]|$)", text):
        return "dangling_subsup"
    if re.match(r"^if\b", lowered) and re.search(r"\bthen\b", lowered):
        match = re.match(r"^if\s+(.*?),?\s+then\s+(.*)$", text, re.IGNORECASE)
        if match:
            antecedent = match.group(1).strip(" ,;:")
            consequent = match.group(2).strip(" ,;:")
            if not clause_has_mathematical_signal(antecedent):
                return "if_then_non_math_antecedent"
            if not clause_has_mathematical_signal(consequent):
                return "if_then_non_math_consequent"
    word_count = len(re.findall(r"\b\w+\b", text))
    if word_count >= 8:
        return ""
    if word_count >= 5 and any(
        phrase in lowered
        for phrase in [
            " implies ",
            " imply ",
            " yields ",
            " yield ",
            " belongs to ",
            " lies in ",
            " estimate ",
            " bound ",
            " regularity ",
            " criterion ",
            " there exists ",
            " for every ",
            " for all ",
        ]
    ):
        return ""
    if len(text) >= 24 and any(token in lowered for token in [" if ", " then ", " suppose ", " there exists ", " for all "]):
        return ""
    if len(text) >= 24 and any(token in text for token in [r"\begin{", r"\label{", r"\Vert", r"\|"]):
        return ""
    if len(text) >= 40 and any(token in text for token in [r"\\", "$", "=", "<", ">", "≤", "≥"]):
        return ""
    return "too_short_or_not_theorem_like"


def has_meaningful_tool_statement(statement: str) -> bool:
    return tool_statement_rejection_reason(statement) == ""


def infer_domain_label(paper: Dict[str, Any], default_category: str = "") -> str:
    explicit_domain = str(paper.get("domain", "") or "").strip()
    title = str(paper.get("published_title") or paper.get("title") or "").strip()
    abstract = str(paper.get("abstract") or paper.get("summary") or "").strip()
    notes = str(paper.get("notes") or "").strip()
    venue = str(paper.get("venue", "") or "").strip()

    scores: Dict[str, int] = {domain: 0 for domain in DOMAIN_HEURISTIC_PATTERNS}
    if explicit_domain in scores:
        scores[explicit_domain] += 1

    if venue in DISTINCTIVE_VENUE_TO_DOMAIN:
        scores[DISTINCTIVE_VENUE_TO_DOMAIN[venue]] += 3

    title_lower = title.lower()
    body_lower = f"{abstract} {notes}".lower()
    for domain, patterns in DOMAIN_HEURISTIC_PATTERNS.items():
        for pattern in patterns:
            if re.search(pattern, title_lower):
                scores[domain] += 2
            if re.search(pattern, body_lower):
                scores[domain] += 1

    ranked_scores = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    best_domain, best_score = ranked_scores[0]
    second_score = ranked_scores[1][1] if len(ranked_scores) > 1 else 0
    if best_score >= 2 and best_score >= second_score + 1:
        return best_domain

    if explicit_domain:
        return explicit_domain
    categories = list(paper.get("categories", []) or [])
    primary = str(paper.get("primary_category", "") or default_category or "")
    for category in [primary] + categories:
        if category in CATEGORY_TO_DOMAIN:
            return CATEGORY_TO_DOMAIN[category]
    if default_category in CATEGORY_TO_DOMAIN:
        return CATEGORY_TO_DOMAIN[default_category]
    return "unknown"


def normalize_tool_family_label(value: str) -> str:
    label = str(value or "").strip()
    return label if label in ALLOWED_TOOL_FAMILIES else "other"


def is_local_context_fragment(block: str) -> bool:
    normalized = normalize_signature_text(block)
    if not normalized:
        return True
    if re.search(r"\\(?:begin|end)\{[A-Za-z*]*$", normalized):
        return True
    if re.search(r"\\(?:begin|end)\{[^}]*\}\s*$", normalized) and len(normalized) < 24:
        return True
    alpha_words = re.findall(r"[A-Za-z]{3,}", normalized)
    if re.match(r"^(\\label|\\left|\\right|\\\\|&|O\\left|\)|\]|\})", normalized):
        return True
    if len(normalized) < 40 and len(alpha_words) < 4:
        return True
    if len(alpha_words) < 3:
        return True
    return False


def coalesce_local_context_blocks(blocks: List[str]) -> List[str]:
    merged: List[str] = []
    pending_prefix = ""

    for block in blocks:
        normalized = normalize_signature_text(block)
        if not normalized:
            continue

        if is_local_context_fragment(normalized):
            pending_prefix = f"{pending_prefix} {normalized}".strip()
            continue

        if pending_prefix:
            normalized = f"{pending_prefix} {normalized}".strip()
            pending_prefix = ""
        merged.append(normalized)

    if pending_prefix:
        if is_local_context_fragment(pending_prefix):
            pending_prefix = ""
        elif merged:
            merged[-1] = f"{merged[-1]} {pending_prefix}".strip()
        else:
            merged.append(pending_prefix)

    return merged


def normalize_local_context_blocks(blocks: Any) -> List[str]:
    if isinstance(blocks, list):
        cleaned = [normalize_signature_text(block) for block in blocks if normalize_signature_text(block)]
    elif blocks is None:
        cleaned = []
    else:
        single = normalize_signature_text(blocks)
        cleaned = [single] if single else []
    coalesced = coalesce_local_context_blocks(cleaned)
    return coalesced[-int(CONFIG["LOCAL_CONTEXT_MAX_BLOCKS"]):]


def proof_opening_locator_is_valid(locator_text: Any) -> bool:
    locator = normalize_signature_text(locator_text)
    if not locator:
        return False
    return bool(PROOF_OPENING_LOCATOR_PATTERN.match(locator))


def empty_local_context_is_acceptable(
    local_stage2_meta: Dict[str, Any],
    locator_text: Any = "",
) -> bool:
    if not isinstance(local_stage2_meta, dict):
        return False
    if str(local_stage2_meta.get("reason") or "") != "ok":
        return False
    prefix_chars = int(local_stage2_meta.get("prefix_chars", -1))
    locator_start = int(local_stage2_meta.get("locator_start", -1))
    slice_start = int(local_stage2_meta.get("slice_start", -1))
    starts_near_proof_head = (
        prefix_chars == 0
        or locator_start == 0
        or (slice_start == 0 and 0 <= prefix_chars <= 80)
    )
    if not starts_near_proof_head:
        return False
    return proof_opening_locator_is_valid(locator_text)


STRUCTURAL_LOCAL_CONTEXT_PATTERN = re.compile(
    r"(?:"
    r"\\(?:sub)*section\*?\{"
    r"|\\paragraph\*?\{"
    r"|\\begin\{section\}"
    r"|\\begin\{(?:figure|table|conj|conjecture)\*?\}"
    r"|\\end\{(?:figure|table|conj|conjecture)\*?\}"
    r"|\\begin\{(?:proof|thm|theorem|lemma|proposition|prop|cor|corollary|remark|rmk|definition|itemize|enumerate)\}"
    r"|\\begin\{(?:tikzpicture|minipage)\}"
    r"|\\end\{(?:tikzpicture|minipage)\}"
    r"|\\caption\{"
    r"|\\node\s+at\b"
    r"|\\draw\b"
    r"|\\(?:vspace|hspace)\{"
    r"|\\approvals\{"
    r")",
    re.IGNORECASE,
)

ALLOWED_LOWERCASE_CONTEXT_STARTERS = {
    "if",
    "for",
    "let",
    "suppose",
    "assume",
    "then",
    "since",
    "because",
    "indeed",
    "thus",
    "hence",
    "therefore",
    "we",
    "by",
    "as",
    "in",
    "on",
    "under",
    "from",
    "when",
    "where",
    "after",
    "before",
    "now",
}

PROOF_OPENING_LOCATOR_PATTERN = re.compile(
    r"^\s*(?:\\begin\{proof\}|proof\.?|we\s+prove|to\s+prove|we\s+now\s+prove|it\s+remains\s+to\s+prove)\b",
    re.IGNORECASE,
)


def local_context_block_rejection_reason(block: Any) -> str:
    cleaned = normalize_signature_text(block)
    if not cleaned:
        return "empty"
    if STRUCTURAL_LOCAL_CONTEXT_PATTERN.search(cleaned):
        return "structural_artifact"
    first_word_match = re.match(r"^([A-Za-z]{3,})\b", cleaned)
    if first_word_match:
        first_word = str(first_word_match.group(1) or "").strip()
        if first_word.islower() and first_word.lower() not in ALLOWED_LOWERCASE_CONTEXT_STARTERS:
            return "truncated_prefix"
    if re.match(r"^(?:Figure|Table)\b", cleaned):
        return "caption_like"
    return ""


def local_context_has_structural_artifacts(blocks: List[str]) -> bool:
    for block in blocks or []:
        if local_context_block_rejection_reason(block):
            return True
    return False


def empty_local_context_row_is_acceptable(row: Dict[str, Any]) -> bool:
    x = row.get("x", {}) or {}
    y = row.get("y", {}) or {}
    z = row.get("z", {}) or {}
    if normalize_local_context_blocks(x.get("local_context", [])):
        return True
    if not bool(y.get("restated_in_citing_paper", False)):
        return False
    locator = z.get("locator_snippet") or z.get("locator")
    if not proof_opening_locator_is_valid(locator):
        return False
    return True


def compact_detail(payload: Dict[str, Any]) -> str:
    return json.dumps(payload, ensure_ascii=False, sort_keys=True)


def normalize_signature_text(text: Any) -> str:
    cleaned = CONTROL_CHAR_PATTERN.sub(" ", str(text or ""))
    cleaned = ARCHIVE_HEADER_ARTIFACT_PATTERN.sub("", cleaned)
    cleaned = SOURCE_FILE_MARKER_PATTERN.sub(" ", cleaned)
    return re.sub(r"\s+", " ", cleaned).strip()


def sanitize_structured_text_field(text: Any) -> str:
    cleaned = normalize_signature_text(text)
    if not cleaned:
        return ""
    if STRUCTURED_ARTIFACT_PATTERN.search(cleaned):
        return ""
    return cleaned


def sanitize_anchor_hint_text(text: Any) -> str:
    raw_cleaned = normalize_signature_text(text)
    cleaned = raw_cleaned.replace("~", " ")
    cleaned = re.sub(r"\\(?:p?label|cref|Cref|ref|eqref)\{[^}]*\}", " ", cleaned)
    cleaned = re.sub(r"\\[A-Za-z]+\*?(?:\[[^\]]*\])?(?:\{[^}]*\})?", " ", cleaned)
    cleaned = cleaned.replace("$ $", " ")
    cleaned = re.sub(r"[{}\[\]]", " ", cleaned)
    cleaned = re.sub(
        r"\b(?:as\s+discussed\s+in|by|using|via|from|according\s+to|applying)\s*$",
        "",
        cleaned,
        flags=re.IGNORECASE,
    )
    cleaned = re.sub(r"\s+", " ", cleaned).strip(" ,;:")
    lowered = cleaned.lower()
    if not cleaned or ANCHOR_NOISE_PATTERN.search(cleaned):
        return ""
    if re.search(r"\\(?:begin|end)\{(?:proof|example|question|remark|rem|enumerate|itemize)\}", raw_cleaned, re.IGNORECASE):
        return ""
    if raw_cleaned.count("$") % 2 == 1:
        return ""
    if re.search(r"(?:\\\.$|\\$|[\^_]\s*$)", raw_cleaned):
        return ""
    if sum(raw_cleaned.count(token) for token in ["\\", "_", "^"]) >= 6:
        return ""
    if len(re.findall(r"[A-Za-z]{3,}", cleaned)) < 4:
        return ""
    if ANCHOR_LOCATOR_ARTIFACT_PATTERN.search(raw_cleaned) or ANCHOR_LOCATOR_ARTIFACT_PATTERN.search(cleaned):
        return ""
    if ANCHOR_DISCOURSE_FRAGMENT_PATTERN.search(cleaned):
        return ""
    if ANCHOR_DANGLING_END_PATTERN.search(cleaned):
        return ""
    if not ANCHOR_ACTION_VERB_PATTERN.search(cleaned):
        return ""
    if any(
        phrase in lowered
        for phrase in [
            "future work",
            "natural question remains unsettled",
            "as a consequence of",
            "the argument in",
            "aligned where",
            "proof-local citation",
            "theorem_like",
            "trigger_word",
            "explicit_locator",
            "result_transition",
            "as discussed in",
            "recent work of",
        ]
    ):
        return ""
    if re.match(r"^(?:proof\s+of|lemma\b|theorem\b|corollary\b|proposition\b|remark\b|section\b)", lowered):
        return ""
    if cleaned and cleaned[0].islower():
        cleaned = cleaned[0].upper() + cleaned[1:]
    return cleaned.rstrip(".") + "."


def sanitize_anchor_hint_fallback_text(text: Any) -> str:
    raw_cleaned = normalize_signature_text(text)
    cleaned = raw_cleaned.replace("~", " ")
    cleaned = re.sub(r"\\(?:p?label|cref|Cref|ref|eqref)\{[^}]*\}", " ", cleaned)
    cleaned = re.sub(r"\\[A-Za-z]+\*?(?:\[[^\]]*\])?(?:\{[^}]*\})?", " ", cleaned)
    cleaned = cleaned.replace("$ $", " ")
    cleaned = re.sub(r"[{}\[\]]", " ", cleaned)
    cleaned = re.sub(r"^(?:proof[\-\s]local\s+citation\s*:?)", "", cleaned, flags=re.IGNORECASE).strip(" ,;:")
    cleaned = re.sub(
        r"\b(?:theorem_like|trigger_word|explicit_locator|result_transition|result_flow|method_reference|background_reference|comparative_reference|remark_context)\b",
        " ",
        cleaned,
        flags=re.IGNORECASE,
    )
    cleaned = re.sub(r"\s+", " ", cleaned).strip(" ,;:")
    lowered = cleaned.lower()
    if not cleaned or ANCHOR_NOISE_PATTERN.search(cleaned):
        return ""
    if re.search(r"\\(?:begin|end)\{(?:proof|example|question|remark|rem|enumerate|itemize)\}", raw_cleaned, re.IGNORECASE):
        return ""
    if raw_cleaned.count("$") % 2 == 1:
        return ""
    if re.search(r"(?:\\\.$|\\$|[\^_]\s*$)", raw_cleaned):
        return ""
    if sum(raw_cleaned.count(token) for token in ["\\", "_", "^"]) >= 8:
        return ""
    if ANCHOR_LOCATOR_ARTIFACT_PATTERN.search(raw_cleaned) or ANCHOR_LOCATOR_ARTIFACT_PATTERN.search(cleaned):
        return ""
    if ANCHOR_DISCOURSE_FRAGMENT_PATTERN.search(cleaned):
        return ""
    if ANCHOR_DANGLING_END_PATTERN.search(cleaned):
        return ""
    if any(
        phrase in lowered
        for phrase in [
            "future work",
            "natural question remains unsettled",
            "as a consequence of",
            "the argument in",
            "proof-local citation",
            "theorem_like",
            "trigger_word",
            "explicit_locator",
            "result_transition",
            "recent work of",
        ]
    ):
        return ""
    if re.match(r"^(?:proof\s+of|lemma\b|theorem\b|corollary\b|proposition\b|remark\b|section\b)", lowered):
        return ""
    if len(re.findall(r"[A-Za-z]{3,}", cleaned)) < 4:
        return ""
    if len(cleaned) > 220:
        cleaned = cleaned[:220].rstrip(" ,;:")
    if cleaned and cleaned[0].islower():
        cleaned = cleaned[0].upper() + cleaned[1:]
    return cleaned.rstrip(".") + "."


def anchor_hint_is_acceptable(text: Any) -> bool:
    return bool(sanitize_anchor_hint_text(text))


def select_anchor_hint(*candidates: Any) -> str:
    for candidate in candidates:
        cleaned = sanitize_anchor_hint_text(candidate)
        if cleaned:
            return cleaned
    for candidate in candidates:
        cleaned = sanitize_anchor_hint_fallback_text(candidate)
        if cleaned:
            return cleaned
    return ""


def build_anchor_hint_from_reason(usage_reason: Any) -> str:
    reason = normalize_signature_text(usage_reason)
    if not reason:
        return ""
    reason = re.sub(r"^proof[\-\s]local\s+citation\s*:?\s*", "", reason, flags=re.IGNORECASE)
    reason = re.sub(
        r"\b(?:theorem_like|trigger_word|explicit_locator|result_transition|result_flow|method_reference|background_reference|comparative_reference|remark_context)\b",
        " ",
        reason,
        flags=re.IGNORECASE,
    )
    reason = re.sub(r"\s+", " ", reason).strip(" ,;:")
    return reason


def build_anchor_hint_from_local_context(local_context_blocks: Optional[List[str]]) -> str:
    for block in reversed(normalize_local_context_blocks(local_context_blocks or [])):
        candidate_block = normalize_proof_clause(block)
        if not candidate_block:
            continue
        sentence_chunks = [chunk.strip() for chunk in re.split(r"(?<=[.?!])\s+", candidate_block) if chunk.strip()]
        for candidate in reversed(sentence_chunks or [candidate_block]):
            if sanitize_anchor_hint_text(candidate) or sanitize_anchor_hint_fallback_text(candidate):
                return candidate
    return ""


def extract_informative_overlap_tokens(text: Any) -> List[str]:
    cleaned = normalize_signature_text(text).lower()
    cleaned = re.sub(r"\\[A-Za-z]+\*?(?:\[[^\]]*\])?(?:\{[^}]*\})?", " ", cleaned)
    cleaned = re.sub(r"[^a-z0-9]+", " ", cleaned)
    stopwords = {
        "the", "and", "for", "with", "that", "this", "then", "from", "into", "such", "under", "over",
        "have", "has", "are", "is", "was", "were", "be", "being", "been", "let", "suppose", "assume",
        "there", "exists", "every", "each", "some", "any", "our", "their", "your", "proof", "result",
        "theorem", "lemma", "proposition", "corollary", "claim", "criterion", "bound", "estimate",
    }
    tokens: List[str] = []
    for token in cleaned.split():
        if len(token) < 3 or token in stopwords:
            continue
        tokens.append(token)
    return tokens


def restated_statement_supported_by_evidence(
    statement: Any,
    locator_snippet: Any,
    local_context_blocks: Optional[List[str]] = None,
    local_slice_text: Any = "",
) -> bool:
    statement_tokens = list(dict.fromkeys(extract_informative_overlap_tokens(statement)))
    if len(statement_tokens) < 4:
        return False
    evidence_parts = [str(locator_snippet or "")]
    evidence_parts.extend(local_context_blocks or [])
    if local_slice_text:
        evidence_parts.append(str(local_slice_text)[:1200])
    evidence = normalize_signature_text(" ".join(part for part in evidence_parts if part))
    if not evidence:
        return False
    evidence_lower = evidence.lower()
    matched = sum(1 for token in statement_tokens if token in evidence_lower)
    overlap_ratio = matched / max(1, len(statement_tokens))
    return matched >= 3 and overlap_ratio >= 0.3


def score_setup_block(block: str) -> int:
    cleaned = normalize_signature_text(block)
    if not cleaned:
        return -999
    rejection_reason = setup_text_rejection_reason(cleaned)
    if rejection_reason:
        return -999
    score = 0
    if SETUP_SIGNAL_PATTERN.search(cleaned):
        score += 2
    if any(token in str(block or "") for token in ["$", "\\", r"\begin{definition}", r"\begin{assumption}"]):
        score += 1
    if count_citations_in_text(cleaned) >= 3:
        score -= 1
    if len(cleaned) >= 80:
        score += 1
    if len(re.findall(r"[A-Za-z]{3,}", cleaned)) < 8:
        score -= 1
    return score


def setup_text_rejection_reason(text: Any) -> str:
    cleaned = normalize_signature_text(text)
    if not cleaned:
        return "empty"
    lowered = cleaned.lower()
    if "\\begin{abstract}" in lowered:
        return "abstract_block"
    if "\\end{proof}" in lowered:
        return "proof_tail"
    if re.search(r"\\(?:begin|end)\{(?:figure|table|tikzpicture|minipage)\}", lowered):
        return "figure_or_table_env"
    if re.search(r"\\(?:caption|draw|node)\b", lowered):
        return "figure_artifact"
    if re.search(r"\\begin\{(?:proof|theorem|thm|lemma|lem|proposition|prop|corollary|cor|claim)\}", lowered):
        return "theorem_like_env"
    if re.search(r"\\begin\{(?:example|question|remark|rem)\}", lowered):
        return "non_setup_env"
    if re.match(r"^\s*\\item\s*\[", cleaned):
        return "list_item"
    return ""


def recover_setup_text_from_anchor(full_text: str, anchor_start: int) -> Tuple[str, Dict[str, Any]]:
    text = str(full_text or "")
    if not text:
        return "", {"reason": "empty_full_text"}
    if anchor_start < 0:
        return "", {"reason": "target_anchor_missing"}

    window_chars = int(CONFIG.get("SETUP_RECOVERY_WINDOW_CHARS", 12000))
    setup_max_chars = int(CONFIG.get("SETUP_MAX_CHARS", 2200))
    slice_start = max(0, anchor_start - window_chars)
    window_text = text[slice_start:anchor_start]
    raw_blocks = re.split(r"\n\s*\n+", window_text)
    scored_blocks: List[Tuple[int, int, str]] = []
    for idx, block in enumerate(raw_blocks):
        cleaned = normalize_signature_text(block)
        score = score_setup_block(block)
        if not cleaned or score < 2:
            continue
        scored_blocks.append((idx, score, cleaned))

    if not scored_blocks:
        fallback_blocks: List[Tuple[int, int, str]] = []
        for idx, block in enumerate(raw_blocks):
            cleaned = normalize_signature_text(block)
            score = score_setup_block(block)
            if not cleaned or score < 1:
                continue
            fallback_blocks.append((idx, score, cleaned))
        scored_blocks = fallback_blocks

    if not scored_blocks:
        return "", {
            "reason": "no_setup_like_blocks",
            "slice_start": slice_start,
            "anchor_start": anchor_start,
        }

    chosen = sorted(scored_blocks[-3:], key=lambda item: item[0])
    setup = normalize_signature_text(" ".join(block for _, _, block in chosen))
    if len(setup) > setup_max_chars:
        setup = setup[:setup_max_chars].rstrip(" ,;:")
    if not setup:
        return "", {
            "reason": "setup_empty_after_join",
            "slice_start": slice_start,
            "anchor_start": anchor_start,
        }
    rejection_reason = setup_text_rejection_reason(setup)
    if rejection_reason:
        return "", {
            "reason": f"recovered_setup_rejected:{rejection_reason}",
            "slice_start": slice_start,
            "anchor_start": anchor_start,
            "selected_block_count": len(chosen),
            "candidate_block_count": len(scored_blocks),
        }
    return setup, {
        "reason": "ok",
        "slice_start": slice_start,
        "anchor_start": anchor_start,
        "selected_block_count": len(chosen),
        "candidate_block_count": len(scored_blocks),
    }


def decode_loose_json_string_fragment(fragment: str) -> str:
    raw = str(fragment or "")
    if not raw:
        return ""
    try:
        return sanitize_structured_text_field(json.loads(f'"{raw}"'))
    except Exception:
        try:
            escaped = re.sub(r'(?<!\\)\\(?!["\\/bfnrtu])', r"\\\\", raw)
            return sanitize_structured_text_field(json.loads(f'"{escaped}"'))
        except Exception:
            fallback = raw.replace('\\"', '"').replace("\\n", " ").replace("\\t", " ").replace("\\r", " ")
            return sanitize_structured_text_field(fallback)


def extract_loose_json_string_field(raw_text: str, field_name: str, next_keys: Optional[List[str]] = None) -> str:
    haystack = str(raw_text or "")
    if not haystack:
        return ""
    key = re.escape(field_name)
    patterns: List[str] = []
    for next_key in next_keys or []:
        patterns.append(
            rf'"{key}"\s*:\s*"(?P<value>.*?)"\s*,\s*"{re.escape(next_key)}"'
        )
    patterns.append(rf'"{key}"\s*:\s*"(?P<value>.*?)"\s*(?:,|\}})')
    for pattern in patterns:
        match = re.search(pattern, haystack, flags=re.DOTALL)
        if match:
            value = decode_loose_json_string_fragment(match.group("value"))
            if value:
                return value
    return ""


def extract_loose_json_bool_field(raw_text: str, field_name: str) -> Optional[bool]:
    haystack = str(raw_text or "")
    if not haystack:
        return None
    key = re.escape(field_name)
    match = re.search(
        rf'"{key}"\s*:\s*(?P<value>true|false|1|0|"true"|"false"|"1"|"0")',
        haystack,
        flags=re.IGNORECASE,
    )
    if not match:
        return None
    value = match.group("value").strip().strip('"').lower()
    if value in {"true", "1"}:
        return True
    if value in {"false", "0"}:
        return False
    return None


def salvage_stage2_payload_from_raw_text(raw_text: str) -> Optional[Dict[str, Any]]:
    payload = {
        "anchor_hint": extract_loose_json_string_field(
            raw_text,
            "anchor_hint",
            next_keys=["reference_tool_latex", "reference_tool_type", "restated_in_citing_paper"],
        ),
        "reference_tool_latex": extract_loose_json_string_field(
            raw_text,
            "reference_tool_latex",
            next_keys=["reference_tool_type", "restated_in_citing_paper", "citation_locator", "tool_family"],
        ),
        "reference_tool_type": extract_loose_json_string_field(
            raw_text,
            "reference_tool_type",
            next_keys=["restated_in_citing_paper", "citation_locator", "tool_family"],
        ),
    }
    restated = extract_loose_json_bool_field(raw_text, "restated_in_citing_paper")
    if restated is not None:
        payload["restated_in_citing_paper"] = restated
    if any(payload.values()):
        return payload
    return None


def extract_named_array_segment(raw_text: str, field_name: str) -> str:
    haystack = str(raw_text or "")
    if not haystack:
        return ""
    match = re.search(rf'"{re.escape(field_name)}"\s*:\s*\[', haystack)
    if not match:
        return ""
    start = match.end() - 1
    depth = 0
    in_string = False
    escape = False
    for idx in range(start, len(haystack)):
        ch = haystack[idx]
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
            continue
        if ch == "[":
            depth += 1
            continue
        if ch == "]":
            depth -= 1
            if depth == 0:
                return haystack[start + 1 : idx]
    return ""


def salvage_stage1_payload_from_raw_text(raw_text: str) -> Optional[Dict[str, Any]]:
    global_context = {
        "setup": extract_loose_json_string_field(
            raw_text,
            "setup",
            next_keys=["target_theorem", "proof_span", "proof_citations"],
        ),
        "target_theorem": extract_loose_json_string_field(
            raw_text,
            "target_theorem",
            next_keys=["proof_span", "proof_citations"],
        ),
    }
    proof_span = {
        "start_snippet": extract_loose_json_string_field(
            raw_text,
            "start_snippet",
            next_keys=["end_snippet", "proof_citations"],
        ),
        "end_snippet": extract_loose_json_string_field(
            raw_text,
            "end_snippet",
            next_keys=["proof_citations"],
        ),
    }

    proof_citations: List[Dict[str, str]] = []
    citations_array = extract_named_array_segment(raw_text, "proof_citations")
    if citations_array:
        key_matches = list(re.finditer(r'"citation_key"\s*:\s*"', citations_array))
        for idx, key_match in enumerate(key_matches):
            block_start = citations_array.rfind("{", 0, key_match.start())
            if block_start == -1:
                block_start = key_match.start()
            block_end = key_matches[idx + 1].start() if idx + 1 < len(key_matches) else len(citations_array)
            block = citations_array[block_start:block_end]
            citation_key = extract_loose_json_string_field(
                block,
                "citation_key",
                next_keys=["locator_snippet", "reason"],
            )
            locator_snippet = extract_loose_json_string_field(
                block,
                "locator_snippet",
                next_keys=["reason", "citation_key"],
            )
            reason = extract_loose_json_string_field(
                block,
                "reason",
                next_keys=["citation_key", "locator_snippet"],
            )
            if citation_key:
                proof_citations.append(
                    {
                        "citation_key": citation_key,
                        "locator_snippet": locator_snippet,
                        "reason": reason,
                    }
                )

    if global_context["setup"] or global_context["target_theorem"] or proof_citations:
        return {
            "global_context": global_context,
            "proof_span": proof_span,
            "proof_citations": proof_citations,
        }
    return None


def structured_json_response_format() -> Dict[str, str]:
    return {"type": "json_object"}


def is_reasoning_model_name(model_name: Any) -> bool:
    lowered = str(model_name or "").strip().lower()
    return "reasoner" in lowered or "thinking" in lowered


def extract_json_candidate_from_text(text: Any, required_keys: Optional[List[str]] = None) -> str:
    haystack = str(text or "").strip()
    if not haystack:
        return ""
    start = haystack.find("{")
    end = haystack.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return ""
    candidate = haystack[start:end + 1]
    if required_keys and not all(key in candidate for key in required_keys):
        return ""
    return candidate


TARGET_THEOREM_LABEL_RE = re.compile(
    r"^\s*(?P<label>main theorem|theorem|lemma|proposition|corollary|claim|step|fact|observation|"
    r"remark|example|question|definition|assumption|construction)\b"
    r"\s*(?:~?\s*\\ref\{[^}]+\}|(?:\d+(?:\.\d+)*[A-Za-z]?))?\s*[:.]?\s*",
    re.IGNORECASE,
)
SUSPICIOUS_LOCAL_TARGET_LABELS = {"lemma", "claim", "step", "fact", "observation"}
STRICT_QC_NON_MAIN_TARGET_LABELS = {
    "lemma",
    "claim",
    "step",
    "fact",
    "observation",
    "proposition",
    "corollary",
    "remark",
    "example",
    "question",
    "definition",
    "assumption",
    "construction",
}
TARGET_THEOREM_ENV_RE = re.compile(
    r"^\s*\\begin\{(?P<env>lemma|claim|step|fact|observation|prop|proposition|cor|corollary|"
    r"remark|rem|example|question|definition|defn|assumption|construction|cons)\*?\}",
    re.IGNORECASE,
)
TARGET_THEOREM_MAIN_ENV_ANY_RE = re.compile(
    r"\\begin\{(?P<env>theorem|thm|maintheorem|mainthm|mainresult|result|proposition|prop|corollary|cor)\*?\}",
    re.IGNORECASE,
)
TARGET_THEOREM_POLLUTION_RE = re.compile(
    r"(?:"
    r"\\(?:begin|end)\{(?:remark|rem|proof|figure|table|tikzpicture|minipage)\*?\}"
    r"|\\caption\{"
    r"|\\draw\b"
    r"|\\node\b"
    r")",
    re.IGNORECASE,
)


def normalize_target_theorem_text(text: Any) -> str:
    raw = sanitize_structured_text_field(text)
    if not raw:
        return ""
    env_match = TARGET_THEOREM_MAIN_ENV_ANY_RE.search(raw)
    if env_match and env_match.start() > 0:
        raw = raw[env_match.start() :].strip()
        env_name = str(env_match.group("env") or "").strip()
        if env_name:
            end_match = re.search(rf"\\end\{{{re.escape(env_name)}\*?\}}", raw, re.IGNORECASE)
            if end_match:
                raw = raw[: end_match.end()].strip()
    match = TARGET_THEOREM_LABEL_RE.match(raw)
    if not match:
        return raw
    remainder = raw[match.end() :].strip()
    return remainder or raw


def target_theorem_looks_local(text: Any) -> bool:
    raw = sanitize_structured_text_field(text)
    if not raw:
        return False
    match = TARGET_THEOREM_LABEL_RE.match(raw)
    if not match:
        return False
    label = str(match.group("label") or "").strip().lower()
    return label in SUSPICIOUS_LOCAL_TARGET_LABELS


def target_theorem_rejection_reason(text: Any, strict_mode: bool = False) -> str:
    raw = sanitize_structured_text_field(text)
    if not raw:
        return ""
    if TARGET_THEOREM_POLLUTION_RE.search(raw):
        normalized = normalize_target_theorem_text(raw)
        if not normalized or TARGET_THEOREM_POLLUTION_RE.search(normalized):
            return "stage1_target_theorem_polluted"
        raw = normalized

    env_match = TARGET_THEOREM_ENV_RE.match(raw)
    if env_match:
        env_label = str(env_match.group("env") or "").strip().lower()
        return f"stage1_target_theorem_env_not_main:{env_label}"

    match = TARGET_THEOREM_LABEL_RE.match(raw)
    if not match:
        return ""

    label = str(match.group("label") or "").strip().lower()
    if label in SUSPICIOUS_LOCAL_TARGET_LABELS:
        return f"stage1_target_theorem_looks_local:{label}"
    if strict_mode and label in STRICT_QC_NON_MAIN_TARGET_LABELS:
        return f"stage1_target_theorem_not_main_in_strict_qc:{label}"
    return ""


def normalize_stage1_result(payload: Any) -> Optional[Dict[str, Any]]:
    if not isinstance(payload, dict):
        return None

    normalized: Dict[str, Any] = {
        "global_context": {"setup": "", "target_theorem": ""},
        "proof_span": {"start_snippet": "", "end_snippet": ""},
        "proof_citations": [],
    }

    raw_global = payload.get("global_context")
    if isinstance(raw_global, dict):
        normalized["global_context"] = {
            "setup": sanitize_structured_text_field(raw_global.get("setup") or payload.get("setup") or ""),
            "target_theorem": sanitize_structured_text_field(
                raw_global.get("target_theorem") or payload.get("target_theorem") or ""
            ),
        }
    elif isinstance(raw_global, str):
        normalized["global_context"] = {
            "setup": sanitize_structured_text_field(payload.get("setup") or ""),
            "target_theorem": sanitize_structured_text_field(raw_global),
        }
    else:
        normalized["global_context"] = {
            "setup": sanitize_structured_text_field(payload.get("setup") or ""),
            "target_theorem": sanitize_structured_text_field(payload.get("target_theorem") or ""),
        }

    raw_proof_span = payload.get("proof_span")
    if isinstance(raw_proof_span, dict):
        normalized["proof_span"] = {
            "start_snippet": sanitize_structured_text_field(
                raw_proof_span.get("start_snippet") or payload.get("start_snippet") or ""
            ),
            "end_snippet": sanitize_structured_text_field(
                raw_proof_span.get("end_snippet") or payload.get("end_snippet") or ""
            ),
        }
    else:
        normalized["proof_span"] = {
            "start_snippet": sanitize_structured_text_field(payload.get("start_snippet") or ""),
            "end_snippet": sanitize_structured_text_field(payload.get("end_snippet") or ""),
        }

    raw_citations = payload.get("proof_citations")
    if isinstance(raw_citations, dict):
        raw_citations = [raw_citations]
    if isinstance(raw_citations, list):
        cleaned_citations: List[Dict[str, str]] = []
        for item in raw_citations:
            if not isinstance(item, dict):
                continue
            cleaned_citations.append(
                {
                    "citation_key": sanitize_structured_text_field(item.get("citation_key") or ""),
                    "locator_snippet": sanitize_structured_text_field(item.get("locator_snippet") or ""),
                    "reason": sanitize_structured_text_field(item.get("reason") or ""),
                    "_local_score": float(item.get("_local_score", 0.0) or 0.0),
                }
            )
        normalized["proof_citations"] = [item for item in cleaned_citations if item["citation_key"]]

    return normalized


def normalize_stage1_citation_list(payload: Any) -> List[Dict[str, Any]]:
    normalized = normalize_stage1_result(payload)
    if isinstance(normalized, dict):
        return normalized.get("proof_citations", []) or []

    if isinstance(payload, dict):
        raw_citations = payload.get("proof_citations")
        if isinstance(raw_citations, dict):
            raw_citations = [raw_citations]
        if isinstance(raw_citations, list):
            output: List[Dict[str, Any]] = []
            for item in raw_citations:
                if not isinstance(item, dict):
                    continue
                citation_key = sanitize_structured_text_field(item.get("citation_key") or "")
                if not citation_key:
                    continue
                output.append(
                    {
                        "citation_key": citation_key,
                        "locator_snippet": sanitize_structured_text_field(item.get("locator_snippet") or ""),
                        "reason": sanitize_structured_text_field(item.get("reason") or ""),
                        "_local_score": float(item.get("_local_score", 0.0) or 0.0),
                    }
                )
            return output
    return []


TARGET_HINT_STOPWORDS = {
    "theorem",
    "lemma",
    "proposition",
    "corollary",
    "claim",
    "proof",
    "there",
    "exists",
    "such",
    "that",
    "under",
    "with",
    "from",
    "into",
    "then",
    "this",
    "these",
    "those",
    "local",
    "global",
    "space",
    "metric",
}
TARGET_MATCH_TOKEN_STOPWORDS = TARGET_HINT_STOPWORDS | {
    "let",
    "then",
    "such",
    "that",
    "have",
    "holds",
    "hold",
    "where",
    "which",
    "with",
    "from",
    "into",
    "under",
    "over",
    "ball",
    "balls",
    "open",
    "some",
    "each",
    "there",
    "moreover",
    "satisfies",
    "satisfy",
    "proof",
    "text",
    "label",
    "ref",
    "eqref",
    "begin",
    "end",
    "left",
    "right",
    "quad",
    "qquad",
    "mathbf",
    "mathrm",
    "mathbb",
    "operatorname",
    "subseteq",
    "infty",
    "local",
    "loc",
    "equation",
}


def build_stage1_bib_context(bib_mapping: Dict[str, str]) -> str:
    keys = [str(key).strip() for key in bib_mapping.keys() if str(key).strip()]
    return "\n".join(f"Key: '{key}'" for key in keys[:400])


def merge_intervals(intervals: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
    cleaned = sorted((max(0, int(start)), max(0, int(end))) for start, end in intervals if end > start)
    if not cleaned:
        return []
    merged: List[Tuple[int, int]] = [cleaned[0]]
    for start, end in cleaned[1:]:
        prev_start, prev_end = merged[-1]
        if start <= prev_end:
            merged[-1] = (prev_start, max(prev_end, end))
            continue
        merged.append((start, end))
    return merged


def extract_target_hint_keywords(target_hint: str, max_keywords: int = 8) -> List[str]:
    raw_tokens = re.findall(r"[A-Za-z]{4,}", str(target_hint or ""))
    keywords: List[str] = []
    seen: set[str] = set()
    for token in raw_tokens:
        lowered = token.lower()
        if lowered in TARGET_HINT_STOPWORDS:
            continue
        if lowered in seen:
            continue
        seen.add(lowered)
        keywords.append(lowered)
        if len(keywords) >= max_keywords:
            break
    return keywords


def extract_target_match_tokens(target_hint: str, max_tokens: int = 16) -> List[str]:
    raw_tokens = re.findall(r"\\[A-Za-z]+|[A-Za-z]{2,}", str(target_hint or ""))
    tokens: List[str] = []
    seen: set[str] = set()
    for token in raw_tokens:
        normalized = str(token).lower().lstrip("\\")
        if len(normalized) < 2:
            continue
        if normalized in TARGET_MATCH_TOKEN_STOPWORDS:
            continue
        if normalized in seen:
            continue
        seen.add(normalized)
        tokens.append(normalized)
        if len(tokens) >= max_tokens:
            break
    return tokens


def collect_keyword_windows(text: str, keywords: List[str], radius: int, max_hits: int) -> List[Tuple[int, int]]:
    lowered_text = text.lower()
    windows: List[Tuple[int, int]] = []
    hit_count = 0
    for keyword in keywords:
        search_from = 0
        while hit_count < max_hits:
            idx = lowered_text.find(keyword, search_from)
            if idx < 0:
                break
            windows.append((max(0, idx - radius), min(len(text), idx + len(keyword) + radius)))
            hit_count += 1
            search_from = idx + len(keyword)
        if hit_count >= max_hits:
            break
    return windows


def collect_stage1_focus_windows(text: str, radius: int, max_excerpts: int) -> List[Tuple[int, int]]:
    windows: List[Tuple[int, int]] = []

    for pattern in [THEOREM_LIKE_PATTERN, PROOF_HEADING_PATTERN, PROOF_START_PATTERN]:
        for idx, match in enumerate(pattern.finditer(text)):
            if idx >= max_excerpts:
                break
            windows.append((max(0, match.start() - radius), min(len(text), match.end() + radius)))

    theorem_env_hits = 0
    for match in THEOREM_ENV_BEGIN_PATTERN.finditer(text):
        env_name = str(match.group("env") or "")
        if not is_target_theorem_env_name(env_name):
            continue
        windows.append((max(0, match.start() - radius), min(len(text), match.end() + radius)))
        theorem_env_hits += 1
        if theorem_env_hits >= max_excerpts:
            break

    for _, pattern, _ in TOOL_CITATION_PATTERNS:
        for idx, match in enumerate(pattern.finditer(text)):
            if idx >= max_excerpts:
                break
            windows.append((max(0, match.start() - radius), min(len(text), match.end() + radius)))

    return windows


def build_stage1_paper_view(full_text: str, target_hint: str = "") -> Tuple[str, Dict[str, Any]]:
    text = str(full_text or "")
    if not text:
        return "", {"head_chars": 0, "excerpt_count": 0, "total_chars": 0, "target_keywords": []}

    front_chars = min(len(text), int(CONFIG["STAGE1_FRONT_MATTER_CHARS"]))
    radius = int(CONFIG["STAGE1_EXCERPT_RADIUS_CHARS"])
    max_excerpts = int(CONFIG["STAGE1_MAX_EXCERPTS"])
    dossier_max_chars = int(CONFIG["STAGE1_DOSSIER_MAX_CHARS"])
    target_keywords = extract_target_hint_keywords(target_hint)

    windows: List[Tuple[int, int]] = collect_stage1_focus_windows(text, radius=radius, max_excerpts=max_excerpts)
    windows.extend(
        collect_keyword_windows(
            text,
            target_keywords,
            radius=radius,
            max_hits=int(CONFIG["STAGE1_TARGET_HINT_MAX_HITS"]),
        )
    )
    merged_windows = merge_intervals(windows)

    head_excerpt = text[:front_chars].strip()
    parts: List[str] = []
    total_chars = 0
    if head_excerpt:
        head_block = "[Paper Head]\n" + head_excerpt
        parts.append(head_block)
        total_chars += len(head_block)

    excerpt_count = 0
    for start, end in merged_windows:
        if excerpt_count >= max_excerpts:
            break
        if end <= front_chars:
            continue
        excerpt = text[start:end].strip()
        if not excerpt:
            continue
        block = f"\n\n[Focused Excerpt {excerpt_count + 1} | chars {start}:{end}]\n{excerpt}"
        if total_chars + len(block) > dossier_max_chars:
            break
        parts.append(block)
        total_chars += len(block)
        excerpt_count += 1

    if excerpt_count == 0 and len(text) > front_chars:
        tail_span = min(max(radius * 2, 5000), max(0, len(text) - front_chars))
        tail_start = max(front_chars, len(text) - tail_span)
        tail_excerpt = text[tail_start:].strip()
        if tail_excerpt:
            block = f"\n\n[Focused Excerpt 1 | chars {tail_start}:{len(text)} | tail_fallback]\n{tail_excerpt}"
            allowed_remaining = max(0, dossier_max_chars - total_chars)
            if allowed_remaining > 100:
                parts.append(block[:allowed_remaining])
                total_chars = min(dossier_max_chars, total_chars + len(block[:allowed_remaining]))
                excerpt_count = 1

    compact_view = "".join(parts)[:dossier_max_chars]
    return compact_view, {
        "head_chars": front_chars,
        "excerpt_count": excerpt_count,
        "total_chars": len(compact_view),
        "target_keywords": target_keywords,
    }


def has_proof_span_snippets(proof_span: Dict[str, Any]) -> bool:
    if not isinstance(proof_span, dict):
        return False
    start_snippet = str(proof_span.get("start_snippet") or "").strip()
    end_snippet = str(proof_span.get("end_snippet") or "").strip()
    return bool(start_snippet and end_snippet)


def build_instance_signature(local_result: Dict[str, Any]) -> Dict[str, str]:
    local_context = normalize_local_context_blocks(local_result.get("local_context", []) or [])
    local_context_text = "\n".join(local_context)
    return {
        "tool": normalize_signature_text(local_result.get("reference_tool_latex")),
        "tool_type": normalize_signature_text(local_result.get("reference_tool_type")),
        "anchor": normalize_signature_text(local_result.get("anchor_hint")),
        "local_context": local_context_text,
    }


def text_similarity(a: str, b: str) -> float:
    left = normalize_signature_text(a)
    right = normalize_signature_text(b)
    if not left or not right:
        return 0.0
    return SequenceMatcher(None, left, right).ratio()


def is_near_duplicate_instance(
    instance_signature: Dict[str, str], seen_instance_signatures: List[Dict[str, str]]
) -> bool:
    current_tool = instance_signature.get("tool", "")
    current_type = instance_signature.get("tool_type", "")
    current_anchor = instance_signature.get("anchor", "")
    current_context = instance_signature.get("local_context", "")
    if not current_tool:
        return False
    for seen in seen_instance_signatures:
        if current_tool != seen.get("tool", ""):
            continue
        if current_type and seen.get("tool_type", "") and current_type != seen.get("tool_type", ""):
            continue
        if text_similarity(current_anchor, seen.get("anchor", "")) >= 0.70 and text_similarity(
            current_context, seen.get("local_context", "")
        ) >= 0.80:
            return True
    return False


def scout_external_tool_usage(full_text: str) -> Dict[str, Any]:
    text = str(full_text or "")[: int(CONFIG["EXTERNAL_TOOL_SCOUT_MAX_CHARS"])]
    lowered = text.lower()
    cite_keys: List[str] = []
    for raw_keys in LATEX_CITE_PATTERN.findall(text):
        for item in raw_keys.split(","):
            key = normalize_key(item)
            if key:
                cite_keys.append(key)

    unique_cite_keys = sorted(set(cite_keys))
    tool_name_hits = len(TOOL_NAME_PATTERN.findall(text))
    proof_heading_hits = len(PROOF_HEADING_PATTERN.findall(text))
    proof_word_hits = len(re.findall(r"\bproof\b", lowered))
    proof_mentions = proof_heading_hits + proof_word_hits

    pattern_hits: Dict[str, int] = {}
    example_snippets: List[str] = []
    score = 0.0
    for label, pattern, weight in TOOL_CITATION_PATTERNS:
        matches = pattern.findall(text)
        count = len(matches)
        if not count:
            continue
        pattern_hits[label] = count
        score += min(3, count) * weight
        if len(example_snippets) < 3:
            for match in matches[: 3 - len(example_snippets)]:
                snippet = re.sub(r"\s+", " ", str(match)).strip()
                if snippet:
                    example_snippets.append(snippet[:180])

    if proof_heading_hits:
        score += 0.5
    if proof_heading_hits and unique_cite_keys:
        score += 0.5
    if len(unique_cite_keys) >= 8:
        score += 0.5
    if len(unique_cite_keys) >= 15:
        score += 0.5

    return {
        "score": round(score, 2),
        "unique_cite_count": len(unique_cite_keys),
        "tool_name_hits": tool_name_hits,
        "proof_heading_hits": proof_heading_hits,
        "proof_mentions": proof_mentions,
        "pattern_hits": pattern_hits,
        "examples": example_snippets,
    }


def collapse_whitespace_with_mapping(text: str) -> Tuple[str, List[int]]:
    chars: List[str] = []
    positions: List[int] = []
    last_was_space = True
    for idx, ch in enumerate(str(text or "")):
        if ch.isspace():
            if not last_was_space:
                chars.append(" ")
                positions.append(idx)
            last_was_space = True
            continue
        chars.append(ch)
        positions.append(idx)
        last_was_space = False

    while chars and chars[-1] == " ":
        chars.pop()
        positions.pop()
    return "".join(chars), positions


def normalize_snippet(snippet: str) -> str:
    normalized, _ = collapse_whitespace_with_mapping(snippet)
    return normalized.strip().lower()


def extract_main_proof_text(full_text: str, proof_span: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
    start_snippet = str((proof_span or {}).get("start_snippet") or "").strip()
    end_snippet = str((proof_span or {}).get("end_snippet") or "").strip()
    if not start_snippet or not end_snippet:
        return "", {"reason": "missing_proof_span_snippets"}

    normalized_text, positions = collapse_whitespace_with_mapping(full_text)
    lowered_text = normalized_text.lower()
    start_norm = normalize_snippet(start_snippet)
    end_norm = normalize_snippet(end_snippet)
    if not normalized_text or not start_norm or not end_norm:
        return "", {"reason": "empty_normalized_span"}

    start_idx = lowered_text.find(start_norm)
    if start_idx < 0:
        return "", {"reason": "start_snippet_not_found", "start_snippet": start_snippet[:160]}

    end_search_from = start_idx + len(start_norm)
    end_idx = lowered_text.find(end_norm, end_search_from)
    if end_idx < 0:
        return "", {
            "reason": "end_snippet_not_found",
            "start_snippet": start_snippet[:160],
            "end_snippet": end_snippet[:160],
        }

    orig_start = positions[start_idx]
    orig_end = positions[end_idx + len(end_norm) - 1] + 1
    if orig_end <= orig_start:
        return "", {"reason": "invalid_proof_span_order"}

    proof_text = str(full_text or "")[orig_start:orig_end]
    return proof_text, {
        "reason": "ok",
        "char_start": orig_start,
        "char_end": orig_end,
        "proof_chars": len(proof_text),
    }


def locate_snippet_span(text: str, snippet: str) -> Optional[Tuple[int, int]]:
    snippet = str(snippet or "").strip()
    if not text or not snippet:
        return None
    normalized_text, positions = collapse_whitespace_with_mapping(text)
    lowered_text = normalized_text.lower()
    normalized_snippet = normalize_snippet(snippet)
    if not normalized_snippet:
        return None
    start_idx = lowered_text.find(normalized_snippet)
    if start_idx < 0:
        return None
    end_idx = start_idx + len(normalized_snippet) - 1
    orig_start = positions[start_idx]
    orig_end = positions[end_idx] + 1
    return orig_start, orig_end


def extract_local_proof_slice(proof_text: str, locator_snippet: str) -> Tuple[str, Dict[str, Any]]:
    span = locate_snippet_span(proof_text, locator_snippet)
    if span is None:
        return "", {"reason": "locator_not_found", "used_fallback": False, "slice_chars": 0}

    start_idx, end_idx = span
    before = int(CONFIG["STAGE2_LOCAL_SLICE_BEFORE_CHARS"])
    after = int(CONFIG["STAGE2_LOCAL_SLICE_AFTER_CHARS"])
    slice_start = max(0, start_idx - before)
    slice_end = min(len(proof_text), end_idx + after)
    slice_start, slice_end = align_span_to_text_boundaries(
        str(proof_text or ""),
        slice_start,
        slice_end,
        max_adjust=96,
    )
    proof_slice = str(proof_text or "")[slice_start:slice_end]
    return proof_slice, {
        "reason": "ok",
        "used_fallback": False,
        "slice_start": slice_start,
        "slice_end": slice_end,
        "slice_chars": len(proof_slice),
    }


def split_proof_prefix_into_blocks(prefix_text: str) -> List[str]:
    prefix = str(prefix_text or "")
    if not prefix.strip():
        return []

    raw_blocks = re.split(r"\n\s*\n+", prefix)
    cleaned_blocks = [normalize_signature_text(block) for block in raw_blocks if normalize_signature_text(block)]
    if len(cleaned_blocks) >= 2:
        return cleaned_blocks

    normalized = normalize_signature_text(prefix)
    if not normalized:
        return []

    sentence_like = re.split(
        r"(?:(?<=[.?!])\s+(?=[A-Z\\$])|(?<=\})\s+(?=\\[A-Za-z])|(?<=\])\s+(?=[A-Z\\$]))",
        normalized,
    )
    return [normalize_signature_text(block) for block in sentence_like if normalize_signature_text(block)]


def extract_local_context_blocks_from_slice(context_text: str, locator_snippet: str) -> Tuple[List[str], Dict[str, Any]]:
    span = locate_snippet_span(context_text, locator_snippet)
    if span is None:
        return [], {"reason": "locator_not_found_in_local_slice", "block_count": 0}

    start_idx, _ = span
    prefix_text = str(context_text or "")[:start_idx]
    raw_blocks = split_proof_prefix_into_blocks(prefix_text)
    coalesced_blocks = coalesce_local_context_blocks(raw_blocks)
    last_boundary_idx = -1
    for idx, block in enumerate(coalesced_blocks):
        if STRUCTURAL_LOCAL_CONTEXT_PATTERN.search(str(block or "")):
            last_boundary_idx = idx
    if last_boundary_idx >= 0:
        coalesced_blocks = coalesced_blocks[last_boundary_idx + 1 :]
    local_blocks = coalesced_blocks[-int(CONFIG["LOCAL_CONTEXT_MAX_BLOCKS"]):]
    return local_blocks, {
        "reason": "ok",
        "block_count": len(local_blocks),
        "prefix_chars": len(prefix_text),
        "locator_start": start_idx,
        "last_structural_boundary_idx": last_boundary_idx,
    }


def extract_pre_citation_focus_snippet(locator_snippet: str) -> str:
    locator = str(locator_snippet or "")
    if not locator.strip():
        return ""

    cite_match = re.search(r"\\cite[a-zA-Z*]*\s*(?:\[[^\]]*\]\s*)?\{", locator)
    if cite_match:
        locator = locator[: cite_match.start()].rstrip().rstrip("\\").rstrip()

    normalized = normalize_signature_text(locator)
    if not normalized:
        return ""
    normalized = re.sub(r"\\+$", "", normalized).strip()

    normalized = re.sub(
        r"(?:by|using|via|from|applying|according to)\s+(?:the\s+)?(?:results?|theorem|lemma|proposition|corollary|criterion)?\s*(?:of|in)?\s*$",
        "",
        normalized,
        flags=re.IGNORECASE,
    ).strip(" ,;:")
    sentence_chunks = [chunk.strip() for chunk in re.split(r"(?<=[.?!])\s+", normalized) if chunk.strip()]
    if sentence_chunks:
        tail = sentence_chunks[-2:] if len(sentence_chunks[-1]) < 80 and len(sentence_chunks) >= 2 else sentence_chunks[-1:]
        return " ".join(tail)[-500:]
    return normalized[-500:]


def align_span_to_text_boundaries(text: str, left: int, right: int, max_adjust: int = 48) -> Tuple[int, int]:
    source = str(text or "")
    if not source:
        return 0, 0

    left = max(0, min(len(source), int(left)))
    right = max(left, min(len(source), int(right)))

    left_adjust = 0
    while left > 0 and left_adjust < max_adjust:
        prev = source[left - 1]
        if prev.isspace() or prev in r"()[]{}<>.,;:!?/\\|\"'`~-=+":
            break
        left -= 1
        left_adjust += 1

    right_adjust = 0
    while right < len(source) and right_adjust < max_adjust:
        curr = source[right]
        if curr.isspace() or curr in r"()[]{}<>.,;:!?/\\|\"'`~-=+":
            break
        right += 1
        right_adjust += 1

    return left, right


def split_locator_around_first_citation(locator_snippet: str) -> Tuple[str, str]:
    locator = str(locator_snippet or "")
    cite_match = re.search(r"\\cite[a-zA-Z*]*\s*(?:\[[^\]]*\]\s*)?\{[^}]+\}", locator)
    if not cite_match:
        normalized = normalize_signature_text(locator)
        return normalized, ""
    before = normalize_signature_text(locator[: cite_match.start()])
    after = normalize_signature_text(locator[cite_match.end() :])
    return before, after


def normalize_proof_clause(text: str) -> str:
    cleaned = normalize_signature_text(text)
    cleaned = DISCOURSE_PREFIX_PATTERN.sub("", cleaned).strip(" ,;:")
    return cleaned


def clause_starts_with_predicate(text: str) -> bool:
    cleaned = normalize_proof_clause(text)
    return bool(cleaned and PREDICATE_HEAD_PATTERN.match(cleaned))


def extract_clause_subject(text: str) -> str:
    cleaned = normalize_proof_clause(text)
    if not cleaned:
        return ""
    match = PREDICATE_HEAD_PATTERN.search(cleaned)
    if not match:
        return ""
    subject = cleaned[: match.start()].strip(" ,;:")
    if not subject:
        return ""
    if len(re.findall(r"\b\w+\b", subject)) > 12:
        return ""
    return subject


def build_implication_statement_from_clause(text: str) -> str:
    cleaned = normalize_proof_clause(text)
    if not cleaned or METHOD_REFERENCE_PATTERN.search(cleaned):
        return ""
    match = IMPLICATION_SPLIT_PATTERN.search(cleaned)
    if not match:
        return ""
    antecedent = cleaned[: match.start()].strip(" ,;:")
    consequent = cleaned[match.end() :].strip(" ,;:")
    if not antecedent or not consequent:
        return ""
    subject = extract_clause_subject(antecedent)
    if subject and clause_starts_with_predicate(consequent):
        consequent = f"{subject} {consequent}"
    if len(re.findall(r"\b\w+\b", consequent)) < 4:
        return ""
    return f"If {antecedent}, then {consequent}."


def build_post_citation_statement(locator_snippet: str) -> str:
    _, after = split_locator_around_first_citation(locator_snippet)
    cleaned = normalize_proof_clause(after)
    if not cleaned:
        return ""
    cleaned = re.split(r"(?<=[.?!])\s+", cleaned)[0].strip(" ,;:")
    cleaned = re.sub(
        r"^(?:shows?|ensures?|implies?|yields?|gives?|proves?|states?|asserts?)\s+that\s+",
        "",
        cleaned,
        flags=re.IGNORECASE,
    ).strip(" ,;:")
    if not cleaned or METHOD_REFERENCE_PATTERN.search(cleaned) or METHOD_FALLBACK_REJECT_PATTERN.search(cleaned):
        return ""
    if any(token in cleaned for token in [r"\begin{", r"\end{", r"\ali{", r"\label{", "&="]):
        return ""
    word_count = len(re.findall(r"\b\w+\b", cleaned))
    if word_count < 4:
        return ""
    if re.match(r"^(?:if|for all|for every|there exists)\b", cleaned, re.IGNORECASE):
        return cleaned.rstrip(".") + "."
    return f"Under the current proof hypotheses, {cleaned.rstrip('.')}."


def build_cross_citation_implication(locator_snippet: str, pre_citation_focus: str = "") -> str:
    antecedent = normalize_proof_clause(pre_citation_focus or extract_pre_citation_focus_snippet(locator_snippet))
    _, after = split_locator_around_first_citation(locator_snippet)
    consequent = normalize_proof_clause(after)
    antecedent = re.sub(r"^(?:suppose|assume|assuming|if|when|whenever)\s+", "", antecedent, flags=re.IGNORECASE)
    antecedent = re.sub(
        r"\b(?:by|using|via|from|according\s+to|applying)\b\s*$",
        "",
        antecedent,
        flags=re.IGNORECASE,
    )
    consequent = re.sub(
        r"^(?:then\s+)?(?:we\s+)?(?:have|obtain|conclude|deduce|see|know)\s+that\s+",
        "",
        consequent,
        flags=re.IGNORECASE,
    )
    consequent = re.sub(r"\bthen\b\s+", "", consequent, count=1, flags=re.IGNORECASE)
    consequent = re.sub(r"^(?:then\s+)", "", consequent, flags=re.IGNORECASE)
    antecedent = antecedent.strip(" ,;:.")
    consequent = consequent.strip(" ,;:.")
    if not antecedent or not consequent:
        return ""
    if METHOD_REFERENCE_PATTERN.search(antecedent) or METHOD_REFERENCE_PATTERN.search(consequent):
        return ""
    if METHOD_FALLBACK_REJECT_PATTERN.search(consequent):
        return ""
    if any(token in consequent for token in [r"\begin{", r"\end{", r"\label{", r"\cite"]):
        return ""
    if len(re.findall(r"\b\w+\b", antecedent)) < 3 or len(re.findall(r"\b\w+\b", consequent)) < 3:
        return ""
    return f"If {antecedent}, then {consequent.rstrip('.')}."


def build_anchor_hint_from_evidence(locator_snippet: str, pre_citation_focus: str = "") -> str:
    clause = normalize_proof_clause(pre_citation_focus or extract_pre_citation_focus_snippet(locator_snippet))
    if not clause:
        before, after = split_locator_around_first_citation(locator_snippet)
        clause = normalize_proof_clause(before or after)
    clause = re.sub(r"\\(?:p?label|cref|Cref|ref|eqref)\{[^}]*\}", " ", clause)
    clause = re.sub(r"\\[A-Za-z]+\*?(?:\[[^\]]*\])?(?:\{[^}]*\})?", " ", clause)
    clause = re.sub(
        r"^(?:it\s+remains\s+to|it\s+suffices\s+to|we\s+need\s+to|enough\s+to|to)\s+",
        "",
        clause,
        flags=re.IGNORECASE,
    ).strip(" ,;:")
    clause = re.sub(
        r"\b(?:by|using|via|from|according\s+to|applying)\b\s*$",
        "",
        clause,
        flags=re.IGNORECASE,
    ).strip(" ,;:")
    clause = re.sub(r"\b(?:see|cf\.)\b.*$", "", clause, flags=re.IGNORECASE).strip(" ,;:")
    clause = re.sub(r"\s+", " ", clause).strip(" ,;:")
    if not clause or METHOD_REFERENCE_PATTERN.search(clause) or BACKGROUND_REFERENCE_PATTERN.search(clause):
        clause = ""
    if len(re.findall(r"\b\w+\b", clause)) < 4:
        post_statement = build_post_citation_statement(locator_snippet)
        prefix = "Under the current proof hypotheses, "
        if post_statement.startswith(prefix):
            clause = post_statement[len(prefix):].strip()
    clause = sanitize_structured_text_field(clause)
    if not clause:
        return ""
    if len(clause) > 220:
        clause = clause[:220].rstrip(" ,;:")
    if clause and clause[0].islower():
        clause = clause[0].upper() + clause[1:]
    return clause.rstrip(".") + "."


def resolve_anchor_hint(
    llm_anchor: Any = "",
    locator_snippet: str = "",
    pre_citation_focus: str = "",
    usage_reason: Any = "",
    local_context_blocks: Optional[List[str]] = None,
) -> str:
    clause = normalize_proof_clause(pre_citation_focus or extract_pre_citation_focus_snippet(locator_snippet))
    implication_hint = build_implication_statement_from_clause(clause)
    post_statement = build_post_citation_statement(locator_snippet)
    heuristic_anchor = build_anchor_hint_from_evidence(locator_snippet, pre_citation_focus=pre_citation_focus)
    reason_anchor = build_anchor_hint_from_reason(usage_reason)
    local_context_hint = build_anchor_hint_from_local_context(local_context_blocks)
    return select_anchor_hint(
        llm_anchor,
        heuristic_anchor,
        implication_hint,
        post_statement,
        clause,
        local_context_hint,
        reason_anchor,
    )


def infer_tool_type_from_statement(statement: str) -> str:
    lowered = normalize_signature_text(statement).lower()
    if not lowered:
        return "other"
    if "criterion" in lowered:
        return "criterion"
    if lowered.startswith("if ") or lowered.startswith("for all ") or lowered.startswith("for every ") or lowered.startswith("there exists "):
        return "theorem"
    if any(token in lowered for token in ["inequality", "estimate", "bound"]):
        return "inequality"
    if any(token in statement for token in [r"\le", r"\ge", "≤", "≥"]) or (
        "=" in statement and not lowered.startswith("under the current proof hypotheses")
    ):
        return "inequality"
    return "theorem"


ALLOWED_REFERENCE_TOOL_TYPES = {
    "theorem",
    "lemma",
    "proposition",
    "corollary",
    "claim",
    "inequality",
    "criterion",
    "other",
}


def finalize_reference_tool_type(tool_type: Any, statement: Any) -> str:
    cleaned = normalize_signature_text(tool_type).lower()
    if cleaned in ALLOWED_REFERENCE_TOOL_TYPES:
        return cleaned
    return infer_tool_type_from_statement(str(statement or ""))


def heuristic_tool_payload_from_evidence(
    locator_snippet: str,
    pre_citation_focus: str = "",
) -> Optional[Dict[str, Any]]:
    clause = normalize_proof_clause(pre_citation_focus or extract_pre_citation_focus_snippet(locator_snippet))
    statement = build_implication_statement_from_clause(clause)
    if not statement:
        statement = build_cross_citation_implication(locator_snippet, pre_citation_focus=pre_citation_focus)
    if not statement:
        statement = build_post_citation_statement(locator_snippet)
    if not has_meaningful_tool_statement(statement):
        return None
    return {
        "reference_tool_latex": statement,
        "reference_tool_type": infer_tool_type_from_statement(statement),
        "restated_in_citing_paper": False,
    }


def is_target_theorem_env_name(env_name: str) -> bool:
    cleaned = str(env_name or "").strip().lower().replace("*", "")
    if not cleaned:
        return False
    strong_names = {
        "theorem",
        "thm",
        "maintheorem",
        "mainthm",
        "mainresult",
        "proposition",
        "prop",
        "corollary",
        "result",
    }
    if cleaned in strong_names:
        return True
    return any(token in cleaned for token in ("theorem", "thm", "proposition", "prop", "corollary"))


def count_keyword_hits(text: str, keywords: List[str]) -> int:
    lowered = str(text or "").lower()
    return sum(1 for keyword in keywords if keyword and keyword in lowered)


def get_target_candidate_source_priority(source: str) -> float:
    lowered = str(source or "").lower()
    if lowered.startswith("env:maintheorem") or lowered.startswith("env:mainthm") or lowered.startswith("env:mainresult"):
        return 3.5
    if lowered.startswith("env:theorem") or lowered.startswith("env:thm"):
        return 3.0
    if lowered == "text_heading":
        return 2.0
    if lowered.startswith("env:proposition") or lowered.startswith("env:prop"):
        return 1.4
    if lowered.startswith("env:corollary"):
        return 1.1
    if lowered.startswith("keyword_window_"):
        return 0.6
    return 0.8


def count_citations_in_text(text: str) -> int:
    return len(LATEX_CITE_PATTERN.findall(str(text or "")))


def proof_text_is_usable_for_stage2(proof_text: str) -> bool:
    text = str(proof_text or "")
    proof_chars = len(text)
    if proof_chars >= 500:
        return True
    if proof_chars >= 250 and count_citations_in_text(text) > 0:
        return True
    return False


def extract_latex_labels(text: str, max_labels: int = 6) -> List[str]:
    labels: List[str] = []
    seen: set[str] = set()
    for match in LATEX_LABEL_PATTERN.finditer(str(text or "")):
        label = str(match.group(1) or "").strip()
        if not label or label in seen:
            continue
        seen.add(label)
        labels.append(label)
        if len(labels) >= max(1, int(max_labels)):
            break
    return labels


def iter_label_reference_spans(text: str, label: str, start_pos: int = 0) -> List[Tuple[int, int, str]]:
    target = str(label or "").strip()
    if not text or not target:
        return []
    matches: List[Tuple[int, int, str]] = []
    for match in LATEX_REF_COMMAND_PATTERN.finditer(text, max(0, int(start_pos))):
        raw_labels = str(match.group(1) or "")
        pieces = [piece.strip() for piece in raw_labels.split(",") if piece.strip()]
        if target not in pieces:
            continue
        matches.append((match.start(), match.end(), match.group(0)))
    return matches


def find_enclosing_proof_span(full_text: str, position: int) -> Tuple[Optional[Tuple[int, int]], Dict[str, Any]]:
    text = str(full_text or "")
    if not text:
        return None, {"reason": "empty_text"}

    pos = max(0, min(len(text), int(position)))
    search_start = max(0, pos - 50000)
    proof_starts = list(PROOF_START_PATTERN.finditer(text, search_start, pos + 1))
    if not proof_starts:
        return None, {"reason": "no_proof_start_before_position", "position": pos}

    for proof_match in reversed(proof_starts[-16:]):
        proof_start = proof_match.start()
        proof_search_from = proof_match.end()
        proof_end_match = PROOF_END_PATTERN.search(text, proof_search_from, min(len(text), proof_start + 160000))
        if proof_end_match and proof_end_match.end() >= pos:
            return (proof_start, proof_end_match.end()), {
                "reason": "ok",
                "proof_start": proof_start,
                "proof_end": proof_end_match.end(),
                "end_strategy": "enclosing_explicit_proof_end",
            }
        boundary_match = PROOF_FALLBACK_BOUNDARY_PATTERN.search(
            text,
            min(len(text), proof_search_from + 600),
            min(len(text), proof_start + 80000),
        )
        if boundary_match and boundary_match.start() >= pos:
            return (proof_start, boundary_match.start()), {
                "reason": "ok",
                "proof_start": proof_start,
                "proof_end": boundary_match.start(),
                "end_strategy": "enclosing_next_boundary",
            }

    return None, {"reason": "no_enclosing_proof_span", "position": pos}


def rank_proof_candidate_key(item: Dict[str, Any]) -> Tuple[int, int, int, float, int, int, int]:
    return (
        int(bool(item.get("target_conclusion_hit"))),
        int(item.get("target_ref_hits", 0)),
        int(bool(item.get("target_proof_cue_hit"))),
        float(item.get("strategy_priority", 0.0)),
        int(item.get("proof_cite_hits", 0)),
        int(item.get("proof_chars", 0)),
        -int(item.get("distance_from_anchor", 0)),
    )


def find_recent_structural_boundary(full_text: str, anchor_end: int, ref_start: int) -> int:
    text = str(full_text or "")
    if not text:
        return max(0, int(anchor_end))

    left = max(0, int(anchor_end))
    right = max(left, min(len(text), int(ref_start)))
    if right <= left:
        return left

    search_start = max(left, right - 24000)
    last_proof_end: Optional[int] = None
    for match in PROOF_END_PATTERN.finditer(text, search_start, right):
        last_proof_end = match.end()
    if last_proof_end is not None:
        return max(left, last_proof_end)

    last_boundary: Optional[int] = None
    for pattern in (PROOF_FALLBACK_BOUNDARY_PATTERN,):
        for match in pattern.finditer(text, search_start, right):
            candidate = match.start()
            if candidate <= left:
                continue
            if last_boundary is None or candidate > last_boundary:
                last_boundary = candidate
    if last_boundary is not None:
        return max(left, last_boundary)

    return max(left, right - 12000)


def summarize_proof_candidates(candidates: List[Dict[str, Any]], limit: int = 4) -> List[Dict[str, Any]]:
    summary: List[Dict[str, Any]] = []
    for item in (candidates or [])[: max(0, int(limit))]:
        summary.append(
            {
                "strategy": item.get("strategy"),
                "proof_start": item.get("proof_start"),
                "proof_end": item.get("proof_end"),
                "proof_chars": item.get("proof_chars"),
                "proof_cite_hits": item.get("proof_cite_hits"),
                "target_ref_hits": item.get("target_ref_hits"),
                "target_proof_cue_hit": bool(item.get("target_proof_cue_hit")),
                "target_conclusion_hit": bool(item.get("target_conclusion_hit")),
                "distance_from_anchor": item.get("distance_from_anchor"),
            }
        )
    return summary


def collect_targeted_proof_candidates(
    full_text: str,
    anchor_start: int,
    anchor_end: int,
    theorem_labels: List[str],
) -> List[Dict[str, Any]]:
    text = str(full_text or "")
    if not text:
        return []

    candidate_map: Dict[Tuple[int, int], Dict[str, Any]] = {}

    def add_candidate(
        proof_start: int,
        proof_end: int,
        *,
        strategy: str,
        target_ref_hits: int = 0,
        target_proof_cue_hit: bool = False,
        target_conclusion_hit: bool = False,
        strategy_priority: float = 0.0,
    ) -> None:
        proof_start = max(0, int(proof_start))
        proof_end = min(len(text), int(proof_end))
        if proof_end <= proof_start:
            return
        proof_text = text[proof_start:proof_end]
        key = (proof_start, proof_end)
        merged = candidate_map.get(key)
        if merged is None:
            candidate_map[key] = {
                "strategy": strategy,
                "strategy_priority": float(strategy_priority),
                "proof_start": proof_start,
                "proof_end": proof_end,
                "proof_chars": len(proof_text),
                "proof_cite_hits": count_citations_in_text(proof_text),
                "target_ref_hits": int(target_ref_hits),
                "target_proof_cue_hit": bool(target_proof_cue_hit),
                "target_conclusion_hit": bool(target_conclusion_hit),
                "distance_from_anchor": max(0, proof_start - int(anchor_end)),
            }
            return
        merged["target_ref_hits"] = max(int(merged.get("target_ref_hits", 0)), int(target_ref_hits))
        merged["target_proof_cue_hit"] = bool(merged.get("target_proof_cue_hit")) or bool(target_proof_cue_hit)
        merged["target_conclusion_hit"] = bool(merged.get("target_conclusion_hit")) or bool(target_conclusion_hit)
        if float(strategy_priority) > float(merged.get("strategy_priority", 0.0)):
            merged["strategy_priority"] = float(strategy_priority)
            merged["strategy"] = strategy

    immediate_span, immediate_meta = locate_following_proof_span(text, anchor_end)
    if immediate_span:
        add_candidate(
            immediate_span[0],
            immediate_span[1],
            strategy="immediate_following_proof",
            strategy_priority=0.5,
        )

    for label in theorem_labels or []:
        ref_spans = iter_label_reference_spans(text, label, start_pos=anchor_end)
        for ref_start, ref_end, _ in ref_spans:
            context_start = max(0, ref_start - 220)
            context_end = min(len(text), ref_end + 220)
            context = text[context_start:context_end]
            cue_hit = TARGET_PROOF_CUE_PATTERN.search(context) is not None
            conclusion_hit = TARGET_PROOF_CONCLUSION_PATTERN.search(context) is not None

            enclosing_span, enclosing_meta = find_enclosing_proof_span(text, ref_start)
            if enclosing_span:
                add_candidate(
                    enclosing_span[0],
                    enclosing_span[1],
                    strategy="label_ref_enclosing_proof",
                    target_ref_hits=1,
                    target_proof_cue_hit=cue_hit,
                    target_conclusion_hit=conclusion_hit,
                    strategy_priority=2.0 if cue_hit else 1.4,
                )

            if cue_hit or conclusion_hit:
                if conclusion_hit:
                    window_start = find_recent_structural_boundary(text, anchor_end, ref_start)
                    strategy = "anchor_to_target_conclusion"
                    priority = 2.6
                else:
                    window_start = max(0, ref_start - 12000)
                    strategy = "target_ref_window"
                    priority = 1.7
                window_end = min(len(text), ref_end + 1600)
                add_candidate(
                    window_start,
                    window_end,
                    strategy=strategy,
                    target_ref_hits=1,
                    target_proof_cue_hit=cue_hit,
                    target_conclusion_hit=conclusion_hit,
                    strategy_priority=priority,
                )

    ranked = sorted(candidate_map.values(), key=rank_proof_candidate_key, reverse=True)
    return ranked


def summarize_target_candidates(candidates: List[Dict[str, Any]], limit: int = 3) -> List[Dict[str, Any]]:
    summary: List[Dict[str, Any]] = []
    for item in (candidates or [])[: max(0, int(limit))]:
        summary.append(
            {
                "source": item.get("source"),
                "score": item.get("score"),
                "source_priority": item.get("source_priority"),
                "keyword_hits": item.get("keyword_hits"),
                "proof_cite_hits": item.get("proof_cite_hits"),
                "proof_target_ref_hits": item.get("proof_target_ref_hits"),
                "proof_target_conclusion_hit": bool(item.get("proof_target_conclusion_hit")),
                "proof_strategy": item.get("proof_strategy"),
                "has_proof_span": bool(item.get("proof_span")),
                "anchor_start": item.get("anchor_start"),
            }
        )
    return summary


def score_target_block(
    block_text: str,
    target_hint: str,
    match_tokens: List[str],
    has_following_proof: bool,
    source: str = "",
    proof_cite_hits: int = 0,
) -> float:
    token_hits = count_keyword_hits(block_text, match_tokens)
    similarity = text_similarity(target_hint[:2000], str(block_text or "")[:3000]) if target_hint else 0.0
    source_priority = get_target_candidate_source_priority(source)
    score = token_hits * 1.2 + similarity * 4.0 + source_priority
    if has_following_proof:
        score += 1.25
    if proof_cite_hits:
        score += min(1.5, 0.2 * int(proof_cite_hits))
    return round(score, 4)


def locate_following_proof_span(full_text: str, anchor_end: int) -> Tuple[Optional[Tuple[int, int]], Dict[str, Any]]:
    text = str(full_text or "")
    if not text:
        return None, {"reason": "empty_text"}

    lookahead_end = min(len(text), max(0, int(anchor_end)) + 40000)
    proof_match = PROOF_START_PATTERN.search(text, max(0, int(anchor_end)), lookahead_end)
    if not proof_match:
        return None, {"reason": "proof_start_not_found", "anchor_end": int(anchor_end)}

    proof_start = proof_match.start()
    proof_search_from = proof_match.end()
    proof_end_match = PROOF_END_PATTERN.search(text, proof_search_from, min(len(text), proof_start + 120000))
    if proof_end_match:
        proof_end = proof_end_match.end()
        return (proof_start, proof_end), {
            "reason": "ok",
            "proof_start": proof_start,
            "proof_end": proof_end,
            "end_strategy": "explicit_proof_end",
        }

    boundary_match = PROOF_FALLBACK_BOUNDARY_PATTERN.search(
        text,
        min(len(text), proof_search_from + 600),
        min(len(text), proof_start + 50000),
    )
    if boundary_match:
        proof_end = boundary_match.start()
        return (proof_start, proof_end), {
            "reason": "ok",
            "proof_start": proof_start,
            "proof_end": proof_end,
            "end_strategy": "next_boundary",
        }

    proof_end = min(len(text), proof_start + 20000)
    return (proof_start, proof_end), {
        "reason": "ok",
        "proof_start": proof_start,
        "proof_end": proof_end,
        "end_strategy": "fixed_window",
    }


def recover_main_proof_text_from_target(full_text: str, target_hint: str) -> Tuple[str, Dict[str, Any]]:
    text = str(full_text or "")
    hint = str(target_hint or "").strip()
    if not text:
        return "", {"reason": "empty_text"}
    if not hint:
        return "", {"reason": "missing_target_hint"}

    match_tokens = extract_target_match_tokens(hint, max_tokens=20)
    candidates: List[Dict[str, Any]] = []
    light_candidates: List[Dict[str, Any]] = []
    max_anchor_candidates = max(1, int(CONFIG.get("TARGET_PROOF_RECOVERY_MAX_ANCHORS", 12)))

    def add_light_candidate(
        *,
        source: str,
        anchor_start: int,
        anchor_end: int,
        block_text: str,
        theorem_labels: Optional[List[str]] = None,
    ) -> None:
        keyword_hits = count_keyword_hits(block_text, match_tokens)
        similarity = text_similarity(hint[:2000], str(block_text or "")[:3000]) if hint else 0.0
        source_priority = get_target_candidate_source_priority(source)
        light_candidates.append(
            {
                "source": source,
                "anchor_start": int(anchor_start),
                "anchor_end": int(anchor_end),
                "block_text": block_text,
                "theorem_labels": list(theorem_labels or []),
                "keyword_hits": keyword_hits,
                "similarity": similarity,
                "source_priority": source_priority,
                "light_score": round(keyword_hits * 1.2 + similarity * 4.0 + source_priority, 4),
            }
        )

    for match in THEOREM_ENV_BEGIN_PATTERN.finditer(text):
        env_name = str(match.group("env") or "")
        if not is_target_theorem_env_name(env_name):
            continue
        end_tag = f"\\end{{{env_name}}}"
        block_end = text.find(end_tag, match.end())
        if block_end >= 0:
            block_end += len(end_tag)
        else:
            block_end = min(len(text), match.end() + 6000)
        block_text = text[match.start():block_end]
        theorem_labels = extract_latex_labels(block_text)
        source = f"env:{env_name}"
        add_light_candidate(
            source=source,
            anchor_start=match.start(),
            anchor_end=block_end,
            block_text=block_text,
            theorem_labels=theorem_labels,
        )

    for match in TEXTUAL_TARGET_HEADING_PATTERN.finditer(text):
        anchor_start = max(0, match.start())
        anchor_end = min(len(text), match.end() + 3000)
        block_text = text[anchor_start:anchor_end]
        add_light_candidate(
            source="text_heading",
            anchor_start=anchor_start,
            anchor_end=anchor_end,
            block_text=block_text,
            theorem_labels=[],
        )

    if light_candidates:
        ranked_light = sorted(
            light_candidates,
            key=lambda item: (
                int(item.get("keyword_hits", 0) > 0),
                float(item.get("light_score", 0.0)),
                float(item.get("source_priority", 0.0)),
                -int(item.get("anchor_start", 0)),
            ),
            reverse=True,
        )
        selected_light = ranked_light[:max_anchor_candidates]
        for seed in selected_light:
            proof_candidates = collect_targeted_proof_candidates(
                text,
                int(seed["anchor_start"]),
                int(seed["anchor_end"]),
                list(seed.get("theorem_labels") or []),
            )
            best_proof = proof_candidates[0] if proof_candidates else None
            proof_span = (
                (int(best_proof["proof_start"]), int(best_proof["proof_end"]))
                if best_proof
                else None
            )
            proof_cite_hits = int(best_proof.get("proof_cite_hits", 0)) if best_proof else 0
            candidates.append(
                {
                    "source": seed["source"],
                    "anchor_start": int(seed["anchor_start"]),
                    "anchor_end": int(seed["anchor_end"]),
                    "block_text": str(seed["block_text"]),
                    "proof_span": proof_span,
                    "proof_meta": {
                        "reason": "ok" if best_proof else "proof_not_found",
                        "theorem_labels": list(seed.get("theorem_labels") or []),
                        "proof_strategy": best_proof.get("strategy") if best_proof else "",
                        "proof_candidates": summarize_proof_candidates(proof_candidates),
                    },
                    "score": score_target_block(
                        str(seed["block_text"]),
                        hint,
                        match_tokens,
                        proof_span is not None,
                        source=str(seed["source"]),
                        proof_cite_hits=proof_cite_hits,
                    ),
                    "keyword_hits": int(seed.get("keyword_hits", 0)),
                    "source_priority": float(seed.get("source_priority", 0.0)),
                    "proof_cite_hits": proof_cite_hits,
                    "proof_target_ref_hits": int(best_proof.get("target_ref_hits", 0)) if best_proof else 0,
                    "proof_target_proof_cue_hit": bool(best_proof.get("target_proof_cue_hit")) if best_proof else False,
                    "proof_target_conclusion_hit": bool(best_proof.get("target_conclusion_hit")) if best_proof else False,
                    "proof_strategy": best_proof.get("strategy") if best_proof else "",
                }
            )

    if not candidates and match_tokens:
        keyword_windows = collect_keyword_windows(text, match_tokens, radius=1800, max_hits=10)
        for idx, (start, end) in enumerate(merge_intervals(keyword_windows)):
            anchor_end = min(len(text), end + 1600)
            block_text = text[start:anchor_end]
            proof_candidates = collect_targeted_proof_candidates(text, start, anchor_end, [])
            best_proof = proof_candidates[0] if proof_candidates else None
            proof_span = (
                (int(best_proof["proof_start"]), int(best_proof["proof_end"]))
                if best_proof
                else None
            )
            proof_text = text[proof_span[0]:proof_span[1]] if proof_span else ""
            proof_cite_hits = int(best_proof.get("proof_cite_hits", 0)) if best_proof else 0
            source = f"keyword_window_{idx + 1}"
            candidates.append(
                {
                    "source": source,
                    "anchor_start": start,
                    "anchor_end": anchor_end,
                    "block_text": block_text,
                    "proof_span": proof_span,
                    "proof_meta": {
                        "reason": "ok" if best_proof else "proof_not_found",
                        "proof_strategy": best_proof.get("strategy") if best_proof else "",
                        "proof_candidates": summarize_proof_candidates(proof_candidates),
                    },
                    "score": score_target_block(
                        block_text,
                        hint,
                        match_tokens,
                        proof_span is not None,
                        source=source,
                        proof_cite_hits=proof_cite_hits,
                    ),
                    "keyword_hits": count_keyword_hits(block_text, match_tokens),
                    "source_priority": get_target_candidate_source_priority(source),
                    "proof_cite_hits": proof_cite_hits,
                    "proof_target_ref_hits": int(best_proof.get("target_ref_hits", 0)) if best_proof else 0,
                    "proof_target_proof_cue_hit": bool(best_proof.get("target_proof_cue_hit")) if best_proof else False,
                    "proof_target_conclusion_hit": bool(best_proof.get("target_conclusion_hit")) if best_proof else False,
                    "proof_strategy": best_proof.get("strategy") if best_proof else "",
                }
            )

    if not candidates:
        return "", {"reason": "no_target_candidates"}

    ranked = sorted(
        candidates,
        key=lambda item: (
            item["proof_span"] is not None,
            bool(item.get("proof_target_conclusion_hit")),
            int(item.get("proof_target_ref_hits", 0)),
            bool(item.get("proof_target_proof_cue_hit")),
            float(item["score"]),
            float(item.get("source_priority", 0.0)),
            int(item.get("proof_cite_hits", 0)),
            int(item["keyword_hits"]),
            -int(item["anchor_start"]),
        ),
        reverse=True,
    )
    best = ranked[0]
    selection_rule = "ranked_best"
    if best.get("proof_span"):
        for alt in ranked[1:5]:
            if not alt.get("proof_span"):
                continue
            if float(alt.get("source_priority", 0.0)) <= float(best.get("source_priority", 0.0)):
                continue
            if float(alt.get("score", 0.0)) + 1.25 < float(best.get("score", 0.0)):
                continue
            if int(best.get("proof_cite_hits", 0)) > 0 and int(alt.get("proof_cite_hits", 0)) + 1 < int(best.get("proof_cite_hits", 0)):
                continue
            best = alt
            selection_rule = "close_score_higher_priority"
            break
    if best.get("proof_span") and int(best.get("proof_cite_hits", 0)) == 0:
        for alt in ranked[1:8]:
            if not alt.get("proof_span"):
                continue
            alt_cite_hits = int(alt.get("proof_cite_hits", 0))
            best_cite_hits = int(best.get("proof_cite_hits", 0))
            if (
                bool(best.get("proof_target_conclusion_hit"))
                and not bool(alt.get("proof_target_conclusion_hit"))
                and alt_cite_hits < max(3, best_cite_hits + 2)
            ):
                continue
            if alt_cite_hits <= 0:
                continue
            if (
                float(alt.get("source_priority", 0.0)) + 1.5 < float(best.get("source_priority", 0.0))
                and alt_cite_hits < 3
            ):
                continue
            if int(alt.get("proof_target_ref_hits", 0)) < int(best.get("proof_target_ref_hits", 0)):
                continue
            if float(alt.get("score", 0.0)) + 0.75 < float(best.get("score", 0.0)):
                continue
            best = alt
            selection_rule = "prefer_citation_bearing_proof"
            break
    if best.get("proof_span") and int(best.get("proof_cite_hits", 0)) <= 2:
        for alt in ranked[1:8]:
            if not alt.get("proof_span"):
                continue
            alt_cite_hits = int(alt.get("proof_cite_hits", 0))
            best_cite_hits = int(best.get("proof_cite_hits", 0))
            if (
                bool(best.get("proof_target_conclusion_hit"))
                and not bool(alt.get("proof_target_conclusion_hit"))
                and alt_cite_hits < max(best_cite_hits + 3, 4)
            ):
                continue
            if (
                float(alt.get("source_priority", 0.0)) + 1.5 < float(best.get("source_priority", 0.0))
                and alt_cite_hits < best_cite_hits + 3
            ):
                continue
            if int(alt.get("proof_target_ref_hits", 0)) < int(best.get("proof_target_ref_hits", 0)):
                continue
            if alt_cite_hits < best_cite_hits + 2:
                continue
            alt_score = float(alt.get("score", 0.0))
            best_score = float(best.get("score", 0.0))
            alt_keywords = int(alt.get("keyword_hits", 0))
            best_keywords = int(best.get("keyword_hits", 0))
            if alt_score + 1.0 < best_score and alt_keywords < best_keywords + 4:
                continue
            best = alt
            selection_rule = "prefer_richer_proof_scope"
            break
    if not best.get("proof_span"):
        return "", {
            "reason": "no_following_proof",
            "best_source": best.get("source"),
            "best_score": best.get("score"),
            "best_source_priority": best.get("source_priority"),
            "best_keyword_hits": best.get("keyword_hits"),
            "best_proof_cite_hits": best.get("proof_cite_hits"),
            "selection_rule": selection_rule,
            "top_candidates": summarize_target_candidates(ranked),
        }

    proof_start, proof_end = best["proof_span"]
    if proof_end <= proof_start:
        return "", {
            "reason": "invalid_recovered_span",
            "best_source": best.get("source"),
            "proof_start": proof_start,
            "proof_end": proof_end,
        }

    proof_text = text[proof_start:proof_end]
    return proof_text, {
        "reason": "ok",
        "strategy": "target_theorem_fallback",
        "candidate_source": best.get("source"),
        "candidate_score": best.get("score"),
        "candidate_source_priority": best.get("source_priority"),
        "candidate_keyword_hits": best.get("keyword_hits"),
        "candidate_proof_cite_hits": best.get("proof_cite_hits"),
        "candidate_proof_target_ref_hits": best.get("proof_target_ref_hits"),
        "candidate_proof_target_conclusion_hit": bool(best.get("proof_target_conclusion_hit")),
        "candidate_proof_strategy": best.get("proof_strategy"),
        "selection_rule": selection_rule,
        "top_candidates": summarize_target_candidates(ranked),
        "anchor_start": best.get("anchor_start"),
        "anchor_end": best.get("anchor_end"),
        "proof_start": proof_start,
        "proof_end": proof_end,
        "proof_chars": len(proof_text),
        "end_strategy": (best.get("proof_meta") or {}).get("end_strategy"),
    }


def recover_target_theorem_text_from_hint(full_text: str, target_hint: str) -> Tuple[str, Dict[str, Any]]:
    _, meta = recover_main_proof_text_from_target(full_text, target_hint)
    if not isinstance(meta, dict) or meta.get("reason") != "ok":
        return "", meta if isinstance(meta, dict) else {"reason": "target_theorem_fallback_failed"}
    anchor_start = int(meta.get("anchor_start", -1))
    anchor_end = int(meta.get("anchor_end", -1))
    if anchor_start < 0 or anchor_end <= anchor_start:
        return "", {**meta, "reason": "target_theorem_anchor_not_found"}
    block_text = normalize_signature_text(str(full_text or "")[anchor_start:anchor_end])
    theorem_text = normalize_target_theorem_text(block_text)
    if not theorem_text:
        return "", {**meta, "reason": "target_theorem_empty_after_fallback"}
    return theorem_text, meta


def fast_prefilter_no_proof_citations(
    full_text: str,
    target_hint: str,
    norm_bib: Dict[str, str],
    canonical_bib_keys: Dict[str, str],
    scout: Optional[Dict[str, Any]] = None,
) -> Tuple[bool, Dict[str, Any]]:
    proof_text, proof_meta = recover_main_proof_text_from_target(full_text, target_hint)
    if not proof_text:
        return False, {"reason": "proof_not_recovered", **(proof_meta or {})}

    proof_chars = len(proof_text)
    raw_cite_count = count_citations_in_text(proof_text)
    scoped_citations = scout_proof_citations(proof_text, norm_bib, canonical_bib_keys)
    scoped_count = len(scoped_citations)
    scout_score = float((scout or {}).get("score", 0.0))
    scout_unique_cites = int((scout or {}).get("unique_cite_count", 0))
    source_priority = float((proof_meta or {}).get("candidate_source_priority", 0.0))
    candidate_source = str((proof_meta or {}).get("candidate_source", "") or "")
    decision_meta = {
        "reason": "checked",
        "proof_chars": proof_chars,
        "raw_proof_cite_count": raw_cite_count,
        "scoped_proof_citation_count": scoped_count,
        "scout_score": scout_score,
        "scout_unique_cite_count": scout_unique_cites,
        "candidate_source": candidate_source,
        "candidate_source_priority": source_priority,
        "proof_scope": proof_meta,
    }

    if proof_chars < int(CONFIG["FAST_PREFILTER_MIN_PROOF_CHARS"]):
        return False, {"reason": "proof_too_short", **decision_meta}
    if bool(CONFIG.get("FAST_PREFILTER_RESPECT_SCOUT_SCORE", True)) and scout_score > float(CONFIG["FAST_PREFILTER_MAX_SCOUT_SCORE"]):
        return False, {"reason": "scout_score_too_high_for_fast_skip", **decision_meta}
    if source_priority < float(CONFIG["FAST_PREFILTER_MIN_SOURCE_PRIORITY"]):
        return False, {"reason": "proof_source_too_weak_for_fast_skip", **decision_meta}
    if not bool(CONFIG.get("FAST_PREFILTER_REQUIRE_ZERO_PROOF_CITES", True)):
        return False, decision_meta
    if raw_cite_count == 0 and scoped_count == 0:
        alternative_candidates = list((proof_meta or {}).get("top_candidates") or [])[1:]
        if any(int(item.get("proof_cite_hits", 0)) > 0 for item in alternative_candidates):
            return False, {"reason": "alternate_candidate_has_citations", **decision_meta}
        return True, decision_meta
    return False, decision_meta


def score_citation_candidate(citation_info: Dict[str, Any], bib_text: str) -> float:
    locator = normalize_signature_text(citation_info.get("locator_snippet"))
    reason = normalize_signature_text(citation_info.get("reason"))
    haystack = f"{locator} {reason}".lower()
    scout_score = float(citation_info.get("_local_score", 0.0) or 0.0)
    local_score, _ = score_local_citation_context(locator or haystack)
    score = max(scout_score, local_score)
    if re.search(r"\b(?:according to|it follows from|thanks to|ensures?|shows?|implies?|yields?|gives?|provides?)\b", haystack):
        score += 1.0
    if RESULT_FLOW_PATTERN.search(haystack):
        score += 1.0
    if re.search(
        r"\\cite[a-zA-Z*]*\[[^\]]*(?:Theorem|Lemma|Proposition|Corollary|Claim|Criterion|Inequality|Estimate|Bound)",
        locator,
        re.IGNORECASE,
    ):
        score += 1.0
    if heuristic_tool_payload_from_evidence(locator):
        score += 1.5
    if METHOD_REFERENCE_PATTERN.search(haystack):
        score -= 2.0
    if BACKGROUND_REFERENCE_PATTERN.search(haystack):
        score -= 2.5
    if COMPARATIVE_REFERENCE_PATTERN.search(haystack):
        score -= 1.25
    if REMARK_CONTEXT_PATTERN.search(locator):
        score -= 2.0
    if NON_TOOL_STATEMENT_PATTERN.search(haystack):
        score -= 2.5
    if re.search(r"\b(?:background|historical|survey|overview|definition|notation)\b", haystack):
        score -= 0.75
    if has_bibliography_metadata(bib_text):
        score += 0.5
    if extract_doi(bib_text) or extract_arxiv_id(bib_text):
        score += 0.5
    return score


def build_locator_snippet(text: str, start_idx: int, end_idx: int, radius: int = 180) -> str:
    left = max(0, int(start_idx) - int(radius))
    right = min(len(str(text or "")), int(end_idx) + int(radius))
    left, right = align_span_to_text_boundaries(str(text or ""), left, right, max_adjust=48)
    snippet = normalize_signature_text(str(text or "")[left:right])
    if len(snippet) > 420:
        snippet = snippet[:420].rstrip()
    return snippet


def score_local_citation_context(context: str) -> Tuple[float, List[str]]:
    lowered = str(context or "").lower()
    reasons: List[str] = []
    score = 0.0
    if TOOL_NAME_PATTERN.search(lowered):
        score += 2.0
        reasons.append("theorem_like")
    if LOCAL_CITATION_TRIGGER_PATTERN.search(lowered):
        score += 1.25
        reasons.append("trigger_word")
    if LOCATOR_LABEL_PATTERN.search(lowered):
        score += 1.0
        reasons.append("explicit_locator")
    if re.search(r"\b(?:obtain|deduce|imply|yields?|gives?)\b", lowered):
        score += 0.75
        reasons.append("result_transition")
    if RESULT_FLOW_PATTERN.search(lowered):
        score += 1.0
        reasons.append("result_flow")
    if METHOD_REFERENCE_PATTERN.search(lowered):
        score -= 2.0
        reasons.append("method_reference")
    if BACKGROUND_REFERENCE_PATTERN.search(lowered):
        score -= 2.5
        reasons.append("background_reference")
    if COMPARATIVE_REFERENCE_PATTERN.search(lowered):
        score -= 1.25
        reasons.append("comparative_reference")
    if REMARK_CONTEXT_PATTERN.search(context):
        score -= 2.0
        reasons.append("remark_context")
    return score, reasons


def scout_proof_citations(
    proof_text: str,
    norm_bib: Dict[str, str],
    canonical_bib_keys: Optional[Dict[str, str]] = None,
) -> List[Dict[str, Any]]:
    text = str(proof_text or "")
    if not text:
        return []

    candidates: List[Dict[str, Any]] = []
    seen: set[Tuple[str, str]] = set()
    for match in LATEX_CITE_PATTERN.finditer(text):
        raw_keys = [normalize_key(part) for part in str(match.group(1) or "").split(",")]
        valid_keys = [key for key in raw_keys if key and key in norm_bib]
        if not valid_keys:
            continue

        context = build_locator_snippet(text, match.start(), match.end(), radius=220)
        local_score, reason_tags = score_local_citation_context(context)
        if local_score <= 0:
            local_score = 0.5
            reason_tags = ["proof_local_citation"]

        for norm_key in valid_keys:
            canonical_key = str((canonical_bib_keys or {}).get(norm_key) or norm_key)
            dedup_key = (norm_key, context)
            if dedup_key in seen:
                continue
            seen.add(dedup_key)
            candidates.append(
                {
                    "citation_key": canonical_key,
                    "locator_snippet": context,
                    "reason": "proof-local citation: " + ", ".join(reason_tags),
                    "_local_score": local_score,
                }
            )

    ranked = sorted(
        candidates,
        key=lambda item: (float(item.get("_local_score", 0.0)), len(str(item.get("locator_snippet") or ""))),
        reverse=True,
    )
    output: List[Dict[str, Any]] = []
    for item in ranked:
        output.append(
            {
                "citation_key": str(item.get("citation_key") or ""),
                "locator_snippet": str(item.get("locator_snippet") or ""),
                "reason": str(item.get("reason") or ""),
                "_local_score": float(item.get("_local_score", 0.0) or 0.0),
            }
        )
    return output


def merge_citation_lists(primary: List[Dict[str, Any]], secondary: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    merged: List[Dict[str, Any]] = []
    index_by_key: Dict[Tuple[str, str], int] = {}
    for source in [primary or [], secondary or []]:
        for item in source:
            key = sanitize_structured_text_field(item.get("citation_key") or "")
            locator = normalize_signature_text(item.get("locator_snippet"))
            if not key or not locator:
                continue
            dedup_key = (normalize_key(key), locator)
            reason = sanitize_structured_text_field(item.get("reason") or "")
            local_score = float(item.get("_local_score", 0.0) or 0.0)
            existing_idx = index_by_key.get(dedup_key)
            if existing_idx is None:
                index_by_key[dedup_key] = len(merged)
                merged.append(
                    {
                        "citation_key": key,
                        "locator_snippet": locator,
                        "reason": reason,
                        "_local_score": local_score,
                    }
                )
                continue
            existing = merged[existing_idx]
            if reason:
                existing_reason = str(existing.get("reason") or "")
                if reason not in existing_reason:
                    existing["reason"] = f"{existing_reason}; {reason}".strip("; ")
            existing["_local_score"] = max(float(existing.get("_local_score", 0.0) or 0.0), local_score)
    return merged


def require_api_key() -> str:
    """Ensure an API key is provided via environment variables."""
    api_key = str(CONFIG.get("API_KEY", "") or "").strip()
    if not api_key:
        raise RuntimeError(
            "Missing API key. Set environment variable OPENAI_API_KEY before running.\n"
            "Example (bash): export OPENAI_API_KEY='...'\n"
            "Do NOT hardcode secrets in source code or commit them to GitHub."
        )
    return api_key


def test_api_connection() -> bool:
    """Quick sanity check for API connectivity."""
    api_key = require_api_key()
    base_url = str(CONFIG["BASE_URL"])
    model_name = str(CONFIG["MODEL_NAME"])

    last_error: Optional[Exception] = None
    for attempt in range(1, 4):
        try:
            client = OpenAI(api_key=api_key, base_url=base_url, timeout=10.0)
            client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": "ping"}],
                max_tokens=5,
            )
            logger.info("API connection OK.")
            return True
        except Exception as e:
            last_error = e
            if attempt < 4:
                wait_seconds = 2 * attempt
                logger.warning(
                    "API connection attempt %d failed: %s. Retrying in %d seconds...",
                    attempt,
                    e,
                    wait_seconds,
                )
                time.sleep(wait_seconds)

    logger.error("API connection failed after retries: %s", last_error)
    return False


# ============================================================
# 1) Extractor (adds bibliographic metadata when possible)
# ============================================================

class ReasoningAwareExtractor(ArxivLatexExtractor):
    """
    Extends ArxivLatexExtractor with bibliography extraction:
    - Prefer .bib entries (title/author/year)
    - Fallback to .bbl/.tex \\bibitem blocks
    """

    def __init__(self):
        super().__init__()
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": "BenchmarkBuilder/1.0 (+https://github.com/)"})

    def _clean_tex_text(self, text: str) -> str:
        if not text:
            return ""
        text = re.sub(r"%.*?$", "", text, flags=re.MULTILINE)
        text = re.sub(r"[\n\r\t]+", " ", text)
        while text.startswith("{") and text.endswith("}"):
            text = text[1:-1]
        while text.startswith('"') and text.endswith('"'):
            text = text[1:-1]
        text = text.replace(r"\'e", "é").replace(r"\'a", "á")
        text = text.replace("{", "").replace("}", "")
        return re.sub(r"\s+", " ", text).strip()

    def _parse_bib_entry_value(self, content: str, start_idx: int) -> Optional[str]:
        """
        Parse a BibTeX field value after "field =".
        Supports {...}, "..." and bare tokens.
        """
        try:
            i = start_idx
            while i < len(content) and content[i].isspace():
                i += 1
            if i >= len(content):
                return None

            delimiter = content[i]
            if delimiter == "{":
                brace_count = 1
                i += 1
                start_read = i
                while i < len(content) and brace_count > 0:
                    if content[i] == "{":
                        brace_count += 1
                    elif content[i] == "}":
                        brace_count -= 1
                    i += 1
                return content[start_read: i - 1]

            if delimiter == '"':
                i += 1
                start_read = i
                while i < len(content):
                    if content[i] == '"' and content[i - 1] != "\\":
                        break
                    i += 1
                return content[start_read:i]

            # bare token
            start_read = i
            while i < len(content) and content[i] not in {",", "}"}:
                i += 1
            return content[start_read:i]

        except Exception:
            return None

    def extract_bib_mapping(self, extract_dir: str) -> Dict[str, str]:
        bib_mapping: Dict[str, str] = {}
        all_files: List[str] = []

        for root, _, files in os.walk(extract_dir):
            for fname in files:
                if fname.endswith((".bib", ".bbl", ".tex")):
                    all_files.append(os.path.join(root, fname))

        # Strategy A: .bib files (extract Title/Author/Year)
        bib_files = [p for p in all_files if p.endswith(".bib")]
        for bib_file in bib_files:
            try:
                with open(bib_file, "r", errors="ignore") as f:
                    content = f.read()

                for match in re.finditer(r"@(\w+)\s*\{\s*([^,\s]+)", content):
                    key = match.group(2).strip()
                    search_scope = content[match.end(): match.end() + 2500]

                    title, author, year = "", "", ""

                    m_t = re.search(r"\btitle\s*=\s*", search_scope, re.IGNORECASE)
                    if m_t:
                        raw = self._parse_bib_entry_value(search_scope, m_t.end())
                        if raw:
                            title = self._clean_tex_text(raw)

                    m_a = re.search(r"\bauthor\s*=\s*", search_scope, re.IGNORECASE)
                    if m_a:
                        raw = self._parse_bib_entry_value(search_scope, m_a.end())
                        if raw:
                            author = self._clean_tex_text(raw)

                    m_y = re.search(r"\byear\s*=\s*", search_scope, re.IGNORECASE)
                    if m_y:
                        raw = self._parse_bib_entry_value(search_scope, m_y.end())
                        if raw:
                            year = self._clean_tex_text(raw)

                    if title:
                        meta_parts = []
                        if author:
                            meta_parts.append(f"Authors: {author}")
                        if year:
                            meta_parts.append(f"Year: {year}")
                        full_info = title + (f" ({'; '.join(meta_parts)})" if meta_parts else "")
                        bib_mapping[key] = full_info

            except Exception as e:
                logger.warning("Error parsing bib file %s: %s", bib_file, e)

        # Strategy B: .bbl/.tex fallback (\\bibitem blocks)
        other_files = [p for p in all_files if p.endswith((".bbl", ".tex"))]
        for fpath in other_files:
            try:
                with open(fpath, "r", errors="ignore") as f:
                    content = f.read()

                items = re.findall(
                    r"\\bibitem(?:\[[^\]]*\])?\{([^}]+)\}(.*?)(?=\\bibitem|\\end\{thebibliography\}|$)",
                    content,
                    re.DOTALL,
                )
                for key, text in items:
                    clean_key = key.strip()
                    if clean_key not in bib_mapping:
                        clean_text = self._clean_tex_text(text)
                        if len(clean_text) > 10:
                            bib_mapping[clean_key] = clean_text[:800]
            except Exception:
                pass

        return bib_mapping

    def process_paper_with_bib(
        self, paper_id: str, latex_link: str
    ) -> Tuple[bool, str, Dict[str, str]]:
        """
        Download source, extract, build bib mapping, and concatenate extracted TeX content.
        """
        try:
            temp_root = os.path.join(tempfile.gettempdir(), "benchmark_builder_tmp")
            os.makedirs(temp_root, exist_ok=True)

            with tempfile.TemporaryDirectory(dir=temp_root) as temp_work_dir:
                download_dir = os.path.join(temp_work_dir, "download")
                extract_dir = os.path.join(temp_work_dir, "extract")
                os.makedirs(download_dir, exist_ok=True)
                os.makedirs(extract_dir, exist_ok=True)

                logger.info("Downloading LaTeX source for %s ...", paper_id)
                archive_path = self.download_latex_source(latex_link, download_dir)
                if not archive_path:
                    logger.error("Download failed for %s.", paper_id)
                    return False, "", {}

                logger.info("Extracting archive for %s ...", paper_id)
                if not self.extract_archive(archive_path, extract_dir):
                    logger.error("Extraction failed for %s.", paper_id)
                    return False, "", {}

                bib_mapping = self.extract_bib_mapping(extract_dir)

                ordered_tex_files = self.determine_tex_file_order(extract_dir)
                if not ordered_tex_files:
                    tex_files = self.find_tex_files(extract_dir)
                    if tex_files:
                        ordered_tex_files = tex_files
                    else:
                        logger.warning("No .tex files found for %s.", paper_id)
                        return False, "", {}

                full_text_parts: List[str] = []
                for tex_file in ordered_tex_files:
                    text = self.extract_text_from_tex(tex_file)
                    full_text_parts.append(
                        f"\n% --- File: {os.path.basename(tex_file)} ---\n{text}\n"
                    )

                full_text = "\n".join(full_text_parts)
                return True, full_text, bib_mapping

        except Exception as e:
            logger.error("Error processing %s: %s", paper_id, e)
            return False, "", {}


# ============================================================
# 2) Dataset generator (LLM stages)
# ============================================================

class DatasetGenerator:
    def __init__(self, api_key: str, base_url: str, model_name: str):
        timeout_config = httpx.Timeout(
            float(CONFIG["REQUEST_TIMEOUT_SECONDS"]),
            connect=float(CONFIG["REQUEST_CONNECT_TIMEOUT_SECONDS"]),
        )
        custom_http_client = httpx.Client(timeout=timeout_config)

        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url,
            http_client=custom_http_client,
            max_retries=int(CONFIG["OPENAI_MAX_RETRIES"]),
        )
        self.model_name = model_name

    def is_reasoning_model(self) -> bool:
        return is_reasoning_model_name(self.model_name)

    def structured_max_tokens(self, stage: str, default_tokens: int) -> int:
        if not self.is_reasoning_model():
            return int(default_tokens)
        if stage == "stage1":
            return max(int(default_tokens), int(CONFIG.get("STAGE1_REASONER_MAX_TOKENS", default_tokens)))
        if stage == "stage1_retry":
            return max(int(default_tokens), int(CONFIG.get("STAGE1_REASONER_RETRY_MAX_TOKENS", default_tokens)))
        if stage == "stage2":
            return max(int(default_tokens), int(CONFIG.get("STAGE2_REASONER_MAX_TOKENS", default_tokens)))
        if stage == "stage2_retry":
            return max(int(default_tokens), int(CONFIG.get("STAGE2_REASONER_RETRY_MAX_TOKENS", default_tokens)))
        return int(default_tokens)

    def request_structured_completion(
        self,
        *,
        messages: List[Dict[str, str]],
        max_tokens: int,
        stage: str,
        required_keys: Optional[List[str]] = None,
    ) -> Tuple[str, str, str]:
        request_kwargs: Dict[str, Any] = {
            "model": self.model_name,
            "messages": messages,
            "temperature": 0.0,
            "max_tokens": self.structured_max_tokens(stage, int(max_tokens)),
        }
        if not self.is_reasoning_model():
            request_kwargs["response_format"] = structured_json_response_format()
        response = self.client.chat.completions.create(**request_kwargs)
        choice = response.choices[0]
        message = choice.message
        raw = str(getattr(message, "content", "") or "")
        reasoning = str(getattr(message, "reasoning_content", "") or "")
        finish_reason = str(getattr(choice, "finish_reason", "") or "")
        if not raw.strip() and reasoning:
            reasoning_json = extract_json_candidate_from_text(reasoning, required_keys=required_keys)
            if reasoning_json:
                raw = reasoning_json
        if self.is_reasoning_model() and (not raw.strip() or finish_reason == "length"):
            logger.info(
                "Structured completion for %s using %s returned finish_reason=%s content_chars=%d reasoning_chars=%d.",
                stage,
                self.model_name,
                finish_reason or "unknown",
                len(raw),
                len(reasoning),
            )
        return raw, reasoning, finish_reason

    def clean_json(self, raw_text: str) -> Optional[Dict]:
        """
        Extract and repair JSON from model output.
        Returns a dict or None.
        """
        try:
            text = re.sub(r"```(?:json)?", "", raw_text)
            text = text.replace("```", "")
            text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()

            start_dict = text.find("{")
            start_list = text.find("[")

            if start_dict != -1 and (start_list == -1 or start_dict < start_list):
                start, end = start_dict, text.rfind("}")
            elif start_list != -1:
                start, end = start_list, text.rfind("]")
            else:
                return None

            if start == -1 or end == -1 or end < start:
                return None

            content = text[start:end + 1]

            parsed = None
            parse_errors: List[str] = []

            if repair_json is not None:
                try:
                    parsed = repair_json(content, return_objects=True)
                except Exception as e:
                    parse_errors.append(f"repair_json:{e}")

            if parsed is None:
                try:
                    parsed = json.loads(content)
                except Exception as e:
                    parse_errors.append(f"json.loads:{e}")

            if parsed is None:
                try:
                    escaped_content = re.sub(r'(?<!\\)\\(?!["\\/bfnrtu])', r"\\\\", content)
                    parsed = json.loads(escaped_content)
                except Exception as e:
                    parse_errors.append(f"escape_backslash_fallback:{e}")

            if parsed is None:
                raise ValueError("; ".join(parse_errors) if parse_errors else "unknown_json_parse_failure")

            if isinstance(parsed, list):
                if parsed and isinstance(parsed[0], dict):
                    return parsed[0]
                return None

            if isinstance(parsed, dict):
                return parsed

            return None

        except Exception as e:
            logger.error("JSON parsing error: %s", e)
            return None

    def repair_structured_output(self, raw_text: str, schema_name: str) -> Optional[Dict]:
        if str(schema_name).startswith("stage1"):
            if not bool(CONFIG.get("STAGE1_STRUCTURED_REPAIR_ENABLED", True)):
                return None
        elif str(schema_name).startswith("stage2"):
            if not bool(CONFIG.get("STAGE2_STRUCTURED_REPAIR_ENABLED", True)):
                return None
        elif not bool(CONFIG.get("LLM_STRUCTURED_REPAIR_ENABLED", True)):
            return None
        schema_map = {
            "stage1": (
                "Repair the malformed output into valid JSON with keys "
                "`global_context`, `proof_span`, and `proof_citations`."
            ),
            "stage1_citations_only": (
                "Repair the malformed output into valid JSON with key "
                "`proof_citations`, where each item has keys `citation_key`, `locator_snippet`, and `reason`."
            ),
            "stage2_context": (
                "Repair the malformed output into valid JSON with keys "
                "`reference_tool_latex`, `reference_tool_type`, and `restated_in_citing_paper`."
            ),
            "stage2_tool": (
                "Repair the malformed output into valid JSON with keys "
                "`reference_tool_latex`, `reference_tool_type`, and `restated_in_citing_paper`."
            ),
            "stage2_single": (
                "Repair the malformed output into valid JSON with keys "
                "`local_context`, `reference_tool_latex`, "
                "`reference_tool_type`, and `restated_in_citing_paper`."
            ),
            "stage2_batch": (
                "Repair the malformed output into valid JSON with top-level key `items`, "
                "where each item has keys `citation_key`, "
                "`reference_tool_latex`, `reference_tool_type`, and `restated_in_citing_paper`."
            ),
        }
        instruction = schema_map.get(schema_name, "Repair the malformed output into valid JSON.")

        try:
            repaired_raw, repaired_reasoning, _ = self.request_structured_completion(
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You repair malformed structured outputs. "
                            "Return valid JSON only. Preserve content when possible. "
                            "Do not invent new mathematical content beyond minimal syntax repair. "
                            + instruction
                        ),
                    },
                    {"role": "user", "content": f"MALFORMED_OUTPUT:\n{str(raw_text or '')[:12000]}"},
                ],
                max_tokens=2000,
                stage="stage2_retry",
                required_keys=["{"],
            )
            parsed = self.clean_json(repaired_raw)
            if not parsed and repaired_reasoning:
                parsed = self.clean_json(repaired_reasoning)
            if not parsed:
                logger.debug("Structured repair failed for %s. Raw head=%r", schema_name, repaired_raw[:400])
            return parsed
        except Exception as e:
            logger.error("Structured repair error for %s: %s", schema_name, e)
            return None

    # --------------------------
    # Stage 1
    # --------------------------
    def stage1_analyze_structure(self, full_text: str, bib_mapping: Dict[str, str]) -> Optional[Dict]:
        bib_context = build_stage1_bib_context(bib_mapping)
        paper_view = str(full_text or "")[: int(CONFIG["STAGE1_MAX_CHARS"])]
        paper_view_meta = {"total_chars": len(paper_view), "excerpt_count": 0, "head_chars": len(paper_view)}
        if bool(CONFIG.get("STAGE1_COMPACT_VIEW_ENABLED", True)):
            paper_view, paper_view_meta = build_stage1_paper_view(full_text)

        system_prompt = r"""
You extract one benchmark instance from one mathematical paper.
Return JSON only. Do not add any prose outside JSON.

Required fields:
- global_context.setup: concise setup needed for the paper's main theorem.
- global_context.target_theorem: the paper's main theorem only; if unsure return "".
- proof_span.start_snippet / end_snippet: verbatim 20-40 word snippets from the proof of the target theorem; if unsure return "".
- proof_citations: citations used as external mathematical tools inside that proof only.

Rules:
- Use citation keys from BIB_MAPPING only.
- Exclude citations used only for background, setup, history, or definitions.
- Prefer empty strings / empty list over guessing.
- Keep output compact: setup <= 1200 chars, target_theorem <= 1200 chars, max 6 proof_citations.
- Each proof_citation must contain citation_key, locator_snippet, and a short reason.

Return exactly:
{
  "global_context": {"setup": "...", "target_theorem": "..."},
  "proof_span": {"start_snippet": "...", "end_snippet": "..."},
  "proof_citations": [
    {"citation_key": "...", "locator_snippet": "...", "reason": "..."}
  ]
}
"""
        reasoner_retry_prompt = r"""
Return one compact JSON object only.
No reasoning, no commentary, no markdown.

Need:
- global_context.setup: concise setup for the main theorem.
- global_context.target_theorem: main theorem only; else "".
- proof_span.start_snippet / end_snippet: verbatim proof snippets; else "".
- proof_citations: max 4 citations from the proof of the target theorem only.

Prefer empty fields over guessing.
"""

        try:
            logger.info("Stage 1: analyzing structure and scouting citations (paper_view_chars=%d excerpts=%d) ...",
                        int(paper_view_meta.get("total_chars", 0)),
                        int(paper_view_meta.get("excerpt_count", 0)))
            messages = [
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": (
                        f"BIBLIOGRAPHY:\n{bib_context}\n\n"
                        f"PAPER_VIEW:\n{paper_view}"
                    ),
                },
            ]
            raw, reasoning, finish_reason = self.request_structured_completion(
                messages=messages,
                max_tokens=int(CONFIG.get("STAGE1_MAX_TOKENS", 3000)),
                stage="stage1",
                required_keys=["global_context", "proof_span", "proof_citations"],
            )
            parsed = self.clean_json(raw)
            if not parsed and reasoning:
                parsed = self.clean_json(reasoning)
            if not parsed:
                parsed = salvage_stage1_payload_from_raw_text(raw or reasoning)
            if not parsed and self.is_reasoning_model() and (finish_reason == "length" or not str(raw or "").strip()):
                logger.info("Stage 1 compact retry triggered for %s.", self.model_name)
                retry_messages = [
                    {"role": "system", "content": reasoner_retry_prompt},
                    {
                        "role": "user",
                        "content": (
                            f"BIBLIOGRAPHY:\n{bib_context}\n\n"
                            f"PAPER_VIEW:\n{paper_view}"
                        ),
                    },
                ]
                raw, reasoning, _ = self.request_structured_completion(
                    messages=retry_messages,
                    max_tokens=int(CONFIG.get("STAGE1_REASONER_RETRY_MAX_TOKENS", 8500)),
                    stage="stage1_retry",
                    required_keys=["global_context", "proof_span", "proof_citations"],
                )
                parsed = self.clean_json(raw)
                if not parsed and reasoning:
                    parsed = self.clean_json(reasoning)
                if not parsed:
                    parsed = salvage_stage1_payload_from_raw_text(raw or reasoning)
            if parsed:
                return parsed
            logger.info("Stage 1 structured repair fallback triggered.")
            return self.repair_structured_output(raw or reasoning, "stage1")

        except Exception as e:
            logger.error("Stage 1 error: %s", e)
            return None

    def stage1_targeted_recall(
        self,
        full_text: str,
        bib_mapping: Dict[str, str],
        target_hint: str,
        setup: str = "",
    ) -> Optional[Dict]:
        bib_context = build_stage1_bib_context(bib_mapping)
        paper_view = str(full_text or "")[: int(CONFIG["STAGE1_MAX_CHARS"])]
        paper_view_meta = {"total_chars": len(paper_view), "excerpt_count": 0, "head_chars": len(paper_view)}
        if bool(CONFIG.get("STAGE1_COMPACT_VIEW_ENABLED", True)):
            paper_view, paper_view_meta = build_stage1_paper_view(full_text, target_hint=target_hint)

        system_prompt = r"""
You are a Mathematical Proof Citation Scout.
Your only job is to recover the proof span of a given target theorem and find
external mathematical tool citations used inside that proof.

INPUT
1) TARGET_HINT: theorem statement when available; otherwise a precise paper-title hint
2) OPTIONAL_SETUP: global setup for disambiguation
3) BIB_MAPPING: available citation keys
4) PAPER_VIEW: a compact dossier of verbatim excerpts from the paper

TASK
- Find the proof of TARGET_HINT in PAPER_VIEW.
- Return start/end snippets copied verbatim from the proof.
- Inside that proof only, recall plausible external-tool citations.
- Favor recall over precision, but discard citations used only for background,
  history, definitions, or non-proof discussion.

For each valid tool usage, output:
- citation_key: exact BibTeX key from BIB_MAPPING
- locator_snippet: unique 20-40 word snippet surrounding the citation
- reason: short explanation of the tool usage

Return JSON only:
{
  "proof_span": {"start_snippet": "...", "end_snippet": "..."},
  "proof_citations": [
    {"citation_key": "...", "locator_snippet": "...", "reason": "..."}
  ]
}
"""

        try:
            logger.info(
                "Stage 1 targeted recall: paper_view_chars=%d excerpts=%d target_hint_keywords=%d",
                int(paper_view_meta.get("total_chars", 0)),
                int(paper_view_meta.get("excerpt_count", 0)),
                len(paper_view_meta.get("target_keywords", []) or []),
            )
            raw, reasoning, _ = self.request_structured_completion(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {
                        "role": "user",
                        "content": (
                            f"TARGET_HINT:\n{target_hint}\n\n"
                            f"OPTIONAL_SETUP:\n{setup[:2000]}\n\n"
                            f"BIBLIOGRAPHY:\n{bib_context}\n\n"
                            f"PAPER_VIEW:\n{paper_view}"
                        ),
                    },
                ],
                max_tokens=2500,
                stage="stage1",
                required_keys=["proof_span", "proof_citations"],
            )
            parsed = self.clean_json(raw)
            if not parsed and reasoning:
                parsed = self.clean_json(reasoning)
            if parsed:
                return parsed
            logger.info("Stage 1 targeted recall structured repair fallback triggered.")
            return self.repair_structured_output(raw or reasoning, "stage1")
        except Exception as e:
            logger.error("Stage 1 targeted recall error: %s", e)
            return None

    def stage1_proof_local_recall(
        self,
        proof_text: str,
        bib_mapping: Dict[str, str],
        target_theorem: str,
        setup: str = "",
    ) -> List[Dict[str, str]]:
        bib_context = build_stage1_bib_context(bib_mapping)
        proof_view = str(proof_text or "")[: int(CONFIG["STAGE1_PROOF_LOCAL_RECALL_MAX_CHARS"])]
        if not proof_view.strip():
            return []

        system_prompt = r"""
You are a Mathematical Proof Citation Scout.
The input is already restricted to the proof of the target theorem.
Your only job is to recover citations used as external mathematical tools inside this proof.

TASK
- Read only the provided proof text.
- Identify citations that supply a non-trivial external result used to justify a proof step.
- Keep citations when the proof uses an external theorem, lemma, proposition, criterion, inequality, estimate, or structural fact.
- Exclude citations used only for background, definitions, historical remarks, or generic attribution.

For each retained citation, output:
- citation_key: exact BibTeX key from BIB_MAPPING
- locator_snippet: a unique 20-40 word snippet surrounding the citation, copied from the proof text
- reason: one short explanation of why this is tool usage

Return JSON only:
{
  "proof_citations": [
    {"citation_key": "...", "locator_snippet": "...", "reason": "..."}
  ]
}
"""
        try:
            logger.info(
                "Stage 1 proof-local recall: proof_chars=%d target_theorem_chars=%d",
                len(proof_view),
                len(str(target_theorem or "")),
            )
            raw, reasoning, _ = self.request_structured_completion(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {
                        "role": "user",
                        "content": (
                            f"TARGET_THEOREM:\n{target_theorem[:3000]}\n\n"
                            f"OPTIONAL_SETUP:\n{setup[:2000]}\n\n"
                            f"BIBLIOGRAPHY:\n{bib_context}\n\n"
                            f"PROOF_TEXT:\n{proof_view}"
                        ),
                    },
                ],
                max_tokens=2200,
                stage="stage1",
                required_keys=["proof_citations"],
            )
            parsed = self.clean_json(raw)
            if not parsed and reasoning:
                parsed = self.clean_json(reasoning)
            citations = normalize_stage1_citation_list(parsed)
            if citations:
                return citations
            logger.info("Stage 1 proof-local recall structured repair fallback triggered.")
            repaired = self.repair_structured_output(raw or reasoning, "stage1_citations_only")
            return normalize_stage1_citation_list(repaired)
        except Exception as e:
            logger.error("Stage 1 proof-local recall error: %s", e)
            return []

    # --------------------------
    # Stage 2
    # --------------------------
    def stage2_extract_local_context(
        self, context_text: str, citation_info: Dict, bib_content: str, target_theorem: str = ""
    ) -> Optional[Dict]:
        target_key = citation_info.get("citation_key", "")
        snippet = citation_info.get("locator_snippet", "")
        usage_reason = citation_info.get("reason", "")
        pre_citation_focus = extract_pre_citation_focus_snippet(snippet)
        local_context_blocks, local_context_meta = extract_local_context_blocks_from_slice(context_text, snippet)
        heuristic_tool = heuristic_tool_payload_from_evidence(snippet, pre_citation_focus=pre_citation_focus)
        throughput_mode = bool(CONFIG.get("THROUGHPUT_MODE", False))
        if local_context_meta.get("reason") != "ok":
            logger.info(
                "Stage 2 local-context heuristic could not align %s inside local slice: %s",
                target_key,
                compact_detail(local_context_meta),
            )
            return None

        system_prompt = r"""
You are a Mathematical Logic Expert.
Extract only the grounded external tool for one citation occurrence.
The local proof context blocks have already been extracted from the proof.

INPUT
- Target theorem statement
- Target citation key
- Pre-citation local context blocks
- Citation-focus snippet immediately before the citation
- Short reason why this citation looks like tool usage
- Citation bibliography content

IMPORTANT
- Do NOT rewrite the local context blocks.
- Focus on the proof step immediately around the citation.
- Recover only the best theorem-like tool statement supported by the citing text.
- Prefer an explicit implication / estimate / criterion over vague prose.
- Do NOT return a theorem title only.
- Reject discourse wrappers such as "In addition ...", "Recall that ...", "future work ...", or case-analysis narration.
- Set restated_in_citing_paper=true only when the citing text explicitly states the mathematical implication/estimate itself.
- reference_tool_latex must not include theorem labels/reporting wrappers such as "Proposition 1.5 states that" or any proof/theorem environment markers.
- reference_tool_latex must be a clean theorem-like statement only; never include phrases like "we follow the argument", "our solution is", "all we can say", or enumerate/example/remark blocks.

OUTPUT FIELDS
1) reference_tool_latex: the minimal theorem-like statement, estimate, or criterion used at this step.
2) reference_tool_type: one of theorem, lemma, proposition, corollary, claim, inequality, criterion, or other.
3) restated_in_citing_paper: true if the cited tool statement is explicitly restated in the citing paper.

Return JSON only:
{
  "reference_tool_latex": "...",
  "reference_tool_type": "theorem|lemma|proposition|corollary|claim|inequality|criterion|other",
  "restated_in_citing_paper": false
}
"""
        user_msg = (
            f"TARGET_THEOREM: {target_theorem}\n"
            f"TARGET: {target_key}\n"
            f"TOOL_USAGE_REASON: {usage_reason}\n"
            f"CITATION_CONTENT: {bib_content}\n"
            f"PRE_CITATION_FOCUS: {pre_citation_focus}\n"
            f"LOCATOR: {snippet}\n\n"
            f"LOCAL_CONTEXT_BLOCKS:\n{json.dumps({'local_context': local_context_blocks}, ensure_ascii=False)}"
        )

        def build_stage2_result(payload: Optional[Dict[str, Any]]) -> Dict[str, Any]:
            payload = payload or {}
            tool_statement = str(payload.get("reference_tool_latex", "") or "").strip()
            if not has_meaningful_tool_statement(tool_statement):
                tool_statement = str((heuristic_tool or {}).get("reference_tool_latex") or "").strip()
            return {
                "local_context": local_context_blocks,
                "anchor_hint": resolve_anchor_hint(
                    locator_snippet=snippet,
                    pre_citation_focus=pre_citation_focus,
                    usage_reason=usage_reason,
                    local_context_blocks=local_context_blocks,
                ),
                "reference_tool_latex": tool_statement,
                "reference_tool_type": finalize_reference_tool_type(
                    payload.get("reference_tool_type") or (heuristic_tool or {}).get("reference_tool_type") or "",
                    tool_statement,
                ),
                "restated_in_citing_paper": bool(
                    payload.get("restated_in_citing_paper", (heuristic_tool or {}).get("restated_in_citing_paper", False))
                ),
                "citation_locator": "",
                "tool_family": "",
            }

        try:
            raw, reasoning, finish_reason = self.request_structured_completion(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_msg},
                ],
                max_tokens=int(CONFIG.get("STAGE2_SINGLE_MAX_TOKENS", 1800)),
                stage="stage2",
                required_keys=["reference_tool_latex", "reference_tool_type", "restated_in_citing_paper"],
            )
            parsed = self.clean_json(raw)
            if not parsed and reasoning:
                parsed = self.clean_json(reasoning)
            if not parsed:
                parsed = salvage_stage2_payload_from_raw_text(raw or reasoning)
            if (
                not parsed
                and self.is_reasoning_model()
                and (finish_reason == "length" or not str(raw or "").strip())
            ):
                logger.info("Stage 2 compact retry triggered for key=%s.", target_key)
                raw, reasoning, _ = self.request_structured_completion(
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_msg},
                    ],
                    max_tokens=int(CONFIG.get("STAGE2_REASONER_RETRY_MAX_TOKENS", 3600)),
                    stage="stage2_retry",
                    required_keys=["reference_tool_latex", "reference_tool_type", "restated_in_citing_paper"],
                )
                parsed = self.clean_json(raw)
                if not parsed and reasoning:
                    parsed = self.clean_json(reasoning)
                if not parsed:
                    parsed = salvage_stage2_payload_from_raw_text(raw or reasoning)
            if parsed:
                return build_stage2_result(parsed)
            logger.debug("Stage 2 JSON parse failed for key=%s. Raw head=%r", target_key, raw[:400])
            if throughput_mode:
                return build_stage2_result(None)
            logger.info("Stage 2 context structured repair fallback triggered for key=%s.", target_key)
            repaired = self.repair_structured_output(raw or reasoning, "stage2_context")
            if not repaired:
                repaired = salvage_stage2_payload_from_raw_text(raw or reasoning)
            if repaired:
                return build_stage2_result(repaired)
            return build_stage2_result(None)

        except Exception as e:
            logger.error("Stage 2 error for key=%s: %s", target_key, e)
            return build_stage2_result(None)

    def stage2_focus_tool_statement(
        self,
        context_text: str,
        citation_info: Dict,
        bib_content: str,
        target_theorem: str = "",
        anchor_hint: str = "",
    ) -> Optional[Dict]:
        target_key = citation_info.get("citation_key", "")
        snippet = citation_info.get("locator_snippet", "")
        usage_reason = citation_info.get("reason", "")
        pre_citation_focus = extract_pre_citation_focus_snippet(snippet)
        heuristic_tool = heuristic_tool_payload_from_evidence(snippet, pre_citation_focus=pre_citation_focus)

        def normalize_tool_payload(parsed: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
            if not parsed or not isinstance(parsed, dict):
                return None
            raw_restated = parsed.get("restated_in_citing_paper", False)
            if isinstance(raw_restated, str):
                restated = raw_restated.strip().lower() in {"true", "1", "yes"}
            else:
                restated = bool(raw_restated)
            return {
                "reference_tool_latex": str(parsed.get("reference_tool_latex", "") or "").strip(),
                "reference_tool_type": str(parsed.get("reference_tool_type", "") or "").strip(),
                "restated_in_citing_paper": restated,
            }

        def request_tool_payload(system_prompt: str, user_msg: str, max_tokens: int = 1800) -> Optional[Dict[str, Any]]:
            raw, reasoning, _ = self.request_structured_completion(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_msg},
                ],
                max_tokens=max_tokens,
                stage="stage2",
                required_keys=["reference_tool_latex", "reference_tool_type", "restated_in_citing_paper"],
            )
            parsed = self.clean_json(raw)
            if not parsed and reasoning:
                parsed = self.clean_json(reasoning)
            if not parsed:
                parsed = salvage_stage2_payload_from_raw_text(raw or reasoning)
            if parsed:
                return normalize_tool_payload(parsed)
            logger.info("Stage 2 focused-tool structured repair fallback triggered for key=%s.", target_key)
            repaired = self.repair_structured_output(raw or reasoning, "stage2_tool")
            if not repaired:
                repaired = salvage_stage2_payload_from_raw_text(raw or reasoning)
            return normalize_tool_payload(repaired)

        primary_system_prompt = r"""
You are a Mathematical Logic Expert.
Focus only on recovering the external tool statement used by a specific citation in a proof.

IMPORTANT
- The citation has already been filtered as plausible tool usage inside the proof of the target theorem.
- Your job is to recover the minimal operative theorem-like statement, estimate, or criterion needed at this step.
- Do NOT return a theorem title only.
- Do NOT return an empty string unless the provided proof text genuinely gives no recoverable theorem-like content.
- If the paper only reveals the consequence used at this step, return that consequence as a faithful theorem-like paraphrase.
- Reject narrative wrappers such as "In addition", "Recall that", case labels, motivation, or future-work sentences.
- Set restated_in_citing_paper=true only when the citing text explicitly states the mathematical implication/estimate itself.
- Do not wrap the answer in proof/theorem environments and do not write source-reporting text such as "Theorem 2 states that ...".

Return JSON only:
{
  "reference_tool_latex": "...",
  "reference_tool_type": "theorem|lemma|proposition|corollary|claim|inequality|criterion|other",
  "restated_in_citing_paper": false
}
"""
        primary_user_msg = (
            f"TARGET_THEOREM: {target_theorem}\n"
            f"TARGET: {target_key}\n"
            f"TOOL_USAGE_REASON: {usage_reason}\n"
            f"ANCHOR_HINT: {anchor_hint}\n"
            f"CITATION_CONTENT: {bib_content}\n"
            f"PRE_CITATION_FOCUS: {pre_citation_focus}\n"
            f"LOCATOR: {snippet}\n\n"
            f"PROOF_TEXT:\n{str(context_text or '')[:int(CONFIG['STAGE2_MAX_CHARS'])]}"
        )

        fallback_system_prompt = r"""
You are a Mathematical Logic Expert.
Recover the cited external tool as a short usable implication, estimate, or criterion.

IMPORTANT
- Stage 1 has already validated that this citation is real tool usage inside the proof of the target theorem.
- Do NOT return a theorem title only.
- Do NOT return an empty string unless there is truly no recoverable mathematical implication.
- If the citation sentence states a consequence such as "... belongs to A, and therefore belongs to B by [X]",
  rewrite it as a theorem-like implication of the form "If ... A ..., then ... B ...".
- A short theorem-like implication is preferred over an empty answer.
- Reject motivation/discussion fragments such as "In addition", "Recall that", case-analysis narration, or future-work remarks.
- Do not return source-reporting wrappers such as "Proposition 1.5 states that ..." and do not include proof/theorem environment markers.

Return JSON only:
{
  "reference_tool_latex": "...",
  "reference_tool_type": "theorem|lemma|proposition|corollary|claim|inequality|criterion|other",
  "restated_in_citing_paper": false
}
"""
        fallback_user_msg = (
            f"TARGET_THEOREM: {target_theorem}\n"
            f"TARGET: {target_key}\n"
            f"TOOL_USAGE_REASON: {usage_reason}\n"
            f"CITATION_CONTENT: {bib_content}\n"
            f"PRE_CITATION_FOCUS: {pre_citation_focus}\n"
            f"LOCATOR_SNIPPET:\n{snippet}\n\n"
            f"LOCAL_PROOF_EVIDENCE:\n{str(context_text or '')[:int(CONFIG['STAGE2_MAX_CHARS'])]}"
        )

        try:
            primary_result = request_tool_payload(primary_system_prompt, primary_user_msg, max_tokens=1800)
            if primary_result and has_meaningful_tool_statement(primary_result.get("reference_tool_latex", "")):
                return primary_result

            logger.info(
                "Stage 2 focused-tool fallback implication pass triggered for key=%s (initial_statement=%r).",
                target_key,
                str((primary_result or {}).get("reference_tool_latex", ""))[:200],
            )
            fallback_result = request_tool_payload(fallback_system_prompt, fallback_user_msg, max_tokens=1600)
            if fallback_result and has_meaningful_tool_statement(fallback_result.get("reference_tool_latex", "")):
                return fallback_result
            return heuristic_tool or fallback_result or primary_result
        except Exception as e:
            logger.error("Stage 2 focused-tool error for key=%s: %s", target_key, e)
            return heuristic_tool

    def stage2_extract_multiple_local_contexts(
        self, context_text: str, target_theorem: str, citations_payload: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        system_prompt = r"""
You are a Mathematical Logic Expert.
Each item below is a candidate external-tool citation from the proof of the target theorem.
For each item, extract a benchmark-ready proof-gap instance draft.

INPUT
- The proof text of the target theorem
- The target theorem statement
- A list of candidate citations.
- Each item includes:
  - citation_key
  - locator_snippet
  - reason
  - citation_content

OUTPUT FIELDS FOR EACH ITEM
1) citation_key: copy from input exactly.
2) reference_tool_latex: the specific theorem/inequality being applied from the cited work.
   - Prefer an explicit theorem-like statement restated in the citing paper when available.
   - Otherwise faithfully restate the minimal external tool that the proof step needs, grounded in the proof text and citation metadata.
   - It must be more than a theorem name.
   - A displayed inequality or a one-sentence criterion stated in the citing proof is valid.
   - Do NOT include discourse wrappers such as "we follow the argument", "our solution is", "all we can say", or example/remark/enumerate blocks.
   - Only return an empty string if there is truly no theorem-like or estimate-like content recoverable from the provided evidence.
3) reference_tool_type: one of theorem, lemma, proposition, corollary, claim, inequality, criterion, or other.
4) restated_in_citing_paper: true if the cited tool statement is explicitly restated in the citing paper.

Return JSON only:
{
  "items": [
    {
      "citation_key": "...",
      "reference_tool_latex": "...",
      "reference_tool_type": "...",
      "restated_in_citing_paper": false
    }
  ]
}
"""

        try:
            raw, reasoning, _ = self.request_structured_completion(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {
                        "role": "user",
                        "content": (
                            f"TARGET_THEOREM:\n{target_theorem}\n\n"
                            f"PROOF_TEXT:\n{str(context_text or '')[: int(CONFIG['STAGE2_MAX_CHARS'])]}\n\n"
                            f"CANDIDATE_CITATIONS:\n{json.dumps({'items': citations_payload}, ensure_ascii=False)}"
                        ),
                    },
                ],
                max_tokens=int(CONFIG.get("STAGE2_BATCH_MAX_TOKENS", 4500)),
                stage="stage2",
                required_keys=["items"],
            )
            parsed = self.clean_json(raw)
            if not parsed and reasoning:
                parsed = self.clean_json(reasoning)
            if not parsed:
                logger.debug("Stage 2 batch JSON parse failed. Raw head=%r", raw[:400])
                logger.info("Stage 2 batch structured repair fallback triggered.")
                parsed = self.repair_structured_output(raw or reasoning, "stage2_batch")
                if not parsed:
                    return []
            items = parsed.get("items", []) if isinstance(parsed, dict) else []
            return items if isinstance(items, list) else []
        except Exception as e:
            logger.error("Stage 2 batch error: %s", e)
            return []


# ============================================================
# Main
# ============================================================

def print_cli_help() -> None:
    print(
        "Usage: python3 construction/mine_dataset.py\n\n"
        "This script is configured primarily through environment variables.\n\n"
        "Core environment variables:\n"
        "  OPENAI_API_KEY           Required LLM API key.\n"
        "  OPENAI_BASE_URL          Optional API base URL.\n"
        "  OPENAI_MODEL             Optional model name.\n"
        "  ARXIV_ID_LIST_FILE       Optional ID list file. Default: configs/arxiv_ids.txt\n"
        "  PAPER_MANIFEST_FILE      Optional manifest file for published-first pipeline.\n"
        "  OUTPUT_FILE              Output JSONL. Default: construction/outputs/benchmark_dataset_v2.jsonl\n"
        "  PROGRESS_FILE            Optional resume log path.\n"
        "  PAPER_SHARD_COUNT        Optional worker shard count. Default: 1\n"
        "  PAPER_SHARD_INDEX        Optional worker shard index [0, count). Default: 0\n"
        "  PAPER_SORT_BY_PROOF_RICH Sort manifest papers by proof-rich score first.\n"
        "  MAX_PAPERS               Max papers for time-window retrieval mode.\n"
        "  TIME_WINDOW_DAYS         Lookback window for time-window retrieval mode.\n"
        "  STAGE1_COMPACT_VIEW_ENABLED\n"
        "  STAGE1_DOSSIER_MAX_CHARS\n"
        "  EXTERNAL_TOOL_SCOUT_ENABLED\n"
        "  EXTERNAL_TOOL_SCOUT_MIN_SCORE\n"
        "  MAX_STAGE2_CITATIONS_PER_PAPER\n"
        "  MAX_STAGE2_SINGLE_CALLS_PER_PAPER\n"
        "  STAGE2_BATCH_ENABLED\n\n"
        "Strictness policy:\n"
        "  - If the main-theorem proof span cannot be aligned, skip the whole paper.\n"
        "  - If a citation locator cannot be aligned inside the main proof, skip that citation.\n\n"
        "Modes:\n"
        "  1. Manifest-backed: set PAPER_MANIFEST_FILE\n"
        "  2. Explicit IDs:    set ARXIV_ID_LIST_FILE\n"
        "  3. Time-window:     set ARXIV_CATEGORY / MAX_PAPERS / TIME_WINDOW_DAYS\n"
    )


def main() -> None:
    if any(flag in sys.argv[1:] for flag in {"-h", "--help"}):
        print_cli_help()
        return

    if bool(CONFIG.get("STARTUP_API_SANITY_CHECK_ENABLED", True)):
        if not test_api_connection():
            logger.warning("Startup API sanity check failed; continuing anyway and relying on per-request retries.")
    else:
        logger.info("Startup API sanity check disabled.")

    api_key = require_api_key()
    base_url = str(CONFIG["BASE_URL"])
    model_name = str(CONFIG["MODEL_NAME"])

    start_time = datetime.datetime.now() - datetime.timedelta(days=int(CONFIG["TIME_WINDOW_DAYS"]))
    retriever = ArxivMathPaperRetriever(start_time=start_time, category=str(CONFIG["CATEGORY"]))

    manifest_file = str(CONFIG.get("PAPER_MANIFEST_FILE", "") or "").strip()
    id_list_file = str(CONFIG.get("ID_LIST_FILE", "") or "").strip()
    explicit_id_list_env = str(os.getenv("ARXIV_ID_LIST_FILE", "") or "").strip()

    if manifest_file:
        # In manifest-backed mode, only apply an ID filter if the user explicitly
        # requested one via ARXIV_ID_LIST_FILE. Otherwise the default sample file
        # would silently collapse the manifest to an unrelated singleton list.
        target_ids = load_ids_from_file(id_list_file) if explicit_id_list_env else []
        if explicit_id_list_env:
            logger.info(
                "Manifest mode: applying explicit ID filter from %s (%d ids).",
                id_list_file,
                len(target_ids),
            )
        papers_metadata = load_papers_from_manifest(manifest_file, target_ids=target_ids or None)
        logger.info(
            "Mode: manifest-backed retrieval (%d papers from %s).",
            len(papers_metadata),
            manifest_file,
        )
    else:
        target_ids = load_ids_from_file(id_list_file)

    if not manifest_file and target_ids:
        logger.info("Mode: explicit ID list (%d papers).", len(target_ids))
        papers_metadata = retriever.retrieve_papers_by_ids(list(dict.fromkeys(target_ids)))
    elif not manifest_file:
        logger.info("Mode: time window retrieval (category=%s).", CONFIG["CATEGORY"])
        papers_metadata = retriever.retrieve_papers(max_results=int(CONFIG["MAX_PAPERS"]))

    papers_metadata = apply_paper_order_and_shard(papers_metadata)
    logger.info(
        "Mining queue prepared with %d papers (shard %d/%d, proof-rich sort=%s).",
        len(papers_metadata),
        int(CONFIG.get("PAPER_SHARD_INDEX", 0)) + 1,
        int(CONFIG.get("PAPER_SHARD_COUNT", 1)),
        bool(CONFIG.get("PAPER_SORT_BY_PROOF_RICH", True)),
    )

    if not papers_metadata:
        logger.warning("No papers retrieved.")
        return

    extractor = ReasoningAwareExtractor()
    generator = DatasetGenerator(api_key, base_url, model_name)

    output_file = str(CONFIG["OUTPUT_FILE"])
    progress_file = str(CONFIG.get("PROGRESS_FILE", "") or "").strip() or f"{output_file}.progress.jsonl"
    processed_paper_ids = load_processed_paper_ids(progress_file)
    if processed_paper_ids:
        logger.info("Resume mode: %d papers already recorded in %s.", len(processed_paper_ids), progress_file)
    total_gaps_saved = 0

    with open(output_file, "a", encoding="utf-8") as f_out, open(progress_file, "a", encoding="utf-8") as f_progress:
        for i, paper in enumerate(papers_metadata):
            paper_id = paper.get("id", "")
            title = paper.get("title", "")
            if paper_id in processed_paper_ids:
                logger.info("[%d/%d] Skipping already processed paper %s: %s", i + 1, len(papers_metadata), paper_id, title)
                continue

            logger.info("[%d/%d] Processing %s: %s", i + 1, len(papers_metadata), paper_id, title)
            paper_started_at = time.perf_counter()
            paper_status = "started"
            paper_detail = ""
            valid_gaps_count = 0
            seen_instance_signatures: List[Dict[str, str]] = []

            try:
                # 1) Download and extract text + bibliography
                success, full_text, full_bib = extractor.process_paper_with_bib(paper_id, paper.get("latex_link", ""))
                if not success or len(full_text) < 1000:
                    logger.warning("Content extraction failed for %s.", paper_id)
                    paper_status = "extract_failed"
                    paper_detail = "content_extraction_failed"
                    continue

                scout: Optional[Dict[str, Any]] = None
                if bool(CONFIG.get("EXTERNAL_TOOL_SCOUT_ENABLED", True)):
                    scout = scout_external_tool_usage(full_text)
                    scout_threshold = float(CONFIG["EXTERNAL_TOOL_SCOUT_MIN_SCORE"])
                    logger.info(
                        "Scout for %s: score=%.2f unique_cites=%d pattern_hits=%s",
                        paper_id,
                        float(scout["score"]),
                        int(scout["unique_cite_count"]),
                        scout.get("pattern_hits", {}),
                    )
                    if float(scout["score"]) < scout_threshold:
                        logger.info("Skipping %s due to weak external-tool evidence.", paper_id)
                        paper_status = "external_tool_scout_reject"
                        paper_detail = compact_detail(
                            {
                                "score": scout["score"],
                                "threshold": scout_threshold,
                                "unique_cite_count": scout["unique_cite_count"],
                                "pattern_hits": scout["pattern_hits"],
                                "examples": scout["examples"],
                            }
                        )
                        continue

                # Normalize bib keys for lookup
                norm_bib = {normalize_key(k): v for k, v in full_bib.items()}
                canonical_bib_keys = {normalize_key(k): str(k) for k in full_bib.keys() if normalize_key(k)}

                if bool(CONFIG.get("FAST_PREFILTER_ENABLED", False)):
                    prefilter_target_hint = str(paper.get("published_title") or title or "").strip()
                    fast_skip, fast_skip_meta = fast_prefilter_no_proof_citations(
                        full_text,
                        prefilter_target_hint,
                        norm_bib,
                        canonical_bib_keys,
                        scout=scout,
                    )
                    if fast_skip:
                        logger.info(
                            "Skipping %s via fast prefilter: recovered target-theorem proof has no citations.",
                            paper_id,
                        )
                        paper_status = "fast_prefilter_no_proof_citations"
                        paper_detail = compact_detail(fast_skip_meta)
                        continue

                # Stage 1
                stage1_result = generator.stage1_analyze_structure(full_text, full_bib)
                if not stage1_result and bool(CONFIG.get("STAGE1_RETRY_ENABLED", True)):
                    logger.info("Retrying Stage 1 once for %s due to empty/invalid structured output.", paper_id)
                    stage1_result = generator.stage1_analyze_structure(full_text, full_bib)
                stage1_result = normalize_stage1_result(stage1_result)
                if not stage1_result:
                    logger.warning("Stage 1 failed for %s.", paper_id)
                    paper_status = "stage1_failed"
                    paper_detail = "stage1_failed"
                    continue

                global_context = stage1_result.get("global_context", {}) or {}
                proof_span = stage1_result.get("proof_span", {}) or {}
                raw_target_theorem = str(global_context.get("target_theorem", "") or "").strip()
                if raw_target_theorem:
                    global_context["target_theorem"] = normalize_target_theorem_text(raw_target_theorem)
                target_theorem = str(global_context.get("target_theorem", "") or "").strip()
                target_rejection_reason = target_theorem_rejection_reason(
                    raw_target_theorem,
                    strict_mode=bool(CONFIG.get("STRICT_QC_MODE", False)),
                )
                if target_rejection_reason:
                    logger.info(
                        "Skipping %s because Stage 1 target theorem is not acceptable under QC: %r",
                        paper_id,
                        raw_target_theorem[:240],
                    )
                    paper_status = "target_theorem_not_main"
                    paper_detail = compact_detail(
                        {
                            "reason": target_rejection_reason,
                            "target_theorem": raw_target_theorem[:240],
                        }
                    )
                    continue
                citations_list = stage1_result.get("proof_citations", []) or []
                target_hint = target_theorem or title
                should_targeted_recall = False
                if bool(CONFIG.get("STAGE1_TARGETED_RECALL_ENABLED", True)) and target_hint:
                    if not has_proof_span_snippets(proof_span):
                        should_targeted_recall = True
                    elif not citations_list and bool(CONFIG.get("STAGE1_TARGETED_RECALL_ON_EMPTY_CITATIONS", True)):
                        should_targeted_recall = True

                if should_targeted_recall:
                    logger.info(
                        "Stage 1 targeted recall fallback triggered for %s (citations=%d, proof_span=%s).",
                        paper_id,
                        len(citations_list),
                        has_proof_span_snippets(proof_span),
                    )
                    recall_result = generator.stage1_targeted_recall(
                        full_text,
                        full_bib,
                        target_hint=target_hint,
                        setup=str(global_context.get("setup", "") or ""),
                    )
                    recall_result = normalize_stage1_result(recall_result)
                    if isinstance(recall_result, dict):
                        recall_proof_span = recall_result.get("proof_span", {}) or {}
                        recall_citations = recall_result.get("proof_citations", []) or []
                        if not has_proof_span_snippets(proof_span) and has_proof_span_snippets(recall_proof_span):
                            proof_span = recall_proof_span
                        if len(recall_citations) > len(citations_list):
                            citations_list = recall_citations
                proof_text = ""
                proof_scope_meta: Dict[str, Any] = {}
                if has_proof_span_snippets(proof_span):
                    extracted_proof_text, proof_scope_meta = extract_main_proof_text(full_text, proof_span)
                    if extracted_proof_text and proof_text_is_usable_for_stage2(extracted_proof_text):
                        proof_text = extracted_proof_text
                    else:
                        logger.info(
                            "Snippet-based proof extraction was insufficient for %s; trying target-theorem fallback.",
                            paper_id,
                        )

                if not proof_text:
                    recovered_proof_text, recovered_meta = recover_main_proof_text_from_target(full_text, target_hint)
                    if recovered_proof_text and proof_text_is_usable_for_stage2(recovered_proof_text):
                        proof_text = recovered_proof_text
                        proof_scope_meta = recovered_meta
                    else:
                        failure_meta = recovered_meta or proof_scope_meta or {"reason": "proof_recovery_failed"}
                        if failure_meta.get("reason") == "ok":
                            logger.info(
                                "Skipping %s because the recovered main proof span is too short.",
                                paper_id,
                            )
                            paper_status = "proof_span_too_short"
                            failure_meta = {
                                **failure_meta,
                                "min_required_chars": 500,
                                "min_chars_with_citation": 250,
                                "recovered_citation_count": count_citations_in_text(recovered_proof_text or ""),
                            }
                        elif not has_proof_span_snippets(proof_span):
                            logger.info(
                                "Skipping %s because the main-theorem proof span could not be identified by Stage 1 or fallback.",
                                paper_id,
                            )
                            paper_status = "main_proof_not_found"
                        else:
                            logger.info(
                                "Skipping %s because the main-theorem proof span could not be aligned and fallback failed.",
                                paper_id,
                            )
                            paper_status = "proof_span_not_aligned"
                        paper_detail = compact_detail(failure_meta)
                        continue

                if not target_theorem and target_hint:
                    recovered_target_theorem, recovered_target_meta = recover_target_theorem_text_from_hint(
                        full_text,
                        target_hint,
                    )
                    if recovered_target_theorem:
                        global_context["target_theorem"] = recovered_target_theorem
                        target_theorem = recovered_target_theorem
                        logger.info(
                            "Recovered target theorem text for %s via target-hint fallback.",
                            paper_id,
                        )
                    else:
                        logger.info(
                            "Target theorem fallback did not recover a theorem block for %s: %s",
                            paper_id,
                            compact_detail(recovered_target_meta if isinstance(recovered_target_meta, dict) else {"reason": "unknown"}),
                        )
                final_target_theorem = str(global_context.get("target_theorem", "") or "").strip()
                final_target_rejection_reason = target_theorem_rejection_reason(
                    final_target_theorem,
                    strict_mode=bool(CONFIG.get("STRICT_QC_MODE", False)),
                )
                if final_target_rejection_reason:
                    logger.info(
                        "Skipping %s because final target theorem is not acceptable under QC: %r",
                        paper_id,
                        final_target_theorem[:240],
                    )
                    paper_status = "target_theorem_not_main"
                    paper_detail = compact_detail(
                        {
                            "reason": final_target_rejection_reason,
                            "target_theorem": final_target_theorem[:240],
                        }
                    )
                    continue

                current_setup = str(global_context.get("setup", "") or "").strip()
                current_setup_rejection_reason = setup_text_rejection_reason(current_setup)
                if current_setup_rejection_reason:
                    logger.info(
                        "Rejecting Stage 1 setup for %s before fallback: %s",
                        paper_id,
                        current_setup_rejection_reason,
                    )
                    global_context["setup"] = ""
                if not str(global_context.get("setup", "") or "").strip():
                    recovered_setup, recovered_setup_meta = recover_setup_text_from_anchor(
                        full_text,
                        int((proof_scope_meta or {}).get("anchor_start", -1)),
                    )
                    if recovered_setup:
                        global_context["setup"] = recovered_setup
                        logger.info(
                            "Recovered setup context for %s near target theorem anchor: %s",
                            paper_id,
                            compact_detail(recovered_setup_meta),
                        )
                    else:
                        logger.info(
                            "Setup recovery did not find usable setup context for %s: %s",
                            paper_id,
                            compact_detail(recovered_setup_meta),
                        )

                logger.info("Proof scope for %s: %s", paper_id, compact_detail(proof_scope_meta))
                proof_local_citations = scout_proof_citations(proof_text, norm_bib, canonical_bib_keys)
                if proof_local_citations:
                    citations_list = merge_citation_lists(citations_list, proof_local_citations)
                    logger.info(
                        "Proof-local citation scout added/merged %d candidates for %s (total=%d).",
                        len(proof_local_citations),
                        paper_id,
                        len(citations_list),
                    )
                desired_citation_count = int(CONFIG["MAX_STAGE2_CITATIONS_PER_PAPER"])
                proof_raw_citation_count = count_citations_in_text(proof_text)
                if (
                    bool(CONFIG.get("STAGE1_PROOF_LOCAL_RECALL_ENABLED", True))
                    and len(citations_list) < desired_citation_count
                    and proof_raw_citation_count > 0
                ):
                    proof_recall_citations = generator.stage1_proof_local_recall(
                        proof_text,
                        full_bib,
                        target_theorem=target_theorem or target_hint,
                        setup=str(global_context.get("setup", "") or ""),
                    )
                    if proof_recall_citations:
                        before_count = len(citations_list)
                        citations_list = merge_citation_lists(citations_list, proof_recall_citations)
                        logger.info(
                            "Proof-local LLM recall added/merged %d candidates for %s (total=%d).",
                            max(0, len(citations_list) - before_count),
                            paper_id,
                            len(citations_list),
                        )
                elif len(citations_list) < desired_citation_count and proof_raw_citation_count == 0:
                    logger.info(
                        "Skipping proof-local LLM recall for %s because recovered proof text has no citation markers.",
                        paper_id,
                    )
                original_count = len(citations_list)
                citations_list = [
                    item for item in citations_list
                    if normalize_key(str(item.get("citation_key", "") or "")) in norm_bib
                ]
                if len(citations_list) < original_count:
                    logger.info(
                        "Filtered out %d citations whose keys were missing from bibliography for %s.",
                        original_count - len(citations_list),
                        paper_id,
                    )
                if not citations_list:
                    logger.info("No proof citations found for %s.", paper_id)
                    paper_status = "no_citations"
                    paper_detail = "no_valid_main_theorem_citations_after_stage1"
                    continue

                logger.info(
                    "Stage 1 done for %s. Found %d candidate citations for the target theorem.",
                    paper_id,
                    len(citations_list),
                )

                # Stage 2
                citation_candidates: List[Dict[str, Any]] = []
                for idx, cit_item in enumerate(citations_list):
                    raw_key = str(cit_item.get("citation_key", "") or "")
                    norm_k = normalize_key(raw_key)
                    bib_text = norm_bib.get(norm_k, "Citation content not found")
                    if not has_bibliography_metadata(bib_text):
                        logger.info("  Skipping citation %s due to missing bibliography metadata.", raw_key)
                        continue

                    citation_candidates.append(
                        {
                            "idx": idx,
                            "citation_key": raw_key,
                            "citation_info": cit_item,
                            "bib_text": bib_text,
                            "score": score_citation_candidate(cit_item, bib_text),
                        }
                    )

                if not citation_candidates:
                    logger.info("No Stage 2-ready citations remained for %s.", paper_id)
                    paper_status = "no_stage2_candidates"
                    paper_detail = "no_main_theorem_citations_with_bibliography_metadata"
                    continue

                citation_candidates = sorted(
                    citation_candidates,
                    key=lambda item: (float(item["score"]), -int(item["idx"])),
                    reverse=True,
                )[: int(CONFIG["MAX_STAGE2_CITATIONS_PER_PAPER"])]

                batch_results: List[Dict[str, Any]] = []
                if bool(CONFIG.get("STAGE2_BATCH_ENABLED", True)) and len(citation_candidates) > 1:
                    logger.info(
                        "Stage 2 batching %d citations for %s.",
                        len(citation_candidates),
                        paper_id,
                    )
                    batch_payload = [
                        {
                            "citation_key": item["citation_key"],
                            "locator_snippet": item["citation_info"].get("locator_snippet", ""),
                            "reason": item["citation_info"].get("reason", ""),
                            "citation_content": item["bib_text"],
                        }
                        for item in citation_candidates
                    ]
                    batch_results = generator.stage2_extract_multiple_local_contexts(
                        proof_text, target_theorem, batch_payload
                    )
                else:
                    logger.info(
                        "Stage 2 batch disabled or unnecessary for %s; using single-citation extraction only.",
                        paper_id,
                    )
                batch_by_key = {
                    str(item.get("citation_key") or ""): item
                    for item in batch_results
                    if isinstance(item, dict) and str(item.get("citation_key") or "").strip()
                }
                stage2_rejections: Counter[str] = Counter()
                single_call_budget = int(CONFIG["MAX_STAGE2_SINGLE_CALLS_PER_PAPER"])
                single_call_count = 0

                for item in citation_candidates:
                    raw_key = str(item["citation_key"])
                    logger.info("  Mining citation: %s", raw_key)
                    locator_snippet = str(item["citation_info"].get("locator_snippet", "") or "")
                    local_stage2_text, local_stage2_meta = extract_local_proof_slice(
                        proof_text,
                        locator_snippet,
                    )
                    logger.info(
                        "  Stage 2 local slice for %s: %s",
                        raw_key,
                        compact_detail(local_stage2_meta),
                    )
                    if local_stage2_meta.get("reason") != "ok":
                        logger.info("  Skipping citation %s due to unresolved locator alignment.", raw_key)
                        stage2_rejections["locator_not_aligned"] += 1
                        continue
                    deterministic_local_context, deterministic_local_meta = extract_local_context_blocks_from_slice(
                        local_stage2_text,
                        locator_snippet,
                    )
                    local_result = batch_by_key.get(raw_key)
                    if not (local_result and isinstance(local_result, dict)):
                        if single_call_count >= single_call_budget:
                            logger.info(
                                "  Skipping single-citation extraction for %s due to Stage 2 single-call budget.",
                                raw_key,
                            )
                            stage2_rejections["single_call_budget_exhausted"] += 1
                            continue
                        single_call_count += 1
                        local_result = generator.stage2_extract_local_context(
                            local_stage2_text,
                            item["citation_info"],
                            item["bib_text"],
                            target_theorem=target_theorem,
                        )
                    if not (local_result and isinstance(local_result, dict)):
                        stage2_rejections["stage2_empty"] += 1
                        continue
                    local_result["local_context"] = deterministic_local_context

                    tool_statement = str(local_result.get("reference_tool_latex") or "").strip()
                    if not has_meaningful_tool_statement(tool_statement):
                        logger.info(
                            "  Weak tool statement from batch/local result for %s: %r (reason=%s)",
                            raw_key,
                            tool_statement[:240],
                            tool_statement_rejection_reason(tool_statement),
                        )
                        if bool(CONFIG.get("STAGE2_WEAK_TOOL_RETRY_ENABLED", False)):
                            if single_call_count >= single_call_budget:
                                logger.info(
                                    "  Skipping weak-tool retry for %s due to Stage 2 single-call budget.",
                                    raw_key,
                                )
                                stage2_rejections["weak_tool_retry_budget_exhausted"] += 1
                            else:
                                logger.info(
                                    "  Retrying focused tool extraction for %s on proof-local slice.",
                                    raw_key,
                                )
                                logger.info(
                                    "  Stage 2 local slice for %s: %s",
                                    raw_key,
                                    compact_detail(local_stage2_meta),
                                )
                                single_call_count += 1
                                retry_result = generator.stage2_focus_tool_statement(
                                    local_stage2_text,
                                    item["citation_info"],
                                    item["bib_text"],
                                    target_theorem=target_theorem,
                                    anchor_hint=str(local_result.get("anchor_hint") or ""),
                                )
                                if retry_result and isinstance(retry_result, dict):
                                    local_result = {
                                        **local_result,
                                        **retry_result,
                                    }
                                    tool_statement = str(local_result.get("reference_tool_latex") or "").strip()
                    if not has_meaningful_tool_statement(tool_statement):
                        logger.info(
                            "  Skipping citation %s due to weak reference tool statement (%s).",
                            raw_key,
                            tool_statement_rejection_reason(tool_statement),
                        )
                        stage2_rejections["weak_tool_statement"] += 1
                        continue

                    tool_type = normalize_signature_text(local_result.get("reference_tool_type")).lower()
                    if bool(CONFIG.get("STRICT_QC_REJECT_OTHER_TOOL_TYPE", False)) and tool_type == "other":
                        logger.info("  Skipping citation %s due to generic reference tool type 'other'.", raw_key)
                        stage2_rejections["generic_tool_type"] += 1
                        continue

                    pre_citation_focus = extract_pre_citation_focus_snippet(locator_snippet)
                    local_result["local_context"] = normalize_local_context_blocks(local_result.get("local_context", []))
                    local_result["anchor_hint"] = resolve_anchor_hint(
                        llm_anchor=local_result.get("anchor_hint", ""),
                        locator_snippet=locator_snippet,
                        pre_citation_focus=pre_citation_focus,
                        usage_reason=item["citation_info"].get("reason", ""),
                        local_context_blocks=local_result.get("local_context", []),
                    )
                    local_result["reference_tool_type"] = finalize_reference_tool_type(
                        local_result.get("reference_tool_type", ""),
                        tool_statement,
                    )
                    if bool(CONFIG.get("STRICT_QC_REJECT_EMPTY_ANCHOR", True)) and not local_result["anchor_hint"]:
                        logger.info(
                            "  Skipping citation %s because anchor hint is empty after fallback.",
                            raw_key,
                        )
                        stage2_rejections["empty_anchor"] += 1
                        continue
                    if bool(local_result.get("restated_in_citing_paper", False)) and bool(
                        CONFIG.get("STRICT_QC_REQUIRE_RESTATED_SUPPORT", True)
                    ):
                        if not restated_statement_supported_by_evidence(
                            tool_statement,
                            locator_snippet,
                            local_result["local_context"],
                            local_stage2_text,
                        ):
                            logger.info(
                                "  Downgrading restated_in_citing_paper=false for %s due to weak textual support.",
                                raw_key,
                            )
                            local_result["restated_in_citing_paper"] = False
                    if (
                        bool(CONFIG.get("STRICT_QC_REJECT_STRUCTURAL_LOCAL_CONTEXT", False))
                        and local_context_has_structural_artifacts(local_result["local_context"])
                    ):
                        logger.info(
                            "  Skipping citation %s due to structural local-context artifacts.",
                            raw_key,
                        )
                        stage2_rejections["structural_local_context"] += 1
                        continue
                    if (
                        bool(CONFIG.get("STRICT_QC_REJECT_EMPTY_LOCAL_CONTEXT", False))
                        and not local_result["local_context"]
                    ):
                        empty_context_meta = {
                            **local_stage2_meta,
                            **deterministic_local_meta,
                        }
                        restated_supported = restated_statement_supported_by_evidence(
                            tool_statement,
                            locator_snippet,
                            local_result["local_context"],
                            local_stage2_text,
                        )
                        if not (
                            empty_local_context_is_acceptable(
                                empty_context_meta,
                                locator_snippet,
                            )
                            and bool(local_result.get("restated_in_citing_paper", False))
                            and restated_supported
                        ):
                            logger.info(
                                "  Skipping citation %s due to empty local context without explicit in-paper restatement.",
                                raw_key,
                            )
                            stage2_rejections["empty_local_context"] += 1
                            continue
                    if bool(CONFIG.get("STRICT_QC_REJECT_EMPTY_SETUP", False)) and not str(
                        (global_context or {}).get("setup", "") or ""
                    ).strip():
                        logger.info(
                            "  Skipping citation %s because global setup context is empty after recovery.",
                            raw_key,
                        )
                        stage2_rejections["empty_setup"] += 1
                        continue
                    setup_rejection_reason = setup_text_rejection_reason(
                        str((global_context or {}).get("setup", "") or "")
                    )
                    if setup_rejection_reason:
                        logger.info(
                            "  Skipping citation %s because global setup context failed QC (%s).",
                            raw_key,
                            setup_rejection_reason,
                        )
                        stage2_rejections["bad_setup"] += 1
                        continue
                    instance_signature = build_instance_signature(local_result)
                    if is_near_duplicate_instance(instance_signature, seen_instance_signatures):
                        logger.info("  Skipping near-duplicate instance for citation %s.", raw_key)
                        stage2_rejections["near_duplicate"] += 1
                        continue
                    seen_instance_signatures.append(instance_signature)

                    seed_domain = str(paper.get("domain") or "").strip()
                    domain_label = infer_domain_label(paper, default_category=str(CONFIG["CATEGORY"]))
                    if seed_domain and domain_label != seed_domain:
                        logger.info(
                            "  Domain relabeled for %s: %s -> %s",
                            paper_id,
                            seed_domain,
                            domain_label,
                        )
                    entry = {
                        "instance_id": f"{paper_id}_gap_{int(item['idx'])}",
                        "split": "unassigned",
                        "paper": {
                            "paper_id": paper_id,
                            "title": title,
                            "publication_date": str(paper.get("published", ""))[:10],
                        },
                        "x": {
                            "global_context": global_context,
                            "local_context": local_result.get("local_context", []) or [],
                            "anchor_hint": local_result.get("anchor_hint"),
                        },
                        "y": {
                            "reference_tool_latex": tool_statement,
                            "reference_tool_type": local_result.get("reference_tool_type"),
                            "restated_in_citing_paper": bool(local_result.get("restated_in_citing_paper", False)),
                        },
                        "z": {
                            "citation_key": raw_key,
                            "citation_content": item["bib_text"],
                            "source_type": infer_source_type(item["bib_text"]),
                            "locator": local_result.get("citation_locator") or "",
                            "locator_snippet": locator_snippet,
                            "doi": extract_doi(item["bib_text"]),
                            "arxiv_id": extract_arxiv_id(item["bib_text"]),
                        },
                        "strata": {
                            "domain": domain_label,
                            "tool_family": normalize_tool_family_label(local_result.get("tool_family", "other")),
                        }
                    }

                    f_out.write(json.dumps(entry, ensure_ascii=False) + "\n")
                    f_out.flush()
                    valid_gaps_count += 1

                if valid_gaps_count > 0:
                    paper_status = "completed"
                    paper_detail = "ok"
                elif stage2_rejections.get("locator_not_aligned", 0) == len(citation_candidates):
                    paper_status = "no_locator_aligned_citations"
                    paper_detail = compact_detail(
                        {
                            "candidate_count": len(citation_candidates),
                            "rejections": dict(stage2_rejections),
                        }
                    )
                else:
                    paper_status = "no_valid_stage2_instances"
                    paper_detail = compact_detail(
                        {
                            "candidate_count": len(citation_candidates),
                            "rejections": dict(stage2_rejections),
                        }
                    )
                logger.info(
                    "Paper done for %s. Saved %d valid gaps in %.1fs.",
                    paper_id,
                    valid_gaps_count,
                    time.perf_counter() - paper_started_at,
                )
                total_gaps_saved += valid_gaps_count

            except Exception as e:
                logger.exception("Paper processing failed for %s: %s", paper_id, e)
                paper_status = "error"
                paper_detail = str(e)
            finally:
                elapsed_seconds = time.perf_counter() - paper_started_at
                append_progress_record(
                    f_progress,
                    paper_id=paper_id,
                    title=title,
                    status=paper_status,
                    valid_gaps=valid_gaps_count,
                    detail=paper_detail,
                    elapsed_seconds=elapsed_seconds,
                )
                processed_paper_ids.add(str(paper_id))

    logger.info("All done. Total gaps generated: %d. Output: %s", total_gaps_saved, output_file)


if __name__ == "__main__":
    main()
