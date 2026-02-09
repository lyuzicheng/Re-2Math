import os
import re
import json
import logging
import datetime
import tempfile
import sys
from typing import List, Dict, Optional, Tuple

import requests
import httpx
from openai import OpenAI
from json_repair import repair_json

# Local modules (expected to be in the same repo)
from arxiv_retriever import ArxivMathPaperRetriever
from extract_latex_text import ArxivLatexExtractor


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
    "ID_LIST_FILE": os.getenv("ARXIV_ID_LIST_FILE", "arxiv_ids.txt"),
    "CATEGORY": os.getenv("ARXIV_CATEGORY", "math.PR"),
    "MAX_PAPERS": int(os.getenv("MAX_PAPERS", "50")),
    "TIME_WINDOW_DAYS": int(os.getenv("TIME_WINDOW_DAYS", "180")),
    "OUTPUT_FILE": os.getenv("OUTPUT_FILE", "benchmark_dataset.jsonl"),

    # Safety / resource limits
    "STAGE1_MAX_CHARS": int(os.getenv("STAGE1_MAX_CHARS", "100000")),
    "STAGE2_MAX_CHARS": int(os.getenv("STAGE2_MAX_CHARS", "150000")),
}


# ============================================================
# Logging
# ============================================================

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("BenchmarkBuilder")


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


def normalize_key(key: str) -> str:
    """Normalize BibTeX keys to improve matching across files."""
    if not key:
        return ""
    return re.sub(r"[^a-z0-9]", "", key.lower())


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
    try:
        api_key = require_api_key()
        base_url = str(CONFIG["BASE_URL"])
        model_name = str(CONFIG["MODEL_NAME"])

        client = OpenAI(api_key=api_key, base_url=base_url, timeout=10.0)
        client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": "ping"}],
            max_tokens=5,
        )
        logger.info("API connection OK.")
        return True
    except Exception as e:
        logger.error("API connection failed: %s", e)
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
        timeout_config = httpx.Timeout(1800.0, connect=60.0)
        custom_http_client = httpx.Client(timeout=timeout_config)

        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url,
            http_client=custom_http_client,
            max_retries=3,
        )
        self.model_name = model_name

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
            parsed = repair_json(content, return_objects=True)

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

    # --------------------------
    # Stage 1
    # --------------------------
    def stage1_analyze_structure(self, full_text: str, bib_mapping: Dict[str, str]) -> Optional[Dict]:
        bib_context = "\n".join(
            [f"Key: '{k}' -> {v[:200]}" for k, v in list(bib_mapping.items())[:200]]
        )

        system_prompt = r"""
You are a Mathematical Structure Analyzer.
Your goal is to curate a dataset for "Retrieval-Augmented Theorem Proving".
We ONLY want citations where the author invokes an external mathematical result
(Lemma, Theorem, Inequality) to bridge a logical gap.

INPUT DATA
1) BIB_MAPPING: available citations
2) FULL_TEXT: raw paper content

PART 1: GLOBAL CONTEXT
Extract the common mathematical setup needed to understand the paper's main theorem.
- setup: global definitions, probability spaces, and active assumptions
  - Must be self-contained; resolve internal references like "Assumption A".
- goal: the main theorem statement (mathematical statement only)

PART 2: PROOF CITATION SCOUTING (STRICT FILTER)
Scan proofs and identify citations used as mathematical tools.

KEEP: citation provides a specific result/bound/inequality that advances the proof.
Examples: "By Lemma 3.1 of [X] ...", "Using the concentration inequality from [Y] ..."

DISCARD: citation used only for definitions/setup/history.
DISCARD: standard/foundational references unless a specific non-trivial theorem is invoked.

For each valid tool usage, output:
- citation_key: exact BibTeX key
- locator_snippet: unique 20-40 word snippet surrounding the citation
- reason: short justification that this is tool usage

OUTPUT (JSON ONLY)
{
  "global_context": {"setup": "...", "goal": "..."},
  "proof_citations": [
    {"citation_key": "...", "locator_snippet": "...", "reason": "..."}
  ]
}
"""

        try:
            logger.info("Stage 1: analyzing structure and scouting citations ...")
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {
                        "role": "user",
                        "content": (
                            f"BIBLIOGRAPHY:\n{bib_context}\n\n"
                            f"FULL TEXT:\n{full_text[:int(CONFIG['STAGE1_MAX_CHARS'])]}"
                        ),
                    },
                ],
                temperature=0.0,
                max_tokens=8000,
            )
            return self.clean_json(response.choices[0].message.content)

        except Exception as e:
            logger.error("Stage 1 error: %s", e)
            return None

    # --------------------------
    # Stage 2
    # --------------------------
    def stage2_extract_local_context(
        self, full_text: str, citation_info: Dict, bib_content: str
    ) -> Optional[Dict]:
        target_key = citation_info.get("citation_key", "")
        snippet = citation_info.get("locator_snippet", "")

        system_prompt = r"""
You are a Mathematical Logic Expert.
Extract the local logical state immediately before a specific citation is applied.

INPUT
- Target citation key
- Location snippet

OUTPUT FIELDS
1) anchor_latex: statement established immediately before the citation
2) gap_objective: what needs to be shown next (do NOT leak the cited result)
3) target_lemma_latex: the specific theorem/inequality being applied from the cited work
   If cited content is provided, extract the exact bound/theorem statement.

Return JSON only:
{
  "anchor_latex": "...",
  "gap_objective": "...",
  "target_lemma_latex": "..."
}
"""
        user_msg = (
            f"TARGET: {target_key}\n"
            f"CITATION_CONTENT: {bib_content}\n"
            f"LOCATOR: {snippet}\n\n"
            f"TEXT:\n{full_text[:int(CONFIG['STAGE2_MAX_CHARS'])]}"
        )

        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_msg},
                ],
                temperature=0.0,
                max_tokens=10000,
            )
            raw = response.choices[0].message.content
            parsed = self.clean_json(raw)
            if not parsed:
                logger.debug("Stage 2 JSON parse failed for key=%s. Raw head=%r", target_key, raw[:400])
            return parsed

        except Exception as e:
            logger.error("Stage 2 error for key=%s: %s", target_key, e)
            return None


# ============================================================
# Main
# ============================================================

def main() -> None:
    if not test_api_connection():
        sys.exit(1)

    api_key = require_api_key()
    base_url = str(CONFIG["BASE_URL"])
    model_name = str(CONFIG["MODEL_NAME"])

    start_time = datetime.datetime.now() - datetime.timedelta(days=int(CONFIG["TIME_WINDOW_DAYS"]))
    retriever = ArxivMathPaperRetriever(start_time=start_time, category=str(CONFIG["CATEGORY"]))

    # Load IDs if provided; otherwise use time window retrieval.
    target_ids = load_ids_from_file(str(CONFIG.get("ID_LIST_FILE", "")))

    if target_ids:
        logger.info("Mode: explicit ID list (%d papers).", len(target_ids))
        papers_metadata = retriever.retrieve_papers_by_ids(list(set(target_ids)))
    else:
        logger.info("Mode: time window retrieval (category=%s).", CONFIG["CATEGORY"])
        papers_metadata = retriever.retrieve_papers(max_results=int(CONFIG["MAX_PAPERS"]))

    if not papers_metadata:
        logger.warning("No papers retrieved.")
        return

    extractor = ReasoningAwareExtractor()
    generator = DatasetGenerator(api_key, base_url, model_name)

    output_file = str(CONFIG["OUTPUT_FILE"])
    total_gaps_saved = 0

    with open(output_file, "a", encoding="utf-8") as f_out:
        for i, paper in enumerate(papers_metadata):
            paper_id = paper.get("id", "")
            title = paper.get("title", "")
            logger.info("[%d/%d] Processing %s: %s", i + 1, len(papers_metadata), paper_id, title)

            # 1) Download and extract text + bibliography
            success, full_text, full_bib = extractor.process_paper_with_bib(paper_id, paper.get("latex_link", ""))
            if not success or len(full_text) < 1000:
                logger.warning("Content extraction failed for %s.", paper_id)
                continue

            # Normalize bib keys for lookup
            norm_bib = {normalize_key(k): v for k, v in full_bib.items()}

            # Stage 1
            stage1_result = generator.stage1_analyze_structure(full_text, full_bib)
            if not stage1_result:
                logger.warning("Stage 1 failed for %s.", paper_id)
                continue

            global_context = stage1_result.get("global_context", {}) or {}
            citations_list = stage1_result.get("proof_citations", []) or []
            if not citations_list:
                logger.info("No proof citations found for %s.", paper_id)
                continue

            logger.info("Stage 1 done for %s. Found %d citations.", paper_id, len(citations_list))

            # Stage 2
            valid_gaps_count = 0
            for idx, cit_item in enumerate(citations_list):
                raw_key = str(cit_item.get("citation_key", "") or "")
                norm_k = normalize_key(raw_key)
                bib_text = norm_bib.get(norm_k, "Citation content not found")

                logger.info("  [%d/%d] Mining citation: %s", idx + 1, len(citations_list), raw_key)

                local_result = generator.stage2_extract_local_context(full_text, cit_item, bib_text)
                if not (local_result and isinstance(local_result, dict)):
                    continue

                entry = {
                    "id": f"{paper_id}_gap_{idx}",
                    "arxiv_id": paper_id,
                    "title": title,
                    "publication_date": str(paper.get("published", ""))[:10],
                    "problem_input": {
                        "global_context": global_context,
                        "local_context": {
                            "anchor_latex": local_result.get("anchor_latex"),
                            "gap_objective": local_result.get("gap_objective"),
                        },
                    },
                    "ground_truth": {
                        "target_citation_key": raw_key,
                        "target_lemma_latex": local_result.get("target_lemma_latex"),
                        "target_citation_content": bib_text,
                    },
                }

                f_out.write(json.dumps(entry, ensure_ascii=False) + "\n")
                f_out.flush()
                valid_gaps_count += 1

            logger.info("Paper done for %s. Saved %d valid gaps.", paper_id, valid_gaps_count)
            total_gaps_saved += valid_gaps_count

    logger.info("All done. Total gaps generated: %d. Output: %s", total_gaps_saved, output_file)


if __name__ == "__main__":
    main()
