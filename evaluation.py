import os
import json
import re
import time
import logging
import argparse
from dataclasses import dataclass
from typing import Dict, Optional, Any, List, Tuple

import fitz  # PyMuPDF
from difflib import SequenceMatcher
from tqdm import tqdm
from openai import OpenAI


# ============================================================
# Logging
# ============================================================

def setup_logging(level: str) -> logging.Logger:
    logging.basicConfig(
        level=level.upper(),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    return logging.getLogger("SemanticEval")


logger = setup_logging(os.getenv("LOG_LEVEL", "INFO"))


# ============================================================
# Config / Secrets (env-only, safe for GitHub)
# ============================================================

@dataclass(frozen=True)
class LLMConfig:
    api_key: str
    base_url: str
    model_name: str
    timeout_sec: float = 120.0


def getenv_required(name: str) -> str:
    val = (os.getenv(name) or "").strip()
    if not val:
        raise RuntimeError(
            f"Missing required environment variable: {name}\n"
            f"Do NOT hardcode API keys in source. Set it via env vars."
        )
    return val


def build_llm_config_from_env(prefix: str, fallback: Optional[LLMConfig] = None) -> LLMConfig:
    """
    Build an LLM config from environment variables.

    Required if fallback is None:
        {prefix}_API_KEY, {prefix}_BASE_URL, {prefix}_MODEL_NAME

    If fallback is provided:
        any missing field falls back to fallback's value.
    """
    api_key = (os.getenv(f"{prefix}_API_KEY") or "").strip()
    base_url = (os.getenv(f"{prefix}_BASE_URL") or "").strip()
    model_name = (os.getenv(f"{prefix}_MODEL_NAME") or "").strip()
    timeout = (os.getenv(f"{prefix}_TIMEOUT_SEC") or "").strip()

    if fallback is None:
        if not api_key:
            api_key = getenv_required(f"{prefix}_API_KEY")
        if not base_url:
            base_url = getenv_required(f"{prefix}_BASE_URL")
        if not model_name:
            model_name = getenv_required(f"{prefix}_MODEL_NAME")
    else:
        api_key = api_key or fallback.api_key
        base_url = base_url or fallback.base_url
        model_name = model_name or fallback.model_name

    timeout_sec = float(timeout) if timeout else (fallback.timeout_sec if fallback else 120.0)

    return LLMConfig(
        api_key=api_key,
        base_url=base_url,
        model_name=model_name,
        timeout_sec=timeout_sec,
    )


class LLMJsonCaller:
    """
    A small wrapper so we can use different models/clients for:
      - solver (full-text scanning / extraction)
      - judge (title extraction + math equivalence)
    """
    def __init__(self, cfg: LLMConfig):
        self.cfg = cfg
        self.client = OpenAI(api_key=cfg.api_key, base_url=cfg.base_url, timeout=cfg.timeout_sec)

    @staticmethod
    def robust_json_parse(llm_output: str) -> Dict[str, Any]:
        if not llm_output:
            return {}

        text = re.sub(r"^```json\s*", "", llm_output, flags=re.MULTILINE)
        text = re.sub(r"^```\s*", "", text, flags=re.MULTILINE)
        text = re.sub(r"```$", "", text, flags=re.MULTILINE).strip()

        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        # Try Python literal dict (some models do this)
        try:
            import ast
            obj = ast.literal_eval(text)
            if isinstance(obj, dict):
                return obj
            return {}
        except Exception:
            pass

        # Last resort: try to salvage key fields
        try:
            score_match = re.search(r'"relevance_score"\s*:\s*(\d+)', text)
            score = int(score_match.group(1)) if score_match else 0

            theorem_match = re.search(r'"extracted_theorem"\s*:\s*"(.*)"', text, re.DOTALL)
            theorem = theorem_match.group(1) if theorem_match else None

            return {
                "relevance_score": score,
                "reasoning": "JSON parse failed (regex recovered).",
                "extracted_theorem": theorem,
            }
        except Exception:
            return {}

    def call_json(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.0,
        max_tokens: int = 4000,
    ) -> Dict[str, Any]:
        try:
            resp = self.client.chat.completions.create(
                model=self.cfg.model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=temperature,
                max_tokens=max_tokens,
            )
            content = (resp.choices[0].message.content or "").strip()
            return self.robust_json_parse(content)
        except Exception as e:
            logger.error("LLM call failed (model=%s): %s", self.cfg.model_name, e)
            return {}


# ============================================================
# Utils: PDF reading, text cleaning
# ============================================================

class Utils:
    @staticmethod
    def clean_text(text: str) -> str:
        if not text:
            return ""
        return re.sub(r"\s+", " ", text).strip()

    @staticmethod
    def read_pdf_header(pdf_path: str, char_limit: int = 4000) -> str:
        try:
            doc = fitz.open(pdf_path)
            text = ""
            for i in range(min(2, len(doc))):
                text += doc[i].get_text()
            doc.close()
            return Utils.clean_text(text)[:char_limit]
        except Exception:
            return ""

    @staticmethod
    def read_full_pdf(pdf_path: str, char_limit: int = 2_000_000) -> str:
        try:
            doc = fitz.open(pdf_path)
            text = ""
            for page in doc:
                text += page.get_text()
            doc.close()
            return Utils.clean_text(text)[:char_limit]
        except Exception:
            return ""


# ============================================================
# Solver: scan full texts and select a candidate paper
# ============================================================

class FullTextSolver:
    def __init__(
        self,
        task_id: str,
        query_context: str,
        folder_path: str,
        solver_llm: LLMJsonCaller,
        per_paper_prompt_chars: int = 150_000,
        early_stop_score: int = 9,
    ):
        self.task_id = task_id
        self.query_context = query_context
        self.folder_path = folder_path
        self.solver_llm = solver_llm
        self.per_paper_prompt_chars = per_paper_prompt_chars
        self.early_stop_score = early_stop_score

        self.pdf_files = sorted(
            [f for f in os.listdir(folder_path) if f.lower().endswith(".pdf")]
        )

    def evaluate_single_paper(self, filename: str) -> Dict[str, Any]:
        path = os.path.join(self.folder_path, filename)
        content = Utils.read_full_pdf(path)

        if not content:
            return {
                "filename": filename,
                "relevance_score": 0,
                "reasoning": "Empty content",
                "extracted_theorem": None,
            }

        prompt = f"""
You are a mathematical assistant for a "Math Gap-Filling" task.

=== PROBLEM CONTEXT ===
{self.query_context}

=== CANDIDATE PAPER ===
Filename: "{filename}"
Content (truncated):
{content[:self.per_paper_prompt_chars]}

=== TASK ===
1) Decide whether this paper contains the specific theorem/lemma needed to bridge the gap described in the Local Context.
2) Output a relevance score (0-10):
   - 10 = high confidence this is the exact source containing the needed lemma.
   - 0 = irrelevant.
3) If score > 5, extract the relevant theorem/lemma LaTeX (as faithfully as possible).

Return JSON ONLY:
{{
  "filename": "{filename}",
  "relevance_score": <int 0-10>,
  "reasoning": "<short explanation>",
  "extracted_theorem": "<latex string or null>"
}}
"""

        data = self.solver_llm.call_json(
            system_prompt="You are a math expert assistant.",
            user_prompt=prompt,
            temperature=0.0,
            max_tokens=4000,
        )

        if not isinstance(data, dict):
            return {
                "filename": filename,
                "relevance_score": 0,
                "reasoning": "Invalid LLM output format",
                "extracted_theorem": None,
            }

        return {
            "filename": filename,
            "relevance_score": int(data.get("relevance_score", 0) or 0),
            "reasoning": str(data.get("reasoning", "") or ""),
            "extracted_theorem": data.get("extracted_theorem", None),
        }

    def run(self) -> Dict[str, Any]:
        if not self.pdf_files:
            return {"error": "no_pdfs"}

        logger.info("Task %s: scanning %d PDFs...", self.task_id, len(self.pdf_files))

        candidates: List[Dict[str, Any]] = []
        for f in self.pdf_files:
            result = self.evaluate_single_paper(f)
            candidates.append(result)

            if result.get("relevance_score", 0) >= self.early_stop_score:
                break

        if not candidates:
            return {"error": "no_candidates"}

        best = max(candidates, key=lambda x: int(x.get("relevance_score", 0) or 0))

        return {
            "selected_filename": best.get("filename"),
            "score": best.get("relevance_score"),
            "reasoning": best.get("reasoning"),
            "extracted_theorem": best.get("extracted_theorem"),
            "all_candidates": candidates,
        }


# ============================================================
# Evaluator: title matching + math equivalence (use judge model)
# ============================================================

class SemanticEvaluator:
    def __init__(self, folder_path: str, gt_data: Dict[str, str], judge_llm: LLMJsonCaller):
        self.folder_path = folder_path
        self.gt_latex = gt_data.get("target_lemma_latex", "") or ""
        self.gt_citation_str = gt_data.get("target_citation_content", "") or ""
        self.judge_llm = judge_llm

    def extract_title_from_string(self, text: str, is_citation: bool = False) -> str:
        context_desc = "citation string" if is_citation else "first page of a PDF"
        prompt = f"""
Extract the academic paper/book TITLE from the following text.

Text ({context_desc}):
{text[:2000]}

Return JSON ONLY:
{{ "extracted_title": "<string>" }}
"""
        res = self.judge_llm.call_json(
            system_prompt="You extract paper titles from messy text.",
            user_prompt=prompt,
            temperature=0.0,
            max_tokens=256,
        )
        if isinstance(res, dict):
            return (res.get("extracted_title", "") or "").strip()
        return ""

    def check_retrieval_correctness(self, selected_filename: Optional[str]) -> Tuple[bool, str, str, str]:
        if not selected_filename:
            return False, "No file selected", "", ""

        path = os.path.join(self.folder_path, selected_filename)

        # GT title
        gt_title = self.extract_title_from_string(self.gt_citation_str, is_citation=True)
        if not gt_title:
            gt_title = (self.gt_citation_str[:100] or "").strip()

        # PDF title from header
        pdf_header_text = Utils.read_pdf_header(path)
        pdf_title = self.extract_title_from_string(pdf_header_text, is_citation=False)

        norm_gt = re.sub(r"[^a-z0-9]", "", gt_title.lower())
        norm_pdf = re.sub(r"[^a-z0-9]", "", pdf_title.lower())

        if not norm_gt or not norm_pdf:
            return False, "Title extraction failed", gt_title, pdf_title

        reason = f"GT title: '{gt_title}' vs PDF title: '{pdf_title}'"

        if norm_gt in norm_pdf or norm_pdf in norm_gt:
            return True, reason, gt_title, pdf_title

        similarity = SequenceMatcher(None, norm_gt, norm_pdf).ratio()
        if similarity > 0.8:
            return True, reason + f" (similarity={similarity:.2f})", gt_title, pdf_title

        return False, reason + f" (similarity={similarity:.2f})", gt_title, pdf_title

    def judge_math_equivalence(self, extracted_latex: Optional[str]) -> Tuple[bool, str]:
        if not extracted_latex or not self.gt_latex:
            return False, "Missing content"

        prompt = f"""
Ground Truth:
{self.gt_latex}

Student Extraction:
{extracted_latex}

Task:
Is the Student Extraction mathematically equivalent to (or stronger than) the Ground Truth?
Ignore trivial notation differences (variable renaming, rearrangements).

Return JSON ONLY:
{{ "is_match": <bool>, "reason": "<short reason>" }}
"""
        res = self.judge_llm.call_json(
            system_prompt="You are a strict mathematical equivalence judge.",
            user_prompt=prompt,
            temperature=0.0,
            max_tokens=512,
        )
        if isinstance(res, dict):
            return bool(res.get("is_match", False)), str(res.get("reason", "") or "")
        return False, "Judge JSON parse failed"


# ============================================================
# Main
# ============================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Full-text semantic evaluation with separate solver and judge models.")
    p.add_argument("--pdf-root", default="scholar_all_top20_downloads_gemini", help="Root folder containing per-task PDF folders.")
    p.add_argument("--dataset-file", default="benchmark_dataset.jsonl", help="Path to dataset JSONL.")
    p.add_argument("--result-file", default="evaluation_results_full_scan.json", help="Where to save results JSON.")
    p.add_argument("--per-paper-prompt-chars", type=int, default=150_000, help="Max chars per paper passed to solver prompt.")
    p.add_argument("--early-stop-score", type=int, default=9, help="Early stop if solver finds a candidate with >= this score.")
    p.add_argument("--log-level", default=os.getenv("LOG_LEVEL", "INFO"))
    return p.parse_args()


def load_dataset_as_dict(path: str) -> Dict[str, Any]:
    dataset: Dict[str, Any] = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            d = json.loads(line)
            dataset[d["id"]] = d
    return dataset


def main() -> None:
    args = parse_args()
    global logger
    logger = setup_logging(args.log_level)

    if not os.path.exists(args.dataset_file):
        raise FileNotFoundError(f"Dataset file not found: {args.dataset_file}")
    if not os.path.exists(args.pdf_root):
        raise FileNotFoundError(f"PDF root not found: {args.pdf_root}")

    # Build separate model configs:
    # - solver: required
    # - judge: optional (fallback to solver if not provided)
    solver_cfg = build_llm_config_from_env("SOLVER", fallback=None)
    judge_cfg = build_llm_config_from_env("JUDGE", fallback=solver_cfg)

    logger.info("Solver model: %s", solver_cfg.model_name)
    logger.info("Judge  model: %s", judge_cfg.model_name)

    solver_llm = LLMJsonCaller(solver_cfg)
    judge_llm = LLMJsonCaller(judge_cfg)

    dataset = load_dataset_as_dict(args.dataset_file)

    task_folders = [
        d for d in os.listdir(args.pdf_root)
        if os.path.isdir(os.path.join(args.pdf_root, d))
    ]

    results: List[Dict[str, Any]] = []
    stats = {"total": 0, "retrieved_correctly": 0, "latex_matched": 0}

    logger.info("Starting evaluation on %d task folders.", len(task_folders))

    for case_id in tqdm(task_folders, desc="Tasks"):
        if case_id not in dataset:
            continue

        folder_path = os.path.join(args.pdf_root, case_id)
        pdfs = [f for f in os.listdir(folder_path) if f.lower().endswith(".pdf")]
        if not pdfs:
            continue

        stats["total"] += 1
        task_data = dataset[case_id]

        pi = task_data.get("problem_input", {})
        gc = pi.get("global_context", {})
        lc = pi.get("local_context", {})

        setup_text = gc.get("setup", "") or ""
        goal_text = gc.get("goal", "") or ""
        anchor_text = lc.get("anchor_latex", "") or ""
        objective_text = lc.get("gap_objective", "") or ""

        query_context = (
            "[Global Context]\n"
            f"Setup: {setup_text}\n"
            f"Goal: {goal_text}\n\n"
            "[Local Context / The Gap]\n"
            f"Anchor Latex: {anchor_text}\n"
            f"Objective: {objective_text}\n"
        )

        gt_data = {
            "title": task_data.get("title", "") or "",
            "target_lemma_latex": (task_data.get("ground_truth", {}) or {}).get("target_lemma_latex", "") or "",
            "target_citation_content": (task_data.get("ground_truth", {}) or {}).get("target_citation_content", "") or "",
        }

        # 1) Solver uses SOLVER model
        solver = FullTextSolver(
            task_id=case_id,
            query_context=query_context,
            folder_path=folder_path,
            solver_llm=solver_llm,
            per_paper_prompt_chars=args.per_paper_prompt_chars,
            early_stop_score=args.early_stop_score,
        )
        solver_out = solver.run()
        if "error" in solver_out:
            logger.warning("Solver error for %s: %s", case_id, solver_out["error"])
            continue

        # 2) Evaluator uses JUDGE model
        evaluator = SemanticEvaluator(folder_path, gt_data, judge_llm=judge_llm)

        sel_file = solver_out.get("selected_filename")
        is_retrieved, retr_reason, gt_title, pdf_title = evaluator.check_retrieval_correctness(sel_file)

        ext_latex = solver_out.get("extracted_theorem")
        is_matched, match_reason = evaluator.judge_math_equivalence(ext_latex) if ext_latex else (False, "No extraction")

        if is_retrieved:
            stats["retrieved_correctly"] += 1
        if is_matched:
            stats["latex_matched"] += 1

        logger.info(
            "Case=%s file=%s retrieved=%s matched=%s",
            case_id, sel_file, is_retrieved, is_matched
        )

        results.append(
            {
                "id": case_id,
                "selected_file": sel_file,
                "is_retrieved": is_retrieved,
                "retrieval_reason": retr_reason,
                "gt_title": gt_title,
                "pdf_title": pdf_title,
                "is_matched": is_matched,
                "match_reason": match_reason,
                "gt_latex": gt_data["target_lemma_latex"],
                "extracted_latex": ext_latex,
                "solver_output": solver_out,
                "meta": {
                    "solver_model": solver_cfg.model_name,
                    "judge_model": judge_cfg.model_name,
                },
            }
        )

    total = stats["total"]
    if total > 0:
        logger.info("Retrieval accuracy: %.2f%%", 100.0 * stats["retrieved_correctly"] / total)
        logger.info("Latex match accuracy: %.2f%%", 100.0 * stats["latex_matched"] / total)

    with open(args.result_file, "w", encoding="utf-8") as f:
        json.dump(
            {
                "stats": stats,
                "results": results,
                "meta": {
                    "pdf_root": args.pdf_root,
                    "dataset_file": args.dataset_file,
                    "solver_model": solver_cfg.model_name,
                    "judge_model": judge_cfg.model_name,
                },
            },
            f,
            indent=2,
            ensure_ascii=False,
        )

    logger.info("Saved to %s", args.result_file)


if __name__ == "__main__":
    main()
