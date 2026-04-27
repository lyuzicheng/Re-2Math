from __future__ import annotations

import json
import logging
import math
import re
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import requests


logger = logging.getLogger("RetrievalBackends")


def normalize_text(text: str) -> str:
    return " ".join(str(text or "").strip().split())


def normalize_title(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(text or "").lower())


def tokenize(text: str) -> List[str]:
    return re.findall(r"[a-z0-9]+", str(text or "").lower())


def extract_year(value: Any) -> Optional[int]:
    if value is None:
        return None
    match = re.search(r"(\d{4})", str(value))
    return int(match.group(1)) if match else None


def load_snapshot(path: str | Path) -> List[Dict[str, Any]]:
    snapshot_path = Path(path)
    text = snapshot_path.read_text(encoding="utf-8")
    stripped = text.lstrip()
    if stripped.startswith("["):
        payload = json.loads(text)
        return payload if isinstance(payload, list) else []

    docs: List[Dict[str, Any]] = []
    for line in text.splitlines():
        line = line.strip()
        if line:
            docs.append(json.loads(line))
    return docs


class ScholarRetriever:
    def __init__(self, serpapi_key: str, *, batch_size: int = 20, sleep_sec: float = 0.5):
        self.serpapi_key = serpapi_key
        self.batch_size = batch_size
        self.sleep_sec = sleep_sec

    def search(self, query: str, *, top_k: int, max_year: Optional[int] = None) -> List[Dict[str, Any]]:
        all_results: List[Dict[str, Any]] = []
        for start in range(0, top_k, self.batch_size):
            params: Dict[str, Any] = {
                "engine": "google_scholar",
                "q": query,
                "api_key": self.serpapi_key,
                "num": min(self.batch_size, top_k - start),
                "start": start,
                "hl": "en",
                "gl": "us",
            }
            if max_year:
                params["as_yhi"] = max_year

            response = requests.get("https://serpapi.com/search.json", params=params, timeout=60)
            response.raise_for_status()
            payload = response.json()
            if "error" in payload:
                error_text = str(payload.get("error") or "").strip()
                lowered = error_text.lower()
                if "hasn't returned any results" in lowered or "has not returned any results" in lowered:
                    logger.warning("Scholar returned no results for query=%r start=%d: %s", query, start, error_text)
                    break
                raise RuntimeError(error_text)

            organic = payload.get("organic_results", []) or []
            if not organic:
                break

            for item in organic:
                position = int(item.get("position", 0) or 0)
                publication_info = item.get("publication_info", {}) or {}
                all_results.append(
                    {
                        "title": item.get("title", "") or "",
                        "link": item.get("link", "") or "",
                        "snippet": item.get("snippet", "") or "",
                        "rank": start + position + 1,
                        "backend": "scholar",
                        "result_id": item.get("result_id", "") or "",
                        "resources": item.get("resources", []) or [],
                        "publication_info": publication_info,
                        "publication_summary": publication_info.get("summary", "") or "",
                        "inline_links": item.get("inline_links", {}) or {},
                        "year": extract_year(publication_info.get("summary", "") or item.get("snippet", "")),
                    }
                )

            if self.sleep_sec > 0:
                time.sleep(self.sleep_sec)

        return all_results[:top_k]


class OfflineBM25Retriever:
    def __init__(self, snapshot_path: str | Path, *, field: str):
        if field not in {"metadata_text", "fulltext_text"}:
            raise ValueError(f"Unsupported field: {field}")
        self.snapshot_path = str(snapshot_path)
        self.field = field
        self.docs = load_snapshot(snapshot_path)
        self._build_index()

    def _build_index(self) -> None:
        self.term_freqs: List[Counter[str]] = []
        self.doc_lengths: List[int] = []
        self.doc_freqs: Dict[str, int] = defaultdict(int)
        self.postings: Dict[str, List[int]] = defaultdict(list)

        for idx, doc in enumerate(self.docs):
            tokens = tokenize(doc.get(self.field, ""))
            tf = Counter(tokens)
            self.term_freqs.append(tf)
            self.doc_lengths.append(len(tokens))
            for token in tf:
                self.doc_freqs[token] += 1
                self.postings[token].append(idx)

        self.total_docs = len(self.docs)
        self.avg_doc_length = sum(self.doc_lengths) / self.total_docs if self.total_docs else 0.0

    def _idf(self, token: str) -> float:
        df = self.doc_freqs.get(token, 0)
        if df <= 0:
            return 0.0
        return math.log(1 + (self.total_docs - df + 0.5) / (df + 0.5))

    def _candidate_indices(self, query_tokens: Iterable[str], max_year: Optional[int]) -> List[int]:
        candidates = set()
        for token in query_tokens:
            candidates.update(self.postings.get(token, []))
        if not candidates:
            return []
        if max_year is None:
            return sorted(candidates)
        filtered: List[int] = []
        for idx in sorted(candidates):
            year = extract_year(self.docs[idx].get("year"))
            if year is None or year <= max_year:
                filtered.append(idx)
        return filtered

    def search(self, query: str, *, top_k: int, max_year: Optional[int] = None) -> List[Dict[str, Any]]:
        query_tokens = tokenize(query)
        if not query_tokens:
            return []

        k1 = 1.5
        b = 0.75
        scores: List[tuple[int, float]] = []
        for idx in self._candidate_indices(query_tokens, max_year):
            tf = self.term_freqs[idx]
            doc_len = self.doc_lengths[idx] or 1
            score = 0.0
            for token in query_tokens:
                freq = tf.get(token, 0)
                if not freq:
                    continue
                denom = freq + k1 * (1 - b + b * doc_len / max(self.avg_doc_length, 1.0))
                score += self._idf(token) * (freq * (k1 + 1)) / denom
            if score > 0:
                scores.append((idx, score))

        scores.sort(key=lambda item: item[1], reverse=True)
        results: List[Dict[str, Any]] = []
        for rank, (idx, score) in enumerate(scores[:top_k], start=1):
            doc = self.docs[idx]
            snippet_source = normalize_text(doc.get(self.field, ""))[:500]
            results.append(
                {
                    "title": doc.get("title", "") or "",
                    "link": doc.get("source_url", "") or doc.get("source_path", "") or "",
                    "snippet": snippet_source,
                    "rank": rank,
                    "backend": f"offline_bm25_{'metadata' if self.field == 'metadata_text' else 'fulltext'}",
                    "backend_doc_id": doc.get("doc_id", ""),
                    "backend_score": score,
                    "local_pdf_path": doc.get("source_path", "") or "",
                    "source_type": doc.get("source_type", "") or "",
                    "year": doc.get("year"),
                }
            )
        return results


def build_retriever(
    backend: str,
    *,
    serpapi_key: Optional[str] = None,
    snapshot_path: Optional[str] = None,
    batch_size: int = 20,
    sleep_sec: float = 0.5,
):
    if backend == "scholar":
        if not serpapi_key:
            raise RuntimeError("Scholar backend requires SERPAPI_API_KEY or SERPAPI_KEY.")
        return ScholarRetriever(serpapi_key, batch_size=batch_size, sleep_sec=sleep_sec)
    if backend == "offline_metadata_bm25":
        if not snapshot_path:
            raise RuntimeError("Offline BM25 backend requires --snapshot.")
        return OfflineBM25Retriever(snapshot_path, field="metadata_text")
    if backend == "offline_fulltext_bm25":
        if not snapshot_path:
            raise RuntimeError("Offline BM25 backend requires --snapshot.")
        return OfflineBM25Retriever(snapshot_path, field="fulltext_text")
    raise ValueError(f"Unsupported backend: {backend}")
