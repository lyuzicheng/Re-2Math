import re
from difflib import SequenceMatcher
from typing import Any, Dict, List, Tuple

TITLE_HINT_WORDS = {
    "of",
    "on",
    "for",
    "with",
    "from",
    "through",
    "under",
    "over",
    "via",
    "in",
    "to",
    "by",
    "without",
    "between",
    "various",
}

VENUE_PATTERNS = [
    r"\bacm transactions\b",
    r"\bjournal\b",
    r"\btransactions\b",
    r"\bproceedings\b",
    r"\bselecta\b",
    r"\bimrn\b",
    r"\bprobab\b",
    r"\btheory related fields\b",
    r"\bint\.\s*math\b",
    r"\bres\.\s*not\b",
    r"\bj\.\s*[a-z]",
    r"\bj\.\s*algebra\b",
    r"\bj\.\s*amer\b",
    r"\bmath\.\s*res\b",
    r"\bmath\.\s*sci\b",
    r"\bcambridge\b",
    r"\bpress\b",
    r"\bpublisher\b",
    r"\bvol\.?\b",
    r"\bvolume\b",
    r"\bpp\.?\b",
    r"\bpages\b",
    r"\bpaper no\.?\b",
    r"\bpublished online\b",
    r"\bdoi\b",
    r"\barxiv\b",
]

BOOK_OR_COLLECTION_MARKERS = {
    "sampler",
    "handbook",
    "lecture",
    "lectures",
    "notes",
    "survey",
    "proceedings",
    "collection",
    "series",
    "volume",
}


def _tokenize_words(text: str) -> List[str]:
    return re.findall(r"[A-Za-z]+(?:'[A-Za-z]+)?", text or "")


def _split_segments(text: str) -> List[str]:
    return [part.strip(" ,.;:") for part in (text or "").split(",") if part.strip(" ,.;:")]


def clean_latex_text(text: str) -> str:
    if not text:
        return ""
    text = re.sub(r"\\[a-zA-Z]+\*?(?:\[[^\]]*\])?\{([^{}]*)\}", r" \1 ", text)
    text = re.sub(r"\\penalty\d+", " ", text)
    text = re.sub(
        r"\\(?:newblock|bgroup|egroup|scshape|bysame|emph|em|textit|textbf|textsc|itshape|bfseries|href|nolinkurl|doi|MR)\s*",
        " ",
        text,
    )
    text = re.sub(r"https?://\S+", " ", text)
    text = text.replace("~", " ")
    text = text.replace("--", " ")
    text = text.replace("–", " ")
    text = text.replace("\\", " ")
    text = re.sub(r'[{}"\u201c\u201d`´]', " ", text)
    text = re.sub(r"[^A-Za-z0-9\s]", " ", text)
    return " ".join(text.split()).lower()


def preprocess_citation_text(text: str) -> str:
    text = text or ""
    text = text.replace("\\newblock", " ||NEWBLOCK|| ")
    text = re.sub(r"\\penalty\d+", " ", text)
    text = re.sub(r"\\[a-zA-Z]+\*?(?:\[[^\]]*\])?\{([^{}]*)\}", r" \1 ", text)
    text = re.sub(
        r"\\(?:bgroup|egroup|scshape|bysame|emph|em|textit|textbf|textsc|itshape|bfseries|href|nolinkurl|doi|MR)\s*",
        " ",
        text,
    )
    text = re.sub(r"https?://\S+", " ", text)
    text = text.replace("~", " ")
    text = text.replace("--", " - ")
    text = text.replace("–", " - ")
    text = text.replace("\\", " ")
    text = re.sub(r'[{}"\u201c\u201d`´]', " ", text)
    text = re.sub(r"\s+", " ", text).strip(" ,.;")
    return text


def looks_like_venue(text: str) -> bool:
    lowered = (text or "").lower()
    if re.search(r"\b(19|20)\d{2}\b", lowered):
        return True
    if any(re.search(pattern, lowered, flags=re.IGNORECASE) for pattern in VENUE_PATTERNS):
        return True
    if len(re.findall(r"[A-Za-z]{1,6}\.", text or "")) >= 2:
        return True
    return False


def is_authorish_segment(segment: str) -> bool:
    tokens = _tokenize_words(segment)
    lower_tokens = [token.lower() for token in tokens]
    if not tokens:
        return False
    if any(token in TITLE_HINT_WORDS for token in lower_tokens):
        return False
    if len(tokens) == 1 and len(tokens[0]) <= 2:
        return True
    if len(tokens) <= 2 and sum(len(token) <= 2 for token in tokens) >= 1:
        return True
    if len(tokens) <= 8 and lower_tokens[0] == "and":
        return True
    if len(tokens) <= 8:
        initialish = sum(len(token) <= 2 for token in tokens)
        capitalized = sum(token[:1].isupper() for token in tokens)
        if initialish >= 1:
            return True
        if capitalized >= max(2, len(tokens) - 1):
            return True
    return False


def strip_leading_author_segments(text: str) -> str:
    parts = _split_segments(text)
    while len(parts) >= 2 and is_authorish_segment(parts[0]):
        parts = parts[1:]
    out = ", ".join(parts).strip(" ,.;:")
    return re.sub(r"^(?:and|with)\s+", "", out, flags=re.IGNORECASE)


def _is_collection_like(segment: str) -> bool:
    words = {token.lower() for token in _tokenize_words(segment)}
    return bool(words & BOOK_OR_COLLECTION_MARKERS)


def _expand_title_candidates(text: str, base_bonus: int, source_label: str) -> List[Tuple[str, int, str]]:
    stripped = strip_leading_author_segments(text)
    if not stripped:
        return []

    segments = _split_segments(stripped)
    if not segments:
        segments = [stripped.strip(" ,.;:")]

    candidates: List[Tuple[str, int, str]] = []
    max_segments = min(4, len(segments))
    for span in range(1, max_segments + 1):
        prefix_segments = segments[:span]
        if span >= 2 and looks_like_venue(prefix_segments[-1]):
            break

        candidate = ", ".join(prefix_segments).strip(" ,.;:")
        if not candidate:
            continue

        bonus = base_bonus
        if span >= 2 and _is_collection_like(prefix_segments[-1]):
            bonus -= 6
        if span < len(segments) and looks_like_venue(segments[span]):
            bonus += 1
        if span >= 2 and looks_like_venue(candidate):
            bonus -= 6
        candidates.append((candidate, bonus, f"{source_label}:span{span}"))

        if span < len(segments) and looks_like_venue(segments[span]):
            break

    return candidates


def _score_title_candidate(title: str, bonus: int = 0) -> int:
    tokens = _tokenize_words(title)
    lower_tokens = [token.lower() for token in tokens]
    if len(tokens) < 2:
        return -999

    score = bonus
    if 4 <= len(tokens) <= 18:
        score += 8
    elif 3 <= len(tokens) <= 24:
        score += 4
    else:
        score -= 4

    if any(token in TITLE_HINT_WORDS for token in lower_tokens):
        score += 2
    if looks_like_venue(title):
        score -= 8
    if is_authorish_segment(title):
        score -= 10
    if len(title) < 15:
        score -= 4
    if title.count(",") >= 2:
        score -= 2
    return score


def extract_citation_title(citation: str) -> str:
    if not citation:
        return ""

    cleaned = preprocess_citation_text(citation)
    candidates: List[Tuple[str, int, str]] = []

    for match in re.finditer(r'"([^"]{8,})"', cleaned):
        candidates.extend(_expand_title_candidates(match.group(1), 18, "quoted"))

    if "||NEWBLOCK||" in cleaned:
        blocks = [block.strip(" ,.;:") for block in cleaned.split("||NEWBLOCK||") if block.strip(" ,.;:")]
        if len(blocks) >= 2:
            candidates.extend(_expand_title_candidates(blocks[1], 20, "newblock"))

    sentences = [segment.strip(" ,.;:") for segment in re.split(r"\.\s+", cleaned) if segment.strip(" ,.;:")]
    if len(sentences) >= 2 and is_authorish_segment(sentences[0]):
        candidates.extend(_expand_title_candidates(sentences[1], 14, "sentence_after_authors"))

    if ":" in cleaned:
        left, right = cleaned.split(":", 1)
        if is_authorish_segment(left):
            candidates.extend(_expand_title_candidates(right, 16, "colon_after_authors"))

    candidates.extend(_expand_title_candidates(cleaned, 10, "fallback"))
    if not candidates:
        return ""

    best_title, _, _ = max(candidates, key=lambda item: (_score_title_candidate(item[0], item[1]), len(item[0])))
    return re.sub(r"\s+", " ", best_title).strip(" ,.;:")


def extract_gt_title_strict(gt_str: str) -> str:
    return clean_latex_text(extract_citation_title(gt_str))


def build_citation_title_oracle_query(gt_citation: str, max_keywords: int = 5) -> str:
    title = extract_gt_title_strict(gt_citation)
    tokens = [token for token in title.split() if len(token) >= 3]
    deduped: List[str] = []
    seen = set()
    for token in tokens:
        if token in seen:
            continue
        seen.add(token)
        deduped.append(token)
        if len(deduped) >= max_keywords:
            break
    if len(deduped) < 2:
        deduped = (title.split() or [])[:max_keywords]
    return " ".join(deduped).strip()


def analyze_title_alignment(candidate_title: str, target_title: str) -> Dict[str, Any]:
    candidate_clean = clean_latex_text(candidate_title)
    target_clean = clean_latex_text(target_title)
    if not candidate_clean or not target_clean:
        return {
            "candidate_clean": candidate_clean,
            "target_clean": target_clean,
            "exact": False,
            "containment": False,
            "subset": False,
            "overlap": 0,
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "jaccard": 0.0,
            "sequence": 0.0,
            "score": 0.0,
        }

    candidate_tokens = set(candidate_clean.split())
    target_tokens = set(target_clean.split())
    overlap = len(candidate_tokens & target_tokens)
    precision = overlap / max(1, len(candidate_tokens))
    recall = overlap / max(1, len(target_tokens))
    union = len(candidate_tokens | target_tokens)
    jaccard = overlap / max(1, union)
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0

    shorter = candidate_clean if len(candidate_clean) <= len(target_clean) else target_clean
    longer = target_clean if shorter == candidate_clean else candidate_clean
    shorter_tokens = shorter.split()
    exact = candidate_clean == target_clean
    containment = len(shorter_tokens) >= 4 and shorter in longer
    subset = overlap >= 4 and (
        candidate_tokens.issubset(target_tokens) or target_tokens.issubset(candidate_tokens)
    )
    sequence = SequenceMatcher(None, candidate_clean[:1500], target_clean[:1500]).ratio()
    score = max(jaccard, f1, sequence if containment else 0.0)

    return {
        "candidate_clean": candidate_clean,
        "target_clean": target_clean,
        "exact": exact,
        "containment": containment,
        "subset": subset,
        "overlap": overlap,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "jaccard": jaccard,
        "sequence": sequence,
        "score": score,
    }


def compare_titles(candidate_title: str, target_title: str, threshold: float = 0.99) -> Tuple[bool, float, str]:
    metrics = analyze_title_alignment(candidate_title, target_title)
    if not metrics["candidate_clean"] or not metrics["target_clean"]:
        return False, 0.0, "empty_title_after_clean"
    if metrics["exact"]:
        return True, 1.0, "exact_normalized_match"
    if metrics["containment"]:
        return True, 0.99, "normalized_containment"
    if metrics["subset"]:
        return True, metrics["f1"], (
            f"token_subset overlap={metrics['overlap']} precision={metrics['precision']:.3f} recall={metrics['recall']:.3f}"
        )
    if metrics["f1"] >= 0.92 and metrics["overlap"] >= 4:
        return True, metrics["f1"], f"high_token_f1={metrics['f1']:.3f}"
    if metrics["jaccard"] >= threshold and metrics["overlap"] >= 4:
        return True, metrics["jaccard"], f"jaccard={metrics['jaccard']:.3f} >= {threshold:.3f}"
    return False, metrics["score"], (
        f"jaccard={metrics['jaccard']:.3f}; f1={metrics['f1']:.3f}; overlap={metrics['overlap']}"
    )


def check_title_match(
    result_title: str,
    gt_citation: str,
    result_snippet: str = "",
    threshold: float = 0.99,
) -> Tuple[bool, float, str]:
    del result_snippet
    gt_title = extract_citation_title(gt_citation)
    if not gt_title or not result_title:
        return False, 0.0, "empty_input"
    return compare_titles(result_title, gt_title, threshold=threshold)


def is_strict_title_match(candidate_title: str, target_title: str) -> Tuple[bool, float, str]:
    metrics = analyze_title_alignment(candidate_title, target_title)
    if not metrics["candidate_clean"] or not metrics["target_clean"]:
        return False, 0.0, "empty_title_after_clean"
    if metrics["exact"]:
        return True, 1.0, "exact_normalized_match"
    if metrics["containment"]:
        return True, 0.99, "normalized_containment"
    if metrics["overlap"] >= 5 and metrics["precision"] >= 0.90 and metrics["recall"] >= 0.90:
        return True, metrics["f1"], (
            f"strict_token_match overlap={metrics['overlap']} precision={metrics['precision']:.3f} recall={metrics['recall']:.3f}"
        )
    return False, metrics["score"], (
        f"strict_reject jaccard={metrics['jaccard']:.3f}; f1={metrics['f1']:.3f}; overlap={metrics['overlap']}"
    )
