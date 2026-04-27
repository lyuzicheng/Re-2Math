from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from construction.apply_manual_anchor_fixes import MANUAL_ANCHOR_FIXES


EXTRA_MANUAL_ANCHOR_FIXES: dict[str, str] = {
    "1312.6371_gap_5": "Define weak admissibility for $(\phi,N)$-modules with Hodge-Pink structures and use openness of the weakly admissible locus.",
    "1811.09859v2_gap_0": "Complete the computation of $\widetilde{\chi}(\Quot_X(\F,n))$ by combining the previous decomposition and Euler-characteristic formulas.",
    "1811.09859v2_gap_1": "Complete the computation of $\widetilde{\chi}(\Quot_X(\F,n))$ by combining the previous decomposition and Euler-characteristic formulas.",
    "2001.08649_gap_0": "Reduce to the smooth case and use the $C^{1+}$ semiconjugacy to an expanding Cantor model preserving the unstable dimension.",
    "2002.01922_gap_0": "Show that the completion of $(\mathcal{H},d)$ is a CAT(0) space by passing to Cauchy sequences and verifying the induced metric structure.",
    "1710.03736v1_gap_3": "Use rationality of the rotation angle to produce an $S^1$-action on $S^3$ and conclude via rigidity of circle actions.",
    "2108.01546_gap_1": "Apply the rank-parity homomorphism to transfer parity invariance from the enhanced Clifford module to the middle cohomology group.",
    "2102.08097_gap_1": "Compare minimal weak upper gradients using the monotonicity $|Df|_p \le |Df|_q$ for $1 \le p < q$.",
    "2104.11383v2_gap_2": "Use Bouchet's theorem that nonsingular principal submatrices of a symmetric or skew-symmetric matrix determine a delta-matroid.",
    "2005.08904v5_gap_1": "Invoke the Feldman-Hajek theorem to conclude that the two Gaussian measures are either equivalent or mutually singular.",
    "1904.09216_gap_0": "Use the known PTAS for the negative-type distance case to complete the approximation step.",
    "2412.09447v3_gap_1": "Use the distinguished summand to obtain the induced embedding in cohomology.",
    "1910.09214_gap_1": "Use the Hochschild-Serre spectral sequence along the nilpotent subgroup filtration to propagate the required cohomology vanishing from $H_w$ to $H$.",
    "2101.08090_gap_4": "Apply the finite-determinacy lemma for the Tjurina ideal to identify the singularity with the quotient model after adding a sufficiently high-degree perturbation.",
    "1801.05701v5_gap_3": "Use Weil's height machine to equip the fixed abelian variety with a logarithmic height compatible with the chosen principal polarization.",
    "1801.08112v5_gap_1": "Choose a small closed ball around the boundary point so that its preimage in $X$ is compact, using density of the image in its closure.",
    "1809.09253_gap_0": "Apply the calculus of paired Lagrangian distributions to determine the precise order of the singularity created by the triple interaction.",
    "2003.04597_gap_0": "Use the dynamical tube-separation criterion to split the family into good and bad tubes and verify the hypothesis of the main bound theorem.",
    "2010.11583v3_gap_5": "Invoke Ostrovskii's embedding theorem to place the hyperbolic group metric bi-Lipschitzly into the James space.",
    "2311.08058v2_gap_0": "Use the cone-avoidance criterion for the decomposability bundle to extract a positive-measure compact set that is null in the forbidden cone direction.",
    "2010.16137v1_gap_2": "Use the characterization of instability by nontrivial two-fold automorphisms to pass from the pair graph to the desired stability conclusion.",
    "1703.10067v2_gap_0": "Define the relevant eigenvalue for the singular metric by variational characterization so the spectral argument still applies.",
    "2312.16950v3_gap_1": "Transfer the loop-equation and pole analysis through the $x$-$y$ swap description of log topological recursion.",
    "1512.04403v3_gap_0": "Reformulate the discounted control problem as a linear program over state-action occupation measures.",
    "2002.00689v1_gap_0": "Use the invariance criterion $H\mathscr M \subseteq E\mathscr M$ to characterize the admissible subspace for the restricted differential-algebraic system.",
    "2011.09782v4_gap_0": "Use boundedness of limiting subgradients along a convergent bounded subsequence to pass to the cluster-point criticality argument.",
    "2012.12859v2_gap_2": "Combine the entropy-to-Wasserstein estimate with compactness to show that the large-deviation rate constant is strictly positive.",
    "2202.13744v5_gap_0": "Use the full-measure differentiability set for definable Lipschitz functions to replace the Clarke field by the classical gradient almost everywhere.",
    "2208.13475v1_gap_0": "Combine the uniform propagator estimate with the small $L^1$ bound on $G_n$ to show that the perturbed evolution stays close to the unperturbed one.",
    "2302.13550v2_gap_0": "Apply the compact-sublevel existence criterion to deduce existence of optimal input maps for the dynamic programming problem.",
}

MANUAL_TOOL_TYPE_FIXES: dict[str, str] = {
    "1710.03132_gap_1": "theorem",
    "1807.02178_gap_3": "theorem",
}


TITLE_REPLACEMENTS: dict[str, str] = {
    "integralBernstein": "integral Bernstein",
    "ofprojective": "of projective",
    "Onsager’sconjecture": "Onsager's conjecture",
    "Onsager'sconjecture": "Onsager's conjecture",
    "liquidcrystals": "liquid crystals",
    "anapplication": "an application",
    "withwedge": "with wedge",
    "ofspectra": "of spectra",
}


ARXIV_PATTERNS = [
    re.compile(r"arxiv[:\s]*([0-9]{4}\.[0-9]{4,5}(?:v\d+)?)", re.I),
    re.compile(r"ar\$\s*\\chi\$\s*iv[:\s]*([0-9]{4}\.[0-9]{4,5}(?:v\d+)?)", re.I),
    re.compile(r"abs/([0-9]{4}\.[0-9]{4,5}(?:v\d+)?)", re.I),
    re.compile(r"arxiv[:\s]*([a-z.\-]+/[0-9]{7}(?:v\d+)?)", re.I),
    re.compile(r"abs/([a-z.\-]+/[0-9]{7}(?:v\d+)?)", re.I),
]

DOI_PATTERN = re.compile(r"(10\.\d{4,9}/[-._;()/:A-Z0-9]+)", re.I)
SUSPICIOUS_ANCHOR_PATTERN = re.compile(r"\(\s*,")
KNOWN_SOURCE_TYPES = {
    "journal_paper",
    "preprint",
    "book",
    "lecture_notes",
    "conference_paper",
    "survey",
    "thesis",
    "unknown",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Clean a merged benchmark dataset JSONL in-place into a new snapshot.")
    parser.add_argument("--input", required=True, help="Input dataset JSONL.")
    parser.add_argument("--output", required=True, help="Output cleaned dataset JSONL.")
    parser.add_argument("--summary", required=True, help="Output cleaning summary JSON.")
    parser.add_argument("--review", required=True, help="Output review JSON for remaining manual checks.")
    return parser.parse_args()


def normalize_space(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def normalize_title(title: str) -> str:
    updated = normalize_space(title)
    for old, new in TITLE_REPLACEMENTS.items():
        updated = updated.replace(old, new)
    return updated


def extract_arxiv_id(text: str) -> str:
    value = text or ""
    for pattern in ARXIV_PATTERNS:
        match = pattern.search(value)
        if match:
            return match.group(1).strip()
    return ""


def extract_doi(text: str) -> str:
    value = text or ""
    match = DOI_PATTERN.search(value)
    if not match:
        return ""
    return match.group(1).rstrip(".,;")


def infer_source_type(citation: str) -> str:
    text = normalize_space(citation)
    lower = text.lower()
    if not lower:
        return ""
    if any(token in lower for token in ("phd thesis", "doctoral thesis", "master thesis", "dissertation")):
        return "thesis"
    if any(token in lower for token in ("lecture notes", "summer school notes", "notes by ")):
        return "lecture_notes"
    if "survey" in lower:
        return "survey"
    if any(token in lower for token in ("private communication", "unpublished", "manuscript")):
        return ""
    if "draft" in lower or "arxiv:" in lower or "ar$\chi$iv:" in lower or "hal-" in lower or ("available as" in lower and "arxiv" in lower):
        return "preprint"
    if any(
        token in lower
        for token in ("proceedings", "proc.", "conference", "symposium", "workshop", "focs", "stoc", "soda", "icalp", "neurips", "nips", "approx", "socg")
    ):
        return "conference_paper"
    journal_cues = (
        "journal",
        "transactions",
        "annals",
        "inventiones",
        "mathematische",
        "discrete mathematics",
        "algebra number theory",
        "geometric and functional analysis",
        "j. inst. math. jussieu",
        "publications de l'ihes",
        "publications de l’ihes",
        "comm. pure appl. math",
        "ergodic theory dynam. systems",
        "phys. rev. lett",
        "arch. rational mech. anal",
        "math. scand",
        "anal. pde",
        "analysis & pde",
        "j. differential geom",
        "j. reine angew. math",
        "duke math",
    )
    if (
        re.search(r"\b\d+\s*\(\d{4}\)", text)
        or re.search(r"\b\d{4}\),\s*(no\.|[0-9]+[:(])", lower)
        or re.search(r"(?:\\textbf\s*)?\d+\s*,\s*\d{4}\b", lower)
        or re.search(r"\bvol\.?\s*\d+\b", lower)
    ):
        return "journal_paper"
    if any(token in lower for token in journal_cues):
        return "journal_paper"
    book_cues = (
        "press",
        "publishing",
        "springer",
        "cambridge university",
        "oxford university",
        "masson",
        "hermann",
        "smf",
        "ams chelsea",
        "tata institute of fundamental research",
        "birkh",
        "elsevier",
        "astérisque",
        "asterisque",
        "book co.",
        "mcgraw-hill",
        "mc graw hill",
        "harvard university",
    )
    if any(token in lower for token in book_cues):
        return "book"
    return ""


def infer_reference_tool_type(instance_id: str, tool_type: str, tool_latex: str) -> str:
    normalized = normalize_space(tool_type)
    if normalized:
        return normalized

    manual = MANUAL_TOOL_TYPE_FIXES.get(instance_id)
    if manual:
        return manual

    statement = normalize_space(tool_latex).lower()
    if not statement:
        return ""
    if "criterion" in statement:
        return "criterion"
    if "corollary" in statement:
        return "corollary"
    if "proposition" in statement:
        return "proposition"
    if "lemma" in statement:
        return "lemma"
    if "claim" in statement:
        return "claim"
    if "inequality" in statement or any(token in statement for token in ("\leq", "\geq", "<=", ">=")):
        return "inequality"
    if statement.startswith(("if ", "let ", "suppose ", "assume ", "there exists", "an optimal solution")):
        return "theorem"
    return "theorem"


def normalize_locator_fields(z_block: dict[str, Any], counts: Counter[str]) -> None:
    locator = normalize_space(str(z_block.get("locator") or ""))
    locator_snippet = normalize_space(str(z_block.get("locator_snippet") or ""))

    if locator:
        z_block["locator"] = locator
        if not locator_snippet:
            z_block["locator_snippet"] = locator[:320]
            counts["locator_snippet_filled_from_locator"] += 1
    elif locator_snippet:
        z_block["locator_snippet"] = locator_snippet[:320]


def is_suspicious_anchor(anchor: str) -> bool:
    if not anchor:
        return False
    if any(token in anchor for token in ("$( ,N)$", "$( , )$", "$( ,d_")):
        return True
    if anchor.startswith((")", "$ to calculate", "Athbb", "Or skew-symmetric")):
        return True
    if anchor.count("$") % 2 == 1:
        return True
    if SUSPICIOUS_ANCHOR_PATTERN.search(anchor):
        return True
    if len(re.findall(r"\b[a-zA-Z]?\s*_\s*[A-Za-z]?", anchor)) > 6:
        return True
    return False


def clean_row(row: dict[str, Any], counts: Counter[str], manual_anchor_fixes: dict[str, str]) -> dict[str, Any]:
    paper = row.setdefault("paper", {})
    x_block = row.setdefault("x", {})
    y_block = row.setdefault("y", {})
    z_block = row.setdefault("z", {})

    old_title = str(paper.get("title") or "")
    new_title = normalize_title(old_title)
    if new_title != old_title:
        paper["title"] = new_title
        counts["title_normalized"] += 1

    citation_content = normalize_space(str(z_block.get("citation_content") or ""))
    if citation_content != str(z_block.get("citation_content") or ""):
        z_block["citation_content"] = citation_content
        counts["citation_content_normalized"] += 1

    tool_latex = normalize_space(str(y_block.get("reference_tool_latex") or ""))
    if tool_latex != str(y_block.get("reference_tool_latex") or ""):
        y_block["reference_tool_latex"] = tool_latex

    tool_type = infer_reference_tool_type(
        str(row.get("instance_id") or "").strip(),
        str(y_block.get("reference_tool_type") or ""),
        tool_latex,
    )
    if tool_type != str(y_block.get("reference_tool_type") or ""):
        y_block["reference_tool_type"] = tool_type
        if tool_type:
            counts["tool_type_inferred"] += 1

    raw_source_type = str(z_block.get("source_type") or "").strip()
    if raw_source_type not in KNOWN_SOURCE_TYPES:
        counts["unexpected_source_type_values"] += 1
    if raw_source_type in {"", "unknown"}:
        inferred_source_type = infer_source_type(citation_content)
        if inferred_source_type:
            z_block["source_type"] = inferred_source_type
            counts["source_type_inferred"] += 1

    if not str(z_block.get("arxiv_id") or "").strip():
        arxiv_id = extract_arxiv_id(citation_content)
        if arxiv_id:
            z_block["arxiv_id"] = arxiv_id
            counts["arxiv_id_filled"] += 1

    if not str(z_block.get("doi") or "").strip():
        doi = extract_doi(citation_content)
        if doi:
            z_block["doi"] = doi
            counts["doi_filled"] += 1

    normalize_locator_fields(z_block, counts)

    instance_id = str(row.get("instance_id") or "").strip()
    fixed_anchor = manual_anchor_fixes.get(instance_id)
    if fixed_anchor and x_block.get("anchor_hint") != fixed_anchor:
        x_block["anchor_hint"] = fixed_anchor
        counts["anchor_manually_fixed"] += 1

    return row


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)
    summary_path = Path(args.summary)
    review_path = Path(args.review)

    manual_anchor_fixes = dict(MANUAL_ANCHOR_FIXES)
    manual_anchor_fixes.update(EXTRA_MANUAL_ANCHOR_FIXES)

    counts: Counter[str] = Counter()
    rows: list[dict[str, Any]] = []

    with input_path.open("r", encoding="utf-8") as f_in:
        for line in f_in:
            if not line.strip():
                continue
            row = json.loads(line)
            rows.append(clean_row(row, counts, manual_anchor_fixes))

    unresolved_anchor_rows: list[dict[str, Any]] = []
    remaining_unknown_source_rows: list[dict[str, Any]] = []
    remaining_missing_arxiv_rows: list[dict[str, Any]] = []
    remaining_missing_doi_rows: list[dict[str, Any]] = []
    domain_counts: Counter[str] = Counter()
    source_type_counts: Counter[str] = Counter()

    for row in rows:
        paper = row.get("paper") or {}
        x_block = row.get("x") or {}
        z_block = row.get("z") or {}
        domain_counts[(row.get("strata") or {}).get("domain", "<missing>")] += 1
        source_type_counts[str(z_block.get("source_type") or "<missing>")] += 1

        anchor = str(x_block.get("anchor_hint") or "")
        if is_suspicious_anchor(anchor):
            unresolved_anchor_rows.append(
                {
                    "instance_id": row.get("instance_id"),
                    "paper_id": paper.get("paper_id"),
                    "title": paper.get("title"),
                    "anchor_hint": anchor,
                    "local_context": x_block.get("local_context"),
                }
            )

        source_type = str(z_block.get("source_type") or "")
        if source_type in {"", "unknown"} and len(remaining_unknown_source_rows) < 200:
            remaining_unknown_source_rows.append(
                {
                    "instance_id": row.get("instance_id"),
                    "paper_id": paper.get("paper_id"),
                    "citation_content": z_block.get("citation_content"),
                }
            )

        if not str(z_block.get("arxiv_id") or "").strip() and len(remaining_missing_arxiv_rows) < 200:
            remaining_missing_arxiv_rows.append(
                {
                    "instance_id": row.get("instance_id"),
                    "paper_id": paper.get("paper_id"),
                    "citation_content": z_block.get("citation_content"),
                }
            )

        if not str(z_block.get("doi") or "").strip() and len(remaining_missing_doi_rows) < 200:
            remaining_missing_doi_rows.append(
                {
                    "instance_id": row.get("instance_id"),
                    "paper_id": paper.get("paper_id"),
                    "citation_content": z_block.get("citation_content"),
                }
            )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f_out:
        for row in rows:
            f_out.write(json.dumps(row, ensure_ascii=False) + "\n")

    summary = {
        "input_file": str(input_path),
        "output_file": str(output_path),
        "rows_seen": len(rows),
        "title_normalized": counts["title_normalized"],
        "citation_content_normalized": counts["citation_content_normalized"],
        "tool_type_inferred": counts["tool_type_inferred"],
        "source_type_inferred": counts["source_type_inferred"],
        "arxiv_id_filled": counts["arxiv_id_filled"],
        "doi_filled": counts["doi_filled"],
        "locator_snippet_filled_from_locator": counts["locator_snippet_filled_from_locator"],
        "anchor_manually_fixed": counts["anchor_manually_fixed"],
        "unexpected_source_type_values": counts["unexpected_source_type_values"],
        "remaining_suspicious_anchor_count": len(unresolved_anchor_rows),
        "remaining_missing_tool_type_count": sum(1 for row in rows if not str((row.get("y") or {}).get("reference_tool_type") or "").strip()),
        "remaining_unknown_source_type_count": sum(1 for row in rows if str((row.get("z") or {}).get("source_type") or "") in {"", "unknown"}),
        "remaining_missing_arxiv_count": sum(1 for row in rows if not str((row.get("z") or {}).get("arxiv_id") or "").strip()),
        "remaining_missing_doi_count": sum(1 for row in rows if not str((row.get("z") or {}).get("doi") or "").strip()),
        "domain_counts": dict(domain_counts),
        "source_type_counts": dict(source_type_counts),
    }
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    review_payload = {
        "input_file": str(input_path),
        "output_file": str(output_path),
        "remaining_suspicious_anchor_rows": unresolved_anchor_rows[:200],
        "remaining_missing_tool_type_rows": [
            {
                "instance_id": row.get("instance_id"),
                "paper_id": (row.get("paper") or {}).get("paper_id"),
                "reference_tool_latex": (row.get("y") or {}).get("reference_tool_latex"),
            }
            for row in rows
            if not str((row.get("y") or {}).get("reference_tool_type") or "").strip()
        ][:200],
        "remaining_unknown_source_type_rows": remaining_unknown_source_rows,
        "remaining_missing_arxiv_rows": remaining_missing_arxiv_rows,
        "remaining_missing_doi_rows": remaining_missing_doi_rows,
    }
    review_path.parent.mkdir(parents=True, exist_ok=True)
    review_path.write_text(json.dumps(review_payload, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
