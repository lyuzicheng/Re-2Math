from __future__ import annotations

import argparse
import json
import re
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import arxiv

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from construction.merge_paper_manifests import dedup_key, load_rows
from construction.prepare_published_manifest import build_manifest_rows


DEFAULT_CONFIG = ROOT / "configs" / "published_paper_config_large_scale.json"
DEFAULT_OUT = ROOT / "construction" / "outputs" / "published_manifest_arxiv_journal_ref_large_20260416.json"


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_exclude_keys(paths: Iterable[str]) -> tuple[set[str], Dict[str, int]]:
    keys: set[str] = set()
    counts: Dict[str, int] = {}
    for raw_path in paths or []:
        path = Path(str(raw_path))
        if not path.is_file():
            continue
        rows = load_rows(path)
        counts[path.name] = len(rows)
        for row in rows:
            keys.add(dedup_key(row))
    return keys, counts


def normalize_space(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip())


def normalize_name(text: str) -> str:
    lowered = normalize_space(text).lower().replace("&", " and ")
    lowered = re.sub(r"\bthe\b", " ", lowered)
    lowered = re.sub(r"\band\b", " ", lowered)
    return re.sub(r"[^a-z0-9]+", "", lowered)


def venue_aliases(venue: str) -> List[str]:
    aliases = [venue]
    special = {
        "Analysis & PDE": ["Anal. PDE", "Analysis and PDE"],
        "Archive for Rational Mechanics and Analysis": ["Arch. Ration. Mech. Anal.", "ARMA"],
        "Calculus of Variations and Partial Differential Equations": ["Calc. Var. Partial Differential Equations", "Calc. Var. PDE"],
        "Communications on Pure and Applied Mathematics": ["Comm. Pure Appl. Math."],
        "Communications in Partial Differential Equations": ["Comm. Partial Differential Equations"],
        "Geometric and Functional Analysis": ["Geom. Funct. Anal.", "GAFA"],
        "Mathematische Annalen": ["Math. Ann."],
        "Algebraic & Geometric Topology": ["Algebr. Geom. Topol.", "AGT"],
        "Algebraic Geometry": ["Algebr. Geom."],
        "Geometry & Topology": ["Geom. Topol."],
        "Groups, Geometry, and Dynamics": ["Groups Geom. Dyn."],
        "Journal of Differential Geometry": ["J. Differential Geom.", "JDG"],
        "Journal of Topology": ["J. Topol."],
        "Selecta Mathematica": ["Selecta Math."],
        "Algebra & Number Theory": ["Algebra Number Theory"],
        "Compositio Mathematica": ["Compos. Math."],
        "Proceedings of the London Mathematical Society": ["Proc. Lond. Math. Soc."],
        "Cambridge Journal of Mathematics": ["Camb. J. Math."],
        "Research in Number Theory": ["Res. Number Theory"],
        "Forum of Mathematics, Sigma": ["Forum Math. Sigma"],
        "International Mathematics Research Notices": ["Int. Math. Res. Not.", "IMRN"],
        "Journal of the American Mathematical Society": ["J. Amer. Math. Soc.", "JAMS"],
        "Mathematical Research Letters": ["Math. Res. Lett."],
        "Advances in Applied Mathematics": ["Adv. Appl. Math."],
        "Algebraic Combinatorics": ["Algebr. Comb."],
        "Combinatorial Theory": ["Combin. Theory"],
        "Discrete Mathematics": ["Discrete Math."],
        "Discrete & Computational Geometry": ["Discrete Comput. Geom.", "DCG"],
        "European Journal of Combinatorics": ["European J. Combin."],
        "Journal of Graph Theory": ["J. Graph Theory"],
        "Journal of Combinatorial Theory, Series A": ["J. Combin. Theory Ser. A"],
        "Journal of Combinatorial Theory, Series B": ["J. Combin. Theory Ser. B"],
        "Random Structures & Algorithms": ["Random Struct. Algorithms"],
        "SIAM Journal on Discrete Mathematics": ["SIAM J. Discrete Math."],
        "Annals of Probability": ["Ann. Probab."],
        "Annals of Applied Probability": ["Ann. Appl. Probab."],
        "Biometrika": ["Biometrika"],
        "Journal of the Royal Statistical Society: Series B (Statistical Methodology)": ["J. R. Stat. Soc. Ser. B", "JRSSB", "Journal of the Royal Statistical Society Series B"],
        "Probability Theory and Related Fields": ["Probab. Theory Relat. Fields"],
        "Stochastic Processes and their Applications": ["Stochastic Process. Appl."],
        "Annals of Statistics": ["Ann. Statist."],
        "Machine Learning": ["Mach. Learn."],
        "Journal of Machine Learning Research": ["JMLR"],
        "Journal of the ACM": ["J. ACM", "JACM"],
        "Mathematical Programming": ["Math. Program."],
        "Mathematics of Operations Research": ["Math. Oper. Res."],
        "Operations Research": ["Oper. Res."],
        "SIAM Journal on Computing": ["SIAM J. Comput.", "SICOMP"],
        "SIAM Journal on Control and Optimization": ["SIAM J. Control Optim."],
        "SIAM Journal on Optimization": ["SIAM J. Optim."],
    }
    aliases.extend(special.get(venue, []))
    return aliases


def resolve_venue(journal_ref: str, seed_venues: Iterable[str]) -> Optional[str]:
    journal_ref_norm = normalize_name(journal_ref)
    if not journal_ref_norm:
        return None
    if "algebrgeometopol" in journal_ref_norm:
        return None
    for venue in seed_venues:
        for alias in venue_aliases(str(venue)):
            alias_norm = normalize_name(alias)
            if not alias_norm:
                continue
            if alias_norm in journal_ref_norm or journal_ref_norm in alias_norm:
                return str(venue)
    return None


def venue_search_aliases(venue: str) -> List[str]:
    special = {
        "Analysis & PDE": ["Analysis & PDE", "Anal. PDE"],
        "Archive for Rational Mechanics and Analysis": ["Archive for Rational Mechanics and Analysis", "Arch. Ration. Mech. Anal."],
        "Calculus of Variations and Partial Differential Equations": ["Calculus of Variations and Partial Differential Equations", "Calc. Var. Partial Differential Equations"],
        "Communications on Pure and Applied Mathematics": ["Communications on Pure and Applied Mathematics", "Comm. Pure Appl. Math."],
        "Communications in Partial Differential Equations": ["Communications in Partial Differential Equations", "Comm. Partial Differential Equations"],
        "Geometric and Functional Analysis": ["Geometric and Functional Analysis", "Geom. Funct. Anal.", "GAFA"],
        "Mathematische Annalen": ["Mathematische Annalen", "Math. Ann."],
        "Algebraic & Geometric Topology": ["Algebraic & Geometric Topology", "Algebr. Geom. Topol."],
        "Algebraic Geometry": ["Algebraic Geometry", "Algebr. Geom."],
        "Geometry & Topology": ["Geometry & Topology", "Geom. Topol."],
        "Groups, Geometry, and Dynamics": ["Groups, Geometry, and Dynamics", "Groups Geom. Dyn."],
        "Journal of Differential Geometry": ["Journal of Differential Geometry", "J. Differential Geom."],
        "Journal of Topology": ["Journal of Topology", "J. Topol."],
        "Selecta Mathematica": ["Selecta Mathematica", "Selecta Math."],
        "Algebra & Number Theory": ["Algebra & Number Theory", "Algebra Number Theory"],
        "Cambridge Journal of Mathematics": ["Cambridge Journal of Mathematics", "Camb. J. Math."],
        "Compositio Mathematica": ["Compositio Mathematica", "Compos. Math."],
        "Proceedings of the London Mathematical Society": ["Proceedings of the London Mathematical Society", "Proc. Lond. Math. Soc."],
        "Research in Number Theory": ["Research in Number Theory", "Res. Number Theory"],
        "Forum of Mathematics, Sigma": ["Forum of Mathematics, Sigma", "Forum Math. Sigma"],
        "International Mathematics Research Notices": ["International Mathematics Research Notices", "Int. Math. Res. Not."],
        "Journal of the American Mathematical Society": ["Journal of the American Mathematical Society", "J. Amer. Math. Soc."],
        "Mathematical Research Letters": ["Mathematical Research Letters", "Math. Res. Lett."],
        "Advances in Applied Mathematics": ["Advances in Applied Mathematics", "Adv. Appl. Math."],
        "Algebraic Combinatorics": ["Algebraic Combinatorics", "Algebr. Comb."],
        "Combinatorial Theory": ["Combinatorial Theory", "Combin. Theory"],
        "Discrete Mathematics": ["Discrete Mathematics", "Discrete Math."],
        "Discrete & Computational Geometry": ["Discrete & Computational Geometry", "Discrete Comput. Geom.", "DCG"],
        "European Journal of Combinatorics": ["European Journal of Combinatorics", "European J. Combin."],
        "Journal of Graph Theory": ["Journal of Graph Theory", "J. Graph Theory"],
        "Journal of Combinatorial Theory, Series A": ["Journal of Combinatorial Theory, Series A", "J. Combin. Theory Ser. A"],
        "Journal of Combinatorial Theory, Series B": ["Journal of Combinatorial Theory, Series B", "J. Combin. Theory Ser. B"],
        "Random Structures & Algorithms": ["Random Structures & Algorithms", "Random Struct. Algorithms"],
        "SIAM Journal on Discrete Mathematics": ["SIAM Journal on Discrete Mathematics", "SIAM J. Discrete Math."],
        "Annals of Probability": ["Annals of Probability", "Ann. Probab."],
        "Annals of Applied Probability": ["Annals of Applied Probability", "Ann. Appl. Probab."],
        "Biometrika": ["Biometrika"],
        "Journal of the Royal Statistical Society: Series B (Statistical Methodology)": ["Journal of the Royal Statistical Society: Series B (Statistical Methodology)", "Journal of the Royal Statistical Society Series B", "J. R. Stat. Soc. Ser. B", "JRSSB"],
        "Probability Theory and Related Fields": ["Probability Theory and Related Fields", "Probab. Theory Relat. Fields"],
        "Stochastic Processes and their Applications": ["Stochastic Processes and their Applications", "Stochastic Process. Appl."],
        "Annals of Statistics": ["Annals of Statistics", "Ann. Statist."],
        "Machine Learning": ["Machine Learning", "Mach. Learn."],
        "Journal of Machine Learning Research": ["Journal of Machine Learning Research", "JMLR"],
        "Journal of the ACM": ["Journal of the ACM", "J. ACM"],
        "Mathematical Programming": ["Mathematical Programming", "Math. Program."],
        "Mathematics of Operations Research": ["Mathematics of Operations Research", "Math. Oper. Res."],
        "Operations Research": ["Operations Research", "Oper. Res."],
        "SIAM Journal on Computing": ["SIAM Journal on Computing", "SIAM J. Comput."],
        "SIAM Journal on Control and Optimization": ["SIAM Journal on Control and Optimization", "SIAM J. Control Optim."],
        "SIAM Journal on Optimization": ["SIAM Journal on Optimization", "SIAM J. Optim."],
    }
    return special.get(venue, [venue])


def parse_published_year(journal_ref: str, fallback_year: Optional[int]) -> Optional[int]:
    years = [int(match) for match in re.findall(r"(?:19|20)\d{2}", journal_ref or "")]
    if years:
        return max(years)
    return fallback_year


def to_raw_row(domain: str, venue: str, result: arxiv.Result) -> Dict[str, Any]:
    journal_ref = normalize_space(getattr(result, "journal_ref", "") or "")
    doi = normalize_space(getattr(result, "doi", "") or "")
    published_year = parse_published_year(journal_ref, result.published.year if getattr(result, "published", None) else None)
    published_url = f"https://doi.org/{doi}" if doi else (result.entry_id or f"https://arxiv.org/abs/{result.get_short_id()}")
    return {
        "domain": domain,
        "title": normalize_space(result.title or ""),
        "authors": [normalize_space(getattr(author, "name", "") or "") for author in (result.authors or [])],
        "venue": venue,
        "year": published_year,
        "publication_date": "",
        "doi": doi,
        "published_url": published_url,
        "published_source_route": "arxiv_journal_ref",
        "abstract": normalize_space(result.summary or ""),
        "notes": normalize_space(journal_ref),
        "openalex_work_id": "",
        "openalex_arxiv_id": result.get_short_id(),
        "openalex_has_arxiv_link": True,
        "arxiv_id": result.get_short_id(),
        "arxiv_title": normalize_space(result.title or ""),
        "arxiv_url": result.entry_id or f"https://arxiv.org/abs/{result.get_short_id()}",
        "latex_link": f"https://arxiv.org/e-print/{result.get_short_id()}",
        "match_type": "journal_ref_seed",
        "match_confidence": 1.0,
        "match_notes": f"Matched from arXiv journal_ref: {journal_ref}",
        "status": "matched_auto",
    }


def iter_venue_results(client: arxiv.Client, venue_query: str, max_results: int) -> List[arxiv.Result]:
    search = arxiv.Search(
        query=f'jr:"{venue_query}"',
        max_results=max_results,
        sort_by=arxiv.SortCriterion.SubmittedDate,
        sort_order=arxiv.SortOrder.Descending,
    )
    last_error: Optional[Exception] = None
    for attempt in range(5):
        try:
            return list(client.results(search))
        except Exception as exc:
            last_error = exc
            if "429" not in str(exc):
                break
            if attempt == 4:
                break
            time.sleep(10.0 * (attempt + 1))
    if last_error is not None:
        raise last_error
    return []


def main() -> None:
    parser = argparse.ArgumentParser(description="Bootstrap an arXiv-backed published paper pool from journal_ref metadata.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument("--domains", default="", help="Comma-separated domain filter.")
    parser.add_argument("--max-results-per-venue-query", type=int, default=120)
    parser.add_argument("--arxiv-delay-seconds", type=float, default=3.0)
    parser.add_argument("--arxiv-retries", type=int, default=2)
    parser.add_argument("--exclude-manifest", action="append", default=[], help="Manifest file whose rows should be excluded from the rescue results.")
    args = parser.parse_args()

    config = load_json(Path(args.config))
    selected_domains = {item.strip() for item in args.domains.split(",") if item.strip()}
    year_start = int((config.get("selection_defaults", {}) or {}).get("publication_year_start") or 2020)
    year_end = int((config.get("selection_defaults", {}) or {}).get("publication_year_end") or 2026)
    exclude_keys, exclude_counts = load_exclude_keys(args.exclude_manifest)

    client = arxiv.Client(
        page_size=100,
        delay_seconds=max(args.arxiv_delay_seconds, 0.0),
        num_retries=max(args.arxiv_retries, 0),
    )

    raw_rows: List[Dict[str, Any]] = []
    seen_keys = set()
    query_counts = Counter()
    excluded_existing = 0

    for domain_item in config.get("domains", []) or []:
        domain = str(domain_item.get("domain", "") or "")
        if selected_domains and domain not in selected_domains:
            continue
        seed_venues = list(domain_item.get("seed_venues", []) or [])
        for venue in seed_venues:
            for venue_query in venue_search_aliases(str(venue)):
                print(f"[arxiv-journal-ref] querying {domain} | {venue_query}", flush=True)
                try:
                    venue_results = iter_venue_results(client, venue_query, args.max_results_per_venue_query)
                except Exception as exc:
                    print(
                        f"[arxiv-journal-ref] failed {domain} | {venue_query} | {exc}",
                        flush=True,
                    )
                    continue
                print(
                    f"[arxiv-journal-ref] fetched {len(venue_results)} records for {domain} | {venue_query}",
                    flush=True,
                )
                for result in venue_results:
                    journal_ref = normalize_space(getattr(result, "journal_ref", "") or "")
                    doi = normalize_space(getattr(result, "doi", "") or "")
                    if not journal_ref and not doi:
                        continue
                    resolved_venue = resolve_venue(journal_ref, seed_venues)
                    if not resolved_venue:
                        continue
                    published_year = parse_published_year(journal_ref, result.published.year if getattr(result, "published", None) else None)
                    if published_year is None or published_year < year_start or published_year > year_end:
                        continue
                    key = (domain, result.get_short_id())
                    if key in seen_keys:
                        continue
                    raw_row = to_raw_row(domain=domain, venue=resolved_venue, result=result)
                    if exclude_keys and dedup_key(raw_row) in exclude_keys:
                        excluded_existing += 1
                        continue
                    seen_keys.add(key)
                    raw_rows.append(raw_row)
                    query_counts[(domain, venue_query)] += 1
                time.sleep(2.0)

    manifest_rows = build_manifest_rows(raw_rows, config)
    payload = {
        "meta": {
            "config": Path(args.config).name,
            "row_count": len(manifest_rows),
            "domain_counts": dict(Counter(str(row.get("domain", "") or "") for row in manifest_rows)),
            "query_counts": {f"{domain}:{query}": count for (domain, query), count in sorted(query_counts.items())},
            "exclude_manifests": [Path(item).name for item in args.exclude_manifest],
            "exclude_manifest_counts": exclude_counts,
            "excluded_existing_count": excluded_existing,
            "source": "arxiv_journal_ref",
        },
        "papers": manifest_rows,
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload["meta"], ensure_ascii=False, indent=2))
    print(f"Saved manifest to {out_path}")


if __name__ == "__main__":
    main()
