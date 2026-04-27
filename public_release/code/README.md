# Code

This directory contains the core code released with the benchmark.

## Layout

- `common/`
  Shared schema, retrieval, and resolver utilities.
- `construction/`
  Dataset-building pipeline.
- `evaluation/`
  Benchmark evaluation pipeline.
- `configs/`
  Minimal configuration and seed examples.

## Construction entry points

- `construction/bootstrap_published_candidates.py`
- `construction/prepare_published_manifest.py`
- `construction/match_published_to_arxiv.py`
- `construction/triage_linked_papers.py`
- `construction/mine_dataset.py`
- `construction/finalize_coretask_snapshot.py`

## Evaluation entry points

- `evaluation/search.py`
- `evaluation/end_to_end_eval.py`
- `evaluation/oracle_source_eval.py`
- `evaluation/minimal_publishable_eval.py`
- `evaluation/build_latest_main_tables.py`

## Included result utilities

- `evaluation/build_all_tables.py`
- `evaluation/build_latest_main_tables.py`
- `evaluation/failure_decomposition.py`

## Dependencies

A minimal public dependency list is provided in `../requirements.txt`.
The released codebase expects a Python environment with packages such as:

- `openai`
- `requests`
- `httpx`
- `tenacity`
- `tqdm`
- `arxiv`
- `PyMuPDF`
- `json-repair`

Only the core pipeline and the released result-building helpers are included here.
Local caches, raw run artifacts, and exploratory scripts are intentionally excluded.
