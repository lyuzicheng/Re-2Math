# AI4Math Benchmark Public Release

This directory is the cleaned public-release layer for the AI4Math benchmark.
It is intended to be published as an anonymous GitHub repository root.

The release is organized into four readable parts:

- `data/`: canonical dataset snapshots and compact release summaries
- `code/`: core construction and evaluation code
- `results/`: final aggregated evaluation tables and oracle summary
- `docs/`: task, schema, workflow, and release-policy notes

## What is included

- full curated dataset: `860` proof-gap instances
- official evaluation subset: `200` instances
- core construction code
- core evaluation code
- final paper-facing result tables
- oracle materialization summary

## What is intentionally excluded

- local caches and downloaded PDFs
- raw search downloads
- exploratory shard outputs
- machine-specific absolute paths
- API keys and private credentials
- unrelated working notes from the local research workspace

## Reading order

1. `docs/TASK_FORMULATION.md`
2. `docs/DATASET_SCHEMA.md`
3. `docs/CONSTRUCTION_WORKFLOW.md`
4. `docs/EVALUATION_WORKFLOW.md`
5. `results/paper_main_tables.md`
6. `results/oracle_materialization_summary.md`

## Release snapshot

- Full dataset: `860` instances
- Eval-200: `200` instances from `167` papers
- Domains: `analysis_pde`, `geometry_topology`, `algebra_number_theory`, `probability_statistics_control`, `combinatorics_discrete`
- Completed model runs in the released tables: `7`

## Publish note

To publish anonymously on GitHub, use the contents of this `public_release/` directory as the repository root rather than pushing the full local workspace.
