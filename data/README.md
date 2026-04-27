# Data

This directory contains the canonical dataset snapshots used by the public release.

## Files

- `benchmark_dataset_full.jsonl`
  Full curated dataset.
- `benchmark_dataset_full.summary.json`
  Compact release summary for the full curated dataset.
- `benchmark_dataset_eval200.jsonl`
  Balanced held-out evaluation subset used in the main multi-model experiments.
- `benchmark_dataset_eval200.summary.json`
  Compact release summary for Eval-200.
- `benchmark_dataset_eval200.structural_audit.json`
  Structural release-time audit for Eval-200.

## Full dataset snapshot

- rows: `860`

### Domain breakdown

| Domain | Rows |
| --- | ---: |
| `geometry_topology` | 215 |
| `analysis_pde` | 170 |
| `probability_statistics_control` | 168 |
| `algebra_number_theory` | 165 |
| `combinatorics_discrete` | 142 |

### Cited-source types

| Source type | Rows |
| --- | ---: |
| `journal_paper` | 492 |
| `book` | 122 |
| `preprint` | 99 |
| `unknown` | 95 |
| `conference_paper` | 38 |
| `lecture_notes` | 11 |
| `survey` | 2 |
| `thesis` | 1 |

### Reference tool types

| Reference tool type | Rows |
| --- | ---: |
| `theorem` | 595 |
| `lemma` | 93 |
| `proposition` | 70 |
| `inequality` | 45 |
| `corollary` | 38 |
| `criterion` | 17 |
| `claim` | 2 |

## Eval-200 snapshot

- rows: `200`
- unique papers: `167`
- strict-quality pass rate before balancing: `0.825`

### Eval-200 domain balance

| Domain | Rows |
| --- | ---: |
| `algebra_number_theory` | 40 |
| `analysis_pde` | 40 |
| `combinatorics_discrete` | 40 |
| `geometry_topology` | 40 |
| `probability_statistics_control` | 40 |

## Eval-200 structural audit

- all journal-paper cited sources: `True`
- all locator evidence present: `True`
- all anchor hints present: `True`
- all local contexts present: `True`
- all reference tools present: `True`
