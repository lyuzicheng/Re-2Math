# Leaderboard And Release Policy

This note fixes the comparison policy for paper-facing evaluation and for future
submissions to the living benchmark.

## Frozen Comparison Unit

Every leaderboard entry must be tied to the tuple

```text
dataset release
+ evaluation split
+ backend configuration
+ retrieval/query-cache artifact
+ evaluation date
```

The paper-facing results in the current draft correspond to:

- dataset release: `benchmark_dataset_coretask_curated_20260422`
- evaluation split: `benchmark_dataset_eval200_balanced_20260422`
- backend configuration: Google Scholar / SerpAPI top-20 metadata shortlist,
  one selected source, one-source full-text reading
- retrieval artifact: the frozen search / shortlist / download artifact used for the paper-facing evaluation (not redistributed in this public release layer)
- evaluation date window: the April 2026 OpenRouter runs bundled in this repo

## Dataset Release ID

A dataset release is identified by:

- the exact JSONL instance file
- the schema version
- the split file(s)
- domain labels and auxiliary strata
- the construction manifest / curation protocol used to produce the release

Different dataset releases should not be mixed in a single leaderboard table.

## Retrieval Artifact ID

A retrieval artifact is identified by:

- backend name and configuration
- search budget and shortlist size
- query-generation protocol
- recorded search outputs
- document-download / materialization logs

For the current paper-facing evaluation, the retrieval artifact is the frozen
search / shortlist / download log attached to each model suite.

## New Submission Policy

Future models may generate new queries. However, those queries must be executed
under the same declared backend configuration, and the resulting retrieval logs
must be saved as a new frozen retrieval artifact.

Therefore, a valid future submission must specify:

- dataset release
- evaluation split
- backend configuration
- retrieval artifact ID
- evaluation date

## Cross-Release Comparison

Results from different dataset releases or different retrieval artifacts are not
directly comparable as a single ranking table unless the compared models are
re-evaluated under the same frozen artifact.

In practice:

- same dataset release + same backend config + same retrieval artifact:
  directly comparable
- same dataset release + same backend config + different retrieval artifact:
  not directly comparable without re-running prior models
- different dataset release:
  report separately or re-evaluate all compared models

## Eval-200 Status

`Eval-200` is the current official quality-controlled held-out evaluation split.
It is domain-balanced and quality-filtered; it is not intended as a random slice
of the full living release.

The paper should therefore describe Eval-200 as:

> a quality-controlled, domain-balanced held-out evaluation split

rather than as a random sample of the full benchmark.

## Oracle Policy

`OracleCoverage` is a source-materialization property of the release protocol,
not a model capability score.

For paper-facing comparison, the normalized oracle-evaluable subset is the
unified materialized union recorded in:

- `results/oracle_materialization_summary.json`

which currently yields:

- unified oracle-evaluable subset: `74/200`

Any future oracle comparison should either:

- use the same frozen oracle-evaluable subset, or
- explicitly declare a new oracle materialization artifact and re-evaluate all
  compared models.
