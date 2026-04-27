# Paper Extraction Plan

This note records the agreed next-step design for dataset construction under the
minimal schema in [DATASET_SCHEMA.md](./DATASET_SCHEMA.md).

The core strategy is now:

- `published-first`
- `arXiv-backed`

In other words, we select candidate citing papers from the published literature,
but we mine proof gaps from an arXiv version with LaTeX source whenever a
high-confidence published-to-arXiv match exists.

## Why this strategy

Using published papers as the selection source gives stronger quality control:

- the paper has passed venue review
- theorem statements and references are more stable
- citation usage is less noisy than early drafts
- the final benchmark can claim that its citing papers come from the published literature

Using arXiv as the extraction source gives practical tractability:

- LaTeX source is often available
- theorem/proof structure is easier to parse than publisher PDFs
- bibliography alignment is easier to recover automatically

So the right split is:

- `selection source`: published paper
- `extraction source`: matched arXiv twin

## High-level pipeline

The recommended pipeline is:

1. build a published-paper candidate pool by domain and venue
2. filter for proof-rich published papers
3. match each published paper to an arXiv twin
4. keep only high-confidence published-arXiv pairs for the automatic main pipeline
5. run the updated `construction/mine_dataset.py` mining pipeline on the matched arXiv source
   - one full-text Stage 1 pass per paper
   - then proof-local Stage 2 passes per retained citation
6. accept mined instances under domain / tool-family balancing

The unit of search is still a citing paper. The unit of acceptance is still a
single mined proof-gap instance.

For the current build target, the collection plan should be treated as:

- about `1000` accepted instances in total
- `5` primary domains
- about `200` accepted instances per domain
- with a small per-paper cap at the mining stage, plan for roughly `125` matched papers per domain
- assuming arXiv-twin match yield around `0.75`, bootstrap about `180` published candidates per domain

## Important distinction

There are now two separate paper identities to track:

### 1. Published identity

This is the identity used to decide whether the paper belongs in the benchmark
candidate pool.

Required fields:

- `title`
- `authors`
- `venue`
- `year`
- `publication_date`
- `doi` when available
- `published_url`
- `domain`

### 2. arXiv extraction identity

This is the identity used to obtain LaTeX source and actually run the mining
pipeline.

Required fields:

- `arxiv_id`
- `arxiv_title`
- `arxiv_url`
- `latex_link`
- `match_type`
- `match_confidence`

The benchmark should only enter the automatic mining pipeline after these two
identities have been aligned with high confidence.

## Candidate pool: published-first

The outer loop should be domain-based, but the actual candidate papers should be
pulled from published venues rather than from recent arXiv listings.

Suggested first-wave domains:

- `analysis_pde`
- `geometry_topology`
- `algebra_number_theory`
- `combinatorics_discrete`
- `probability_statistics_control`

For each domain, maintain a curated venue seed list. The purpose of the list is
not prestige alone; it is proof density, theorem-heavy style, and citation-rich
arguments.

For the mixed `probability_statistics_control` bucket, the current seed policy
is intentionally narrow:

- keep core probability venues
- add `Journal of Machine Learning Research` to cover theorem-heavy theory-ML papers
- keep `Annals of Statistics` as the statistics-side anchor venue

The candidate routes should be:

### 1. Official venue pages

Use journal / proceedings pages as the primary route:

- journal tables of contents
- conference proceedings pages
- official article landing pages

This is the default backbone because it enforces the published-first policy.

### 2. Metadata aggregators

Use metadata providers only as helpers for recall and normalization:

- DOI lookup
- author normalization
- publication date normalization
- venue normalization

These sources help build the manifest, but the conceptual source of truth is the
published version.

### 3. Manual seeds

Keep a small manual-seed route for underfilled domains or rare tool families.
Manual seeds should still be published papers whenever possible.

## Proof-rich filtering

Not every published paper should enter the expensive matching-and-mining stage.
Apply a lightweight filter first.

Recommended positive signals:

- long theorem/proof style article
- non-trivial bibliography
- explicit main theorem proof
- multiple formal citations inside proofs
- abstract / title clearly indicate theoretical rather than expository content

Recommended exclusion rules:

- survey
- lecture notes
- thesis
- monograph
- book chapter
- erratum / correction
- very short note with little proof structure

This filter is still paper-level. It should happen before any expensive LLM
stage.

## Matching published papers to arXiv

This is the critical new step. The benchmark should not pretend that every
published paper has an arXiv twin, and it should not use weak matching rules.

### Match order

Use the following precedence:

1. `doi_exact`
2. `title_author_year`
3. `manual_verified`

### 1. `doi_exact`

Accept immediately if:

- published DOI matches the DOI recorded by the arXiv entry or a linked metadata source

This is the highest-confidence automatic route.

### 2. `title_author_year`

Accept automatically only if all of the following hold:

- normalized title match is exact or near-exact
- first author matches
- publication year and arXiv year are compatible
- no competing arXiv candidate has similar score

This route should record a numeric confidence score.

### 3. `manual_verified`

If DOI is unavailable and title matching is plausible but not clean, send the
paper to a manual verification queue. Only after confirmation should it enter
the main matched manifest.

### Automatic rejection cases

Reject from the automatic main pipeline if:

- multiple arXiv candidates are plausibly tied
- title is substantially changed across versions
- author list mismatch is large
- publication version appears to merge or split earlier preprints

These cases belong in a manual queue, not in the automatic main set.

## Manifest structure

The new manifest should be pair-based rather than arXiv-only. A manifest row is
one matched published-arXiv paper pair.

Suggested fields:

- `paper_key`
- `domain`
- `published_title`
- `published_authors`
- `venue`
- `year`
- `publication_date`
- `doi`
- `published_url`
- `published_source_route`
- `proof_rich_score`
- `proof_rich_signals`
- `arxiv_id`
- `arxiv_title`
- `arxiv_url`
- `latex_link`
- `match_type`
- `match_confidence`
- `match_notes`
- `status`

Recommended `status` values:

- `matched_auto`
- `matched_manual`
- `needs_manual_review`
- `no_arxiv_twin`
- `rejected`

Only `matched_auto` and `matched_manual` should be fed into automatic gap mining.

## Gap mining

After a paper pair is accepted into the matched manifest:

1. use the arXiv source to run Stage 1 once on the full paper:
   - `x.global_context.setup`
   - `x.global_context.target_theorem`
   - internal `proof_span` for the main theorem proof
   - candidate instrumental citations
2. restrict to the main theorem proof span
   - if the main-theorem proof span cannot be aligned back into the source text, skip the paper
3. use Stage 2 on each valid citation:
   - feed only the main-proof text, or a citation-local proof slice
   - if the citation locator cannot be aligned inside the main proof, skip that citation
   - `x.local_context` as an ordered list of up to the last 5 pre-citation proof blocks
   - if the citation appears before any retained pre-citation block, allow `x.local_context = []`
   - `x.anchor_hint`
   - `y.reference_tool_*`
   - `z.*`
   - `strata.tool_family`

This is the preferred efficiency policy because it avoids sending the whole
paper repeatedly to the LLM. The paper should be read globally only once; after
that, all per-gap work should be done inside the identified main-theorem proof.

The mined dataset row should still use the current benchmark schema. The
published/arXiv pairing information belongs in the paper-level metadata used
during construction and, if desired, in a future extension of the `paper` field.

Window convention:

- Canonical local window for official runs: `m = 5`
- Local-window sensitivity: `m \in \{1,3,5\}`
- Therefore construction should preserve enough ordered local blocks for `m=5`, and evaluation should slice the last `m` blocks from that stored list.
- If fewer than `m` blocks exist, evaluation should use the shorter available prefix.
- If zero blocks exist, keep the local context empty instead of backfilling from unrelated text.

## Acceptance logic after mining

Do not accept all mined gaps blindly. Maintain running counts for:

- accepted instances per `domain`
- accepted instances per `tool_family`
- accepted instances per `domain x tool_family`

Accept a mined instance when it helps fill the target strata. This keeps the
expansion stratified rather than merely larger.

## Main dataset vs manual queue

The construction pipeline should explicitly separate:

### Main dataset path

Published paper with a high-confidence arXiv twin:

- `doi_exact`
- strong `title_author_year`
- `manual_verified`

These are eligible for the main benchmark.

### Manual queue

Published paper without a clean arXiv twin:

- `no_arxiv_twin`
- ambiguous candidate set
- weak title alignment
- extraction source unavailable

These can still be valuable later, but they should not block the high-quality
main pipeline.

## Next implementation steps

The next coding steps should be:

1. `construction/prepare_published_manifest.py`
   - collect published-paper candidates by domain and venue
   - assign `proof_rich_score`
   - emit a published-paper manifest
2. `construction/match_published_to_arxiv.py`
   - attach arXiv twins using DOI and title-author-year matching
   - emit a matched manifest plus a manual-review queue
3. `construction/mine_dataset.py`
   - consume only accepted matched pairs
   - run Stage 1 once per paper to identify the main theorem, proof span, and
     candidate tool citations
   - run Stage 2 only inside the proof-local text region
   - emit benchmark rows under the current `x / y / z / strata` schema

This replaces the old logic where recent arXiv papers formed the primary
candidate pool.
