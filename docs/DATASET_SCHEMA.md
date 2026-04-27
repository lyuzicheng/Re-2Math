# Dataset Schema

This file records the current benchmark instance format.

Each line/item in the dataset corresponds to one proof-gap instance.

## Canonical instance format

```json
{
  "instance_id": "2505.01049v2_gap_0",
  "split": "test",

  "paper": {
    "paper_id": "2505.01049v2",
    "title": "Multi-Step Consistency Models: Fast Generation with Theoretical Guarantees",
    "publication_date": "2025-05-02"
  },

  "x": {
    "global_context": {
      "setup": "...",
      "target_theorem": "..."
    },
    "local_context": [
      "...",
      "...",
      "...",
      "..."
    ],
    "anchor_hint": "..."
  },

  "y": {
    "reference_tool_latex": "...",
    "reference_tool_type": "lemma",
    "restated_in_citing_paper": false
  },

  "z": {
    "citation_key": "chen2023improved",
    "citation_content": "...",
    "source_type": "conference_paper",
    "locator": "Lemma 2.1",
    "doi": "10.xxx/xxx",
    "arxiv_id": "2301.12345"
  },

  "strata": {
    "domain": "probability",
    "tool_family": "concentration_tail_bound"
  }
}
```

## Field semantics

- `instance_id`: Unique gap instance identifier.
- `split`: Dataset split such as `train`, `dev`, or `test`.

- `paper`: Metadata for the citing paper from which the gap is mined.
- `paper.paper_id`: Internal paper identifier, currently usually the citing paper arXiv ID.
- `paper.title`: Title of the citing paper.
- `paper.publication_date`: Publication or posted date used in the dataset.

- `x`: Agent input.
- `x.global_context.setup`: Global proof context, including setup/definitions/assumptions as a single merged field.
- `x.global_context.target_theorem`: The target theorem being proved in the citing paper.
- `x.local_context`: Ordered pre-citation local context blocks. These are stored as a plain ordered list, without separating sentence vs displayed math.
  Construction stores at most the last 5 preceding proof blocks, in order, so later evaluation can slice a consistent local window.
  Construction is strict: a gap instance is kept only if the citation can be aligned to a concrete occurrence inside the identified main-theorem proof.
  This list may contain fewer than 5 blocks, and it may be empty if the citation occurs at the beginning of the available proof slice.
- `x.anchor_hint`: Leakage-safe auxiliary hint used only in the Assisted track.

- `y`: Reference tool witness.
- `y.reference_tool_latex`: The reference theorem-like statement `K*` in LaTeX.
- `y.reference_tool_type`: The statement type, for example `theorem`, `lemma`, `proposition`, or `corollary`.
- `y.restated_in_citing_paper`: Whether the reference tool is explicitly restated in the citing paper.

- `z`: Auxiliary citation metadata `d^cite`.
- `z.citation_key`: Citation key in the citing paper.
- `z.citation_content`: Bibliographic citation content.
- `z.source_type`: Type of cited source, such as `journal_paper`, `conference_paper`, `book`, `survey`, or `lecture_notes`.
- `z.locator`: Optional internal locator such as `Lemma 2.1` or `Theorem 4`.
- `z.doi`: DOI of the cited source when available; otherwise `null` or empty.
- `z.arxiv_id`: arXiv ID of the cited source when available; otherwise `null` or empty.

- `strata`: Lightweight stratification metadata for analysis.
- `strata.domain`: Primary domain label.
- `strata.tool_family`: Primary tool-family label.

## Track convention

- Raw track: expose `x.global_context` and `x.local_context`.
- Assisted track: expose `x.global_context`, `x.local_context`, and `x.anchor_hint`.

The dataset stores a single `x`. Track-specific exposure is controlled at evaluation time rather than by duplicating the instance.

## Local window convention

- Canonical local-context window: `m = 5`
- Official local-window sensitivity: `m \in \{1,3,5\}`
- Evaluation should slice the stored `x.local_context` from the end, i.e. use the last `m` ordered blocks.
- If fewer than `m` blocks are available, evaluation uses all available blocks.
- If no pre-citation block exists, `x.local_context = []` is valid and the runtime prompt should render the local section as empty rather than fabricating context.
