# Evaluation Workflow

This note records the public-facing evaluation workflow used for the released results.

## End-to-end protocol

For each Eval-200 instance:

1. the model receives the raw hierarchical context `x_raw`
2. the model produces a retrieval query
3. Google Scholar is queried through SerpAPI
4. the Top-20 metadata candidates are retained
5. the model shortlists one source from metadata
6. the selected source is read in full text
7. the model extracts a theorem-like statement
8. a separate judge evaluates grounding and sufficiency

## Main metrics

- `AnchorAcc(x_raw)`
- `CiteRecall@20`
- `GroundRate`
- `ToolAcc`
- `OracleCoverage`
- `Oracle ToolAcc`
- `AltSourceToolAcc`
- `AltSourceSuccessRate`

## Eval-200

The released evaluation split contains `200` instances and is:

- quality-controlled
- domain-balanced
- held out for paper-facing comparison

It should not be described as a random slice of the full living benchmark.

## Oracle reporting

`OracleCoverage` is a source-materialization property of the release protocol, not a model capability.

In the released paper-facing artifact:

- `OracleCoverage` is normalized to the unified oracle-evaluable subset `74/200`
- `Oracle ToolAcc` remains the per-model oracle-run score

Therefore the oracle gap should be interpreted as an upper-bound-style diagnostic rather than as a perfectly controlled same-subset rerun.

## Audit reporting

Human-audit artifacts are intentionally omitted from this public release layer.
The released package is limited to the dataset, core code, and aggregated benchmark results.
