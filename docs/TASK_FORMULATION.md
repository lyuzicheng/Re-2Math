# Task Formulation

This file records the current paper-facing task definition and metric names.

## Task

We study tool-grounded retrieval. Given a partially completed proof represented
as hierarchical context and access to an external scientific corpus, an agent
must retrieve a grounded external mathematical tool that closes a non-trivial
proof gap.

The benchmark is not citation-locked. The cited source attached to an instance
is a construction signal and diagnostic indicator, not the definition of
success. A prediction is correct if it returns any source-grounded theorem-like
statement that is sufficient for the target proof transition.

## Proof Gap

A proof gap is a logical transition in a proof that cannot be justified from the
current local derivation alone and requires an external mathematical result.

Operationally, gaps are mined from instrumental citations inside the proof of a
main theorem. A citation is treated as instrumental when removing the cited
result would leave the proof transition unjustified.

## Hierarchical Context

For a proof segmented into ordered blocks

```text
P = (s_1, ..., s_T),
```

where each block is either a sentence or displayed mathematical environment,
consider a gap whose citation appears in block `s_c`. The missing justification
is treated as the transition into the citation block, so the pre-citation state
is `k = c - 1`.

The instance context is:

```text
C = (C_global, C_local)
```

where:

- `C_global` contains the paper-level setup, definitions, assumptions, and the
  target theorem being proved.
- `C_local` is the ordered window of proof blocks immediately preceding the
  citation.

The canonical stored local context contains up to 5 pre-citation blocks. At
evaluation time, smaller windows are obtained by taking the last `m` blocks.

## Anchor Hint

Each instance may include a leakage-safe auxiliary anchor hint `a`. The anchor
summarizes the immediate proof intention or the kind of next step to justify.
It must not name theorem titles, author names, citation keys, or bibliographic
metadata.

## Input Tracks

Raw track:

```text
x_raw = (C_global, C_local)
```

Assisted track:

```text
x_assist = (C_global, C_local, a)
```

The Raw track measures the full abstraction burden from proof context alone.
The Assisted track supplies the leakage-safe planning hint while preserving the
same retrieval target.

## Instance Structure

Each benchmark instance stores:

```text
x: agent input
y = K*: reference tool witness
z = d_cite: auxiliary citation metadata
```

`K*` is the reference theorem-like statement aligned to the proof gap. The
citation metadata records the source used by the author, but is not the official
success criterion.

The concrete JSON schema is documented in `docs/DATASET_SCHEMA.md`.

## Agent Output

Given `x` and access to an external corpus `D`, the agent produces:

1. Search queries and a ranked candidate list:

   ```text
   R = (d_1, ..., d_M), d_i in D
   ```

2. A selected source document:

   ```text
   d_hat in R_N
   ```

3. A theorem-like statement:

   ```text
   K_hat
   ```

   extracted from or faithfully restated from `d_hat`.

The benchmark output is:

```text
y_hat = (d_hat, K_hat)
```

## Evaluation Predicates

Document grounding:

```text
Ground(d_hat, K_hat) = 1
```

when `K_hat` is faithfully supported by `d_hat`, either as a verbatim extraction
or a mathematically faithful restatement of a theorem-like statement appearing
in the selected source.

Tool sufficiency:

```text
Suff(K_hat | K*, C) = 1
```

when `K_hat` is sufficient to justify the target proof gap under the available
context. This includes:

- statements equivalent to `K*` up to benign notation changes
- strictly stronger statements that imply the needed proof transition

A stronger statement receives credit only when any additional assumptions are
satisfied by the provided proof context.

## Primary Metric

The main source-invariant task metric is:

```text
ToolAcc = P(Ground(d_hat, K_hat) = 1 and Suff(K_hat | K*, C) = 1)
```

This is the official task-aligned success criterion.

## Auxiliary Metrics

Planning metric:

```text
AnchorAcc(x_raw)
```

measures whether the model-generated planning anchor from raw context matches
the reference anchor.

Citation diagnostic:

```text
CiteRecall@N = P(d_cite in R_N)
```

This reports whether the agent rediscovers the author-used cited source. It is
useful diagnostically but does not define success.

Grounding rate:

```text
GroundRate = P(Ground(d_hat, K_hat) = 1)
```

Alternative-source success:

```text
AltSourceToolAcc = P(ToolAcc = 1 and d_hat != d_cite)
```

Alternative-source success among successes:

```text
AltSourceSuccessRate = P(d_hat != d_cite | ToolAcc = 1)
```

Oracle-source diagnostic:

```text
Oracle ToolAcc
```

measures extraction and sufficiency when the cited source is provided directly,
conditioned on the cited source being materialized/resolved. In the current
paper-facing artifact this should be described as an upper-bound-style
diagnostic, not as a perfectly controlled same-subset rerun for every model.

Oracle coverage:

```text
OracleCoverage
```

is the fraction of evaluation instances for which the cited source can be
materialized for oracle-source evaluation. This is a release-protocol property
rather than a model capability.

## Reporting Convention

Main paper tables should report:

- `AnchorAcc(x_raw)`
- `CiteRecall@20`
- `GroundRate`
- `ToolAcc`
- `OracleCoverage`
- `Oracle ToolAcc`
- `AltSourceToolAcc`
- `AltSourceSuccessRate`

Statistical uncertainty should be reported with Wilson 95% confidence intervals
and paper-cluster bootstrap confidence intervals in the appendix.

For the current no-rerun paper-facing comparison, `OracleCoverage` should be
reported using the unified oracle-evaluable subset from
`evaluation/outputs/revision_oracle_materialization_20260425.json`
(`74/200`). `Oracle ToolAcc` should still be interpreted as the completed
per-model oracle-run score unless a fresh same-subset rerun is performed.

Avoid using `SuffRate` as a headline metric. Sufficiency is mainly meaningful
when paired with grounding, so `ToolAcc` is the clean primary metric.
