# Latest Paper Main Tables

- Generated at: `2026-04-25 23:00:55`
- Completed model summaries: `7/7`
- GLM is intentionally excluded because its planning diagnostics were unstable.
- All listed models are complete.
- `OracleCoverage` is displayed using the unified oracle-evaluable subset from `revision_oracle_materialization_20260425.json` (`74/200`), i.e. the union of cited-source materializations under the release protocol.
- `Oracle ToolAcc` still reflects each model's completed oracle run; only the coverage denominator is normalized in this paper-facing table.

## Main Table 1: Core End-to-End

| Model | Status | Progress | AnchorAcc(x_raw) | CiteRecall@20 | GroundRate | ToolAcc |
| --- | --- | --- | --- | --- | --- | --- |
| GPT-5.2 | completed | completed | 11/200 (5.5%) | 17/200 (8.5%) | 50/200 (25.0%) | 6/200 (3.0%) |
| Gemini 3.1 Pro | completed | completed | 4/200 (2.0%) | 11/200 (5.5%) | 24/200 (12.0%) | 4/200 (2.0%) |
| Claude Opus 4.5 | completed | completed | 48/200 (24.0%) | 21/200 (10.5%) | 48/200 (24.0%) | 14/200 (7.0%) |
| DeepSeek V3.2 | completed | completed | 16/200 (8.0%) | 11/200 (5.5%) | 58/200 (29.0%) | 5/200 (2.5%) |
| Qwen3-235B Thinking | completed | completed | 81/200 (40.5%) | 12/200 (6.0%) | 34/200 (17.0%) | 2/200 (1.0%) |
| Kimi K2 Thinking | completed | completed | 87/200 (43.5%) | 12/200 (6.0%) | 48/200 (24.0%) | 7/200 (3.5%) |
| Grok 4 | completed | completed | 53/200 (26.5%) | 9/200 (4.5%) | 83/200 (41.5%) | 7/200 (3.5%) |

## Main Table 2: Oracle Gap and Source-Invariant Success

| Model | Status | ToolAcc | OracleCoverage | Oracle ToolAcc | Delta to Oracle | AltSourceToolAcc | AltSourceSuccessRate |
| --- | --- | --- | --- | --- | --- | --- | --- |
| GPT-5.2 | completed | 6/200 (3.0%) | 74/200 (37.0%) | 5/71 (7.0%) | 4.0% | 5/200 (2.5%) | 5/6 (83.3%) |
| Gemini 3.1 Pro | completed | 4/200 (2.0%) | 74/200 (37.0%) | 5/67 (7.5%) | 5.5% | 3/200 (1.5%) | 3/4 (75.0%) |
| Claude Opus 4.5 | completed | 14/200 (7.0%) | 74/200 (37.0%) | 9/71 (12.7%) | 5.7% | 13/200 (6.5%) | 13/14 (92.9%) |
| DeepSeek V3.2 | completed | 5/200 (2.5%) | 74/200 (37.0%) | 2/74 (2.7%) | 0.2% | 5/200 (2.5%) | 5/5 (100.0%) |
| Qwen3-235B Thinking | completed | 2/200 (1.0%) | 74/200 (37.0%) | 2/74 (2.7%) | 1.7% | 2/200 (1.0%) | 2/2 (100.0%) |
| Kimi K2 Thinking | completed | 7/200 (3.5%) | 74/200 (37.0%) | 1/74 (1.4%) | -2.1% | 7/200 (3.5%) | 7/7 (100.0%) |
| Grok 4 | completed | 7/200 (3.5%) | 74/200 (37.0%) | 2/74 (2.7%) | -0.8% | 7/200 (3.5%) | 7/7 (100.0%) |
