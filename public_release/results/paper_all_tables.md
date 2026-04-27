# Complete Paper Tables

- Generated at: `2026-04-25 23:00:55`
- Completed model summaries: `7/7`
- GLM is intentionally excluded because its planning diagnostics were unstable.
- All listed models are complete.
- `OracleCoverage` is displayed using the unified oracle-evaluable subset from `revision_oracle_materialization_20260425.json` (`74/200`).
- `Oracle ToolAcc` remains the per-model oracle-run score; only the coverage denominator is normalized in these paper-facing tables.

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

## Appendix Table A1a: Planning Ablation

| Model | Status | Raw local_only m=5 | Raw global_local m=1 | Raw global_local m=3 | Raw global_local m=5 |
| --- | --- | --- | --- | --- | --- |
| GPT-5.2 | completed | 21/200 (10.5%) | 16/200 (8.0%) | 12/200 (6.0%) | 11/200 (5.5%) |
| Gemini 3.1 Pro | completed | 13/200 (6.5%) | 8/200 (4.0%) | 7/200 (3.5%) | 4/200 (2.0%) |
| Claude Opus 4.5 | completed | 46/200 (23.0%) | 60/200 (30.0%) | 47/200 (23.5%) | 48/200 (24.0%) |
| DeepSeek V3.2 | completed | 23/200 (11.5%) | 14/200 (7.0%) | 21/200 (10.5%) | 16/200 (8.0%) |
| Qwen3-235B Thinking | completed | 83/200 (41.5%) | 83/200 (41.5%) | 66/200 (33.0%) | 81/200 (40.5%) |
| Kimi K2 Thinking | completed | 89/200 (44.5%) | 89/200 (44.5%) | 104/200 (52.0%) | 87/200 (43.5%) |
| Grok 4 | completed | 48/197 (24.4%) | 54/200 (27.0%) | 55/200 (27.5%) | 53/200 (26.5%) |

## Appendix Table A1b: Assisted Query Accuracy

| Model | Status | Assist global_local m=5 |
| --- | --- | --- |
| GPT-5.2 | completed | 37/200 (18.5%) |
| Gemini 3.1 Pro | completed | 23/200 (11.5%) |
| Claude Opus 4.5 | completed | 82/200 (41.0%) |
| DeepSeek V3.2 | completed | 28/200 (14.0%) |
| Qwen3-235B Thinking | completed | 60/200 (30.0%) |
| Kimi K2 Thinking | completed | 96/200 (48.0%) |
| Grok 4 | completed | 110/198 (55.6%) |

## Appendix Table A2: Statistical Uncertainty

| Model | Metric | Count/Rate | Wilson 95% CI | Paper-cluster 95% CI | Paper Macro | Domain Macro |
| --- | --- | --- | --- | --- | --- | --- |
| GPT-5.2 | AnchorAcc(x_raw) | 11/200 (5.5%) | [3.1%, 9.6%] | [2.4%, 9.0%] | 5.1% | 5.5% |
| GPT-5.2 | CiteRecall@20 | 17/200 (8.5%) | [5.4%, 13.2%] | [5.2%, 12.3%] | 9.3% | 8.5% |
| GPT-5.2 | GroundRate | 50/200 (25.0%) | [19.5%, 31.4%] | [18.7%, 31.3%] | 24.9% | 25.0% |
| GPT-5.2 | ToolAcc | 6/200 (3.0%) | [1.4%, 6.4%] | [1.0%, 5.5%] | 2.7% | 3.0% |
| GPT-5.2 | OracleCoverage | 74/200 (37.0%) | [30.6%, 43.9%] |  |  |  |
| GPT-5.2 | Oracle ToolAcc | 5/71 (7.0%) | [3.0%, 15.4%] | [1.4%, 14.1%] | 7.5% | 6.3% |
| GPT-5.2 | AltSourceToolAcc | 5/200 (2.5%) | [1.1%, 5.7%] | [0.5%, 4.9%] | 2.1% | 2.5% |
| GPT-5.2 | AltSourceSuccessRate | 5/6 (83.3%) | [43.6%, 97.0%] | [50.0%, 100.0%] | 83.3% | 87.5% |
| Gemini 3.1 Pro | AnchorAcc(x_raw) | 4/200 (2.0%) | [0.8%, 5.0%] | [0.5%, 4.1%] | 2.1% | 2.0% |
| Gemini 3.1 Pro | CiteRecall@20 | 11/200 (5.5%) | [3.1%, 9.6%] | [2.5%, 9.0%] | 5.1% | 5.5% |
| Gemini 3.1 Pro | GroundRate | 24/200 (12.0%) | [8.2%, 17.2%] | [7.7%, 16.6%] | 12.3% | 12.0% |
| Gemini 3.1 Pro | ToolAcc | 4/200 (2.0%) | [0.8%, 5.0%] | [0.5%, 4.0%] | 2.1% | 2.0% |
| Gemini 3.1 Pro | OracleCoverage | 74/200 (37.0%) | [30.6%, 43.9%] |  |  |  |
| Gemini 3.1 Pro | Oracle ToolAcc | 5/67 (7.5%) | [3.2%, 16.3%] | [2.9%, 14.7%] | 7.7% | 8.6% |
| Gemini 3.1 Pro | AltSourceToolAcc | 3/200 (1.5%) | [0.5%, 4.3%] | [0.0%, 3.5%] | 1.5% | 1.5% |
| Gemini 3.1 Pro | AltSourceSuccessRate | 3/4 (75.0%) | [30.1%, 95.4%] | [25.0%, 100.0%] | 75.0% | 83.3% |
| Claude Opus 4.5 | AnchorAcc(x_raw) | 48/200 (24.0%) | [18.6%, 30.4%] | [18.0%, 30.3%] | 25.1% | 24.0% |
| Claude Opus 4.5 | CiteRecall@20 | 21/200 (10.5%) | [7.0%, 15.5%] | [6.4%, 14.5%] | 11.1% | 10.5% |
| Claude Opus 4.5 | GroundRate | 48/200 (24.0%) | [18.6%, 30.4%] | [17.7%, 30.5%] | 24.3% | 24.0% |
| Claude Opus 4.5 | ToolAcc | 14/200 (7.0%) | [4.2%, 11.4%] | [3.6%, 10.7%] | 7.5% | 7.0% |
| Claude Opus 4.5 | OracleCoverage | 74/200 (37.0%) | [30.6%, 43.9%] |  |  |  |
| Claude Opus 4.5 | Oracle ToolAcc | 9/71 (12.7%) | [6.8%, 22.4%] | [5.6%, 21.1%] | 13.2% | 14.2% |
| Claude Opus 4.5 | AltSourceToolAcc | 13/200 (6.5%) | [3.8%, 10.8%] | [3.4%, 10.1%] | 6.9% | 6.5% |
| Claude Opus 4.5 | AltSourceSuccessRate | 13/14 (92.9%) | [68.5%, 98.7%] | [78.6%, 100.0%] | 92.9% | 93.3% |
| DeepSeek V3.2 | AnchorAcc(x_raw) | 16/200 (8.0%) | [5.0%, 12.6%] | [4.4%, 11.9%] | 8.7% | 8.0% |
| DeepSeek V3.2 | CiteRecall@20 | 11/200 (5.5%) | [3.1%, 9.6%] | [2.5%, 8.7%] | 5.4% | 5.5% |
| DeepSeek V3.2 | GroundRate | 58/200 (29.0%) | [23.2%, 35.6%] | [22.6%, 35.6%] | 29.9% | 29.0% |
| DeepSeek V3.2 | ToolAcc | 5/200 (2.5%) | [1.1%, 5.7%] | [0.5%, 5.0%] | 2.7% | 2.5% |
| DeepSeek V3.2 | OracleCoverage | 74/200 (37.0%) | [30.6%, 43.9%] |  |  |  |
| DeepSeek V3.2 | Oracle ToolAcc | 2/74 (2.7%) | [0.7%, 9.3%] | [0.0%, 6.8%] | 2.9% | 2.6% |
| DeepSeek V3.2 | AltSourceToolAcc | 5/200 (2.5%) | [1.1%, 5.7%] | [0.5%, 5.0%] | 2.7% | 2.5% |
| DeepSeek V3.2 | AltSourceSuccessRate | 5/5 (100.0%) | [56.6%, 100.0%] | [100.0%, 100.0%] | 100.0% | 100.0% |
| Qwen3-235B Thinking | AnchorAcc(x_raw) | 81/200 (40.5%) | [33.9%, 47.4%] | [33.3%, 47.2%] | 41.9% | 40.5% |
| Qwen3-235B Thinking | CiteRecall@20 | 12/200 (6.0%) | [3.5%, 10.2%] | [2.8%, 9.5%] | 6.3% | 6.0% |
| Qwen3-235B Thinking | GroundRate | 34/200 (17.0%) | [12.4%, 22.8%] | [11.6%, 22.4%] | 18.0% | 17.0% |
| Qwen3-235B Thinking | ToolAcc | 2/200 (1.0%) | [0.3%, 3.6%] | [0.0%, 2.6%] | 1.2% | 1.0% |
| Qwen3-235B Thinking | OracleCoverage | 74/200 (37.0%) | [30.6%, 43.9%] |  |  |  |
| Qwen3-235B Thinking | Oracle ToolAcc | 2/74 (2.7%) | [0.7%, 9.3%] | [0.0%, 6.9%] | 2.9% | 2.6% |
| Qwen3-235B Thinking | AltSourceToolAcc | 2/200 (1.0%) | [0.3%, 3.6%] | [0.0%, 2.6%] | 1.2% | 1.0% |
| Qwen3-235B Thinking | AltSourceSuccessRate | 2/2 (100.0%) | [34.2%, 100.0%] | [100.0%, 100.0%] | 100.0% | 100.0% |
| Kimi K2 Thinking | AnchorAcc(x_raw) | 87/200 (43.5%) | [36.8%, 50.4%] | [36.6%, 50.7%] | 44.0% | 43.5% |
| Kimi K2 Thinking | CiteRecall@20 | 12/200 (6.0%) | [3.5%, 10.2%] | [3.0%, 9.4%] | 6.6% | 6.0% |
| Kimi K2 Thinking | GroundRate | 48/200 (24.0%) | [18.6%, 30.4%] | [18.1%, 30.3%] | 24.3% | 24.0% |
| Kimi K2 Thinking | ToolAcc | 7/200 (3.5%) | [1.7%, 7.0%] | [1.0%, 6.4%] | 3.3% | 3.5% |
| Kimi K2 Thinking | OracleCoverage | 74/200 (37.0%) | [30.6%, 43.9%] |  |  |  |
| Kimi K2 Thinking | Oracle ToolAcc | 1/74 (1.4%) | [0.2%, 7.3%] | [0.0%, 4.2%] | 1.4% | 1.2% |
| Kimi K2 Thinking | AltSourceToolAcc | 7/200 (3.5%) | [1.7%, 7.0%] | [1.0%, 6.4%] | 3.3% | 3.5% |
| Kimi K2 Thinking | AltSourceSuccessRate | 7/7 (100.0%) | [64.6%, 100.0%] | [100.0%, 100.0%] | 100.0% | 100.0% |
| Grok 4 | AnchorAcc(x_raw) | 53/200 (26.5%) | [20.9%, 33.0%] | [20.2%, 32.5%] | 26.0% | 26.5% |
| Grok 4 | CiteRecall@20 | 9/200 (4.5%) | [2.4%, 8.3%] | [1.5%, 7.5%] | 4.2% | 4.5% |
| Grok 4 | GroundRate | 83/200 (41.5%) | [34.9%, 48.4%] | [34.0%, 48.5%] | 39.8% | 41.5% |
| Grok 4 | ToolAcc | 7/200 (3.5%) | [1.7%, 7.0%] | [1.0%, 6.6%] | 3.3% | 3.5% |
| Grok 4 | OracleCoverage | 74/200 (37.0%) | [30.6%, 43.9%] |  |  |  |
| Grok 4 | Oracle ToolAcc | 2/74 (2.7%) | [0.7%, 9.3%] | [0.0%, 6.9%] | 2.9% | 2.5% |
| Grok 4 | AltSourceToolAcc | 7/200 (3.5%) | [1.7%, 7.0%] | [1.0%, 6.6%] | 3.3% | 3.5% |
| Grok 4 | AltSourceSuccessRate | 7/7 (100.0%) | [64.6%, 100.0%] | [100.0%, 100.0%] | 100.0% | 100.0% |

## Appendix By domain

| Model | Bucket | Papers | Gaps | GroundRate | ToolAcc | Oracle ToolAcc |
| --- | --- | --- | --- | --- | --- | --- |
| GPT-5.2 | algebra_number_theory | 31 | 40 | 22.5% | 0.0% | 6.2% |
| GPT-5.2 | analysis_pde | 32 | 40 | 20.0% | 5.0% | 7.1% |
| GPT-5.2 | combinatorics_discrete | 31 | 40 | 25.0% | 5.0% | 0.0% |
| GPT-5.2 | geometry_topology | 36 | 40 | 35.0% | 2.5% | 6.2% |
| GPT-5.2 | probability_statistics_control | 37 | 40 | 22.5% | 2.5% | 11.8% |
| Gemini 3.1 Pro | algebra_number_theory | 31 | 40 | 7.5% | 0.0% | 0.0% |
| Gemini 3.1 Pro | analysis_pde | 32 | 40 | 15.0% | 5.0% | 7.1% |
| Gemini 3.1 Pro | combinatorics_discrete | 31 | 40 | 10.0% | 0.0% | 16.7% |
| Gemini 3.1 Pro | geometry_topology | 36 | 40 | 17.5% | 2.5% | 13.3% |
| Gemini 3.1 Pro | probability_statistics_control | 37 | 40 | 10.0% | 2.5% | 5.9% |
| Claude Opus 4.5 | algebra_number_theory | 31 | 40 | 17.5% | 5.0% | 13.3% |
| Claude Opus 4.5 | analysis_pde | 32 | 40 | 17.5% | 2.5% | 14.3% |
| Claude Opus 4.5 | combinatorics_discrete | 31 | 40 | 32.5% | 10.0% | 25.0% |
| Claude Opus 4.5 | geometry_topology | 36 | 40 | 27.5% | 7.5% | 13.3% |
| Claude Opus 4.5 | probability_statistics_control | 37 | 40 | 25.0% | 10.0% | 5.3% |
| DeepSeek V3.2 | algebra_number_theory | 31 | 40 | 22.5% | 0.0% | 0.0% |
| DeepSeek V3.2 | analysis_pde | 32 | 40 | 20.0% | 0.0% | 6.7% |
| DeepSeek V3.2 | combinatorics_discrete | 31 | 40 | 30.0% | 5.0% | 0.0% |
| DeepSeek V3.2 | geometry_topology | 36 | 40 | 35.0% | 0.0% | 6.2% |
| DeepSeek V3.2 | probability_statistics_control | 37 | 40 | 37.5% | 7.5% | 0.0% |
| Qwen3-235B Thinking | algebra_number_theory | 31 | 40 | 22.5% | 0.0% | 0.0% |
| Qwen3-235B Thinking | analysis_pde | 32 | 40 | 15.0% | 0.0% | 6.7% |
| Qwen3-235B Thinking | combinatorics_discrete | 31 | 40 | 22.5% | 0.0% | 0.0% |
| Qwen3-235B Thinking | geometry_topology | 36 | 40 | 20.0% | 2.5% | 6.2% |
| Qwen3-235B Thinking | probability_statistics_control | 37 | 40 | 5.0% | 2.5% | 0.0% |
| Kimi K2 Thinking | algebra_number_theory | 31 | 40 | 25.0% | 0.0% | 6.2% |
| Kimi K2 Thinking | analysis_pde | 32 | 40 | 27.5% | 0.0% | 0.0% |
| Kimi K2 Thinking | combinatorics_discrete | 31 | 40 | 15.0% | 7.5% | 0.0% |
| Kimi K2 Thinking | geometry_topology | 36 | 40 | 27.5% | 2.5% | 0.0% |
| Kimi K2 Thinking | probability_statistics_control | 37 | 40 | 25.0% | 7.5% | 0.0% |
| Grok 4 | algebra_number_theory | 31 | 40 | 40.0% | 0.0% | 6.2% |
| Grok 4 | analysis_pde | 32 | 40 | 52.5% | 5.0% | 0.0% |
| Grok 4 | combinatorics_discrete | 31 | 40 | 32.5% | 5.0% | 0.0% |
| Grok 4 | geometry_topology | 36 | 40 | 37.5% | 2.5% | 6.2% |
| Grok 4 | probability_statistics_control | 37 | 40 | 45.0% | 5.0% | 0.0% |

## Appendix By tool family

| Model | Bucket | Papers | Gaps | GroundRate | ToolAcc | Oracle ToolAcc |
| --- | --- | --- | --- | --- | --- | --- |
| GPT-5.2 | other | 167 | 200 | 25.0% | 3.0% | 7.0% |
| Gemini 3.1 Pro | other | 167 | 200 | 12.0% | 2.0% | 7.5% |
| Claude Opus 4.5 | other | 167 | 200 | 24.0% | 7.0% | 12.7% |
| DeepSeek V3.2 | other | 167 | 200 | 29.0% | 2.5% | 2.7% |
| Qwen3-235B Thinking | other | 167 | 200 | 17.0% | 1.0% | 2.7% |
| Kimi K2 Thinking | other | 167 | 200 | 24.0% | 3.5% | 1.4% |
| Grok 4 | other | 167 | 200 | 41.5% | 3.5% | 2.7% |

## Appendix By source type

| Model | Bucket | Papers | Gaps | GroundRate | ToolAcc | Oracle ToolAcc |
| --- | --- | --- | --- | --- | --- | --- |
| GPT-5.2 | journal_paper | 167 | 200 | 25.0% | 3.0% | 7.0% |
| Gemini 3.1 Pro | journal_paper | 167 | 200 | 12.0% | 2.0% | 7.5% |
| Claude Opus 4.5 | journal_paper | 167 | 200 | 24.0% | 7.0% | 12.7% |
| DeepSeek V3.2 | journal_paper | 167 | 200 | 29.0% | 2.5% | 2.7% |
| Qwen3-235B Thinking | journal_paper | 167 | 200 | 17.0% | 1.0% | 2.7% |
| Kimi K2 Thinking | journal_paper | 167 | 200 | 24.0% | 3.5% | 1.4% |
| Grok 4 | journal_paper | 167 | 200 | 41.5% | 3.5% | 2.7% |
