# Oracle Materialization Taxonomy

- Dataset cases: `200`
- Unified oracle-evaluable subset: `74/200`
- Materialized in every run: `66/200`
- Unavailable in every run: `126/200`
- Cases with run-specific availability variation: `8`

## Final Unified Subset

- `shared_oracle_cache:cited_source_resolver`: `73`
- `shared_oracle_cache:cited_arxiv_id`: `1`

## Unavailable Cases Across All Runs

- `doi=False,arxiv=False`: `102`
- `doi=False,arxiv=True`: `5`
- `doi=True,arxiv=False`: `19`

## Unavailable Domain Counts

- `combinatorics_discrete`: `32`
- `analysis_pde`: `25`
- `algebra_number_theory`: `24`
- `geometry_topology`: `24`
- `probability_statistics_control`: `21`

## Availability-Variant Cases

- `1909.01262_gap_0`: `{'gpt': True, 'gemini': False, 'claude': False, 'deepseek': True, 'qwen': True, 'kimi': True, 'grok': True}`
- `2010.13210v2_gap_0`: `{'gpt': True, 'gemini': True, 'claude': False, 'deepseek': True, 'qwen': True, 'kimi': True, 'grok': True}`
- `2011.08321v2_gap_1`: `{'gpt': False, 'gemini': False, 'claude': True, 'deepseek': True, 'qwen': True, 'kimi': True, 'grok': True}`
- `2203.16998v3_gap_0`: `{'gpt': False, 'gemini': False, 'claude': True, 'deepseek': True, 'qwen': True, 'kimi': True, 'grok': True}`
- `2207.14756v2_gap_1`: `{'gpt': False, 'gemini': False, 'claude': True, 'deepseek': True, 'qwen': True, 'kimi': True, 'grok': True}`
- `2309.08509v2_gap_0`: `{'gpt': True, 'gemini': False, 'claude': False, 'deepseek': True, 'qwen': True, 'kimi': True, 'grok': True}`
- `2401.09280v2_gap_1`: `{'gpt': True, 'gemini': False, 'claude': True, 'deepseek': True, 'qwen': True, 'kimi': True, 'grok': True}`
- `2508.14366v1_gap_4`: `{'gpt': True, 'gemini': False, 'claude': True, 'deepseek': True, 'qwen': True, 'kimi': True, 'grok': True}`
