# Retrieval Evaluation Summary

- Questions evaluated: 60 of 60.
- Chunk corpus: 48,109 chunks; SHA256 `b6af8d5df0fee1c59dcca7833e16099652b2a51f5226fab48213c9f46c3927fe`.
- Benchmark SHA256: `1a60f5e235eec5ac24ff8f2bc28c706256a5d6e6ba7de4bce56f520cf79cd0fa`.
- Retrieval top-K: 10; reported cutoffs: 1, 3, 5, 10.
- The retrieval pipeline's baseline default is top-5; this evaluation requests top-10 to calculate @10 as well as @1/@3/@5.
- Query embedding model: `BAAI/bge-base-en-v1.5` at revision `08e3d2c28e3886dfbcd98caebb447d9e23614bc0`.
- Embedding warm-up: 11.3922s using the first original question (`tcs_q01`); excluded from per-query strategy latency.
- Queries: exact original benchmark wording; no planner, rewriting, query expansion, or answer generation.
- Relevance labels: `supports` references from proposed drafts only; `context_only` and other relations are excluded.
- Annotation status: 134 evidence references are `proposed_unverified`; none has independent human verification. These are provisional-label results, not verified-gold performance.
- Human evidence verification has not been performed by this evaluation. Do not present these scores as final research findings until an independent reviewer checks the cited source content and labels.
- Supporting-evidence labels: 49 cases evaluable; 11 cases have no supporting chunk label and are excluded from ranking-metric macro averages.
- Proposed action counts: {'abstain': 5, 'answer': 43, 'current_information_unavailable': 6, 'partial': 6}.

## Metric definitions

- Macro P/R/F1@K: arithmetic mean of the per-question metric over successfully retrieved questions with at least one proposed supporting evidence label. Per-question precision uses K as its denominator, even when fewer than K results are returned.
- Micro P@K: total relevant hits across scored questions divided by K times the number of scored questions.
- Micro R@K: total relevant hits across scored questions divided by the total number of proposed relevant chunk IDs for those questions.
- Micro F1@K: harmonic mean of micro P@K and micro R@K. MRR and nDCG are reported as macro means only.
- Operational rates use all questions in the group as denominator, except empty-retrieval rate, which is empty successful searches divided by successful searches. Failed, unavailable, and successful-empty outcomes are distinct.
- Latency summaries use all measured strategy calls in the group, including failed or unavailable calls; P95 uses the nearest-rank definition (rank = ceil(0.95 * N)).

## Strategy results

| Strategy | Scored | Macro P@5 | Macro R@5 | Macro F1@5 | Micro P@5 | Micro R@5 | Micro F1@5 | Macro MRR | Macro nDCG@10 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline | 48 | 0.0500 | 0.1274 | 0.0688 | 0.0500 | 0.1395 | 0.0736 | 0.1655 | 0.1331 |
| chunk_semantic | 48 | 0.0500 | 0.1274 | 0.0688 | 0.0500 | 0.1395 | 0.0736 | 0.1655 | 0.1331 |
| hybrid_all | 48 | 0.0292 | 0.0684 | 0.0388 | 0.0292 | 0.0814 | 0.0429 | 0.0443 | 0.0533 |
| hybrid_chunk_keyword | 48 | 0.0292 | 0.0684 | 0.0388 | 0.0292 | 0.0814 | 0.0429 | 0.0780 | 0.0730 |
| hybrid_chunk_lexical | 48 | 0.0292 | 0.0684 | 0.0388 | 0.0292 | 0.0814 | 0.0429 | 0.0780 | 0.0730 |
| keyword_lexical | 49 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| keyword_semantic | 48 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |

## Common evaluation subset

- 58 questions were successfully retrieved by every configured strategy; all strategy comparisons below use exactly this subset.
- 48 common-subset questions have proposed supporting evidence labels and contribute to ranking metrics.
- Proposed-label questions excluded from the common subset due to retrieval failures: `tcs_q15`.

| Strategy | Common scored | Macro P@5 | Macro R@5 | Macro F1@5 | Micro P@5 | Micro R@5 | Micro F1@5 | Macro MRR |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline | 48 | 0.0500 | 0.1274 | 0.0688 | 0.0500 | 0.1395 | 0.0736 | 0.1655 |
| chunk_semantic | 48 | 0.0500 | 0.1274 | 0.0688 | 0.0500 | 0.1395 | 0.0736 | 0.1655 |
| hybrid_all | 48 | 0.0292 | 0.0684 | 0.0388 | 0.0292 | 0.0814 | 0.0429 | 0.0443 |
| hybrid_chunk_keyword | 48 | 0.0292 | 0.0684 | 0.0388 | 0.0292 | 0.0814 | 0.0429 | 0.0780 |
| hybrid_chunk_lexical | 48 | 0.0292 | 0.0684 | 0.0388 | 0.0292 | 0.0814 | 0.0429 | 0.0780 |
| keyword_lexical | 48 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| keyword_semantic | 48 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |

## Operational metrics

| Strategy | Successful | Failed | Unavailable | Empty (count/rate) | Success rate | Failure rate | Mean / median / P95 latency (s) | Min / max latency (s) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline | 58 | 2 | 0 | 0 / 0.0000 | 0.9667 | 0.0333 | 0.2887 / 0.2925 / 0.3379 | 0.0886 / 0.3573 |
| chunk_semantic | 58 | 2 | 0 | 0 / 0.0000 | 0.9667 | 0.0333 | 0.2908 / 0.2944 / 0.3354 | 0.0920 / 0.3510 |
| hybrid_all | 58 | 2 | 0 | 0 / 0.0000 | 0.9667 | 0.0333 | 0.3621 / 0.3653 / 0.4137 | 0.0912 / 0.4490 |
| hybrid_chunk_keyword | 58 | 2 | 0 | 0 / 0.0000 | 0.9667 | 0.0333 | 0.3551 / 0.3598 / 0.4206 | 0.0923 / 0.4368 |
| hybrid_chunk_lexical | 58 | 2 | 0 | 0 / 0.0000 | 0.9667 | 0.0333 | 0.2886 / 0.2940 / 0.3347 | 0.0881 / 0.3522 |
| keyword_lexical | 60 | 0 | 0 | 2 / 0.0333 | 1.0000 | 0.0000 | 0.0022 / 0.0021 / 0.0032 | 0.0013 / 0.0041 |
| keyword_semantic | 58 | 2 | 0 | 0 / 0.0000 | 0.9667 | 0.0333 | 0.0712 / 0.0701 / 0.0936 | 0.0022 / 0.1610 |

## Retrieval failures

- Failed question IDs: `infosys_q36`, `tcs_q15`. Chroma-backed strategies reject the requested financial-year filters for these out-of-index periods; the exact per-strategy errors are retained in `raw_retrieval_results.json`. No unfiltered fallback was used.

## Index availability and limitations

- Default chunk-semantic collection: `financial_reports_filtered` at `C:\Kiran\SRM\Semester 4\Project\financial-rag-india\data\chroma_db`; 48063 indexed chunks.
- `baseline` and `chunk_semantic` use the same direct Chroma chunk-semantic path; both are listed to make the configured default explicit.
- Keyword semantic path: available; 150 pilot chunks. Configured default path `C:\Kiran\SRM\Semester 4\Project\financial-rag-india\data\keyword_chroma_db` is missing; pilot results are not full-corpus production results.
- Keyword lexical path: available; 150 pilot chunks. Configured default path `C:\Kiran\SRM\Semester 4\Project\financial-rag-india\data\keyword_index\financial_keyword_index.sqlite` is missing; pilot results are not full-corpus production results.
- Hybrid paths use the existing RRF implementation and its default per-path candidate limit of {'chunk_semantic': 5, 'keyword_semantic': 5, 'keyword_lexical': 5}; no retrieval implementation or index was rebuilt.
- Metadata filtering uses the benchmark's exact `company`, `financial_year`, and `chunk_type` fields. The existing retriever accepts only one company filter; multi-company cases retain the benchmark's primary `company` filter.
- When an index rejects a filter or a query fails, the error is preserved in the raw results; no unfiltered fallback is used.

## Files and reproducibility

- `raw_retrieval_results.json`: per-question rankings, scores, provenance, diagnostics, failures, and provisional relevance labels.
- `metrics_overall.csv`, `metrics_by_company.csv`, `metrics_by_category.csv`, `per_query_metrics.csv`: macro ranking metrics and retrieval coverage.
- `metrics_common_subset.csv`: macro and micro metrics evaluated on the same successful question set for every strategy.
- `common_evaluation_subset.csv`: exact question IDs and inclusion flags used for the fair comparison.
- `strategy_metrics.svg`: comparison chart.
- Evaluated limit: all 60 questions.
- Input file SHA256 values:
  - `financial_rag_60_question_benchmark.json`: `1a60f5e235eec5ac24ff8f2bc28c706256a5d6e6ba7de4bce56f520cf79cd0fa`
  - `all_chunks.jsonl`: `b6af8d5df0fee1c59dcca7833e16099652b2a51f5226fab48213c9f46c3927fe`
  - `tcs_q01_q20_gold_annotations.json`: `beab27813877def7d0daa74e3dce171d86879d7f4c652740375a93aab8274b25`
  - `infosys_gold_annotations.json`: `277a26121b40da4c6248fd72653a2035b8d66e12275f668141cb640ba979411b`
  - `reliance_gold_annotations.json`: `da6e352dc0db8a5055470166760593ff84deecda6aed058fd959987bae19ea3e`

The benchmark and annotation files are inputs only. No claims about answer correctness are made, and metrics must be recomputed after human review changes the evidence labels.
