# Question-Level Retrieval Audit: Data Integrity Validation

- Workbook output: `question_level_retrieval_audit.xlsx`
- Source files unchanged after build: **yes**
- Benchmark questions mapped: **60**
- Retrieval strategies: **7**
- Question/strategy metric rows mapped: **420**
- Actual retrieved chunk records: **3133**
- Expected evidence reference occurrences: **134**

## Mapping and statuses

- Categories preserved: `conflicting_or_reconcilable_evidence` (7), `financial_metric` (12), `insufficient_evidence` (6), `multi_evidence_comparison` (9), `segment_performance` (9), `temporal_ambiguity_or_current_information` (6), `year_over_year_comparison` (11)
- Companies preserved: `Infosys` (20 question/strategy rows), `RelianceIndustries` (20 question/strategy rows), `TataConsultancyServices` (20 question/strategy rows)
- Strategies: `baseline`, `chunk_semantic`, `keyword_semantic`, `keyword_lexical`, `hybrid_chunk_keyword`, `hybrid_chunk_lexical`, `hybrid_all`
- Retrieval statuses: `completed`=408, `failed`=12
- Metric statuses: `failed`=12, `not_evaluable`=71, `scored`=337
- Expected evidence types: `context_only`=23, `supports`=111
- Annotation verification statuses: `proposed_unverified`=134
- Questions without supporting gold IDs (metrics not evaluable): `infosys_q35`, `infosys_q36`, `infosys_q39`, `infosys_q40`, `ril_q55`, `ril_q56`, `ril_q59`, `ril_q60`, `tcs_q16`, `tcs_q19`, `tcs_q20`
- Questions in saved common evaluation subset: **58**

## Integrity checks

- Raw result question IDs match benchmark IDs: **pass**
- Per-query metric rows match every question/strategy pair: **pass**
- Retrieved IDs absent from all_chunks.jsonl: **0**
- Expected evidence IDs absent from all_chunks.jsonl: **0**
- Retrieved passages differing from canonical chunk text: **0**
- Retrieved ranks are consecutive and retrieved counts match both saved result and metric records: **pass**
- Raw relevance-ID sets equal exact `supports` chunk IDs from gold annotations: **pass**
- Blank metric values preserved as unavailable: Precision@5=83, Recall@5=83, Recall@10=83, MRR=83
- Full passage lengths exceeding Excel cell limit in included records: **0**

Workbook worksheet row counts (header rows excluded):

- `Question Performance Summary`: 420
- `Retrieved Chunks`: 3147
- `Expected Gold Chunks`: 134
- `Side-by-Side Comparison`: 4162
- `Error Analysis`: 420
- `Investigation Priorities`: 60

## Input SHA-256

- `benchmarks\answer_evaluation\financial_rag_60_question_benchmark.json`: `1a60f5e235eec5ac24ff8f2bc28c706256a5d6e6ba7de4bce56f520cf79cd0fa`
- `benchmarks\answer_evaluation\tcs_q01_q20_gold_annotations.json`: `beab27813877def7d0daa74e3dce171d86879d7f4c652740375a93aab8274b25`
- `benchmarks\answer_evaluation\infosys_gold_annotations.json`: `277a26121b40da4c6248fd72653a2035b8d66e12275f668141cb640ba979411b`
- `benchmarks\answer_evaluation\reliance_gold_annotations.json`: `da6e352dc0db8a5055470166760593ff84deecda6aed058fd959987bae19ea3e`
- `evaluation_results\raw_retrieval_results.json`: `52e585efde4d5a9cf1797f502ed1f315bec9f03d1b86f94efbf9b8a9802312a0`
- `evaluation_results\per_query_metrics.csv`: `3458a5b487d4839b65498ec1755f257a00a34b05dd1436a4c22de732d7f2f19d`
- `data\chunks\all_chunks.jsonl`: `b6af8d5df0fee1c59dcca7833e16099652b2a51f5226fab48213c9f46c3927fe`

The evidence labels are provisional (`proposed_unverified`). This report validates source relationships and workbook construction; it does not certify annotation correctness or infer retrieval failure causes.
