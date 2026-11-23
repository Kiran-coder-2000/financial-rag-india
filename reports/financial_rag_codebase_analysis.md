# Financial RAG Codebase and Evaluation Readiness Analysis

**Inspection scope:** current working-tree state on 2026-11-10. This was a
read-only audit of source, tests, benchmarks, configuration, local data and
persistent indexes. No source, benchmark, configuration or database was
changed; no embeddings/indexes were rebuilt and no retrieval or LLM benchmark
was run. The only test execution was the existing unit suite.

## 1. Executive summary

The code contains six callable retrieval strategies and an offline evaluator
that implements Precision@5, Recall@5, Recall@10, MRR and a one-pass mean
retrieval latency. The evaluator can instantiate a common seven-variant set
(the six strategies plus the `baseline` alias) and run them on the same
fixed, labeled cases.

**It cannot currently produce valid comparison findings for all six
strategies from the available artifacts.**

The blockers are data and index readiness, not missing metric formulas:

1. The 15-case retrieval gold dataset has 15 pending cases, empty
   `relevant_chunk_ids`, and 156 unverified candidate-ID references in total. The
   evaluator correctly rejects pending-only gold data. The separate
   15-case answer-evaluation seed and 60-case answer benchmark also have
   pending reviews and null gold fields. The 20-case TCS draft is explicitly
   proposed/unverified.
2. The default dense collection, `financial_reports_filtered`, contains
   48,063 of the current corpus's 48,109 chunk IDs. All 46 missing items are
   table chunks. The separate `financial_reports` collection contains all
   48,109 current IDs and matching text/selected metadata.
3. The keyword-vector and FTS5 indexes are pilots of 150 chunks, stored under
   pilot paths/names; the evaluator's default full-index paths/names are not
   present. They do not represent the same complete corpus as the dense
   index.
4. The default evaluation benchmark is one query with five query/filter/table
   variants, not six retrieval strategies. Its two relevance IDs have no
   human-verification status. It is suitable only as an unverified smoke
   fixture, not as evidence for Table 5.8.
5. The latency code measures one search per case. Query normalization,
   query embedding, filter validation, database search and hybrid RRF are
   included; LLM planning/generation are excluded. There is no warm-up or
   repeat option, so a first-use model load can contaminate the first
   measurement.

There are **no persisted retrieval evaluation results** in `reports/`,
`experiments/`, `figures/` or `presentation/`, and no retrieval metric output
was found in the notebooks. The actual unit-test run completed with
**604 passed, 0 failed, 0 skipped**. This confirms tested code paths, not
retrieval quality or live provider behavior.

### Bottom line

**No, the present repository state cannot generate valid, comparable metrics
for all six strategies without preparation.** Minimum work: human-review and
verify an answerable retrieval gold set (and separately label unsupported
questions for answer/abstention evaluation), make all six paths use the same
corpus by selecting the complete dense collection and building full-corpus
keyword indexes, then run the evaluator. The metric formulas themselves do
not need changes. For defensible warmed/repeated latency, the evaluator needs
a small warm-up/repeat harness or an explicitly documented single-pass
protocol.

## 2. Repository architecture

| Layer | Responsibility and evidence |
|---|---|
| Documents and chunks | Annual-report PDFs are stored in `data/raw/`; document processing, text/table parsing and chunk assembly are implemented under `src/` and `src/chunking/`. The current combined chunk file is `data/chunks/all_chunks.jsonl`. |
| Embeddings | `src/embeddings.py` defines `BAAI/bge-base-en-v1.5`, revision and the asymmetric BGE query prefix. The local metadata artifact records 48,109 vectors of dimension 768. See [`src/embeddings.py`](../src/embeddings.py#L44) and [`data/embeddings/embeddings_metadata.json`](../data/embeddings/embeddings_metadata.json). |
| Dense store | `src/chroma_store.py` manages persisted Chroma collections; `src/retrieval.py` opens and queries them. The default retrieval collection is `financial_reports_filtered`; a second collection, `financial_reports`, is also present. See [`src/chroma_store.py`](../src/chroma_store.py#L19) and [`src/retrieval.py`](../src/retrieval.py#L41). |
| Keyword stores | `src/keyword_generator.py` generates per-chunk search terms/descriptions; `src/keyword_store.py` stores keyword vectors and a SQLite FTS5 index. Indexing and generation CLIs are in `scripts/generate_chunk_keywords.py` and `scripts/index_chunk_keywords.py`. |
| Query and retrieval orchestration | `src/query_planner.py` asks Gemini to produce a validated plan; `src/retrieval_pipeline.py` turns its tasks and metadata constraints into calls to the selected retriever. The offline metric evaluator bypasses this planning stage and calls retrievers directly. |
| Answer generation | `src/answer_generation.py` adapts retrieved evidence, requests structured output and constructs citations; the `src/llm/` package supplies Gemini/Groq provider adapters and JSON Schema validation. |
| Evaluation and tests | `tests/evaluate_pipeline_metrics.py` is the retrieval evaluator. `scripts/validate_answer_evaluation_dataset.py` validates answer-annotation data but does not score answers. Pytest tests are mostly isolated/faked or mocked. |
| Local report artifacts | `reports/`, `experiments/`, `figures/` and `presentation/` were empty at inspection. The notebooks are primarily environment, processing, chunking and embedding work; none contained retrieval metric result cells. |

The ordinary runtime path is: user question -> validated planner plan/tasks
-> normalized task query and metadata filters -> selected retrieval path(s)
-> ranked original chunks -> optional answer generator. The evaluator uses a
different, intentionally simpler path: fixed benchmark question or stored
rewrite -> direct retriever call with benchmark filters -> metric calculation.
It does not call the planner, answer generator or either LLM provider.

The code also contains `DenseRetriever`, a brute-force NumPy dot-product
retriever, but the six listed strategies and current evaluator use
`ChromaRetriever` for the semantic chunk path. See
[`src/retrieval.py`](../src/retrieval.py#L158) and
[`tests/evaluate_pipeline_metrics.py`](../tests/evaluate_pipeline_metrics.py#L395).

## 3. Six-strategy retrieval implementation comparison

The strategy map and default runtime candidate limits are in
[`src/hybrid_retrieval.py`](../src/hybrid_retrieval.py#L19). Its `STRATEGY_PATHS`
maps the six names to one or more independent paths. The actual path invocation,
deduplication and fusion are in
[`HybridRetriever.search`](../src/hybrid_retrieval.py#L161).

| Strategy | Implementation and query processing | Retrieval and ranking | Returned count and filters | Duplicate handling / latency |
|---|---|---|---|---|
| `chunk_semantic` | `ChromaRetriever.search` in [`src/retrieval.py`](../src/retrieval.py#L388). Collapses whitespace, applies the BGE query prefix through `RetrievalProcessor`/`embeddings.embed_query`, then queries Chroma. | Cosine-distance results are converted to similarity (`1 - distance`); Chroma's vector order is retained. | `top_k` argument; retriever default 5. Runtime pipeline default is 10 for non-baseline strategies. Optional exact filters: `company`, `financial_year`, `chunk_type`. | No post-query de-duplication; a Chroma collection has unique record IDs. No latency timer inside the retriever; evaluator times its call. |
| `keyword_semantic` | `KeywordSemanticRetriever.search` in [`src/hybrid_retrieval.py`](../src/hybrid_retrieval.py#L50) sends the original query through the same BGE query-embedding processor used by semantic retrieval. | Chroma searches a separate embedding of generated keywords/description; hit IDs are resolved to original source chunks, preserving source text and provenance. Chroma cosine ranking is used. | `top_k` argument; path default 5. Same three optional metadata filters, applied by Chroma. | No post-query de-duplication in this single path; stored IDs are unique. No path-local timer; outer evaluator times the call. |
| `keyword_lexical` | `KeywordLexicalRetriever.search` in [`src/hybrid_retrieval.py`](../src/hybrid_retrieval.py#L102) adapts `KeywordLexicalIndex.search` in [`src/keyword_store.py`](../src/keyword_store.py#L488). Query text is converted by `_fts_expression` to an FTS5 expression. | SQLite FTS5 searches generated keywords and descriptions; ranking is `bm25(keyword_search) ASC`, with `chunk_id` as deterministic tie-break. Results are mapped back to source chunks. | `top_k` argument; path default 5. `company`, `financial_year`, `chunk_type` are exact SQL equality filters. | No post-query de-duplication; `keyword_records.chunk_id` is a primary key. No path-local timer; outer evaluator times the call. |
| `hybrid_chunk_keyword` | Selects `chunk_semantic` and `keyword_semantic`. Both receive the same query and filters. | Rank-only Reciprocal Rank Fusion (RRF), default smoothing `k=60`: sum of `1/(k + rank)` across paths. Raw cosine scores are not mixed with each other or with lexical scores. | Default runtime path budget is 5 per path; default final `top_k` is 10. Evaluator overrides path budget to `candidate_k` (default 100). All selected paths receive the same filters. | Deduplicates repeated chunk IDs within paths and fuses shared IDs once. Final ties use chunk ID. Hybrid internal latency includes each path and RRF; evaluator also times the complete search call. |
| `hybrid_chunk_lexical` | Selects `chunk_semantic` and `keyword_lexical`; query/filter handling follows those respective paths. | Same rank-only RRF and deterministic tie-break. | Same runtime/evaluation output limits as above; all three metadata filters are passed to both paths. | Same explicit chunk-ID de-duplication and latency behavior as other hybrids. |
| `hybrid_all` | Selects all three paths: `chunk_semantic`, `keyword_semantic`, `keyword_lexical`. | Same rank-only RRF over up to three path rankings. Returned result includes path-rank/score evidence; final score is the RRF score. | Same runtime/evaluation output limits; exact filters are passed to every path. | Same explicit de-duplication. Failure of a required path raises `HybridRetrievalError`; it does not silently fall back to another path. |

Common implementation details:

- `RetrievalProcessor.normalize_query_text` collapses whitespace; `embed_query`
  uses `embeddings.embed_query`, which applies the BGE query prefix
  ([`src/retrieval.py`](../src/retrieval.py#L100),
  [`src/embeddings.py`](../src/embeddings.py#L44)).
- The runtime pipeline asks the planner for atomic tasks, derives planned
  company/year/chunk-type filters and passes them to the selected retriever.
  An omitted `top_k` is 5 for `baseline`, and 10 for the other strategies
  ([`src/retrieval_pipeline.py`](../src/retrieval_pipeline.py#L110)).
- The evaluator does not use this runtime default. It sets pure semantic
  retrieval to `candidate_k` and sets every selected hybrid path limit to
  `candidate_k`; `candidate_k` defaults to 100 and must be at least 10
  ([`tests/evaluate_pipeline_metrics.py`](../tests/evaluate_pipeline_metrics.py#L283)).
- `baseline` is accepted as an alias for the same chunk-semantic retriever.
  The evaluator's default gold variants therefore contain seven rows:
  baseline plus the six strategies. Avoid describing baseline and
  `chunk_semantic` as different retrieval algorithms.
- The default runtime path limits (5) and the evaluator limit (100) differ
  intentionally. A reported experiment must state its candidate budget.
- The evaluator supports optional query rewrites, chunk-type restriction and
  table score boosting. These are comparison dimensions, not part of a fair
  strategy-only comparison unless held constant across all six.

## 4. Evaluation framework status

The retrieval evaluator is
[`tests/evaluate_pipeline_metrics.py`](../tests/evaluate_pipeline_metrics.py#L1).
Metric fields and formulas are as follows:

| Metric | Implemented? | Calculation |
|---|---|---|
| Precision@5 | Yes | `|unique(top 5 IDs) intersect relevant IDs| / 5`. Fixed denominator of 5, including when fewer than five results are returned. |
| Recall@5 | Yes | `|unique(top 5 IDs) intersect relevant IDs| / |relevant IDs|`. |
| Recall@10 | Yes | Same formula over the first 10 IDs. |
| MRR | Yes | Reciprocal of the 1-based rank of the first relevant ID; 0 if no relevant ID is retrieved. |
| Average retrieval latency | Yes, one pass only | Arithmetic mean of one measured `total_latency_seconds` per case. Variant metrics are macro-averaged over cases. There are no repeated-run or percentile metrics. |

The metric functions are `score_ranked_ids` and `evaluate_variant`
([`tests/evaluate_pipeline_metrics.py`](../tests/evaluate_pipeline_metrics.py#L269),
[`tests/evaluate_pipeline_metrics.py`](../tests/evaluate_pipeline_metrics.py#L283)).
The formulas are internally consistent for a non-empty, human-verified set of
relevant chunk IDs. Benchmark validation requires each case to have a
non-empty `relevant_chunk_ids` list, so an unanswerable/no-relevant-evidence
case cannot be included as-is. Such cases require separate answerability and
abstention analysis; assigning an empty list would be rejected.

**Same-question support:** yes. A benchmark's cases are evaluated for each
variant; the generated default gold variants contain baseline plus all six
strategies. All variants receive the same question, case filters and
`candidate_k`, subject to variant-level query source, chunk type, collection,
table boost or path-budget options. These options must be held fixed for a
strategy comparison.

**Current default behavior:** `data/benchmark/retrieval_validation.json` has
one case and five variants that change original/rewritten query, chunk-type
filter or table boost. The variants do not specify the six retrieval strategy
names, so `strategy` defaults to `baseline`. Running the evaluator with no
arguments is not an all-six comparison. The `--gold-dataset` route supplies
all six plus the baseline alias, but rejects a dataset with no verified
cases.

**Saved output:** the evaluator prints a summary matrix and per-case
diagnostics to stdout only. It does not write CSV or JSON. The output is a
plain-text table; shell redirection/`Tee-Object` can preserve that text, but
there is no structured export for post-processing. Relevant implementation:
[`format_matrix`](../tests/evaluate_pipeline_metrics.py#L639),
[`format_case_diagnostics`](../tests/evaluate_pipeline_metrics.py#L653) and
[`main`](../tests/evaluate_pipeline_metrics.py#L691).

**Metric caveats and risks:**

- `Precision@5` is fixed at denominator 5, by design; explicitly describe
  this convention if a strategy returns fewer than five chunks.
- Macro-averaging gives every question equal weight, irrespective of how
  many relevant chunks it has. The formulas require exhaustive-enough human
  labels; a candidate generator's shortlist is not a complete relevance
  denominator.
- The evaluator supports optional table boosting and query/type variants.
  The default one-case benchmark is therefore not a strategy-only benchmark.
- Query/result duplicate handling is safe for the current retrievers
  (unique Chroma IDs, lexical primary key and explicit hybrid fusion
  de-duplication). The metric function itself is not a general duplicate
  sanitizer for arbitrary external retrievers.
- No evaluation result is currently saved as a CSV/JSON artifact. No
  measured retrieval scores are reported in this document.

## 5. Benchmark and relevance-label readiness

| Dataset | Available cases | What it contains now | Readiness |
|---|---:|---|---|
| [`data/benchmark/retrieval_gold_dataset.json`](../data/benchmark/retrieval_gold_dataset.json#L1) | 15 | All `verification_status: pending`; `relevant_chunk_ids` is empty for every case; 156 candidate-ID references are suggestions, not labels. One case has zero candidate IDs. | Not usable for P/R/MRR until a human confirms relevant IDs and marks cases verified. The evaluator intentionally ignores candidates. |
| [`data/benchmark/retrieval_validation.json`](../data/benchmark/retrieval_validation.json#L1) | 1 | Two populated `relevant_chunk_ids`; five variants concerning query wording, chunk type and table boost. No explicit human-verification provenance/status. | At most an unverified smoke fixture. It is not a six-strategy benchmark and is insufficient for general conclusions. |
| [`benchmarks/answer_evaluation/answer_evaluation_dataset.json`](../benchmarks/answer_evaluation/answer_evaluation_dataset.json#L1) | 15 | Seed answer-level cases; all pending, `gold: null`, no reviewed answer runs. | Not retrieval gold and not answer-scoring-ready. |
| [`benchmarks/answer_evaluation/financial_rag_60_question_benchmark.json`](../benchmarks/answer_evaluation/financial_rag_60_question_benchmark.json#L1) | 60 | Refined benchmark, 20 per company, all pending, all gold null, candidate expected-action status unverified. | A structured answer-review candidate set, not retrieval relevance gold and not yet answer-scoring-ready. |
| [`benchmarks/answer_evaluation/tcs_q01_q20_gold_draft.json`](../benchmarks/answer_evaluation/tcs_q01_q20_gold_draft.json#L1) and its review packet | 20 | Draft status is proposed/pending human review; evidence references are `proposed_unverified`. The packet says reviewer identity is not recorded. | Not authoritative gold and must not be reported as human-verified. |

The annotation contract is stated in
[`benchmarks/answer_evaluation/ANNOTATION_GUIDELINES.md`](../benchmarks/answer_evaluation/ANNOTATION_GUIDELINES.md#L1):
candidate IDs are search leads, not evidence judgments; evidence for an
answerable gold assertion requires human-verification status, reviewer and
rationale. The refined 60-case file intentionally includes potential
insufficient-evidence, conflicting-evidence, temporal/current and abstention
cases. Its candidate actions are not verified answers: the dataset records
42 candidate `answer`, 8 `abstain`, 6 `partial` and 4
`current_information_unavailable` actions, all unverified. It includes six
current-information periods. Do not present these counts as adjudicated
answerability findings.

Corpus availability also matters. The 60-case benchmark includes six cases
whose requested period is not represented for that company after normalizing
quarter labels: TCS 2009-2010 (one), Infosys 1989-1990 (one), and Reliance
Industries 2023-2024 (four). These may be deliberately unsupported tests,
but their intended action and evidence status remain unverified. TCS
FY2024-Q4 normalizes to 2023-2024 and is represented; it is not counted among
the six missing periods.

**Conclusions:**

- The 15-question seed is a valid *annotation starting point*, not a valid
  metric experiment yet. It has no verified relevance labels.
- The 60-question set is not ready for retrieval P/R/MRR: it is an
  answer-evaluation schema with no gold chunk IDs and all cases pending.
- Questions that may be unsupported/unanswerable are represented in the
  60-case answer benchmark, but candidate actions/categories do not establish
  that any particular question is truly unanswerable.
- Precision/recall/MRR should be reported only for adjudicated,
  answerable retrieval cases. Evaluate no-evidence/current-information
  behavior separately through answer-level metrics and reviewed labels.

The candidate builder has 15 curated questions and explicitly leaves
`relevant_chunk_ids` empty and `verification_status` pending
([`scripts/build_retrieval_gold_dataset.py`](../scripts/build_retrieval_gold_dataset.py#L62),
[`scripts/build_retrieval_gold_dataset.py`](../scripts/build_retrieval_gold_dataset.py#L279)).
Its keyword matcher creates review candidates; it does not decide ground
truth. The evaluator's gold loader accepts only verified cases and rejects a
pending-only dataset
([`tests/evaluate_pipeline_metrics.py`](../tests/evaluate_pipeline_metrics.py#L557)).

## 6. Corpus and database statistics

All counts below were read from local files and SQLite databases in read-only
mode. No database was rebuilt or changed.

### Current source chunks

| Company | Text chunks | Table chunks | Total chunks | Represented company/FY documents |
|---|---:|---:|---:|---:|
| Infosys | 11,802 | 4,530 | 16,332 | 12 |
| RelianceIndustries | 12,986 | 3,809 | 16,795 | 11 |
| TataConsultancyServices | 10,949 | 4,033 | 14,982 | 11 |
| **Total** | **35,737** | **12,372** | **48,109** | **34** |

`data/chunks/all_chunks.jsonl` has 48,109 rows and 48,109 unique chunk IDs.
Its combined financial-year range is 2014-2015 through 2025-2026. Coverage is
not complete for every company/year pair: RelianceIndustries 2023-2024 and
TataConsultancyServices 2019-2020 have no chunks. The raw folder contains 36
annual-report PDFs (12 per company); the chunk corpus represents 34
company/year pairs. In particular, 60-question cases targeting the missing
Reliance period cannot retrieve that company's source report from this
corpus.

The indexed chunk metadata has a generic `document: "Annual Report"` value,
not a unique source document ID. The 34 document count above is therefore
the number of company/year groups represented by the chunk IDs/metadata, not
a distinct document-ID field stored by Chroma.

### Persistent indexes

| Artifact | Observed state | Corpus alignment |
|---|---|---|
| `data/embeddings/embeddings.npy` and metadata | 48,109 vectors; 768 dimensions; BAAI/bge-base-en-v1.5 | Metadata count matches current combined chunk file. |
| Chroma `financial_reports` in `data/chroma_db` | 48,109 records | All chunk IDs and stored text match the current `all_chunks.jsonl`; company, FY, chunk type and document metadata also match. |
| Chroma `financial_reports_filtered` in `data/chroma_db` (retriever default) | 48,063 records | 48,063 IDs are in the current chunk file; 46 current chunks are absent. All 46 missing records are table chunks. |
| Chroma keyword pilot in `data/keyword_chroma_db_pilot` | Collection `financial_report_keywords_pilot`, 150 records | All 150 IDs resolve to the current source corpus, but this is a small partial subset. |
| SQLite FTS5 pilot `data/keyword_index/financial_keyword_index_pilot.sqlite` | 150 keyword records | All 150 IDs resolve to the current source corpus; company counts are Infosys 49, RelianceIndustries 50, TCS 51. |
| Expected full keyword paths | `data/keyword_chroma_db/financial_report_keywords` and `data/keyword_index/financial_keyword_index.sqlite` | Not present at these defaults. The existing pilot path and collection name differ from the defaults. |

The default Chroma collection is defined in
[`src/retrieval.py`](../src/retrieval.py#L41). Keyword default locations and
names are in [`src/keyword_store.py`](../src/keyword_store.py#L21). The
evaluation CLI accepts collection/path overrides
([`tests/evaluate_pipeline_metrics.py`](../tests/evaluate_pipeline_metrics.py#L691)).

**Same-corpus finding:** no. Current `chunk_semantic` runtime uses the
48,063-record filtered collection by default; keyword semantic and lexical
artifacts each cover only 150 chunks. Although a full 48,109-record dense
collection exists and exactly matches current chunk IDs/text, the two keyword
indexes are not full-corpus indexes. A fair all-six comparison requires the
full dense collection and full keyword indexes from the same chunk snapshot.

## 7. Latency measurement readiness

`evaluate_variant` wraps `retriever.search` with `time.perf_counter`; for
hybrid strategies it records the hybrid retriever's total path-plus-fusion
latency where available
([`tests/evaluate_pipeline_metrics.py`](../tests/evaluate_pipeline_metrics.py#L321),
[`src/hybrid_retrieval.py`](../src/hybrid_retrieval.py#L185)).

| Component | Included in current evaluator timer? | Reason |
|---|---|---|
| Evaluator query-source lookup and benchmark JSON parsing | No | These occur before the timer. |
| Runtime query planner / Gemini decomposition | No | Evaluator calls the retriever directly; it does not call `FinancialRetrievalPipeline`. |
| Retrieval-side whitespace normalization and query embedding | Yes | They occur inside `ChromaRetriever.search`. |
| Metadata filter validation | Yes | Chroma exact-value validation is inside `search`; lexical filter/query work is likewise inside its search. |
| Chroma vector query / FTS5 query | Yes | Both execute inside retriever `search`. |
| RRF and fusion bookkeeping | Yes | Included in the hybrid search total. |
| LLM answer generation | No | It is not invoked by this evaluator. |
| First-use embedding-model load | Potentially yes | `RetrievalProcessor` loads lazily on first use; no warm-up is performed. |

Current latency is therefore a **single-call macro mean per case**, not a
repeat-run latency study. The first call may include model initialization,
and strategy iteration order can produce cold/warm bias. No CLI option
supports warm-up counts, repeats, randomized/rotating order, median or
percentiles.

**Recommended protocol:** choose one definition and report it explicitly.
For end-to-end retriever latency, include query normalization, embedding,
filters, database retrieval and RRF for every strategy, but exclude planner
and LLM answer generation. Load the embedding model before timing; execute
at least one untimed warm-up per strategy/path; then run the same fixed
questions and filters repeatedly in a rotating or randomized strategy order,
with identical candidate budgets. Record per-call timings and report the
mean across question/repeat pairs plus median and dispersion (at least
standard deviation or interquartile range). Record hardware, software,
collection/index names, model revision, query set, path limits and cache
state. The existing evaluator can report the one-pass mean; warm-up and
repeated statistics need a small harness/evaluator extension or manual
aggregation.

## 8. Answer-generation and citation-validation test readiness

| Capability | Implemented | Tested in this repository | What the tests do not establish |
|---|---|---|---|
| Citation ID validation | Yes. Corpus-grounded claims must cite IDs supplied in the retrieval result; unknown IDs are rejected. | `tests/test_answer_generation.py` tests valid, unknown and mismatched source IDs with a fake provider. | Does not prove a cited chunk semantically supports the associated claim. |
| Citation source traceability | Yes. Backend constructs citation objects from original retrieved chunk metadata and adds retrieval strategy/rank/path provenance. | Unit tests assert citation metadata is sourced from retrieval results. | No end-to-end report-source audit or live generated answer has been reviewed. |
| Claim-level evidence validation | Partial. Every corpus-grounded claim requires one or more retrieved IDs, and IDs are membership-checked. Human annotation schema supports atomic claims and per-citation support/contradiction assessments. | Answer-generation structure/membership tests and annotation-validator tests exist. | No semantic entailment checker; the answer-evaluation datasets currently have no completed claim/citation reviews. |
| Abstention behavior | Yes. Deterministic no-evidence/current-information/clarification paths and schema constraints require abstained answers to have no claims and a limitation. | Mocked answer-generation tests cover abstention and schema combinations. | No human-rated abstention-quality results. |
| Current vs. historical classification | Yes, by query signals and financial-intent checks in `_requires_current_information`, `_requires_period_clarification` and `_select_answer_mode`; the query planner also normalizes stated fiscal years. | Planner and answer-generation unit tests cover scenarios with fake clients/providers. | Not a validated classifier benchmark or evidence of production accuracy. |
| Structured answer validation | Yes. Draft 2020-12 JSON Schema is applied to provider JSON and mode/status/claim/citation invariants are checked. | `tests/test_llm_validation.py` and `tests/test_answer_generation.py` cover valid/malformed/schema-invalid responses. | Tests use synthetic responses; no actual provider outputs are validated in a live run. |
| Gemini/Groq provider integration | Adapters and configuration exist. Gemini and Groq can be selected for answer generation. The query planner itself has a Gemini-specific client path. | `tests/test_llm_providers.py` mocks SDK calls and tests request building, parsing and error normalization. Planner tests use a fake Gemini client. | No live Gemini/Groq API call was run or recorded. Manual root scripts are not automated integration evidence. |
| Error handling | Yes. Provider errors are normalized/sanitized; planner retries selected transient errors; retrieval task failures are surfaced in pipeline status; invalid evidence/schema is rejected. | Unit tests simulate provider, planner, retrieval and validation failures. | No fault-injected end-to-end run against live stores/providers. |

Implementation references: [`src/answer_generation.py`](../src/answer_generation.py#L260),
[`src/answer_generation.py`](../src/answer_generation.py#L350),
[`src/answer_generation.py`](../src/answer_generation.py#L479),
[`src/answer_generation.py`](../src/answer_generation.py#L558),
[`src/llm/validation.py`](../src/llm/validation.py#L12),
[`src/llm/gemini_provider.py`](../src/llm/gemini_provider.py#L20),
[`src/llm/groq_provider.py`](../src/llm/groq_provider.py#L20), and
[`src/query_planner.py`](../src/query_planner.py#L536). The annotation schema
is defined in
[`benchmarks/answer_evaluation/answer_evaluation.schema.json`](../benchmarks/answer_evaluation/answer_evaluation.schema.json).

The answer data validator
([`scripts/validate_answer_evaluation_dataset.py`](../scripts/validate_answer_evaluation_dataset.py#L240))
checks JSON Schema and cross-field annotation semantics; it does not call an
LLM, validate retrieved citations at runtime or calculate answer-quality
scores.

## 9. Existing test-suite status

Command run from the project root:

```powershell
.venv\Scripts\python.exe -m pytest -q -p no:cacheprovider
```

**Result:** `604 passed in 21.19s`; 0 failed and 0 skipped. The cache provider
was disabled so this run did not update `.pytest_cache`.

The directly retrieval/evaluator-focused modules collected 108 test cases:

| Module | Collected cases | Focus |
|---|---:|---|
| `tests/test_retrieval.py` | 56 | Loader/alignment, query normalization, dense ranking, Chroma filters and search validation. |
| `tests/test_hybrid_retrieval.py` | 7 | RRF, merge behavior, filters/limits, ties, missing/failing paths. |
| `tests/test_retrieval_pipeline.py` | 22 | Planner/pipeline routing, metadata filters, error/status behavior. |
| `tests/test_evaluate_pipeline_metrics.py` | 23 | Metric formulas, benchmark validation, gold loader, strategy wiring and diagnostics. |

Supporting retrieval/index tests include
`tests/test_embeddings.py`, `tests/test_chroma_store.py`,
`tests/test_keyword_store.py`, `tests/test_keyword_generation.py` and
`tests/test_build_retrieval_gold_dataset.py`. Answer/provider coverage
includes `tests/test_answer_generation.py` (64 collected),
`tests/test_llm_providers.py` (37),
`tests/test_llm_validation.py` (11),
`tests/test_query_planner.py` (66),
`tests/test_answer_evaluation_dataset.py` (13) and
`tests/test_refined_answer_evaluation_dataset.py` (15).

These are unit tests, primarily using fake retrievers, temporary indexes,
fake providers and mocked SDK clients. Passing them demonstrates the
assertions under test; it does not establish retrieval effectiveness,
human-label validity, live Gemini/Groq integration, end-to-end citation
faithfulness or production latency.

## 10. Missing components and identified risks

1. **No verified retrieval gold.** Pending candidate IDs must not enter the
   relevant set. The current retrieval gold loader will reject the existing
   15 pending-only cases, as intended.
2. **Corpus gaps.** Two of 36 source company/year reports have no represented
   chunks: RelianceIndustries 2023-2024 and TCS 2019-2020. Some benchmark
   periods (including older historical cases) also lie outside indexed
   coverage.
3. **Default dense collection is incomplete.** It omits 46 table chunks.
   The full `financial_reports` collection is available and matched the
   chunk file during this read-only comparison; the evaluator needs an
   explicit collection override to use it.
4. **Keyword indexes are partial and defaults do not resolve.** Existing
   indexes contain 150 records, not all 48,109 chunks, and use
   `*_pilot` locations/names. A keyword-semantic exact filter may also reject
   company/year values absent from its pilot metadata.
5. **Current benchmark is not an all-six run.** The default JSON has one case
   and five input variants; it does not specify the six strategy names.
6. **No repeated/warmed latency support.** First-call initialization and
   order bias can affect the one-pass mean. No structured output is written.
7. **Unsupported cases are not supported by retrieval metric schema.** The
   evaluator requires non-empty relevant IDs; answerability and abstention
   must be judged separately.
8. **Relevance completeness risk.** The generated candidate matcher and
   keyword pilot only nominate chunks. Human reviewers should judge a pooled
   set from all six strategies and inspect the source/context; otherwise
   unobserved relevant chunks can make recall denominators incomplete.
9. **Live integrations are untested.** Provider and planner tests use mocks.
   Manual connection scripts exist but were not executed.
10. **Working-tree context.** The repository already had modifications and
    untracked files at inspection start. The test result and this report
    describe the current working tree, not necessarily the committed branch
    state. No pre-existing changes were reverted.

## 11. Exact files and commands for an actual retrieval evaluation

### Required files

- Fixed question set and human-verified retrieval labels:
  [`data/benchmark/retrieval_gold_dataset.json`](../data/benchmark/retrieval_gold_dataset.json).
  For each included answerable case, a reviewer must check the original
  source/context, populate `relevant_chunk_ids` with confirmed IDs, and set
  `verification_status` to `verified`. Keep candidate IDs separate. Exclude
  no-evidence/current cases from these P/R/MRR calculations and assess those
  separately.
- Current chunk source:
  [`data/chunks/all_chunks.jsonl`](../data/chunks/all_chunks.jsonl).
- Complete dense index:
  `data/chroma_db`, collection `financial_reports` (not the default
  `financial_reports_filtered` if all chunks are required).
- Complete keyword annotation/index artifacts:
  `data/keyword_metadata/chunk_keywords.jsonl`,
  `data/keyword_chroma_db/financial_report_keywords`, and
  `data/keyword_index/financial_keyword_index.sqlite`. These full-corpus
  artifacts are not present in the current workspace.
- Evaluator:
  [`tests/evaluate_pipeline_metrics.py`](../tests/evaluate_pipeline_metrics.py).

### Future execution commands

Keyword annotations for the full corpus require Gemini and may incur
substantial time/cost. These commands are listed for future execution only;
they were **not run** in this analysis:

```powershell
.venv\Scripts\python.exe scripts\generate_chunk_keywords.py --batch-size 16
.venv\Scripts\python.exe scripts\index_chunk_keywords.py --batch-size 64
```

After human-reviewing the gold set and confirming both keyword indexes have
full coverage, run all six strategies plus the baseline alias on the
identical verified cases:

```powershell
.venv\Scripts\python.exe tests\evaluate_pipeline_metrics.py --gold-dataset data\benchmark\retrieval_gold_dataset.json --persist-directory data\chroma_db --collection financial_reports --keyword-chroma-path data\keyword_chroma_db --keyword-collection financial_report_keywords --lexical-index data\keyword_index\financial_keyword_index.sqlite --candidate-k 100 | Tee-Object -FilePath reports\table_5_8_retrieval_evaluation.txt
```

This prints a text matrix and per-case diagnostic table; `Tee-Object`
preserves the text but does not produce CSV/JSON. The evaluator will fail
before retrieval while the gold dataset has no verified cases. It can also
fail when the keyword full-index paths/collection are absent. Do not present
the one-case default benchmark as Table 5.8 evidence.

Before claiming latency, apply the warm-up/repeated-run protocol in Section
7 or clearly label results as one-pass, cold/warm-state-dependent latency.
Record exact verified case count, gold dataset hash, chunk corpus hash,
collection/index names and sizes, candidate budgets, model/revision,
hardware, query/filter configuration, and test/run timestamp alongside the
table.

## 12. Recommended plan for producing Table 5.8

1. **Freeze the evaluation protocol.** Choose answerable retrieval questions,
   original vs rewritten query policy, candidate budget (100 is the current
   evaluator default), exact filters, tie policy, and whether baseline is
   shown separately from the six strategies.
2. **Review coverage before labels.** Remove or separately route questions
   whose company/FY is missing from the indexed corpus. Decide whether to
   add the missing reports only as a separate, documented corpus update;
   do not silently rebuild for this analysis.
3. **Create verified relevance judgments.** Pool top candidates from each of
   the six strategies plus a broad search, inspect source chunks and
   surrounding table/page context, and record all relevant IDs with reviewer
   rationale. Keep pending/unverified cases out of the gold evaluation.
4. **Build complete keyword indexes.** Generate and review annotations for
   the same corpus snapshot used by dense retrieval, then index them to the
   evaluator's expected paths/names. Verify index counts and ID coverage
   before measuring.
5. **Select the full dense collection.** Use `financial_reports` for the
   current complete chunk-ID/text match, or explicitly justify and report a
   different corpus. Do not compare it against the 150-record keyword pilot.
6. **Run the same cases across all six.** Use the future command in Section
   11 with one benchmark and a fixed candidate budget per path. Preserve the
   printed per-case output and capture the hashes/configuration.
7. **Measure latency consistently.** Warm each strategy, repeat the same
   cases in rotated order, exclude planner/answer generation, include query
   embedding and RRF, and report repeated latency statistics. Extend the
   evaluator or use a reproducible wrapper if needed.
8. **Review and report limitations.** Verify that gold IDs belong to the
   selected corpus and that no incomplete/failed cases were silently
   counted. Report answerability/abstention results separately from
   retrieval P/R/MRR. Do not infer answer correctness from retrieval scores.

## Final recommendation

**Can the current codebase generate valid Precision@5, Recall@5, Recall@10,
MRR and average latency results for all six retrieval strategies without code
changes?**

**Not from the current data/index state.** The metric implementations are
present, but the available gold is unverified, the default dense index is
missing 46 table chunks, and both keyword indexes are 150-record pilots at
non-default paths/names. Minimum required work is human-reviewed relevance
labels plus full-corpus keyword indexes and an explicit common dense
collection selection. No metric-formula source change is needed for a
one-pass comparison after those prerequisites. For a defensible warmed and
repeated latency result, add a small warm-up/repeat harness or clearly
describe the existing metric as a single-pass mean.
