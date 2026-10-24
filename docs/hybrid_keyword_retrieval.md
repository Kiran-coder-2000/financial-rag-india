# Hybrid Keyword Retrieval

This is an experimental extension. The current BGE/Chroma dense path remains the default and is kept available as the measured baseline. No source chunks or baseline Chroma collection are modified by these stages.

## Why Keywords

Dense embeddings are useful for semantic similarity, but exact financial terms, periods, table labels and entity names can be missed or ranked below broadly related narrative passages. This experiment adds two separate representations:

- **Keyword semantic:** a vector of generated keywords and a short description, embedded by the repository's existing BGE model.
- **Keyword lexical:** an SQLite FTS5 index over the keywords and description, searched lexically with company/year/type metadata filters.

The original chunk text remains the evidence returned to the user. Keyword strings are retrieval signals only.

## Storage

- Source: `data/chunks/all_chunks.jsonl` (read-only)
- Existing embeddings: `data/embeddings/` (read-only)
- Existing baseline collection: `data/chroma_db/financial_reports_filtered` (read-only)
- Annotation JSONL: `data/keyword_metadata/chunk_keywords.jsonl`
- Keyword-vector Chroma: `data/keyword_chroma_db/financial_report_keywords`
- Lexical SQLite/FTS5: `data/keyword_index/financial_keyword_index.sqlite`

The `data/` directory is ignored by Git in this repository. Keyword indexers reject the baseline collection name/path. The keyword vector collection records the same BGE model/revision, 768 dimensions, and cosine metric as the baseline. Keyword document embeddings have no document prefix; queries use the existing `RetrievalProcessor` and query prefix.

Each annotation contains the source `chunk_id`, copied provenance, source-text SHA-256, keywords, description, annotation provider/model/version and status. The generator never accepts company names, years, values, or other source metadata from Gemini; it copies those fields from the source record.

## Stage A: Pilot Annotations

Preview a deterministic 200-chunk sample stratified across company, financial year and chunk type, without calling Gemini or writing files:

```powershell
.venv\Scripts\python.exe scripts\generate_chunk_keywords.py --limit 200 --dry-run
```

Generate the pilot into a separate file:

```powershell
.venv\Scripts\python.exe scripts\generate_chunk_keywords.py --limit 200 --batch-size 16 --output data\keyword_metadata\chunk_keywords_pilot.jsonl
```

The runner uses `.env`/environment settings `GEMINI_API_KEY` and `GEMINI_MODEL`, as does the existing query planner. Keys are never printed. It requests structured JSON and validates it with Pydantic, retries transient failures with bounded exponential backoff, checkpoints atomically after each batch, and resumes successful/skipped records only when source text hash, source metadata, provider, model and annotation version still match. Failed records are retried on a later run. `--force` explicitly regenerates selected annotations.

Review the pilot JSONL manually before full processing. Check factual support, consolidated/standalone distinction, reporting periods, table headers/units, useful domain terms and generic/filler keywords. The current format marks generation status, not human approval; do not treat `success` as verified gold evidence.

The runner reports selected, processed, resumed, successful, failed and skipped counts, annotation success rate, average keywords per successful chunk, elapsed time and estimated full-corpus time from observed throughput. Cost remains unestimated because provider pricing/token usage is not configured.

After the pilot is acceptable, process the full source in resumable batches:

```powershell
.venv\Scripts\python.exe scripts\generate_chunk_keywords.py --batch-size 16
```

To use a different annotation model without changing retrieval embeddings, pass `--model MODEL_NAME`; that model identity is recorded and successful records from another model are not reused. Use `--force` only when intentionally regenerating.

## Stage B: Build Indexes

Build both separate indexes from successful annotation JSONL records. This step does not call Gemini:

```powershell
.venv\Scripts\python.exe scripts\index_chunk_keywords.py --batch-size 64
```

Build only one index when needed:

```powershell
.venv\Scripts\python.exe scripts\index_chunk_keywords.py --skip-vectors
.venv\Scripts\python.exe scripts\index_chunk_keywords.py --skip-lexical
```

Indexes use stable original chunk IDs and update in place. The vector representation contains only keywords plus the short description, not the full chunk text. The lexical index maps tokens/phrases to chunk IDs and retains metadata for exact filters. Both resolve results back to original source chunks; unresolved IDs are errors.

## Stage C: Retrieval Paths and Fusion

The manual query tool keeps `baseline` as its default. Experimental paths are opt-in:

```powershell
.venv\Scripts\python.exe tests\test_retrieval_manual.py --strategy baseline
.venv\Scripts\python.exe tests\test_retrieval_manual.py --strategy chunk_semantic
.venv\Scripts\python.exe tests\test_retrieval_manual.py --strategy keyword_semantic
.venv\Scripts\python.exe tests\test_retrieval_manual.py --strategy keyword_lexical
.venv\Scripts\python.exe tests\test_retrieval_manual.py --strategy hybrid_chunk_keyword
.venv\Scripts\python.exe tests\test_retrieval_manual.py --strategy hybrid_chunk_lexical
.venv\Scripts\python.exe tests\test_retrieval_manual.py --strategy hybrid_all
```

Each path is independently callable and takes the same query and metadata filters. Defaults are top 5 per path and final top 10. Hybrid fusion deduplicates by original chunk ID and applies Reciprocal Rank Fusion:

`RRF(chunk) = sum(1 / (k + rank_in_path))`

The default smoothing constant is `k=60`; ties are ordered by chunk ID. Raw cosine and BM25 scores are not combined. Returned chunks include original text/provenance and per-path rank/score evidence. Missing indexes or failed paths are surfaced, not silently replaced by baseline results.

## Stage D: Offline Evaluation

The existing `data/benchmark/retrieval_validation.json` remains supported and unchanged. For generated gold data, a human must set `verification_status` to `verified`, enter the expected answer if known, and move confirmed IDs into `relevant_chunk_ids`. The evaluator ignores `candidate_relevant_chunk_ids` and rejects a gold dataset with no verified cases.

Run the existing benchmark:

```powershell
.venv\Scripts\python.exe tests\evaluate_pipeline_metrics.py
```

Run only manually verified gold cases:

```powershell
.venv\Scripts\python.exe tests\evaluate_pipeline_metrics.py --gold-dataset data\benchmark\retrieval_gold_dataset.json
```

Evaluation compares baseline, chunk semantic, keyword semantic, lexical, two-path and three-path strategies with the same case queries, filters and per-path candidate budget. Reported metrics are macro-averaged: Recall@k is relevant IDs in the first k divided by all gold relevant IDs; Precision@5 is relevant IDs in the first five divided by 5; MRR is the mean reciprocal rank of the first relevant ID. Per-case output includes first relevant rank, retrieved/missing IDs, per-path candidate counts, pairwise overlap, unique candidates, per-path/total latency and final candidate IDs. No LLM is called during evaluation.

The current generated gold dataset has pending candidates and is not yet suitable for gold evaluation. The existing validation benchmark currently has one case, so results from it are a smoke comparison, not evidence that a strategy is generally superior. Use a broad reviewed set and held-out questions before selecting a winner.

## Troubleshooting and Limitations

- Missing `GEMINI_API_KEY` or `GEMINI_MODEL`: set them in the existing `.env` or environment. Annotation stops with an explicit configuration error.
- Rate limit/provider failure: failed annotation records remain retryable; rerun the same command. No placeholder keywords are substituted.
- Annotation/source mismatch: regenerate the affected annotation; indexers verify source text hash and copied metadata.
- Missing keyword vector collection or lexical DB: run the index command before selecting that retrieval path.
- SQLite FTS5 availability: it is part of the Python SQLite build; initialization errors are reported explicitly.
- The project requirements file currently does not declare installed runtime dependencies including `google-genai`, ChromaDB and sentence-transformers. Pin and declare these for clean-environment reproducibility in a separate dependency-management change.
- The baseline collection/model/chunks are not rebuilt or changed by these commands.
