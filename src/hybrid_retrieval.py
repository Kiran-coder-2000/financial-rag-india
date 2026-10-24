"""Independent keyword retrieval paths and reciprocal-rank fusion."""

from dataclasses import dataclass
from time import perf_counter

from .retrieval import ChromaRetriever, RetrievalResult, RetrievalProcessor
from .keyword_store import (
    DEFAULT_KEYWORD_CHROMA_PATH,
    DEFAULT_KEYWORD_COLLECTION,
    KeywordIndexError,
    KeywordLexicalIndex,
)


class HybridRetrievalError(RuntimeError):
    """Raised when a configured retrieval path fails or returns invalid evidence."""


DEFAULT_PATH_CANDIDATE_LIMITS = {
    "chunk_semantic": 5,
    "keyword_semantic": 5,
    "keyword_lexical": 5,
}

STRATEGY_PATHS = {
    "chunk_semantic": ("chunk_semantic",),
    "keyword_semantic": ("keyword_semantic",),
    "keyword_lexical": ("keyword_lexical",),
    "hybrid_chunk_keyword": ("chunk_semantic", "keyword_semantic"),
    "hybrid_chunk_lexical": ("chunk_semantic", "keyword_lexical"),
    "hybrid_all": ("chunk_semantic", "keyword_semantic", "keyword_lexical"),
}


def _merge_source_metadata(existing: dict, incoming: dict, chunk_id: str) -> dict:
    merged = {key: value for key, value in existing.items() if key != "_source_metadata_json"}
    for key, value in incoming.items():
        if key == "_source_metadata_json":
            continue
        current = merged.get(key)
        if current is not None and value is not None and current != value:
            raise HybridRetrievalError(
                f"Retrieval paths returned conflicting {key!r} for source chunk {chunk_id!r}."
            )
        if key not in merged or current is None:
            merged[key] = value
    return merged


class KeywordSemanticRetriever:
    """Search keyword-only vectors, then restore original chunk text/provenance."""

    def __init__(
        self,
        source_chunks_by_id: dict[str, dict],
        processor: RetrievalProcessor | None = None,
        persist_directory=DEFAULT_KEYWORD_CHROMA_PATH,
        collection_name=DEFAULT_KEYWORD_COLLECTION,
        client=None,
        collection=None,
    ):
        self.source_chunks_by_id = source_chunks_by_id
        self._retriever = ChromaRetriever(
            processor=processor,
            persist_directory=persist_directory,
            collection_name=collection_name,
            client=client,
            collection=collection,
        )

    def search(
        self,
        query: str,
        top_k: int = 5,
        company: str | None = None,
        financial_year: str | None = None,
        chunk_type: str | None = None,
    ) -> list[RetrievalResult]:
        keyword_hits = self._retriever.search(
            query=query,
            top_k=top_k,
            company=company,
            financial_year=financial_year,
            chunk_type=chunk_type,
        )
        original_hits = []
        for hit in keyword_hits:
            source_chunk = self.source_chunks_by_id.get(hit.chunk_id)
            if source_chunk is None:
                raise HybridRetrievalError(
                    f"Keyword vector hit {hit.chunk_id!r} does not resolve to an original source chunk."
                )
            original_hits.append(RetrievalResult(
                chunk_id=hit.chunk_id,
                score=hit.score,
                rank=hit.rank,
                chunk=source_chunk,
            ))
        return original_hits


class KeywordLexicalRetriever:
    """Adapt the SQLite keyword index to the shared retrieval-result interface."""

    def __init__(self, index: KeywordLexicalIndex):
        self.index = index

    def search(self, query: str, top_k: int = 5, company=None, financial_year=None, chunk_type=None):
        try:
            return self.index.search(
                query=query,
                top_k=top_k,
                company=company,
                financial_year=financial_year,
                chunk_type=chunk_type,
            )
        except KeywordIndexError as error:
            raise HybridRetrievalError(f"Keyword lexical path failed: {error}") from error


@dataclass(frozen=True)
class RetrievalPath:
    name: str
    retriever: object


class HybridRetriever:
    """Execute configured retrieval paths and fuse unique chunks with RRF."""

    def __init__(
        self,
        chunk_semantic: ChromaRetriever,
        keyword_semantic: KeywordSemanticRetriever | None = None,
        keyword_lexical: KeywordLexicalRetriever | KeywordLexicalIndex | None = None,
        path_candidate_limits: dict[str, int] | None = None,
        rrf_k: int = 60,
    ):
        if isinstance(rrf_k, bool) or not isinstance(rrf_k, int) or rrf_k < 0:
            raise ValueError("rrf_k must be a non-negative integer.")
        self.paths = {
            "chunk_semantic": chunk_semantic,
            "keyword_semantic": keyword_semantic,
            "keyword_lexical": (
                KeywordLexicalRetriever(keyword_lexical)
                if isinstance(keyword_lexical, KeywordLexicalIndex)
                else keyword_lexical
            ),
        }
        self.path_candidate_limits = dict(DEFAULT_PATH_CANDIDATE_LIMITS)
        if path_candidate_limits:
            unknown = set(path_candidate_limits) - set(self.path_candidate_limits)
            if unknown:
                raise ValueError(f"Unknown retrieval path limit(s): {sorted(unknown)}.")
            self.path_candidate_limits.update(path_candidate_limits)
        for name, limit in self.path_candidate_limits.items():
            if isinstance(limit, bool) or not isinstance(limit, int) or limit <= 0:
                raise ValueError(f"Candidate limit for {name} must be a positive integer.")
        self.rrf_k = rrf_k
        self.last_search_diagnostics = {}

    def search(
        self,
        query: str,
        top_k: int = 10,
        strategy: str = "hybrid_all",
        company: str | None = None,
        financial_year: str | None = None,
        chunk_type: str | None = None,
    ) -> list[RetrievalResult]:
        if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k <= 0:
            raise ValueError("top_k must be a positive integer.")
        if strategy not in STRATEGY_PATHS:
            raise ValueError(f"Unknown experimental retrieval strategy {strategy!r}.")
        selected_paths = STRATEGY_PATHS[strategy]
        unavailable = [name for name in selected_paths if self.paths.get(name) is None]
        if unavailable:
            raise HybridRetrievalError(
                f"Strategy {strategy!r} requires unavailable retrieval path(s): {unavailable}."
            )

        fused: dict[str, dict] = {}
        path_ids = {}
        path_counts = {}
        path_latencies = {}
        total_started = perf_counter()
        for path_name in selected_paths:
            retriever = self.paths[path_name]
            candidate_limit = self.path_candidate_limits[path_name]
            path_started = perf_counter()
            try:
                results = retriever.search(
                    query=query,
                    top_k=candidate_limit,
                    company=company,
                    financial_year=financial_year,
                    chunk_type=chunk_type,
                )
            except Exception as error:
                path_latencies[path_name] = perf_counter() - path_started
                self.last_search_diagnostics = {
                    "path_candidate_counts": path_counts,
                    "path_latencies_seconds": path_latencies,
                    "total_latency_seconds": perf_counter() - total_started,
                    "failed_path": path_name,
                }
                raise HybridRetrievalError(
                    f"Retrieval path {path_name!r} failed: {type(error).__name__}: {error}"
                ) from error

            seen_in_path = set()
            for path_rank, result in enumerate(results[:candidate_limit], start=1):
                if result.chunk_id in seen_in_path:
                    continue
                seen_in_path.add(result.chunk_id)
                candidate = fused.setdefault(result.chunk_id, {
                    "rrf_score": 0.0,
                    "chunk": result.chunk,
                    "path_evidence": {},
                })
                if candidate["chunk"] != result.chunk:
                    candidate["chunk"] = _merge_source_metadata(
                        candidate["chunk"], result.chunk, result.chunk_id
                    )
                candidate["rrf_score"] += 1.0 / (self.rrf_k + path_rank)
                candidate["path_evidence"][path_name] = {
                    "rank": path_rank,
                    "score": float(result.score),
                }
            path_ids[path_name] = seen_in_path
            path_counts[path_name] = len(seen_in_path)
            path_latencies[path_name] = perf_counter() - path_started

        ordered = sorted(
            fused.items(),
            key=lambda item: (-item[1]["rrf_score"], item[0]),
        )[:top_k]
        overlaps = {}
        for index, left in enumerate(selected_paths):
            for right in selected_paths[index + 1:]:
                overlaps[f"{left}|{right}"] = len(path_ids[left] & path_ids[right])
        self.last_search_diagnostics = {
            "path_candidate_counts": path_counts,
            "path_overlap_counts": overlaps,
            "unique_candidate_count": len(fused),
            "path_latencies_seconds": path_latencies,
            "total_latency_seconds": perf_counter() - total_started,
            "final_candidate_ids": [chunk_id for chunk_id, _ in ordered],
        }
        return [
            RetrievalResult(
                chunk_id=chunk_id,
                score=evidence["rrf_score"],
                rank=rank,
                chunk={
                    **evidence["chunk"],
                    "retrieval_score_type": "rrf",
                    "retrieval_path_evidence": evidence["path_evidence"],
                },
            )
            for rank, (chunk_id, evidence) in enumerate(ordered, start=1)
        ]