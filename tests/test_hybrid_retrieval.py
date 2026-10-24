import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import hybrid_retrieval, retrieval


def _result(chunk_id, score, rank, **overrides):
    chunk = {
        "chunk_id": chunk_id,
        "text": f"original source text for {chunk_id}",
        "company": "Infosys",
        "document": "Annual Report",
        "financial_year": "2025-2026",
        "page": 77,
        "section": "Revenue",
        "chunk_type": "text",
        **overrides,
    }
    return retrieval.RetrievalResult(chunk_id, score, rank, chunk)


class FakePath:
    def __init__(self, results=None, error=None):
        self.results = results or []
        self.error = error
        self.calls = []

    def search(self, **kwargs):
        self.calls.append(kwargs)
        if self.error:
            raise self.error
        return self.results[:kwargs["top_k"]]


def _engine(chunk=None, keyword=None, lexical=None, **kwargs):
    return hybrid_retrieval.HybridRetriever(
        chunk_semantic=chunk or FakePath(),
        keyword_semantic=keyword,
        keyword_lexical=lexical,
        **kwargs,
    )


def test_rrf_accumulates_evidence_for_shared_chunks_and_preserves_provenance():
    chunk_path = FakePath([_result("shared", 0.8, 1), _result("chunk-only", 0.7, 2)])
    keyword_path = FakePath([_result("shared", 0.91, 1), _result("keyword-only", 0.6, 2)])
    lexical_path = FakePath([_result("lexical-only", 3.0, 1)])
    engine = _engine(chunk_path, keyword_path, lexical_path, rrf_k=10)

    results = engine.search("query", top_k=5, strategy="hybrid_all")

    assert [result.chunk_id for result in results][0] == "shared"
    shared = results[0]
    assert shared.score == pytest.approx(2 / 11)
    assert shared.chunk["text"] == "original source text for shared"
    assert shared.chunk["page"] == 77
    assert set(shared.chunk["retrieval_path_evidence"]) == {
        "chunk_semantic", "keyword_semantic"
    }
    assert shared.chunk["retrieval_score_type"] == "rrf"
    assert engine.last_search_diagnostics["path_candidate_counts"] == {
        "chunk_semantic": 2,
        "keyword_semantic": 2,
        "keyword_lexical": 1,
    }
    assert engine.last_search_diagnostics["path_overlap_counts"][
        "chunk_semantic|keyword_semantic"
    ] == 1
    assert engine.last_search_diagnostics["unique_candidate_count"] == 4
    assert engine.last_search_diagnostics["final_candidate_ids"][0] == "shared"


def test_shared_ids_merge_chroma_metadata_and_original_chunk_metadata():
    chroma_chunk = _result("shared", 0.8, 1).chunk
    chroma_chunk.pop("section")
    chroma_chunk["_source_metadata_json"] = "serialized source metadata"
    original_chunk = {
        **_result("shared", 0.9, 1).chunk,
        "table_id": "revenue-table",
        "table_heading": "Consolidated revenue",
    }
    engine = _engine(
        FakePath([retrieval.RetrievalResult("shared", 0.8, 1, chroma_chunk)]),
        FakePath([retrieval.RetrievalResult("shared", 0.9, 1, original_chunk)]),
        rrf_k=10,
    )

    result = engine.search("revenue", strategy="hybrid_chunk_keyword")[0]

    assert result.chunk["section"] == "Revenue"
    assert result.chunk["table_id"] == "revenue-table"
    assert "_source_metadata_json" not in result.chunk


def test_shared_ids_with_conflicting_source_text_fail_explicitly():
    chunk = _result("shared", 0.8, 1).chunk
    altered = {**chunk, "text": "different source text"}
    engine = _engine(
        FakePath([retrieval.RetrievalResult("shared", 0.8, 1, chunk)]),
        FakePath([retrieval.RetrievalResult("shared", 0.9, 1, altered)]),
    )

    with pytest.raises(hybrid_retrieval.HybridRetrievalError, match="conflicting 'text'"):
        engine.search("revenue", strategy="hybrid_chunk_keyword")


def test_path_filters_and_candidate_limits_are_applied_independently():
    chunk_path = FakePath([_result("c1", 0.8, 1)])
    keyword_path = FakePath([_result("k1", 0.7, 1)])
    engine = _engine(
        chunk_path,
        keyword_path,
        path_candidate_limits={"chunk_semantic": 3, "keyword_semantic": 4},
    )

    engine.search(
        "revenue", top_k=2, strategy="hybrid_chunk_keyword",
        company="Infosys", financial_year="2025-2026", chunk_type="table",
    )

    assert chunk_path.calls[0] == {
        "query": "revenue", "top_k": 3, "company": "Infosys",
        "financial_year": "2025-2026", "chunk_type": "table",
    }
    assert keyword_path.calls[0]["top_k"] == 4
    assert keyword_path.calls[0]["company"] == "Infosys"


def test_final_top_k_and_deterministic_tie_breaking():
    chunk_path = FakePath([_result("z", 0.9, 1), _result("a", 0.9, 2)])
    keyword_path = FakePath([_result("a", 0.8, 1), _result("z", 0.8, 2)])
    engine = _engine(
        chunk_path,
        keyword_path,
        path_candidate_limits={"chunk_semantic": 2, "keyword_semantic": 2},
        rrf_k=10,
    )

    first = engine.search("query", top_k=1, strategy="hybrid_chunk_keyword")
    second = engine.search("query", top_k=1, strategy="hybrid_chunk_keyword")

    assert [result.chunk_id for result in first] == ["a"]
    assert [result.chunk_id for result in second] == ["a"]
    assert first[0].score == pytest.approx(1 / 11 + 1 / 12)


def test_missing_path_and_path_exception_are_explicit():
    engine = _engine()
    with pytest.raises(hybrid_retrieval.HybridRetrievalError, match="unavailable retrieval path"):
        engine.search("query", strategy="hybrid_all")

    failing = _engine(chunk=FakePath(error=ValueError("index offline")))
    with pytest.raises(hybrid_retrieval.HybridRetrievalError, match="chunk_semantic.*index offline"):
        failing.search("query", strategy="chunk_semantic")


def test_invalid_strategy_and_candidate_configuration_rejected():
    with pytest.raises(ValueError, match="positive integer"):
        _engine(path_candidate_limits={"chunk_semantic": 0})
    with pytest.raises(ValueError, match="Unknown experimental"):
        _engine().search("query", strategy="hybrid_mystery")
    with pytest.raises(ValueError, match="positive integer"):
        _engine().search("query", top_k=0, strategy="chunk_semantic")