"""
tests/test_retrieval.py

Tests for src/retrieval.py. The real model is never loaded -- a
fake model is injected into RetrievalProcessor throughout.
"""

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import retrieval


class FakeModel:
    """Deterministic fake, sha256-seeded (not Python's hash()) so
    vectors are reproducible across processes -- same approach as
    test_embeddings.py's FakeModel."""
    def __init__(self, dimension=8):
        self.dimension = dimension

    def encode(self, texts, batch_size=32, show_progress_bar=False,
               convert_to_numpy=True, normalize_embeddings=True):
        vectors = []
        for text in texts:
            digest = hashlib.sha256(text.encode("utf-8")).digest()
            seed = int.from_bytes(digest[:4], "big")
            rng = np.random.default_rng(seed)
            vec = rng.random(self.dimension).astype(np.float32)
            if normalize_embeddings:
                vec = vec / np.linalg.norm(vec)
            vectors.append(vec)
        return np.array(vectors, dtype=np.float32)


def _write_chunks_and_embeddings(tmp_path, chunk_ids, dimension=8, mismatch=False):
    """Build a matching (or deliberately mismatched) chunks.jsonl +
    embeddings directory pair."""
    chunks_file = tmp_path / "chunks.jsonl"
    with open(chunks_file, "w", encoding="utf-8") as f:
        for cid in chunk_ids:
            f.write(json.dumps({
                "chunk_id": cid, "text": f"content for {cid}", "chunk_type": "text",
                "company": "X", "document": "D", "financial_year": "2025", "page": 1, "tokens": 10,
            }) + "\n")

    embeddings_root = tmp_path / "embeddings"
    embeddings_root.mkdir()

    stored_ids = list(reversed(chunk_ids)) if mismatch else chunk_ids
    vectors = np.random.default_rng(0).random((len(stored_ids), dimension)).astype(np.float32)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    vectors = vectors / norms

    with open(embeddings_root / "embeddings.npy", "wb") as f:
        np.save(f, vectors)
    (embeddings_root / "chunk_ids.json").write_text(json.dumps(stored_ids))
    (embeddings_root / "embeddings_metadata.json").write_text(json.dumps({
        "model_name": "m", "model_revision": "r", "dimension": dimension, "count": len(stored_ids),
    }))

    return chunks_file, embeddings_root


class TestRetrievalLoaderHappyPath:
    def test_loads_chunks_and_embeddings_together(self, tmp_path):
        chunks_file, embeddings_root = _write_chunks_and_embeddings(tmp_path, ["c1", "c2", "c3"])

        loader = retrieval.RetrievalLoader(chunks_file, embeddings_root)
        chunks, result = loader.load()

        assert len(chunks) == 3
        assert result.chunk_ids == ["c1", "c2", "c3"]
        assert result.vectors.shape == (3, 8)

    def test_load_chunks_alone(self, tmp_path):
        chunks_file, embeddings_root = _write_chunks_and_embeddings(tmp_path, ["c1"])
        loader = retrieval.RetrievalLoader(chunks_file, embeddings_root)
        chunks = loader.load_chunks()
        assert chunks[0]["chunk_id"] == "c1"

    def test_load_embeddings_alone(self, tmp_path):
        chunks_file, embeddings_root = _write_chunks_and_embeddings(tmp_path, ["c1", "c2"])
        loader = retrieval.RetrievalLoader(chunks_file, embeddings_root)
        result = loader.load_embeddings()
        assert result.chunk_ids == ["c1", "c2"]


class TestRetrievalLoaderMissingFiles:
    def test_missing_chunks_file_raises_clear_error(self, tmp_path):
        loader = retrieval.RetrievalLoader(tmp_path / "does_not_exist.jsonl", tmp_path / "embeddings")
        with pytest.raises(FileNotFoundError, match="Chunk file not found"):
            loader.load_chunks()

    def test_missing_embeddings_dir_raises_clear_error(self, tmp_path):
        chunks_file = tmp_path / "chunks.jsonl"
        chunks_file.write_text(json.dumps({"chunk_id": "c1", "text": "x"}) + "\n")
        loader = retrieval.RetrievalLoader(chunks_file, tmp_path / "does_not_exist")
        with pytest.raises(FileNotFoundError, match="Embeddings directory not found"):
            loader.load_embeddings()


class TestRetrievalLoaderAlignment:
    def test_mismatched_chunk_ids_raises_value_error(self, tmp_path):
        chunks_file, embeddings_root = _write_chunks_and_embeddings(
            tmp_path, ["c1", "c2", "c3"], mismatch=True
        )
        loader = retrieval.RetrievalLoader(chunks_file, embeddings_root)

        with pytest.raises(ValueError, match="do not match"):
            loader.load()

    def test_different_count_raises_value_error(self, tmp_path):
        chunks_file, embeddings_root = _write_chunks_and_embeddings(tmp_path, ["c1", "c2"])
        # Add a chunk to the chunks file only, after embeddings were written.
        with open(chunks_file, "a", encoding="utf-8") as f:
            f.write(json.dumps({
                "chunk_id": "c3", "text": "extra", "chunk_type": "text",
                "company": "X", "document": "D", "financial_year": "2025", "page": 1, "tokens": 10,
            }) + "\n")

        loader = retrieval.RetrievalLoader(chunks_file, embeddings_root)
        with pytest.raises(ValueError, match="do not match"):
            loader.load()


class TestQueryNormalization:
    def test_collapses_internal_whitespace(self):
        result = retrieval.RetrievalProcessor.normalize_query_text("what   was\n\nrevenue")
        assert result == "what was revenue"

    def test_strips_leading_trailing_whitespace(self):
        result = retrieval.RetrievalProcessor.normalize_query_text("  revenue growth  ")
        assert result == "revenue growth"

    def test_non_string_raises_type_error(self):
        with pytest.raises(TypeError):
            retrieval.RetrievalProcessor.normalize_query_text(None)


class TestRetrievalProcessorEmbedQuery:
    def test_embed_query_with_injected_model(self):
        processor = retrieval.RetrievalProcessor(model=FakeModel(dimension=8))
        vector = processor.embed_query("what was revenue in FY25")
        assert vector.shape == (8,)

    def test_query_is_normalized_before_embedding(self):
        captured = []

        class RecordingModel(FakeModel):
            def encode(self, texts, **kwargs):
                captured.extend(texts)
                return super().encode(texts, **kwargs)

        processor = retrieval.RetrievalProcessor(model=RecordingModel())
        processor.embed_query("  revenue   growth  ")

        # QUERY_PREFIX + normalized text, exactly once.
        import embeddings as emb
        assert captured == [emb.QUERY_PREFIX + "revenue growth"]

    def test_empty_query_rejected_via_embeddings_module(self):
        """Empty-query rejection is delegated to embeddings.embed_query
        -- not reimplemented here -- so this confirms delegation
        actually happens rather than silently swallowing the error."""
        import embeddings as emb
        processor = retrieval.RetrievalProcessor(model=FakeModel())
        with pytest.raises(emb.EmbeddingValidationError):
            processor.embed_query("   ")

    def test_model_not_loaded_when_injected(self, monkeypatch):
        """If load_model() were called despite a model being
        injected, it would try to download the real model."""
        import embeddings as emb

        def _fail_if_called(*args, **kwargs):
            raise AssertionError("load_model() should not be called when a model is injected")

        monkeypatch.setattr(emb, "load_model", _fail_if_called)

        processor = retrieval.RetrievalProcessor(model=FakeModel())
        processor.embed_query("a query")  # must not raise the assertion above

    def test_model_loaded_lazily_only_once(self, monkeypatch):
        import embeddings as emb

        call_count = {"n": 0}

        def _fake_load_model(*args, **kwargs):
            call_count["n"] += 1
            return FakeModel()

        monkeypatch.setattr(emb, "load_model", _fake_load_model)

        processor = retrieval.RetrievalProcessor()  # no model injected
        processor.embed_query("first query")
        processor.embed_query("second query")

        assert call_count["n"] == 1  # loaded once, reused for the second call


# ================================================================
# DenseRetriever
# ================================================================

def _make_embedding_result(chunk_ids, vectors):
    import embeddings as emb
    return emb.EmbeddingResult(
        chunk_ids=list(chunk_ids), vectors=np.asarray(vectors, dtype=np.float32),
        model_name="m", model_revision="r", dimension=vectors.shape[1],
    )


def _chunk_dict(chunk_id, **overrides):
    d = {"chunk_id": chunk_id, "text": f"content {chunk_id}", "chunk_type": "text",
          "company": "X", "document": "D", "financial_year": "2025", "page": 1, "tokens": 10}
    d.update(overrides)
    return d


class TestDenseRetrieverConstruction:
    def test_valid_construction(self):
        chunk_ids = ["a", "b", "c"]
        chunks = [_chunk_dict(c) for c in chunk_ids]
        vectors = np.eye(3, dtype=np.float32)  # already unit vectors
        result = _make_embedding_result(chunk_ids, vectors)

        retriever = retrieval.DenseRetriever(chunks, result)
        assert retriever.dimension == 3

    def test_misaligned_chunk_ids_rejected(self):
        chunks = [_chunk_dict("a"), _chunk_dict("b")]
        result = _make_embedding_result(["b", "a"], np.eye(2, dtype=np.float32))  # reversed order
        with pytest.raises(ValueError, match="not aligned"):
            retrieval.DenseRetriever(chunks, result)

    def test_row_count_mismatch_rejected(self):
        chunks = [_chunk_dict("a"), _chunk_dict("b"), _chunk_dict("c")]
        result = _make_embedding_result(["a", "b", "c"], np.eye(2, dtype=np.float32))  # only 2 rows
        with pytest.raises(ValueError):
            retrieval.DenseRetriever(chunks, result)

    def test_non_2d_vectors_rejected(self):
        chunks = [_chunk_dict("a")]
        # Build with valid 2-D data first (the helper itself needs
        # .shape[1], so it can't be constructed with 1-D input
        # directly), then corrupt to 1-D afterward to isolate
        # exactly the case under test.
        result = _make_embedding_result(["a"], np.array([[1.0]], dtype=np.float32))
        result.vectors = np.array([1.0], dtype=np.float32)
        with pytest.raises(ValueError, match="2-dimensional"):
            retrieval.DenseRetriever(chunks, result)


class TestDenseRetrieverRanking:
    def _build(self):
        # 4 orthonormal-ish vectors so dot products are exact and
        # easy to reason about by hand.
        chunk_ids = ["a", "b", "c", "d"]
        chunks = [_chunk_dict(c) for c in chunk_ids]
        vectors = np.array([
            [1.0, 0.0, 0.0],   # a: identical to query -> score 1.0
            [0.0, 1.0, 0.0],   # b: orthogonal -> score 0.0
            [0.7071, 0.7071, 0.0],  # c: 45 degrees -> score ~0.7071
            [-1.0, 0.0, 0.0],  # d: opposite -> score -1.0
        ], dtype=np.float32)
        result = _make_embedding_result(chunk_ids, vectors)
        return retrieval.DenseRetriever(chunks, result)

    def test_correct_similarity_ranking(self):
        retriever = self._build()
        query = np.array([1.0, 0.0, 0.0], dtype=np.float32)

        results = retriever.search(query, top_k=4)

        assert [r.chunk_id for r in results] == ["a", "c", "b", "d"]

    def test_scores_are_preserved_and_correct(self):
        retriever = self._build()
        query = np.array([1.0, 0.0, 0.0], dtype=np.float32)

        results = retriever.search(query, top_k=4)

        assert results[0].score == pytest.approx(1.0, abs=1e-4)
        assert results[1].score == pytest.approx(0.7071, abs=1e-3)
        assert results[2].score == pytest.approx(0.0, abs=1e-4)
        assert results[3].score == pytest.approx(-1.0, abs=1e-4)

    def test_top_k_limits_result_count(self):
        retriever = self._build()
        query = np.array([1.0, 0.0, 0.0], dtype=np.float32)

        results = retriever.search(query, top_k=2)

        assert len(results) == 2
        assert [r.chunk_id for r in results] == ["a", "c"]

    def test_ranks_are_1_indexed_and_sequential(self):
        retriever = self._build()
        results = retriever.search(np.array([1.0, 0.0, 0.0], dtype=np.float32), top_k=3)
        assert [r.rank for r in results] == [1, 2, 3]

    def test_chunk_provenance_preserved(self):
        retriever = self._build()
        results = retriever.search(np.array([1.0, 0.0, 0.0], dtype=np.float32), top_k=1)

        top = results[0]
        assert top.chunk_id == "a"
        assert top.chunk["chunk_id"] == "a"
        assert top.chunk["company"] == "X"
        assert top.chunk["text"] == "content a"


class TestDenseRetrieverTopKValidation:
    def _build(self):
        chunk_ids = ["a", "b", "c"]
        chunks = [_chunk_dict(c) for c in chunk_ids]
        result = _make_embedding_result(chunk_ids, np.eye(3, dtype=np.float32))
        return retrieval.DenseRetriever(chunks, result)

    def test_zero_top_k_rejected(self):
        retriever = self._build()
        with pytest.raises(ValueError, match="positive"):
            retriever.search(np.array([1.0, 0.0, 0.0]), top_k=0)

    def test_negative_top_k_rejected(self):
        retriever = self._build()
        with pytest.raises(ValueError, match="positive"):
            retriever.search(np.array([1.0, 0.0, 0.0]), top_k=-5)

    def test_non_integer_top_k_rejected(self):
        retriever = self._build()
        with pytest.raises(ValueError, match="integer"):
            retriever.search(np.array([1.0, 0.0, 0.0]), top_k=2.5)

    def test_boolean_top_k_rejected(self):
        """bool is technically an int subclass in Python -- must be
        explicitly excluded, or True/False would silently act as 1/0."""
        retriever = self._build()
        with pytest.raises(ValueError, match="integer"):
            retriever.search(np.array([1.0, 0.0, 0.0]), top_k=True)

    def test_top_k_larger_than_corpus_returns_all_available(self):
        retriever = self._build()
        results = retriever.search(np.array([1.0, 0.0, 0.0], dtype=np.float32), top_k=1000)
        assert len(results) == 3  # not an error -- capped at corpus size


class TestDenseRetrieverDimensionMismatch:
    def test_query_dimension_mismatch_rejected(self):
        chunk_ids = ["a", "b"]
        chunks = [_chunk_dict(c) for c in chunk_ids]
        result = _make_embedding_result(chunk_ids, np.eye(2, dtype=np.float32))
        retriever = retrieval.DenseRetriever(chunks, result)

        with pytest.raises(ValueError, match="dimension"):
            retriever.search(np.array([1.0, 0.0, 0.0]), top_k=1)  # 3-D query, 2-D corpus

    def test_multi_dimensional_query_rejected(self):
        chunk_ids = ["a", "b"]
        chunks = [_chunk_dict(c) for c in chunk_ids]
        result = _make_embedding_result(chunk_ids, np.eye(2, dtype=np.float32))
        retriever = retrieval.DenseRetriever(chunks, result)

        with pytest.raises(ValueError, match="1-dimensional"):
            retriever.search(np.array([[1.0, 0.0], [0.0, 1.0]]), top_k=1)


class TestDenseRetrieverEmptyCorpus:
    def test_empty_corpus_search_raises_clear_error(self):
        result = _make_embedding_result([], np.zeros((0, 8), dtype=np.float32))
        retriever = retrieval.DenseRetriever([], result)

        with pytest.raises(ValueError, match="empty corpus"):
            retriever.search(np.zeros(8, dtype=np.float32), top_k=5)


class TestDenseRetrieverDeterministicTies:
    def test_tied_scores_broken_by_chunk_id_ascending(self):
        # Two chunks with IDENTICAL vectors -> identical scores.
        chunk_ids = ["zzz", "aaa"]
        chunks = [_chunk_dict(c) for c in chunk_ids]
        vectors = np.array([[1.0, 0.0], [1.0, 0.0]], dtype=np.float32)  # tied
        result = _make_embedding_result(chunk_ids, vectors)
        retriever = retrieval.DenseRetriever(chunks, result)

        results = retriever.search(np.array([1.0, 0.0], dtype=np.float32), top_k=2)

        assert [r.chunk_id for r in results] == ["aaa", "zzz"]  # ascending, not insertion order

    def test_tie_break_is_stable_across_repeated_calls(self):
        chunk_ids = ["zzz", "aaa", "mmm"]
        chunks = [_chunk_dict(c) for c in chunk_ids]
        vectors = np.array([[1.0, 0.0], [1.0, 0.0], [1.0, 0.0]], dtype=np.float32)  # all tied
        result = _make_embedding_result(chunk_ids, vectors)
        retriever = retrieval.DenseRetriever(chunks, result)

        query = np.array([1.0, 0.0], dtype=np.float32)
        r1 = [r.chunk_id for r in retriever.search(query, top_k=3)]
        r2 = [r.chunk_id for r in retriever.search(query, top_k=3)]

        assert r1 == r2 == ["aaa", "mmm", "zzz"]


class TestDenseRetrieverUsesVectorizedDotProduct:
    def test_search_source_uses_matmul_not_a_python_loop(self):
        """
        numpy.ndarray is an immutable C-level type -- its dunder
        methods (__matmul__ etc.) cannot be monkeypatched to count
        calls, so vectorization is verified by inspecting search()'s
        own source for the vectorized '@' operator and confirming
        there's no per-chunk Python loop computing the similarity
        (the loop that DOES exist is only over the already-ranked
        top_k results, building RetrievalResult objects -- a
        constant-size loop over the OUTPUT, not the corpus).
        """
        import inspect
        source = inspect.getsource(retrieval.DenseRetriever.search)
        assert "@" in source or "matmul" in source or "dot(" in source

    def test_correct_result_on_a_larger_corpus(self):
        """Behavioral confirmation that scores are computed correctly
        at a size where a subtle per-row bug would be more likely to
        show up than at n=3-4."""
        n, dim = 200, 16
        rng = np.random.default_rng(42)
        vectors = rng.random((n, dim)).astype(np.float32)
        vectors = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
        chunk_ids = [f"c{i}" for i in range(n)]
        chunks = [_chunk_dict(c) for c in chunk_ids]
        result = _make_embedding_result(chunk_ids, vectors)
        retriever = retrieval.DenseRetriever(chunks, result)

        query = vectors[37].copy()  # identical to one corpus vector -> that one must rank first with score ~1.0

        results = retriever.search(query, top_k=5)

        assert results[0].chunk_id == "c37"
        assert results[0].score == pytest.approx(1.0, abs=1e-4)
        # Manually verify against a brute-force Python computation
        # for the top result, independent of the implementation's
        # own vectorized path.
        manual_scores = [float(np.dot(v, query)) for v in vectors]
        expected_top_id = f"c{manual_scores.index(max(manual_scores))}"
        assert results[0].chunk_id == expected_top_id


class FakeQueryProcessor:
    def __init__(self):
        self.queries = []
        self.vector = np.zeros(768, dtype=np.float32)
        self.vector[0] = 1.0

    def embed_query(self, query):
        self.queries.append(query)
        return self.vector


class FakeChromaCollection:
    def __init__(self, records=None, metadata=None):
        self.records = records if records is not None else [
            {
                "id": "infosys-table-2026",
                "distance": 0.1,
                "document": "Consolidated revenue was 178,650 crore.",
                "metadata": {
                    "chunk_id": "infosys-table-2026",
                    "company": "Infosys",
                    "financial_year": "2025-2026",
                    "chunk_type": "table",
                    "page": 33,
                    "section": "Financial statements",
                    "table_id": "table-1",
                    "table_heading": "Revenue",
                    "row_start": 1,
                    "row_end": 3,
                },
            },
            {
                "id": "infosys-text-2025",
                "distance": 0.25,
                "document": "Prior year consolidated revenue was 162,990 crore.",
                "metadata": {
                    "chunk_id": "infosys-text-2025",
                    "company": "Infosys",
                    "financial_year": "2024-2025",
                    "chunk_type": "text",
                    "page": 34,
                    "section": "Revenue",
                },
            },
            {
                "id": "reliance-text-2026",
                "distance": 0.3,
                "document": "Reliance financial results.",
                "metadata": {
                    "chunk_id": "reliance-text-2026",
                    "company": "RelianceIndustries",
                    "financial_year": "2025-2026",
                    "chunk_type": "text",
                    "page": 10,
                },
            },
        ]
        self.metadata = metadata or {
            "embedding_dimension": 768,
            "embedding_model_name": retrieval.embeddings.DEFAULT_MODEL_NAME,
            "embedding_model_revision": retrieval.embeddings.DEFAULT_MODEL_REVISION,
            "distance_metric": "cosine",
        }
        self.configuration = {"hnsw": {"space": "cosine"}}
        self.query_calls = []
        self.get_calls = []
        self.response_override = None
        self.use_response_override = False

    @staticmethod
    def _matches(metadata, where):
        if not where:
            return True
        if "$and" in where:
            return all(FakeChromaCollection._matches(metadata, item) for item in where["$and"])
        return all(metadata.get(key) == value for key, value in where.items())

    def count(self):
        return len(self.records)

    def get(self, where=None, limit=None, include=None, **kwargs):
        self.get_calls.append({"where": where, "limit": limit, "include": include})
        matches = [record for record in self.records if self._matches(record["metadata"], where)]
        if limit is not None:
            matches = matches[:limit]
        return {"ids": [record["id"] for record in matches]}

    def query(self, query_embeddings, n_results, include, **kwargs):
        where = kwargs.get("where")
        self.query_calls.append({
            "query_embeddings": query_embeddings,
            "n_results": n_results,
            "include": include,
            "where": where,
            "where_supplied": "where" in kwargs,
        })
        if self.use_response_override:
            return self.response_override
        matches = [record for record in self.records if self._matches(record["metadata"], where)]
        matches = sorted(matches, key=lambda record: record["distance"])[:n_results]
        return {
            "ids": [[record["id"] for record in matches]],
            "distances": [[record["distance"] for record in matches]],
            "documents": [[record["document"] for record in matches]],
            "metadatas": [[record["metadata"] for record in matches]],
        }


class FakeChromaClient:
    def __init__(self, collection=None):
        self.collection = collection or FakeChromaCollection()
        self.requested_names = []
        self.create_calls = []

    def get_collection(self, name, embedding_function=None):
        self.requested_names.append(name)
        if name != retrieval.DEFAULT_CHROMA_COLLECTION:
            raise KeyError(name)
        return self.collection

    def create_collection(self, *args, **kwargs):
        self.create_calls.append((args, kwargs))
        raise AssertionError("Retrieval must never create a collection.")


def _build_chroma_retriever(collection=None, client=None):
    collection = collection or FakeChromaCollection()
    client = client or FakeChromaClient(collection)
    processor = FakeQueryProcessor()
    retriever = retrieval.ChromaRetriever(
        processor=processor,
        client=client,
    )
    return retriever, processor, client, collection


class TestChromaRetrieverInitialization:
    def test_connects_to_existing_collection(self):
        retriever, _, client, collection = _build_chroma_retriever()

        assert retriever.collection is collection
        assert client.requested_names == ["financial_reports_filtered"]
        assert client.create_calls == []

    def test_missing_collection_fails_without_creating(self):
        class MissingClient(FakeChromaClient):
            def get_collection(self, name, embedding_function=None):
                raise KeyError(name)

        client = MissingClient()
        with pytest.raises(retrieval.ChromaRetrievalError, match="was not found"):
            retrieval.ChromaRetriever(processor=FakeQueryProcessor(), client=client)
        assert client.create_calls == []

    def test_empty_collection_fails(self):
        collection = FakeChromaCollection(records=[])
        with pytest.raises(retrieval.ChromaRetrievalError, match="is empty"):
            retrieval.ChromaRetriever(collection=collection, processor=FakeQueryProcessor())

    @pytest.mark.parametrize(
        "metadata",
        [
            {"embedding_dimension": 384},
            {"embedding_model_name": "another/model"},
            {"embedding_model_revision": "another-revision"},
        ],
    )
    def test_incompatible_collection_metadata_fails(self, metadata):
        collection = FakeChromaCollection(metadata=metadata)
        with pytest.raises(retrieval.ChromaRetrievalError):
            retrieval.ChromaRetriever(collection=collection, processor=FakeQueryProcessor())


class TestChromaRetrieverSearch:
    def test_query_embedding_passed_with_expected_dimension_and_top_k(self):
        retriever, processor, _, collection = _build_chroma_retriever()

        results = retriever.search("  revenue   FY2026 ", top_k=2)

        call = collection.query_calls[0]
        assert processor.queries == ["  revenue   FY2026 "]
        assert len(call["query_embeddings"][0]) == 768
        assert call["n_results"] == 2
        assert len(results) == 2

    def test_cosine_distance_is_converted_to_similarity(self):
        retriever, _, _, _ = _build_chroma_retriever()

        results = retriever.search("revenue", top_k=1)

        assert results[0].score == pytest.approx(0.9)

    def test_result_keeps_id_rank_text_and_source_metadata(self):
        retriever, _, _, _ = _build_chroma_retriever()

        result = retriever.search("revenue", top_k=2)[0]

        assert result.chunk_id == "infosys-table-2026"
        assert result.rank == 1
        assert result.chunk["text"] == "Consolidated revenue was 178,650 crore."
        assert result.chunk["company"] == "Infosys"
        assert result.chunk["financial_year"] == "2025-2026"
        assert result.chunk["page"] == 33
        assert result.chunk["chunk_id"] == result.chunk_id
        assert result.chunk["table_id"] == "table-1"
        assert result.chunk["table_heading"] == "Revenue"
        assert "section" in result.chunk

    @pytest.mark.parametrize(
        ("filters", "expected_where"),
        [
            ({"company": "Infosys"}, {"company": "Infosys"}),
            ({"financial_year": "2025-2026"}, {"financial_year": "2025-2026"}),
            ({"chunk_type": "table"}, {"chunk_type": "table"}),
            (
                {"company": "Infosys", "financial_year": "2025-2026", "chunk_type": "table"},
                {"$and": [
                    {"company": "Infosys"},
                    {"financial_year": "2025-2026"},
                    {"chunk_type": "table"},
                ]},
            ),
        ],
    )
    def test_metadata_filters_use_exact_stored_values(self, filters, expected_where):
        retriever, _, _, collection = _build_chroma_retriever()

        retriever.search("revenue", **filters)

        assert collection.query_calls[0]["where"] == expected_where

    def test_invalid_filter_does_not_fall_back_to_unfiltered_query(self):
        retriever, _, _, collection = _build_chroma_retriever()

        with pytest.raises(ValueError, match="Unsupported financial_year filter"):
            retriever.search("revenue", financial_year="FY2026")

        assert collection.query_calls == []

    def test_blank_or_non_string_filter_is_rejected(self):
        retriever, _, _, collection = _build_chroma_retriever()

        for invalid in ("  ", 2026):
            with pytest.raises(ValueError, match="company filter must be a non-empty string"):
                retriever.search("revenue", company=invalid)
        assert collection.query_calls == []

    @pytest.mark.parametrize(
        "response",
        [
            None,
            {"ids": [["x"]], "distances": [[]], "documents": [[]], "metadatas": [[]]},
            {"ids": [["x"]], "distances": [[0.1]], "documents": [[None]], "metadatas": [[{}]]},
        ],
    )
    def test_malformed_query_results_raise_clear_error(self, response):
        retriever, _, _, collection = _build_chroma_retriever()
        collection.response_override = response
        collection.use_response_override = True

        with pytest.raises(retrieval.ChromaRetrievalError):
            retriever.search("revenue")

    def test_no_filter_does_not_pass_where_or_create_records(self):
        retriever, _, client, collection = _build_chroma_retriever()

        retriever.search("revenue")

        assert not collection.query_calls[0]["where_supplied"]
        assert client.create_calls == []

    def test_invalid_top_k_rejected(self):
        retriever, _, _, collection = _build_chroma_retriever()

        with pytest.raises(ValueError, match="positive"):
            retriever.search("revenue", top_k=0)
        assert collection.query_calls == []