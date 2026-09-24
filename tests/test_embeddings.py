"""
tests/test_embeddings.py

Tests for src/embeddings.py. The real model (sentence-transformers,
network download) is never loaded -- a fake model is injected
everywhere, since embed_texts/embed_corpus/embed_query/run_embedding
all accept a `model` object rather than loading one internally
except as an explicit fallback.
"""

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import embeddings


class FakeModel:
    """
    Deterministic fake standing in for a real SentenceTransformer.

    Uses hashlib.sha256 (not Python's built-in hash()) to derive the
    per-text random seed -- hash() on strings is randomized per
    process by default (PYTHONHASHSEED), so two separate test runs
    could otherwise get different "deterministic" vectors for the
    same text. sha256 is stable across processes, machines, and
    Python versions.
    """
    def __init__(self, dimension=8):
        self.dimension = dimension
        self.last_batch_size = None

    def encode(self, texts, batch_size=32, show_progress_bar=False,
               convert_to_numpy=True, normalize_embeddings=True):
        self.last_batch_size = batch_size
        vectors = []
        for text in texts:
            digest = hashlib.sha256(text.encode("utf-8")).digest()
            seed = int.from_bytes(digest[:4], "big")
            rng = np.random.default_rng(seed)
            vec = rng.random(self.dimension).astype(np.float32)
            if normalize_embeddings:
                norm = np.linalg.norm(vec)
                if norm > 0:
                    vec = vec / norm
            vectors.append(vec)
        return np.array(vectors, dtype=np.float32)


def _write_jsonl(path: Path, chunks: list[dict]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(c) + "\n")


def _chunk(chunk_id, text="Some content", **overrides):
    chunk = {"chunk_id": chunk_id, "text": text, "chunk_type": "text",
              "company": "X", "document": "D", "financial_year": "2025", "page": 1, "tokens": 10}
    chunk.update(overrides)
    return chunk


# ================================================================
# FakeModel determinism (meta-test: the fix itself)
# ================================================================

class TestFakeModelDeterminism:
    def test_same_text_same_vector_across_separate_instances(self):
        m1, m2 = FakeModel(dimension=8), FakeModel(dimension=8)
        v1 = m1.encode(["hello world"])[0]
        v2 = m2.encode(["hello world"])[0]
        assert np.array_equal(v1, v2)

    def test_does_not_use_builtin_hash(self):
        """Regression guard: FakeModel must not depend on Python's
        randomized-per-process hash() for strings."""
        import inspect
        source = inspect.getsource(FakeModel.encode)
        assert "hashlib" in source
        assert "hash(text)" not in source


# ================================================================
# Input validation
# ================================================================

class TestInputValidationMissingChunkId:
    def test_missing_chunk_id_raises(self):
        chunk = _chunk("c1")
        del chunk["chunk_id"]
        with pytest.raises(embeddings.EmbeddingValidationError) as exc_info:
            embeddings.validate_chunks_for_embedding([chunk])
        assert "chunk_id" in str(exc_info.value)

    def test_valid_chunks_not_modified_when_error_raised(self):
        good = _chunk("good1")
        bad = _chunk("bad1")
        del bad["chunk_id"]
        chunks = [good, bad]
        try:
            embeddings.validate_chunks_for_embedding(chunks)
        except embeddings.EmbeddingValidationError:
            pass
        assert chunks[0] == good  # untouched, not filtered/reordered


class TestInputValidationEmptyChunkId:
    def test_empty_string_chunk_id_raises(self):
        chunk = _chunk("")
        with pytest.raises(embeddings.EmbeddingValidationError):
            embeddings.validate_chunks_for_embedding([chunk])

    def test_whitespace_only_chunk_id_raises(self):
        chunk = _chunk("   ")
        with pytest.raises(embeddings.EmbeddingValidationError):
            embeddings.validate_chunks_for_embedding([chunk])


class TestInputValidationDuplicateChunkId:
    def test_duplicate_chunk_id_raises(self):
        chunks = [_chunk("dup"), _chunk("dup")]
        with pytest.raises(embeddings.EmbeddingValidationError) as exc_info:
            embeddings.validate_chunks_for_embedding(chunks)
        assert "Duplicate" in str(exc_info.value)
        assert "dup" in str(exc_info.value)


class TestInputValidationMissingText:
    def test_missing_text_raises(self):
        chunk = _chunk("c1")
        del chunk["text"]
        with pytest.raises(embeddings.EmbeddingValidationError) as exc_info:
            embeddings.validate_chunks_for_embedding([chunk])
        assert "text" in str(exc_info.value)


class TestEmptyTextHandling:
    def test_empty_text_allowed_by_default_but_returns_warning(self):
        chunk = _chunk("c1", text="")
        warnings = embeddings.validate_chunks_for_embedding([chunk])
        assert len(warnings) == 1
        assert "c1" in warnings[0]

    def test_whitespace_only_text_also_warned(self):
        chunk = _chunk("c1", text="   ")
        warnings = embeddings.validate_chunks_for_embedding([chunk])
        assert len(warnings) == 1

    def test_empty_text_rejected_when_flag_set(self):
        chunk = _chunk("c1", text="")
        with pytest.raises(embeddings.EmbeddingValidationError):
            embeddings.validate_chunks_for_embedding([chunk], reject_empty_text=True)

    def test_non_empty_text_produces_no_warning(self):
        chunk = _chunk("c1", text="real content")
        warnings = embeddings.validate_chunks_for_embedding([chunk])
        assert warnings == []


class TestMultipleErrorsReportedTogether:
    def test_all_problems_listed_not_just_first(self):
        bad1 = _chunk("c1")
        del bad1["text"]
        bad2 = _chunk("")
        chunks = [bad1, bad2]
        with pytest.raises(embeddings.EmbeddingValidationError) as exc_info:
            embeddings.validate_chunks_for_embedding(chunks)
        assert len(exc_info.value.problems) == 2


# ================================================================
# Output validation
# ================================================================

def _unit_vectors(n: int, dim: int) -> np.ndarray:
    """Properly L2-normalized dummy vectors -- NOT all-zero, since a
    zero vector has norm 0 and correctly fails the normalization
    check (normalize_embeddings=True is this project's default)."""
    rng = np.random.default_rng(0)
    vecs = rng.random((n, dim)).astype(np.float32)
    return vecs / np.linalg.norm(vecs, axis=1, keepdims=True)


class TestOutputValidation:
    def test_valid_result_passes(self):
        result = embeddings.EmbeddingResult(
            chunk_ids=["a", "b"], vectors=_unit_vectors(2, 8),
            model_name="m", model_revision="r", dimension=8,
        )
        embeddings.validate_embedding_output(result)  # must not raise

    def test_vector_count_mismatch_raises(self):
        result = embeddings.EmbeddingResult(
            chunk_ids=["a", "b", "c"], vectors=np.zeros((2, 8), dtype=np.float32),
            model_name="m", model_revision="r", dimension=8,
        )
        with pytest.raises(embeddings.EmbeddingValidationError) as exc_info:
            embeddings.validate_embedding_output(result)
        assert "count" in str(exc_info.value).lower()

    def test_dimension_mismatch_raises(self):
        result = embeddings.EmbeddingResult(
            chunk_ids=["a"], vectors=np.zeros((1, 8), dtype=np.float32),
            model_name="m", model_revision="r", dimension=16,
        )
        with pytest.raises(embeddings.EmbeddingValidationError) as exc_info:
            embeddings.validate_embedding_output(result)
        assert "dimension" in str(exc_info.value).lower()

    def test_duplicate_chunk_ids_in_result_raises(self):
        result = embeddings.EmbeddingResult(
            chunk_ids=["a", "a"], vectors=np.zeros((2, 8), dtype=np.float32),
            model_name="m", model_revision="r", dimension=8,
        )
        with pytest.raises(embeddings.EmbeddingValidationError):
            embeddings.validate_embedding_output(result)


# ================================================================
# Metadata completeness
# ================================================================

class TestMetadataCompleteness:
    REQUIRED_KEYS = {
        "input_chunks_file", "input_file_sha256", "model_name", "model_revision",
        "dimension", "count", "batch_size", "normalize_embeddings", "query_prefix",
        "embedding_config", "format_version", "library_versions", "created_at_utc",
    }

    def test_all_required_fields_present(self, tmp_path):
        chunks_file = tmp_path / "chunks.jsonl"
        _write_jsonl(chunks_file, [_chunk("c1")])

        embeddings.run_embedding(
            chunks_file=chunks_file, output_root=tmp_path / "out",
            model=FakeModel(dimension=8), batch_size=16,
        )

        metadata = json.loads((tmp_path / "out" / "embeddings_metadata.json").read_text())
        assert self.REQUIRED_KEYS.issubset(metadata.keys())
        assert metadata["batch_size"] == 16
        assert metadata["query_prefix"] == embeddings.QUERY_PREFIX
        assert metadata["format_version"] == embeddings.FORMAT_VERSION

    def test_source_hash_recorded_and_correct(self, tmp_path):
        chunks_file = tmp_path / "chunks.jsonl"
        _write_jsonl(chunks_file, [_chunk("c1")])
        expected_hash = embeddings._sha256_file(chunks_file)

        embeddings.run_embedding(chunks_file=chunks_file, output_root=tmp_path / "out", model=FakeModel())

        metadata = json.loads((tmp_path / "out" / "embeddings_metadata.json").read_text())
        assert metadata["input_file_sha256"] == expected_hash


# ================================================================
# Corpus-level caching
# ================================================================

class TestCaching:
    def test_reuse_when_unchanged(self, tmp_path):
        chunks_file = tmp_path / "chunks.jsonl"
        _write_jsonl(chunks_file, [_chunk("c1"), _chunk("c2")])
        output_root = tmp_path / "out"

        s1 = embeddings.run_embedding(chunks_file=chunks_file, output_root=output_root, model=FakeModel())
        assert s1["status"] == "rebuilt"

        s2 = embeddings.run_embedding(chunks_file=chunks_file, output_root=output_root, model=FakeModel())
        assert s2["status"] == "reused"

    def test_cache_invalidated_by_input_change(self, tmp_path):
        chunks_file = tmp_path / "chunks.jsonl"
        _write_jsonl(chunks_file, [_chunk("c1")])
        output_root = tmp_path / "out"

        embeddings.run_embedding(chunks_file=chunks_file, output_root=output_root, model=FakeModel())

        _write_jsonl(chunks_file, [_chunk("c1"), _chunk("c2")])  # content changed

        s2 = embeddings.run_embedding(chunks_file=chunks_file, output_root=output_root, model=FakeModel())
        assert s2["status"] == "rebuilt"

    def test_cache_invalidated_by_model_revision_change(self, tmp_path):
        chunks_file = tmp_path / "chunks.jsonl"
        _write_jsonl(chunks_file, [_chunk("c1")])
        output_root = tmp_path / "out"

        embeddings.run_embedding(chunks_file=chunks_file, output_root=output_root,
                                   model=FakeModel(), model_revision="rev1")

        s2 = embeddings.run_embedding(chunks_file=chunks_file, output_root=output_root,
                                        model=FakeModel(), model_revision="rev2")
        assert s2["status"] == "rebuilt"

    def test_force_bypasses_valid_cache(self, tmp_path):
        chunks_file = tmp_path / "chunks.jsonl"
        _write_jsonl(chunks_file, [_chunk("c1")])
        output_root = tmp_path / "out"

        embeddings.run_embedding(chunks_file=chunks_file, output_root=output_root, model=FakeModel())

        s2 = embeddings.run_embedding(chunks_file=chunks_file, output_root=output_root, model=FakeModel(), force=True)
        assert s2["status"] == "rebuilt"

    def test_real_model_not_loaded_on_cache_hit(self, tmp_path, monkeypatch):
        chunks_file = tmp_path / "chunks.jsonl"
        _write_jsonl(chunks_file, [_chunk("c1")])
        output_root = tmp_path / "out"

        embeddings.run_embedding(chunks_file=chunks_file, output_root=output_root, model=FakeModel())

        def _fail_if_called(*args, **kwargs):
            raise AssertionError("load_model() should not be called on a cache hit")
        monkeypatch.setattr(embeddings, "load_model", _fail_if_called)

        # No model injected this time -- if the cache check didn't
        # short-circuit correctly, this would try to load a real model.
        summary = embeddings.run_embedding(chunks_file=chunks_file, output_root=output_root)
        assert summary["status"] == "reused"


# ================================================================
# Failure-safe behavior
# ================================================================

class TestFailureSafety:
    def test_output_validation_failure_preserves_previous_valid_files(self, tmp_path):
        chunks_file = tmp_path / "chunks.jsonl"
        _write_jsonl(chunks_file, [_chunk("c1")])
        output_root = tmp_path / "out"

        embeddings.run_embedding(chunks_file=chunks_file, output_root=output_root, model=FakeModel())
        before = (output_root / "embeddings.npy").read_bytes()

        # Force invalidation, then make the rebuild attempt fail
        # inside embed_corpus by having the fake model return the
        # wrong number of vectors.
        class BrokenModel(FakeModel):
            def encode(self, texts, **kwargs):
                result = super().encode(texts, **kwargs)
                return result[:-1] if len(result) > 0 else result  # drop one row

        _write_jsonl(chunks_file, [_chunk("c1"), _chunk("c2")])  # invalidate cache

        with pytest.raises(embeddings.EmbeddingValidationError):
            embeddings.run_embedding(chunks_file=chunks_file, output_root=output_root, model=BrokenModel())

        after = (output_root / "embeddings.npy").read_bytes()
        assert before == after  # untouched

    def test_no_tmp_files_left_behind_after_failure(self, tmp_path):
        chunks_file = tmp_path / "chunks.jsonl"
        _write_jsonl(chunks_file, [_chunk("c1"), _chunk("c2")])
        output_root = tmp_path / "out"

        class BrokenModel(FakeModel):
            def encode(self, texts, **kwargs):
                result = super().encode(texts, **kwargs)
                return result[:-1]

        with pytest.raises(embeddings.EmbeddingValidationError):
            embeddings.run_embedding(chunks_file=chunks_file, output_root=output_root, model=BrokenModel())

        assert not (output_root / "embeddings.npy.tmp").exists()
        assert not (output_root / "chunk_ids.json.tmp").exists()
        assert not (output_root / "embeddings_metadata.json.tmp").exists()

    def test_first_run_failure_leaves_no_final_files(self, tmp_path):
        chunks_file = tmp_path / "chunks.jsonl"
        _write_jsonl(chunks_file, [_chunk("c1"), _chunk("c2")])
        output_root = tmp_path / "out"

        class BrokenModel(FakeModel):
            def encode(self, texts, **kwargs):
                return super().encode(texts, **kwargs)[:-1]

        with pytest.raises(embeddings.EmbeddingValidationError):
            embeddings.run_embedding(chunks_file=chunks_file, output_root=output_root, model=BrokenModel())

        assert not (output_root / "embeddings.npy").exists()


# ================================================================
# Query prefix / batch size plumbing
# ================================================================

class TestQueryPrefix:
    def test_prefix_appears_exactly_once(self):
        captured = []

        class RecordingModel(FakeModel):
            def encode(self, texts, **kwargs):
                captured.extend(texts)
                return super().encode(texts, **kwargs)

        embeddings.embed_query("revenue growth", RecordingModel())

        assert len(captured) == 1
        assert captured[0].count(embeddings.QUERY_PREFIX) == 1
        assert captured[0] == embeddings.QUERY_PREFIX + "revenue growth"

    def test_document_text_never_prefixed(self):
        captured = []

        class RecordingModel(FakeModel):
            def encode(self, texts, **kwargs):
                captured.extend(texts)
                return super().encode(texts, **kwargs)

        embeddings.embed_corpus([_chunk("c1", text="plain document text")], RecordingModel())

        assert embeddings.QUERY_PREFIX not in captured[0]


class TestBatchSizePlumbing:
    def test_batch_size_passed_to_model(self):
        model = FakeModel()
        embeddings.embed_texts(model, ["a", "b", "c"], batch_size=7)
        assert model.last_batch_size == 7

    def test_default_batch_size_used_when_unspecified(self):
        model = FakeModel()
        embeddings.embed_texts(model, ["a"])
        assert model.last_batch_size == embeddings.DEFAULT_BATCH_SIZE

    def test_batch_size_flows_through_run_embedding(self, tmp_path):
        chunks_file = tmp_path / "chunks.jsonl"
        _write_jsonl(chunks_file, [_chunk("c1")])
        model = FakeModel()

        embeddings.run_embedding(chunks_file=chunks_file, output_root=tmp_path / "out", model=model, batch_size=4)

        assert model.last_batch_size == 4


# ================================================================
# Persisted-output validation / round-trip
# ================================================================

class TestPersistedOutputValidation:
    def test_roundtrip_preserves_everything(self, tmp_path):
        chunks = [_chunk("c1"), _chunk("c2")]
        model = FakeModel(dimension=6)
        result = embeddings.embed_corpus(chunks, model, model_name="m", model_revision="r1")

        output_root = tmp_path / "embeddings"
        embeddings._atomic_save_embeddings(result, output_root, {
            "input_chunks_file": "x", "input_file_sha256": "y", "model_name": "m",
            "model_revision": "r1", "dimension": 6, "count": 2, "batch_size": 32,
            "normalize_embeddings": True, "query_prefix": embeddings.QUERY_PREFIX,
            "embedding_config": {}, "format_version": "1.0", "library_versions": {},
            "created_at_utc": "now",
        })
        loaded = embeddings.load_embeddings(output_root)

        assert loaded.chunk_ids == result.chunk_ids
        assert np.allclose(loaded.vectors, result.vectors)
        assert loaded.dimension == 6

    def test_chunk_id_order_matches_vector_row_order(self):
        chunks = [_chunk("first", text="AAAA"), _chunk("second", text="BBBB")]
        model = FakeModel(dimension=4)

        result = embeddings.embed_corpus(chunks, model)

        assert result.chunk_ids == ["first", "second"]
        direct = model.encode(["AAAA", "BBBB"])
        assert np.allclose(result.vectors, direct)


# ================================================================
# Orchestration / empty corpus
# ================================================================

class TestRunEmbeddingOrchestration:
    def test_full_run_with_injected_model(self, tmp_path):
        chunks_file = tmp_path / "all_chunks.jsonl"
        _write_jsonl(chunks_file, [_chunk("c1"), _chunk("c2"), _chunk("c3")])

        summary = embeddings.run_embedding(
            chunks_file=chunks_file, output_root=tmp_path / "embeddings", model=FakeModel(dimension=8),
        )

        assert summary["chunk_count"] == 3
        assert summary["dimension"] == 8
        assert summary["status"] == "rebuilt"

    def test_empty_chunks_file_handled_cleanly(self, tmp_path):
        chunks_file = tmp_path / "all_chunks.jsonl"
        chunks_file.write_text("")
        output_root = tmp_path / "embeddings"

        summary = embeddings.run_embedding(chunks_file=chunks_file, output_root=output_root, model=FakeModel())

        assert summary["chunk_count"] == 0
        assert not output_root.exists()


class TestDeterminism:
    def test_same_input_same_model_produces_identical_vectors(self):
        chunks = [_chunk("c1", text="identical text")]
        model = FakeModel(dimension=8)

        r1 = embeddings.embed_corpus(chunks, model)
        r2 = embeddings.embed_corpus(chunks, model)

        assert np.array_equal(r1.vectors, r2.vectors)


# ================================================================
# Priority 1/3: corrupted and inconsistent cache/persisted-file tests
# ================================================================

def _write_valid_embeddings(output_root: Path, n=2, dim=8) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    vectors = _unit_vectors(n, dim)
    with open(output_root / "embeddings.npy", "wb") as f:
        np.save(f, vectors)
    chunk_ids = [f"c{i}" for i in range(n)]
    (output_root / "chunk_ids.json").write_text(json.dumps(chunk_ids))
    (output_root / "embeddings_metadata.json").write_text(json.dumps({
        "input_chunks_file": "x", "input_file_sha256": "hash1", "model_name": "m",
        "model_revision": "r1", "dimension": dim, "count": n, "batch_size": 32,
        "normalize_embeddings": True, "query_prefix": embeddings.QUERY_PREFIX,
        "embedding_config": {"normalize_embeddings": True, "query_prefix": embeddings.QUERY_PREFIX},
        "format_version": embeddings.FORMAT_VERSION, "library_versions": {},
        "created_at_utc": "now",
    }))


class TestCorruptedCacheDetection:
    def test_corrupted_npy_file_rejects_cache(self, tmp_path):
        output_root = tmp_path / "out"
        _write_valid_embeddings(output_root)
        (output_root / "embeddings.npy").write_bytes(b"not a valid npy file at all")

        with pytest.raises(embeddings.EmbeddingValidationError):
            embeddings.load_embeddings(output_root)

    def test_invalid_json_in_chunk_ids_rejects_cache(self, tmp_path):
        output_root = tmp_path / "out"
        _write_valid_embeddings(output_root)
        (output_root / "chunk_ids.json").write_text("{not valid json")

        with pytest.raises(embeddings.EmbeddingValidationError):
            embeddings.load_embeddings(output_root)

    def test_invalid_json_in_metadata_rejects_cache(self, tmp_path):
        output_root = tmp_path / "out"
        _write_valid_embeddings(output_root)
        (output_root / "embeddings_metadata.json").write_text("{not valid json")

        with pytest.raises(embeddings.EmbeddingValidationError):
            embeddings.load_embeddings(output_root)

    def test_metadata_count_mismatch_rejected(self, tmp_path):
        output_root = tmp_path / "out"
        _write_valid_embeddings(output_root, n=2)
        meta = json.loads((output_root / "embeddings_metadata.json").read_text())
        meta["count"] = 999
        (output_root / "embeddings_metadata.json").write_text(json.dumps(meta))

        with pytest.raises(embeddings.EmbeddingValidationError) as exc_info:
            embeddings.load_embeddings(output_root)
        assert "count" in str(exc_info.value).lower()

    def test_metadata_dimension_mismatch_rejected(self, tmp_path):
        output_root = tmp_path / "out"
        _write_valid_embeddings(output_root, dim=8)
        meta = json.loads((output_root / "embeddings_metadata.json").read_text())
        meta["dimension"] = 999
        (output_root / "embeddings_metadata.json").write_text(json.dumps(meta))

        with pytest.raises(embeddings.EmbeddingValidationError) as exc_info:
            embeddings.load_embeddings(output_root)
        assert "dimension" in str(exc_info.value).lower()

    def test_vector_id_count_mismatch_rejected(self, tmp_path):
        output_root = tmp_path / "out"
        _write_valid_embeddings(output_root, n=3)
        (output_root / "chunk_ids.json").write_text(json.dumps(["c0", "c1"]))  # now only 2, vectors still 3

        with pytest.raises(embeddings.EmbeddingValidationError):
            embeddings.load_embeddings(output_root)

    def test_duplicate_chunk_ids_in_cache_rejected(self, tmp_path):
        output_root = tmp_path / "out"
        _write_valid_embeddings(output_root, n=2)
        (output_root / "chunk_ids.json").write_text(json.dumps(["dup", "dup"]))

        with pytest.raises(embeddings.EmbeddingValidationError):
            embeddings.load_embeddings(output_root)

    def test_wrong_model_name_invalidates_cache(self, tmp_path):
        output_root = tmp_path / "out"
        _write_valid_embeddings(output_root)
        meta = json.loads((output_root / "embeddings_metadata.json").read_text())

        is_valid, reason = embeddings._is_cache_valid(
            meta, output_root, meta["input_file_sha256"], "different-model",
            meta["model_revision"], meta["embedding_config"],
        )
        assert is_valid is False
        assert "model_name" in reason

    def test_wrong_config_fingerprint_invalidates_cache(self, tmp_path):
        output_root = tmp_path / "out"
        _write_valid_embeddings(output_root)
        meta = json.loads((output_root / "embeddings_metadata.json").read_text())

        is_valid, reason = embeddings._is_cache_valid(
            meta, output_root, meta["input_file_sha256"], meta["model_name"],
            meta["model_revision"], {"normalize_embeddings": False, "query_prefix": "different"},
        )
        assert is_valid is False
        assert "embedding_config" in reason

    def test_non_2d_embedding_array_rejected(self, tmp_path):
        output_root = tmp_path / "out"
        output_root.mkdir(parents=True)
        with open(output_root / "embeddings.npy", "wb") as f:
            np.save(f, np.zeros(8, dtype=np.float32))  # 1-D, not 2-D
        (output_root / "chunk_ids.json").write_text(json.dumps(["c0"]))
        (output_root / "embeddings_metadata.json").write_text(json.dumps({
            "model_name": "m", "model_revision": "r", "dimension": 8, "count": 1,
        }))

        with pytest.raises(embeddings.EmbeddingValidationError) as exc_info:
            embeddings.load_embeddings(output_root)
        assert "2-dimensional" in str(exc_info.value) or "2-D" in str(exc_info.value)

    def test_nan_values_in_cached_embeddings_rejected(self, tmp_path):
        output_root = tmp_path / "out"
        output_root.mkdir(parents=True)
        vectors = _unit_vectors(2, 8)
        vectors[0, 0] = np.nan
        with open(output_root / "embeddings.npy", "wb") as f:
            np.save(f, vectors)
        (output_root / "chunk_ids.json").write_text(json.dumps(["c0", "c1"]))
        (output_root / "embeddings_metadata.json").write_text(json.dumps({
            "model_name": "m", "model_revision": "r", "dimension": 8, "count": 2,
        }))

        with pytest.raises(embeddings.EmbeddingValidationError) as exc_info:
            embeddings.load_embeddings(output_root)
        assert "non-finite" in str(exc_info.value)

    def test_full_run_embedding_falls_back_to_rebuild_on_corrupted_cache(self, tmp_path):
        """End-to-end: a corrupted cache must not crash run_embedding
        -- it should be detected, logged, and transparently rebuilt."""
        chunks_file = tmp_path / "chunks.jsonl"
        _write_jsonl(chunks_file, [_chunk("c1"), _chunk("c2")])
        output_root = tmp_path / "out"

        embeddings.run_embedding(chunks_file=chunks_file, output_root=output_root, model=FakeModel(dimension=8))
        (output_root / "embeddings.npy").write_bytes(b"corrupted garbage")

        summary = embeddings.run_embedding(chunks_file=chunks_file, output_root=output_root, model=FakeModel(dimension=8))
        assert summary["status"] == "rebuilt"


# ================================================================
# Priority 2: additional output-shape validation tests
# ================================================================

class TestOutputShapeValidation:
    def test_one_dimensional_output_rejected_clearly(self):
        result = embeddings.EmbeddingResult(
            chunk_ids=["a"], vectors=np.zeros(8, dtype=np.float32),
            model_name="m", model_revision="r", dimension=8,
        )
        with pytest.raises(embeddings.EmbeddingValidationError) as exc_info:
            embeddings.validate_embedding_output(result)
        assert "2-dimensional" in str(exc_info.value)

    def test_three_dimensional_output_rejected_clearly(self):
        result = embeddings.EmbeddingResult(
            chunk_ids=["a"], vectors=np.zeros((1, 8, 2), dtype=np.float32),
            model_name="m", model_revision="r", dimension=8,
        )
        with pytest.raises(embeddings.EmbeddingValidationError) as exc_info:
            embeddings.validate_embedding_output(result)
        assert "2-dimensional" in str(exc_info.value)

    def test_zero_dimensional_output_rejected_clearly(self):
        result = embeddings.EmbeddingResult(
            chunk_ids=[], vectors=np.array(5.0, dtype=np.float32),  # 0-D scalar array
            model_name="m", model_revision="r", dimension=8,
        )
        with pytest.raises(embeddings.EmbeddingValidationError) as exc_info:
            embeddings.validate_embedding_output(result)
        assert "2-dimensional" in str(exc_info.value)

    def test_empty_2d_output_with_matching_empty_ids_is_valid(self):
        result = embeddings.EmbeddingResult(
            chunk_ids=[], vectors=np.zeros((0, 8), dtype=np.float32),
            model_name="m", model_revision="r", dimension=8,
        )
        embeddings.validate_embedding_output(result)  # must not raise

    def test_incorrect_row_count_rejected(self):
        result = embeddings.EmbeddingResult(
            chunk_ids=["a", "b", "c"], vectors=_unit_vectors(2, 8),
            model_name="m", model_revision="r", dimension=8,
        )
        with pytest.raises(embeddings.EmbeddingValidationError) as exc_info:
            embeddings.validate_embedding_output(result)
        assert "count" in str(exc_info.value).lower()

    def test_nan_values_rejected(self):
        vectors = _unit_vectors(2, 8)
        vectors[0, 0] = np.nan
        result = embeddings.EmbeddingResult(
            chunk_ids=["a", "b"], vectors=vectors, model_name="m", model_revision="r", dimension=8,
        )
        with pytest.raises(embeddings.EmbeddingValidationError) as exc_info:
            embeddings.validate_embedding_output(result)
        assert "non-finite" in str(exc_info.value)

    def test_infinite_values_rejected(self):
        vectors = _unit_vectors(2, 8)
        vectors[0, 0] = np.inf
        result = embeddings.EmbeddingResult(
            chunk_ids=["a", "b"], vectors=vectors, model_name="m", model_revision="r", dimension=8,
        )
        with pytest.raises(embeddings.EmbeddingValidationError) as exc_info:
            embeddings.validate_embedding_output(result)
        assert "non-finite" in str(exc_info.value)

    def test_valid_two_dimensional_output_passes(self):
        result = embeddings.EmbeddingResult(
            chunk_ids=["a", "b", "c"], vectors=_unit_vectors(3, 8),
            model_name="m", model_revision="r", dimension=8,
        )
        embeddings.validate_embedding_output(result)  # must not raise


# ================================================================
# Priority 5: query validation
# ================================================================

class TestQueryValidation:
    def test_empty_query_rejected(self):
        with pytest.raises(embeddings.EmbeddingValidationError):
            embeddings.embed_query("", FakeModel())

    def test_whitespace_only_query_rejected(self):
        with pytest.raises(embeddings.EmbeddingValidationError):
            embeddings.embed_query("   ", FakeModel())

    def test_non_string_query_rejected(self):
        with pytest.raises(embeddings.EmbeddingValidationError):
            embeddings.embed_query(None, FakeModel())

    def test_valid_query_still_works(self):
        vec = embeddings.embed_query("what was revenue", FakeModel(dimension=8))
        assert vec.shape == (8,)


class TestCliRejectEmptyTextFlag:
    def test_reject_empty_text_flag_propagates_and_raises(self, tmp_path):
        chunks_file = tmp_path / "chunks.jsonl"
        _write_jsonl(chunks_file, [_chunk("c1", text="")])

        with pytest.raises(embeddings.EmbeddingValidationError):
            embeddings.run_embedding(
                chunks_file=chunks_file, output_root=tmp_path / "out",
                model=FakeModel(), reject_empty_text=True,
            )

    def test_default_still_allows_empty_text(self, tmp_path):
        chunks_file = tmp_path / "chunks.jsonl"
        _write_jsonl(chunks_file, [_chunk("c1", text="")])

        summary = embeddings.run_embedding(
            chunks_file=chunks_file, output_root=tmp_path / "out", model=FakeModel(),
        )
        assert summary["status"] == "rebuilt"