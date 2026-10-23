import hashlib
import csv
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import chroma_store


def _chunks(count):
    return [
        {
            "chunk_id": f"chunk-{index}",
            "text": f"Financial report text {index}",
            "company": "Example Ltd",
            "document": "Annual Report",
            "financial_year": "2024-2025",
            "page": index + 1,
            "chunk_type": "text",
            "parser_version": "1.0",
            "section": None,
            "custom": {"origin": "fixture", "index": index},
        }
        for index in range(count)
    ]


def _write_corpus(tmp_path, chunks=None, vectors=None, ids=None):
    chunks = _chunks(3) if chunks is None else chunks
    ids = [chunk["chunk_id"] for chunk in chunks] if ids is None else ids
    vectors = np.zeros((len(ids), 768), dtype=np.float32) if vectors is None else vectors
    if vectors.ndim == 2 and vectors.shape[1] == 768:
        for index in range(len(vectors)):
            vectors[index, index] = 1.0

    chunks_file = tmp_path / "all_chunks.jsonl"
    chunks_file.write_text(
        "".join(json.dumps(chunk, ensure_ascii=False) + "\n" for chunk in chunks),
        encoding="utf-8",
    )
    embeddings_root = tmp_path / "embeddings"
    embeddings_root.mkdir()
    np.save(embeddings_root / "embeddings.npy", vectors)
    (embeddings_root / "chunk_ids.json").write_text(json.dumps(ids), encoding="utf-8")
    source_hash = hashlib.sha256(chunks_file.read_bytes()).hexdigest()
    metadata = {
        "model_name": "BAAI/bge-base-en-v1.5",
        "model_revision": "08e3d2c28e3886dfbcd98caebb447d9e23614bc0",
        "dimension": vectors.shape[1] if vectors.ndim == 2 else 768,
        "count": len(ids),
        "input_file_sha256": source_hash,
        "query_prefix": "Represent this sentence for searching relevant passages: ",
    }
    (embeddings_root / "embeddings_metadata.json").write_text(
        json.dumps(metadata), encoding="utf-8"
    )
    return chunks_file, embeddings_root


class FakeCollection:
    def __init__(self, name, metadata, configuration=None):
        self.name = name
        self.metadata = metadata
        self.configuration = configuration or {"hnsw": {"space": "cosine"}}
        self.records = {}
        self.batch_sizes = []

    def count(self):
        return len(self.records)

    def upsert(self, ids, embeddings, documents, metadatas):
        self.batch_sizes.append(len(ids))
        for index, chunk_id in enumerate(ids):
            self.records[chunk_id] = {
                "embedding": np.asarray(embeddings[index]).copy(),
                "document": documents[index],
                "metadata": metadatas[index],
            }

    def get(self, ids=None, limit=None, offset=0, include=None):
        selected = list(ids) if ids is not None else sorted(self.records)[offset:offset + limit]
        selected = [chunk_id for chunk_id in selected if chunk_id in self.records]
        result = {"ids": selected}
        if "documents" in (include or []):
            result["documents"] = [self.records[key]["document"] for key in selected]
        if "metadatas" in (include or []):
            result["metadatas"] = [self.records[key]["metadata"] for key in selected]
        if "embeddings" in (include or []):
            result["embeddings"] = np.asarray([self.records[key]["embedding"] for key in selected])
        return result


class FakeClient:
    def __init__(self):
        self.collections = {}

    def list_collections(self):
        return list(self.collections)

    def get_collection(self, name, embedding_function=None):
        return self.collections[name]

    def create_collection(self, name, configuration, metadata, embedding_function=None):
        collection = FakeCollection(name, metadata, configuration)
        self.collections[name] = collection
        return collection


def _ingest_args(tmp_path, client, chunks=None, **kwargs):
    chunks_file, embeddings_root = _write_corpus(tmp_path, chunks=chunks)
    options = {
        "chunks_file": chunks_file,
        "embeddings_root": embeddings_root,
        "persist_directory": tmp_path / "db",
        "client": client,
    }
    options.update(kwargs)
    return options


def test_rejects_id_alignment_mismatch_before_creating_collection(tmp_path):
    chunks = _chunks(3)
    chunks_file, embeddings_root = _write_corpus(tmp_path, chunks, ids=["other", "chunk-1", "chunk-2"])
    client = FakeClient()

    with pytest.raises(chroma_store.ChromaIngestionError, match="not aligned exactly"):
        chroma_store.ingest(chunks_file, embeddings_root, tmp_path / "db", client=client)

    assert client.collections == {}


def test_rejects_duplicate_source_ids(tmp_path):
    chunks = _chunks(2)
    chunks[1]["chunk_id"] = chunks[0]["chunk_id"]
    chunks_file, embeddings_root = _write_corpus(tmp_path, chunks)

    with pytest.raises(chroma_store.ChromaIngestionError, match="duplicate chunk IDs"):
        chroma_store.load_validated_corpus(chunks_file, embeddings_root)


@pytest.mark.parametrize(
    ("vectors", "error"),
    [
        (np.zeros((3, 767), dtype=np.float32), "dimension"),
        (np.full((3, 768), np.nan, dtype=np.float32), "non-finite"),
        (np.zeros((2, 768), dtype=np.float32), "row count"),
    ],
)
def test_rejects_invalid_vectors(tmp_path, vectors, error):
    chunks = _chunks(3)
    chunks_file, embeddings_root = _write_corpus(tmp_path, chunks, vectors=vectors)

    with pytest.raises(chroma_store.ChromaIngestionError, match=error):
        chroma_store.load_validated_corpus(chunks_file, embeddings_root)


def test_ingestion_batches_and_losslessly_preserves_metadata(tmp_path):
    client = FakeClient()
    result = chroma_store.ingest(**_ingest_args(tmp_path, client, batch_size=2))
    collection = client.collections[chroma_store.DEFAULT_COLLECTION_NAME]

    assert result["count"] == 3
    assert collection.batch_sizes == [2, 1]
    stored = collection.records["chunk-0"]["metadata"]
    assert "custom" not in stored
    assert json.loads(stored[chroma_store.SOURCE_METADATA_KEY])["custom"] == {
        "origin": "fixture",
        "index": 0,
    }


@pytest.mark.parametrize("text", [""])
def test_empty_text_is_skipped_and_audited(tmp_path, text):
    chunks = _chunks(3)
    chunks[1]["text"] = text
    client = FakeClient()

    result = chroma_store.ingest(**_ingest_args(tmp_path, client, chunks=chunks))

    collection = client.collections[chroma_store.DEFAULT_COLLECTION_NAME]
    assert result["count"] == 2
    assert collection.count() == 2
    audit_path = tmp_path / "embeddings" / "embedding_audit_log.csv"
    with audit_path.open(encoding="utf-8", newline="") as audit_file:
        audit_rows = list(csv.DictReader(audit_file))
    assert len(audit_rows) == 1
    assert audit_rows[0]["chunk_id"] == "chunk-1"
    assert audit_rows[0]["stage"] == "chroma_ingestion"
    assert audit_rows[0]["event"] == "chunk_skipped"
    assert audit_rows[0]["company"] == "Example Ltd"
    assert audit_rows[0]["document"] == "Annual Report"
    assert audit_rows[0]["financial_year"] == "2024-2025"
    assert audit_rows[0]["page"] == "2"
    assert audit_rows[0]["section"] == ""
    assert audit_rows[0]["chunk_type"] == "text"
    assert audit_rows[0]["reason"] == "empty_or_whitespace_text"
    assert audit_rows[0]["action"] == "excluded_from_vector_db"


def test_whitespace_only_text_is_skipped(tmp_path):
    chunks = _chunks(3)
    chunks[1]["text"] = " \t\n "
    client = FakeClient()

    result = chroma_store.ingest(**_ingest_args(tmp_path, client, chunks=chunks))

    collection = client.collections[chroma_store.DEFAULT_COLLECTION_NAME]
    assert result["count"] == 2
    assert set(collection.records) == {"chunk-0", "chunk-2"}


def test_non_string_text_is_skipped_and_audited(tmp_path):
    chunks = _chunks(3)
    chunks[1]["text"] = None
    client = FakeClient()

    result = chroma_store.ingest(**_ingest_args(tmp_path, client, chunks=chunks))

    assert result["count"] == 2
    audit_path = tmp_path / "embeddings" / "embedding_audit_log.csv"
    with audit_path.open(encoding="utf-8", newline="") as audit_file:
        audit_rows = list(csv.DictReader(audit_file))
    assert audit_rows[0]["chunk_id"] == "chunk-1"
    assert audit_rows[0]["reason"] == "non_string_text"


def test_filtered_chunks_ids_metadata_and_vectors_stay_aligned(tmp_path):
    chunks = _chunks(4)
    chunks[1]["text"] = ""
    chunks[3]["text"] = "   "
    client = FakeClient()

    result = chroma_store.ingest(**_ingest_args(tmp_path, client, chunks=chunks))

    collection = client.collections[chroma_store.DEFAULT_COLLECTION_NAME]
    assert result["count"] == 2
    assert result["sample_ids"] == ["chunk-0", "chunk-2"]
    assert collection.records["chunk-0"]["document"] == chunks[0]["text"]
    assert collection.records["chunk-2"]["document"] == chunks[2]["text"]
    assert collection.records["chunk-0"]["embedding"][0] == 1
    assert collection.records["chunk-2"]["embedding"][2] == 1
    assert json.loads(
        collection.records["chunk-2"]["metadata"][chroma_store.SOURCE_METADATA_KEY]
    )["custom"]["index"] == 2
    assert collection.metadata["source_chunk_count"] == 4
    assert collection.metadata["ingestion_count"] == 2


def test_audit_csv_header_is_written_once_and_rows_append(tmp_path):
    chunks = _chunks(2)
    chunks[0]["text"] = ""
    client = FakeClient()
    options = _ingest_args(tmp_path, client, chunks=chunks)

    chroma_store.ingest(**options)
    chroma_store.ingest(**options)

    audit_path = tmp_path / "embeddings" / "embedding_audit_log.csv"
    with audit_path.open(encoding="utf-8", newline="") as audit_file:
        rows = list(csv.reader(audit_file))
    assert rows.count(list(chroma_store.AUDIT_FIELDS)) == 1
    assert len(rows) == 3
    assert [row[3] for row in rows[1:]] == ["chunk-0", "chunk-0"]


def test_audit_write_failure_is_logged_and_raised(tmp_path, monkeypatch, caplog):
    def fail_writer(*args, **kwargs):
        raise RuntimeError("disk failure")

    monkeypatch.setattr(chroma_store.csv, "DictWriter", fail_writer)
    with caplog.at_level(logging.ERROR):
        with pytest.raises(chroma_store.ChromaIngestionError, match="disk failure"):
            chroma_store._audit_skipped_chunks([_chunks(1)[0]], tmp_path / "embeddings")

    assert "Failed to write skipped-chunk audit records" in caplog.text


def test_ingestion_reuses_existing_vectors_without_embedding(tmp_path, monkeypatch):
    chunks = _chunks(3)
    chunks[1]["text"] = ""
    client = FakeClient()

    def fail_if_called(*args, **kwargs):
        raise AssertionError("Ingestion must not regenerate embeddings.")

    monkeypatch.setattr(chroma_store.embeddings, "embed_corpus", fail_if_called)
    result = chroma_store.ingest(**_ingest_args(tmp_path, client, chunks=chunks))

    assert result["count"] == 2
    collection = client.collections[chroma_store.DEFAULT_COLLECTION_NAME]
    assert collection.records["chunk-2"]["embedding"][2] == 1


def test_no_valid_text_chunks_raises_after_auditing(tmp_path):
    chunks = _chunks(2)
    chunks[0]["text"] = "  "
    chunks[1]["text"] = None
    client = FakeClient()
    options = _ingest_args(tmp_path, client, chunks=chunks)

    with pytest.raises(chroma_store.ChromaIngestionError, match="No chunks with non-empty string text"):
        chroma_store.ingest(**options)

    assert client.collections == {}
    audit_path = tmp_path / "embeddings" / "embedding_audit_log.csv"
    with audit_path.open(encoding="utf-8", newline="") as audit_file:
        assert len(list(csv.DictReader(audit_file))) == 2


def test_rerun_upserts_without_duplicate_records(tmp_path):
    client = FakeClient()
    options = _ingest_args(tmp_path, client, batch_size=2)

    chroma_store.ingest(**options)
    chroma_store.ingest(**options)

    collection = client.collections[chroma_store.DEFAULT_COLLECTION_NAME]
    assert collection.count() == 3
    assert collection.batch_sizes == [2, 1, 2, 1]


def test_rejects_collection_with_different_source_fingerprint(tmp_path):
    client = FakeClient()
    options = _ingest_args(tmp_path, client)
    corpus = chroma_store.load_validated_corpus(options["chunks_file"], options["embeddings_root"])
    wrong_metadata = dict(corpus.collection_metadata, source_chunks_sha256="stale")
    client.collections[chroma_store.DEFAULT_COLLECTION_NAME] = FakeCollection(
        chroma_store.DEFAULT_COLLECTION_NAME, wrong_metadata
    )

    with pytest.raises(chroma_store.ChromaIngestionError, match="different corpus/model metadata"):
        chroma_store.ingest(**options)


def test_rejects_stale_existing_ids_before_upsert(tmp_path):
    client = FakeClient()
    options = _ingest_args(tmp_path, client)
    corpus = chroma_store.load_validated_corpus(options["chunks_file"], options["embeddings_root"])
    collection = FakeCollection(chroma_store.DEFAULT_COLLECTION_NAME, corpus.collection_metadata)
    collection.records["old-chunk"] = {
        "embedding": np.zeros(768, dtype=np.float32),
        "document": "stale",
        "metadata": {},
    }
    client.collections[chroma_store.DEFAULT_COLLECTION_NAME] = collection

    with pytest.raises(chroma_store.ChromaIngestionError, match="IDs absent from the source corpus"):
        chroma_store.ingest(**options)

    assert collection.batch_sizes == []


def test_verify_reopens_collection_and_checks_sample_records(tmp_path):
    client = FakeClient()
    options = _ingest_args(tmp_path, client)
    chroma_store.ingest(**options)

    result = chroma_store.verify(**options, sample_size=2)

    assert result["count"] == 3
    assert result["sample_ids"] == ["chunk-0", "chunk-1"]


def test_verification_rejects_missing_or_extra_collection_ids(tmp_path):
    client = FakeClient()
    options = _ingest_args(tmp_path, client)
    chroma_store.ingest(**options)
    collection = client.collections[chroma_store.DEFAULT_COLLECTION_NAME]
    del collection.records["chunk-0"]

    with pytest.raises(chroma_store.ChromaIngestionError, match="count mismatch"):
        chroma_store.verify(**options)


def test_verification_accepts_small_vector_round_trip_drift(tmp_path):
    client = FakeClient()
    options = _ingest_args(tmp_path, client)
    chroma_store.ingest(**options)
    collection = client.collections[chroma_store.DEFAULT_COLLECTION_NAME]
    collection.records["chunk-0"]["embedding"][0] += 1e-8

    result = chroma_store.verify(**options)

    assert result["count"] == 3


def test_verification_rejects_material_vector_change(tmp_path):
    client = FakeClient()
    options = _ingest_args(tmp_path, client)
    chroma_store.ingest(**options)
    collection = client.collections[chroma_store.DEFAULT_COLLECTION_NAME]
    collection.records["chunk-0"]["embedding"][0] += 1e-3

    with pytest.raises(chroma_store.ChromaIngestionError, match="Embedding vector mismatch"):
        chroma_store.verify(**options)


def test_rejects_source_hash_mismatch(tmp_path):
    chunks_file, embeddings_root = _write_corpus(tmp_path)
    metadata_path = embeddings_root / "embeddings_metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["input_file_sha256"] = "stale"
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")

    with pytest.raises(chroma_store.ChromaIngestionError, match="different chunk file"):
        chroma_store.load_validated_corpus(chunks_file, embeddings_root)