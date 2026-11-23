import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import embeddings
from src.keyword_generator import KeywordGenerator
from src.keyword_schema import KeywordSuggestion
from src.keyword_store import (
    DEFAULT_KEYWORD_COLLECTION,
    KeywordIndexError,
    KeywordLexicalIndex,
    _keyword_document,
    index_keyword_vectors,
)


def _chunk(chunk_id, text, **overrides):
    chunk = {
        "chunk_id": chunk_id,
        "company": "Infosys",
        "document": "Annual Report",
        "financial_year": "2025-2026",
        "page": 77,
        "section": "Revenue",
        "chunk_type": "text",
        "text": text,
    }
    chunk.update(overrides)
    return chunk


def _annotation(
    chunk,
    keywords=None,
    description="Revenue reporting details.",
    status="success",
    error=None,
):
    from src.keyword_schema import ChunkKeywordAnnotation

    return ChunkKeywordAnnotation(
        chunk_id=chunk["chunk_id"],
        source_text_sha256=__import__("hashlib").sha256(chunk["text"].encode("utf-8")).hexdigest(),
        source_metadata={key: value for key, value in chunk.items() if key != "text"},
        company=chunk["company"],
        document=chunk["document"],
        financial_year=chunk["financial_year"],
        page=chunk["page"],
        section=chunk["section"],
        chunk_type=chunk["chunk_type"],
        keywords=(
            keywords if keywords is not None
            else ["consolidated revenue", "FY2026", "revenue growth"]
        ),
        description=description if status == "success" else None,
        annotation_model="fake-model",
        annotation_status=status,
        annotation_error=error,
    )


class FakeModel:
    def __init__(self):
        self.calls = []

    def encode(self, texts, **kwargs):
        self.calls.extend(texts)
        vectors = np.zeros((len(texts), 768), dtype=np.float32)
        for index in range(len(texts)):
            vectors[index, index % 768] = 1.0
        return vectors


class FakeCollection:
    def __init__(self, name, metadata, configuration=None):
        self.name = name
        self.metadata = metadata
        self.configuration = configuration or {"hnsw": {"space": "cosine"}}
        self.records = {}

    def upsert(self, ids, embeddings, documents, metadatas):
        for index, record_id in enumerate(ids):
            self.records[record_id] = {
                "embedding": np.asarray(embeddings[index]).copy(),
                "document": documents[index],
                "metadata": metadatas[index],
            }

    def get(self, ids=None, include=None, **kwargs):
        ids = [record_id for record_id in (ids or []) if record_id in self.records]
        result = {
            "ids": ids,
            "metadatas": [self.records[record_id]["metadata"] for record_id in ids],
        }
        if "embeddings" in (include or []):
            result["embeddings"] = [self.records[record_id]["embedding"] for record_id in ids]
        return result

    def count(self):
        return len(self.records)


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


def test_keyword_representation_does_not_include_source_chunk_text():
    annotation = _annotation(
        _chunk("c1", "FULL ORIGINAL CHUNK TEXT MUST NOT BE COPIED"),
        keywords=["consolidated revenue", "FY2026"],
        description="Consolidated revenue for FY2026.",
    )

    document = _keyword_document(annotation)

    assert "consolidated revenue" in document
    assert "Consolidated revenue for FY2026" in document
    assert "FULL ORIGINAL CHUNK TEXT" not in document


def test_keyword_vector_index_uses_stable_chunk_ids_and_original_metadata(tmp_path):
    chunk = _chunk("stable-id", "Original source passage.")
    annotation = _annotation(chunk)
    model = FakeModel()
    client = FakeClient()

    summary = index_keyword_vectors(
        [annotation], {"stable-id": chunk}, persist_directory=tmp_path / "keyword_db",
        client=client, model=model,
    )

    collection = client.collections[DEFAULT_KEYWORD_COLLECTION]
    assert summary["indexed_count"] == 1
    assert collection.count() == 1
    record = collection.records["stable-id"]
    assert record["document"] == _keyword_document(annotation)
    assert record["metadata"]["company"] == "Infosys"
    assert record["metadata"]["financial_year"] == "2025-2026"
    assert record["metadata"]["page"] == 77
    assert record["metadata"]["chunk_id"] == "stable-id"
    assert model.calls == [_keyword_document(annotation)]


def test_keyword_vector_index_excludes_skipped_and_failed_annotations(tmp_path):
    skipped_chunk = _chunk("skipped", "Cover page")
    failed_chunk = _chunk("failed", "Revenue data")
    annotations = [
        _annotation(skipped_chunk, keywords=[], status="skipped"),
        _annotation(failed_chunk, keywords=[], status="failed", error="API failure"),
    ]
    client = FakeClient()
    model = FakeModel()

    summary = index_keyword_vectors(
        annotations,
        {"skipped": skipped_chunk, "failed": failed_chunk},
        persist_directory=tmp_path / "keyword_db",
        client=client,
        model=model,
    )

    assert summary["skipped_count"] == 1
    assert summary["failed_count"] == 1
    assert summary["indexed_count"] == 0
    assert client.collections == {}
    assert model.calls == []


def test_keyword_vector_index_skips_unchanged_ids_without_loading_model(tmp_path, monkeypatch):
    chunk = _chunk("stable-id", "Original source passage.")
    annotation = _annotation(chunk)
    client = FakeClient()
    index_keyword_vectors(
        [annotation], {"stable-id": chunk}, persist_directory=tmp_path / "keyword_db",
        client=client, model=FakeModel(),
    )

    def fail_if_loaded(*args, **kwargs):
        raise AssertionError("Unchanged keyword documents should not be re-embedded.")

    monkeypatch.setattr(embeddings, "load_model", fail_if_loaded)
    summary = index_keyword_vectors(
        [annotation], {"stable-id": chunk}, persist_directory=tmp_path / "keyword_db",
        client=client,
    )

    assert summary["unchanged_count"] == 1
    assert summary["indexed_count"] == 0


def test_keyword_vector_index_updates_changed_annotation_without_duplicate_ids(tmp_path):
    chunk = _chunk("stable-id", "Original source passage.")
    first = _annotation(chunk, keywords=["revenue"])
    second = _annotation(chunk, keywords=["profit after tax"])
    client = FakeClient()
    index_keyword_vectors(
        [first], {"stable-id": chunk}, persist_directory=tmp_path / "keyword_db",
        client=client, model=FakeModel(),
    )

    summary = index_keyword_vectors(
        [second], {"stable-id": chunk}, persist_directory=tmp_path / "keyword_db",
        client=client, model=FakeModel(),
    )

    assert summary["updated_count"] == 1
    assert client.collections[DEFAULT_KEYWORD_COLLECTION].count() == 1
    assert "profit after tax" in client.collections[DEFAULT_KEYWORD_COLLECTION].records["stable-id"]["document"]


def test_keyword_vector_index_refreshes_metadata_without_reembedding(tmp_path):
    chunk = _chunk("stable-id", "Original source passage.")
    annotation = _annotation(chunk)
    client = FakeClient()
    model = FakeModel()
    index_keyword_vectors(
        [annotation], {"stable-id": chunk}, persist_directory=tmp_path / "keyword_db",
        client=client, model=model,
    )
    changed = {**chunk, "page": 99, "section": "Changed section"}
    changed_annotation = _annotation(changed)
    before_calls = len(model.calls)

    summary = index_keyword_vectors(
        [changed_annotation], {"stable-id": changed}, persist_directory=tmp_path / "keyword_db",
        client=client, model=model,
    )

    record = client.collections[DEFAULT_KEYWORD_COLLECTION].records["stable-id"]
    assert summary["updated_count"] == 1
    assert len(model.calls) == before_calls
    assert record["metadata"]["page"] == 99
    assert record["metadata"]["section"] == "Changed section"


def test_keyword_vector_index_refreshes_legacy_record_without_metadata_fingerprint(tmp_path):
    chunk = _chunk("legacy", "Original source passage.")
    annotation = _annotation(chunk)
    client = FakeClient()
    index_keyword_vectors(
        [annotation], {"legacy": chunk}, persist_directory=tmp_path / "keyword_db",
        client=client, model=FakeModel(),
    )
    record = client.collections[DEFAULT_KEYWORD_COLLECTION].records["legacy"]
    record["metadata"].pop("keyword_metadata_sha256", None)
    changed = {**chunk, "company": "TataConsultancyServices"}
    changed_annotation = _annotation(changed)
    model = FakeModel()

    summary = index_keyword_vectors(
        [changed_annotation], {"legacy": changed}, persist_directory=tmp_path / "keyword_db",
        client=client, model=model,
    )

    assert summary["updated_count"] == 1
    assert client.collections[DEFAULT_KEYWORD_COLLECTION].records["legacy"]["metadata"]["company"] == "TataConsultancyServices"


def test_keyword_vector_index_refuses_baseline_collection_and_path(tmp_path):
    chunk = _chunk("c1", "source")
    annotation = _annotation(chunk)
    with pytest.raises(KeywordIndexError, match="baseline Chroma collection"):
        index_keyword_vectors(
            [annotation], {"c1": chunk}, persist_directory=tmp_path / "db",
            collection_name="financial_reports_filtered", client=FakeClient(), model=FakeModel(),
        )
    with pytest.raises(KeywordIndexError, match="overlaps the existing baseline"):
        index_keyword_vectors(
            [annotation], {"c1": chunk}, persist_directory=tmp_path / "chroma_db",
            baseline_persist_directory=tmp_path / "chroma_db", client=FakeClient(), model=FakeModel(),
        )


def test_lexical_index_supports_filtering_and_resolves_original_text(tmp_path):
    text_chunk = _chunk("text-revenue", "Narrative source text contains revenue.", chunk_type="text")
    table_chunk = _chunk(
        "table-revenue",
        "Full financial table source.",
        chunk_type="table",
        table_id="revenue-table",
        table_heading="Consolidated revenue",
    )
    chunks = {item["chunk_id"]: item for item in (text_chunk, table_chunk)}
    annotations = [
        _annotation(text_chunk, ["revenue", "consolidated revenue"], "Narrative revenue evidence."),
        _annotation(table_chunk, ["revenue", "consolidated revenue", "FY2026"], "Revenue table."),
    ]
    index = KeywordLexicalIndex(tmp_path / "financial_keyword_index.sqlite", chunks)

    summary = index.index_annotations(annotations)
    filtered = index.search("consolidated revenue FY2026", top_k=5, chunk_type="table")

    assert summary["indexed_count"] == 2
    assert [result.chunk_id for result in filtered] == ["table-revenue"]
    assert filtered[0].chunk["text"] == table_chunk["text"]
    assert filtered[0].chunk["table_id"] == "revenue-table"


def test_lexical_index_incremental_runs_are_stable_and_filters_are_exact(tmp_path):
    chunk = _chunk("c1", "The consolidated revenue rose.")
    index = KeywordLexicalIndex(tmp_path / "keywords.sqlite", {"c1": chunk})
    annotation = _annotation(chunk)

    first = index.index_annotations([annotation])
    second = index.index_annotations([annotation])
    result = index.search("revenue", company="Infosys", financial_year="2025-2026")
    other_company = index.search("revenue", company="RelianceIndustries")

    assert first["indexed_count"] == 1
    assert second["unchanged_count"] == 1
    assert len(result) == 1 and result[0].chunk_id == "c1"
    assert other_company == []


def test_lexical_index_excludes_skipped_and_failed_annotations(tmp_path):
    skipped_chunk = _chunk("skipped", "Cover page")
    failed_chunk = _chunk("failed", "Revenue data")
    index = KeywordLexicalIndex(
        tmp_path / "keywords.sqlite",
        {"skipped": skipped_chunk, "failed": failed_chunk},
    )
    annotations = [
        _annotation(skipped_chunk, keywords=[], status="skipped"),
        _annotation(failed_chunk, keywords=[], status="failed", error="API failure"),
    ]

    summary = index.index_annotations(annotations)

    assert summary["skipped_count"] == 1
    assert summary["failed_count"] == 1
    assert summary["indexed_count"] == 0
    with index._connect() as connection:
        assert connection.execute("SELECT COUNT(*) FROM keyword_records").fetchone()[0] == 0
        assert connection.execute("SELECT COUNT(*) FROM keyword_search").fetchone()[0] == 0


def test_lexical_index_metadata_only_update_refreshes_table_and_fts_filters(tmp_path):
    chunk = _chunk("c1", "The consolidated revenue rose.")
    index = KeywordLexicalIndex(tmp_path / "keywords.sqlite", {"c1": chunk})
    annotation = _annotation(chunk)
    index.index_annotations([annotation])
    changed = {**chunk, "company": "TataConsultancyServices", "page": 99}
    changed_annotation = _annotation(changed)
    index.source_chunks_by_id = {"c1": changed}

    summary = index.index_annotations([changed_annotation])
    new_results = index.search("revenue", company="TataConsultancyServices")
    old_results = index.search("revenue", company="Infosys")

    assert summary["updated_count"] == 1
    assert [result.chunk_id for result in new_results] == ["c1"]
    assert old_results == []
    assert new_results[0].chunk["page"] == 99


def test_lexical_index_rebuilds_missing_fts_row(tmp_path):
    chunk = _chunk("c1", "The consolidated revenue rose.")
    index = KeywordLexicalIndex(tmp_path / "keywords.sqlite", {"c1": chunk})
    annotation = _annotation(chunk)
    index.index_annotations([annotation])
    with index._connect() as connection:
        connection.execute("DELETE FROM keyword_search WHERE chunk_id = ?", ("c1",))

    summary = index.index_annotations([annotation])

    assert summary["updated_count"] == 1
    assert [result.chunk_id for result in index.search("revenue")] == ["c1"]


def test_lexical_index_rolls_back_partial_batch_on_failure(tmp_path):
    first = _chunk("first", "revenue first")
    second = _chunk("second", "revenue second")
    index = KeywordLexicalIndex(tmp_path / "keywords.sqlite", {"first": first})
    annotations = [_annotation(first), _annotation(second)]

    with pytest.raises(KeywordIndexError, match="Unknown source chunk ID"):
        index.index_annotations(annotations)

    with index._connect() as connection:
        assert connection.execute("SELECT COUNT(*) FROM keyword_records").fetchone()[0] == 0
        assert connection.execute("SELECT COUNT(*) FROM keyword_search").fetchone()[0] == 0


def test_lexical_index_rejects_missing_source_chunk(tmp_path):
    chunk = _chunk("c1", "revenue")
    annotation = _annotation(chunk)
    index = KeywordLexicalIndex(tmp_path / "keywords.sqlite", {})

    with pytest.raises(KeywordIndexError, match="Unknown source chunk ID"):
        index.index_annotations([annotation])


def test_lexical_search_fails_if_original_source_cannot_be_resolved(tmp_path):
    chunk = _chunk("c1", "revenue")
    index = KeywordLexicalIndex(tmp_path / "keywords.sqlite", {"c1": chunk})
    index.index_annotations([_annotation(chunk)])
    index.source_chunks_by_id = {}

    with pytest.raises(KeywordIndexError, match="source text/provenance is unavailable"):
        index.search("revenue")