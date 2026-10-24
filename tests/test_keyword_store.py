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


def _annotation(chunk, keywords=None, description="Revenue reporting details."):
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
        keywords=keywords or ["consolidated revenue", "FY2026", "revenue growth"],
        description=description,
        annotation_model="fake-model",
        annotation_status="success",
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
        return {
            "ids": ids,
            "metadatas": [self.records[record_id]["metadata"] for record_id in ids],
        }

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