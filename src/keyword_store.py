"""Separate vector and lexical indexes for generated chunk keyword metadata."""

import hashlib
import json
import logging
import math
import re
import sqlite3
from pathlib import Path

import numpy as np

from . import embeddings
from .keyword_generator import load_existing_annotations
from .keyword_schema import ChunkKeywordAnnotation
from .retrieval import ChromaRetrievalError, RetrievalResult

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_KEYWORD_CHROMA_PATH = PROJECT_ROOT / "data" / "keyword_chroma_db"
DEFAULT_KEYWORD_COLLECTION = "financial_report_keywords"
DEFAULT_LEXICAL_INDEX_PATH = PROJECT_ROOT / "data" / "keyword_index" / "financial_keyword_index.sqlite"
KEYWORD_INDEX_VERSION = "1.0"
EXPECTED_DIMENSION = 768
LEXICAL_FIELDS = ("company", "financial_year", "chunk_type")


class KeywordIndexError(RuntimeError):
    """Raised when keyword metadata cannot safely be indexed or resolved."""


def _keyword_document(annotation: ChunkKeywordAnnotation) -> str:
    keywords = "; ".join(annotation.keywords)
    if annotation.description:
        return f"{keywords}; description: {annotation.description}"
    return keywords


def _source_text_hash(chunk: dict) -> str:
    text = chunk.get("text")
    encoded = text if isinstance(text, str) else json.dumps(text, ensure_ascii=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def load_successful_annotations(
    annotations_path: str | Path,
    source_chunks_by_id: dict[str, dict],
) -> tuple[list[ChunkKeywordAnnotation], dict[str, int]]:
    try:
        records = load_existing_annotations(annotations_path, set(source_chunks_by_id))
    except ValueError as error:
        raise KeywordIndexError(str(error)) from error
    successes = []
    status_counts = {"failed": 0, "skipped": 0, "success": 0}
    for chunk_id, annotation in records.items():
        if annotation.annotation_status == "failed":
            status_counts["failed"] += 1
            continue
        if annotation.annotation_status == "skipped":
            status_counts["skipped"] += 1
            continue
        chunk = source_chunks_by_id[chunk_id]
        if annotation.source_text_sha256 != _source_text_hash(chunk):
            raise KeywordIndexError(
                f"Keyword annotation for {chunk_id!r} is stale relative to source text; regenerate it first."
            )
        expected_metadata = {key: value for key, value in chunk.items() if key != "text"}
        if annotation.source_metadata != expected_metadata:
            raise KeywordIndexError(
                f"Keyword annotation metadata for {chunk_id!r} does not match the source chunk."
            )
        if not annotation.keywords:
            raise KeywordIndexError(f"Successful keyword annotation for {chunk_id!r} has no keywords.")
        successes.append(annotation)
        status_counts["success"] += 1
    return successes, status_counts


def _resolve_isolated_path(path: str | Path, baseline_path: Path) -> Path:
    resolved = Path(path).resolve()
    baseline = baseline_path.resolve()
    if resolved == baseline or baseline in resolved.parents or resolved in baseline.parents:
        raise KeywordIndexError(
            f"Keyword index path {resolved} overlaps the existing baseline database {baseline}."
        )
    return resolved


def _collection_metadata() -> dict:
    return {
        "embedding_model_name": embeddings.DEFAULT_MODEL_NAME,
        "embedding_model_revision": embeddings.DEFAULT_MODEL_REVISION,
        "embedding_dimension": EXPECTED_DIMENSION,
        "distance_metric": "cosine",
        "keyword_index_version": KEYWORD_INDEX_VERSION,
        "keyword_document_format": "keywords_and_description_only",
    }


def _get_or_create_keyword_collection(client, name: str):
    if name in {"financial_reports", "financial_reports_filtered"}:
        raise KeywordIndexError(f"Refusing to use baseline Chroma collection {name!r} for keyword vectors.")
    listed = client.list_collections()
    names = {collection if isinstance(collection, str) else collection.name for collection in listed}
    expected_metadata = _collection_metadata()
    if name in names:
        collection = client.get_collection(name=name, embedding_function=None)
        if collection.metadata != expected_metadata:
            raise KeywordIndexError(
                f"Keyword collection {name!r} has incompatible metadata; refusing to mix indexes."
            )
        if collection.configuration.get("hnsw", {}).get("space") != "cosine":
            raise KeywordIndexError(f"Keyword collection {name!r} must use cosine distance.")
        return collection
    return client.create_collection(
        name=name,
        configuration={"hnsw": {"space": "cosine"}},
        metadata=expected_metadata,
        embedding_function=None,
    )


def _batched(values: list, batch_size: int):
    for start in range(0, len(values), batch_size):
        yield values[start:start + batch_size]


def index_keyword_vectors(
    annotations: list[ChunkKeywordAnnotation],
    source_chunks_by_id: dict[str, dict],
    persist_directory: str | Path = DEFAULT_KEYWORD_CHROMA_PATH,
    collection_name: str = DEFAULT_KEYWORD_COLLECTION,
    batch_size: int = 64,
    client=None,
    model=None,
    baseline_persist_directory: str | Path | None = None,
) -> dict:
    """Upsert keyword-only embeddings without touching the original Chroma store."""
    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size <= 0:
        raise ValueError("batch_size must be a positive integer.")
    baseline_path = Path(baseline_persist_directory or (PROJECT_ROOT / "data" / "chroma_db"))
    persist_path = _resolve_isolated_path(persist_directory, baseline_path)
    persist_path.mkdir(parents=True, exist_ok=True)

    valid_annotations, status_counts = [], {"failed": 0, "skipped": 0, "success": 0}
    seen_ids = set()
    for annotation in annotations:
        if annotation.chunk_id in seen_ids:
            raise KeywordIndexError(f"Duplicate keyword annotation for {annotation.chunk_id!r}.")
        seen_ids.add(annotation.chunk_id)
        if annotation.annotation_status == "failed":
            status_counts["failed"] += 1
            continue
        if annotation.annotation_status == "skipped":
            status_counts["skipped"] += 1
            continue
        if annotation.chunk_id not in source_chunks_by_id:
            raise KeywordIndexError(f"Annotation references unknown source chunk {annotation.chunk_id!r}.")
        chunk = source_chunks_by_id[annotation.chunk_id]
        if annotation.source_text_sha256 != _source_text_hash(chunk):
            raise KeywordIndexError(f"Annotation for {annotation.chunk_id!r} is stale; regenerate first.")
        if annotation.source_metadata != {key: value for key, value in chunk.items() if key != "text"}:
            raise KeywordIndexError(f"Annotation metadata for {annotation.chunk_id!r} mismatches source metadata.")
        if not annotation.keywords:
            raise KeywordIndexError(f"Successful annotation for {annotation.chunk_id!r} has no keywords.")
        valid_annotations.append(annotation)
        status_counts["success"] += 1

    if not valid_annotations:
        return {
            "indexed_count": 0,
            "updated_count": 0,
            "unchanged_count": 0,
            "failed_count": status_counts["failed"],
            "skipped_count": status_counts["skipped"],
        }

    if client is None:
        try:
            import chromadb
        except ImportError as error:
            raise KeywordIndexError("ChromaDB is unavailable in the selected Python environment.") from error
        client = chromadb.PersistentClient(path=str(persist_path))
    collection = _get_or_create_keyword_collection(client, collection_name)

    existing_metadata = {}
    all_ids = [annotation.chunk_id for annotation in valid_annotations]
    for id_batch in _batched(all_ids, batch_size):
        existing = collection.get(ids=id_batch, include=["metadatas"])
        existing_metadata.update(zip(existing["ids"], existing["metadatas"]))

    pending = []
    unchanged = 0
    for annotation in valid_annotations:
        document = _keyword_document(annotation)
        digest = hashlib.sha256(document.encode("utf-8")).hexdigest()
        current = existing_metadata.get(annotation.chunk_id)
        if current and current.get("keyword_document_sha256") == digest:
            unchanged += 1
            continue
        pending.append((annotation, document, digest, source_chunks_by_id[annotation.chunk_id]))

    if pending and model is None:
        model = embeddings.load_model(embeddings.DEFAULT_MODEL_NAME, embeddings.DEFAULT_MODEL_REVISION)

    indexed = updated = 0
    for batch in _batched(pending, batch_size):
        documents = [item[1] for item in batch]
        vectors = embeddings.embed_texts(model, documents, batch_size=batch_size)
        if vectors.ndim != 2 or vectors.shape != (len(batch), EXPECTED_DIMENSION):
            raise KeywordIndexError(
                f"Keyword embedding batch has shape {vectors.shape}; expected ({len(batch)}, {EXPECTED_DIMENSION})."
            )
        if not np.isfinite(vectors).all():
            raise KeywordIndexError("Keyword embeddings contain non-finite values.")
        metadatas = []
        for annotation, document, digest, source in batch:
            metadata = {
                key: source[key]
                for key in (
                    "company", "financial_year", "document", "page", "section", "chunk_type",
                    "table_id", "table_heading",
                )
                if source.get(key) is not None and isinstance(source.get(key), (str, int, bool))
            }
            metadata.update({
                "chunk_id": annotation.chunk_id,
                "annotation_version": annotation.annotation_version,
                "annotation_provider": annotation.annotation_provider,
                "annotation_model": annotation.annotation_model,
                "keyword_document_sha256": digest,
            })
            metadatas.append(metadata)
        collection.upsert(
            ids=[item[0].chunk_id for item in batch],
            embeddings=vectors,
            documents=documents,
            metadatas=metadatas,
        )
        for annotation, _, _, _ in batch:
            if annotation.chunk_id in existing_metadata:
                updated += 1
            else:
                indexed += 1

    for id_batch in _batched(all_ids, batch_size):
        stored_ids = collection.get(ids=id_batch, include=["metadatas"])["ids"]
        if set(stored_ids) != set(id_batch):
            raise KeywordIndexError("Keyword collection verification found missing indexed IDs.")

    return {
        "indexed_count": indexed,
        "updated_count": updated,
        "unchanged_count": unchanged,
        "failed_count": status_counts["failed"],
        "skipped_count": status_counts["skipped"],
    }


def _lexical_terms(query: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", query.casefold())


def _fts_expression(query: str) -> str:
    tokens = list(dict.fromkeys(_lexical_terms(query)))[:40]
    if not tokens:
        return ""
    phrases = ["\"" + token.replace('"', '""') + "\"" for token in tokens]
    phrases.extend(
        "\"" + f"{first} {second}".replace('"', '""') + "\""
        for first, second in zip(tokens, tokens[1:])
    )
    return " OR ".join(phrases)


class KeywordLexicalIndex:
    """Incremental SQLite FTS5 index with metadata filters and source resolution."""

    def __init__(self, path: str | Path = DEFAULT_LEXICAL_INDEX_PATH, source_chunks_by_id=None):
        self.path = Path(path).resolve()
        baseline_db = (PROJECT_ROOT / "data" / "chroma_db" / "chroma.sqlite3").resolve()
        if self.path == baseline_db or baseline_db in self.path.parents:
            raise KeywordIndexError("Lexical index path must not be inside the baseline Chroma database.")
        self.source_chunks_by_id = source_chunks_by_id or {}
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _connect(self):
        connection = sqlite3.connect(self.path)
        connection.row_factory = sqlite3.Row
        return connection

    def _initialize(self):
        try:
            with self._connect() as connection:
                connection.execute("""
                    CREATE TABLE IF NOT EXISTS keyword_records (
                        chunk_id TEXT PRIMARY KEY,
                        company TEXT,
                        financial_year TEXT,
                        document TEXT,
                        page INTEGER,
                        section TEXT,
                        chunk_type TEXT,
                        table_id TEXT,
                        table_heading TEXT,
                        keywords_json TEXT NOT NULL,
                        description TEXT NOT NULL,
                        annotation_version TEXT NOT NULL,
                        annotation_provider TEXT NOT NULL,
                        annotation_model TEXT NOT NULL,
                        source_text_sha256 TEXT NOT NULL
                    )
                """)
                connection.execute("""
                    CREATE VIRTUAL TABLE IF NOT EXISTS keyword_search USING fts5(
                        chunk_id UNINDEXED,
                        company UNINDEXED,
                        financial_year UNINDEXED,
                        chunk_type UNINDEXED,
                        keywords,
                        description,
                        tokenize='unicode61 remove_diacritics 2'
                    )
                """)
                connection.execute("PRAGMA user_version = 1")
        except sqlite3.Error as error:
            raise KeywordIndexError(f"Could not initialize FTS5 keyword index at {self.path}: {error}") from error

    def index_annotations(self, annotations: list[ChunkKeywordAnnotation]) -> dict:
        inserted = updated = unchanged = skipped = failed = 0
        seen = set()
        try:
            with self._connect() as connection:
                for annotation in annotations:
                    if annotation.chunk_id in seen:
                        raise KeywordIndexError(f"Duplicate keyword annotation for {annotation.chunk_id!r}.")
                    seen.add(annotation.chunk_id)
                    if annotation.annotation_status == "failed":
                        failed += 1
                        continue
                    if annotation.annotation_status == "skipped":
                        skipped += 1
                        continue
                    source = self.source_chunks_by_id.get(annotation.chunk_id)
                    if source is None:
                        raise KeywordIndexError(f"Unknown source chunk ID {annotation.chunk_id!r}.")
                    if annotation.source_text_sha256 != _source_text_hash(source):
                        raise KeywordIndexError(f"Keyword annotation for {annotation.chunk_id!r} is stale.")
                    if annotation.source_metadata != {key: value for key, value in source.items() if key != "text"}:
                        raise KeywordIndexError(f"Keyword annotation metadata for {annotation.chunk_id!r} mismatches source.")
                    if not annotation.keywords:
                        raise KeywordIndexError(f"Successful annotation for {annotation.chunk_id!r} has no keywords.")

                    previous = connection.execute(
                        "SELECT keywords_json, description, annotation_version, source_text_sha256 "
                        "FROM keyword_records WHERE chunk_id = ?",
                        (annotation.chunk_id,),
                    ).fetchone()
                    keywords_json = json.dumps(annotation.keywords, ensure_ascii=False, separators=(",", ":"))
                    if previous and tuple(previous) == (
                        keywords_json,
                        annotation.description or "",
                        annotation.annotation_version,
                        annotation.source_text_sha256,
                    ):
                        unchanged += 1
                        continue

                    if previous:
                        updated += 1
                    else:
                        inserted += 1
                    metadata_values = (
                        annotation.chunk_id,
                        source.get("company"),
                        source.get("financial_year"),
                        source.get("document"),
                        source.get("page"),
                        source.get("section"),
                        source.get("chunk_type"),
                        source.get("table_id"),
                        source.get("table_heading"),
                        keywords_json,
                        annotation.description or "",
                        annotation.annotation_version,
                        annotation.annotation_provider,
                        annotation.annotation_model,
                        annotation.source_text_sha256,
                    )
                    connection.execute("""
                        INSERT OR REPLACE INTO keyword_records VALUES
                        (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, metadata_values)
                    connection.execute("DELETE FROM keyword_search WHERE chunk_id = ?", (annotation.chunk_id,))
                    connection.execute("""
                        INSERT INTO keyword_search
                        (chunk_id, company, financial_year, chunk_type, keywords, description)
                        VALUES (?, ?, ?, ?, ?, ?)
                    """, (
                        annotation.chunk_id,
                        source.get("company") or "",
                        source.get("financial_year") or "",
                        source.get("chunk_type") or "",
                        " ".join(annotation.keywords),
                        annotation.description or "",
                    ))
        except sqlite3.Error as error:
            raise KeywordIndexError(f"SQLite keyword indexing failed at {self.path}: {error}") from error
        return {
            "indexed_count": inserted,
            "updated_count": updated,
            "unchanged_count": unchanged,
            "failed_count": failed,
            "skipped_count": skipped,
        }

    def search(
        self,
        query: str,
        top_k: int = 5,
        company: str | None = None,
        financial_year: str | None = None,
        chunk_type: str | None = None,
    ) -> list[RetrievalResult]:
        if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k <= 0:
            raise ValueError("top_k must be a positive integer.")
        if not isinstance(query, str) or not query.strip():
            raise ValueError("Lexical query must be a non-empty string.")
        for field, value in (("company", company), ("financial_year", financial_year), ("chunk_type", chunk_type)):
            if value is not None and (not isinstance(value, str) or not value.strip()):
                raise ValueError(f"{field} filter must be a non-empty string.")
        expression = _fts_expression(query)
        if not expression:
            return []
        filters = [(field, value) for field, value in (
            ("company", company), ("financial_year", financial_year), ("chunk_type", chunk_type)
        ) if value is not None]
        where_sql = " AND ".join(f"{field} = ?" for field, _ in filters)
        if where_sql:
            where_sql = " AND " + where_sql
        params = [expression, *(value for _, value in filters), top_k]
        try:
            with self._connect() as connection:
                rows = connection.execute(
                    "SELECT chunk_id, bm25(keyword_search) AS bm25_score "
                    "FROM keyword_search WHERE keyword_search MATCH ?" + where_sql +
                    " ORDER BY bm25_score ASC, chunk_id ASC LIMIT ?",
                    params,
                ).fetchall()
        except sqlite3.Error as error:
            raise KeywordIndexError(f"Lexical keyword search failed: {error}") from error

        results = []
        for rank, row in enumerate(rows, start=1):
            chunk_id = row["chunk_id"]
            source = self.source_chunks_by_id.get(chunk_id)
            if source is None:
                raise KeywordIndexError(
                    f"Lexical index returned {chunk_id!r}, but source text/provenance is unavailable."
                )
            results.append(RetrievalResult(
                chunk_id=chunk_id,
                score=-float(row["bm25_score"]),
                rank=rank,
                chunk=source,
            ))
        return results