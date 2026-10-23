"""Persistent ChromaDB ingestion for the existing financial-report corpus."""

import argparse
import csv
import hashlib
import json
import logging
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from . import embeddings

logger = logging.getLogger(__name__)

DEFAULT_CHUNKS_FILE = Path("data/chunks/all_chunks.jsonl")
DEFAULT_EMBEDDINGS_ROOT = Path("data/embeddings")
DEFAULT_PERSIST_DIRECTORY = Path("data/chroma_db")
DEFAULT_COLLECTION_NAME = "financial_reports"
DEFAULT_BATCH_SIZE = 500
EXPECTED_DIMENSION = 768
INGESTION_SCHEMA_VERSION = "1"
INGESTION_FILTER_VERSION = "non-empty-string-text-v1"
SOURCE_METADATA_KEY = "_source_metadata_json"
AUDIT_FIELDS = (
    "timestamp",
    "stage",
    "event",
    "chunk_id",
    "company",
    "document",
    "financial_year",
    "page",
    "section",
    "chunk_type",
    "reason",
    "action",
)
REQUIRED_METADATA_FIELDS = (
    "company",
    "document",
    "financial_year",
    "page",
    "chunk_type",
    "parser_version",
)


class ChromaIngestionError(ValueError):
    """Raised when corpus files or an existing collection are unsafe to ingest."""


@dataclass
class ValidatedCorpus:
    chunks: list[dict]
    embedding_result: embeddings.EmbeddingResult
    skipped_chunks: list[dict]
    collection_metadata: dict


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1 << 16), b""):
            digest.update(block)
    return digest.hexdigest()


def _audit_skipped_chunks(skipped_chunks: list[dict], embeddings_root: str | Path) -> None:
    if not skipped_chunks:
        return

    audit_path = Path(embeddings_root) / "embedding_audit_log.csv"
    try:
        audit_path.parent.mkdir(parents=True, exist_ok=True)
        write_header = not audit_path.exists() or audit_path.stat().st_size == 0
        with audit_path.open("a", encoding="utf-8", newline="") as audit_file:
            writer = csv.DictWriter(audit_file, fieldnames=AUDIT_FIELDS)
            if write_header:
                writer.writeheader()
            for chunk in skipped_chunks:
                text = chunk.get("text")
                reason = "empty_or_whitespace_text" if isinstance(text, str) else "non_string_text"
                writer.writerow({
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "stage": "chroma_ingestion",
                    "event": "chunk_skipped",
                    "chunk_id": chunk.get("chunk_id", ""),
                    "company": chunk.get("company", ""),
                    "document": chunk.get("document", ""),
                    "financial_year": chunk.get("financial_year", ""),
                    "page": chunk.get("page", ""),
                    "section": chunk.get("section", ""),
                    "chunk_type": chunk.get("chunk_type", ""),
                    "reason": reason,
                    "action": "excluded_from_vector_db",
                })
    except Exception as error:
        logger.exception("Failed to write skipped-chunk audit records to %s", audit_path)
        raise ChromaIngestionError(
            f"Could not write skipped-chunk audit records to {audit_path}: {error}"
        ) from error


def load_validated_corpus(
    chunks_file: str | Path = DEFAULT_CHUNKS_FILE,
    embeddings_root: str | Path = DEFAULT_EMBEDDINGS_ROOT,
) -> ValidatedCorpus:
    """Load existing project artifacts and prove exact row alignment before use."""
    chunks_file = Path(chunks_file)
    embeddings_root = Path(embeddings_root)
    try:
        chunks = embeddings.load_chunks(chunks_file)
        embedding_result = embeddings.load_embeddings(embeddings_root)
    except (OSError, json.JSONDecodeError, embeddings.EmbeddingValidationError) as error:
        raise ChromaIngestionError(f"Could not load source artifacts: {error}") from error

    try:
        warnings = embeddings.validate_chunks_for_embedding(chunks)
    except embeddings.EmbeddingValidationError as error:
        raise ChromaIngestionError(str(error)) from error
    for warning in warnings:
        logger.warning("%s", warning)

    if not chunks:
        raise ChromaIngestionError("The source chunk corpus is empty.")

    for index, chunk in enumerate(chunks):
        for field in REQUIRED_METADATA_FIELDS:
            value = chunk.get(field)
            if field == "page":
                valid = isinstance(value, int) and not isinstance(value, bool)
            else:
                valid = isinstance(value, str) and bool(value.strip())
            if not valid:
                raise ChromaIngestionError(
                    f"Chunk at index {index} has missing or invalid required metadata '{field}'."
                )
    source_ids = [chunk["chunk_id"] for chunk in chunks]
    if len(set(source_ids)) != len(source_ids):
        raise ChromaIngestionError("The source chunk file contains duplicate chunk IDs.")
    if source_ids != embedding_result.chunk_ids:
        mismatch_index = next(
            (
                index
                for index, (source_id, vector_id) in enumerate(
                    zip(source_ids, embedding_result.chunk_ids)
                )
                if source_id != vector_id
            ),
            min(len(source_ids), len(embedding_result.chunk_ids)),
        )
        raise ChromaIngestionError(
            "Chunk IDs and embedding rows are not aligned exactly in source order "
            f"(chunks={len(source_ids)}, embeddings={len(embedding_result.chunk_ids)}, "
            f"first mismatch index={mismatch_index})."
        )

    vectors = embedding_result.vectors
    if vectors.ndim != 2 or vectors.shape[0] != len(chunks):
        raise ChromaIngestionError(
            f"Expected one embedding row per chunk; got shape {vectors.shape} for {len(chunks)} chunks."
        )
    if vectors.shape[1] != EXPECTED_DIMENSION or embedding_result.dimension != EXPECTED_DIMENSION:
        raise ChromaIngestionError(
            f"Expected {EXPECTED_DIMENSION}-dimensional embeddings, got {vectors.shape[1]}."
        )
    if not np.isfinite(vectors).all():
        raise ChromaIngestionError("Embedding vectors contain non-finite values.")

    metadata_path = embeddings_root / "embeddings_metadata.json"
    try:
        embedding_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ChromaIngestionError(f"Could not read embedding metadata: {error}") from error

    source_hash = _sha256_file(chunks_file)
    recorded_source_hash = embedding_metadata.get("input_file_sha256")
    if source_hash != recorded_source_hash:
        raise ChromaIngestionError(
            "Embedding metadata refers to a different chunk file "
            f"(recorded SHA-256={recorded_source_hash!r}, current SHA-256={source_hash}). "
            "Regenerate embeddings for the current chunks before ingestion."
        )

    if not embedding_result.model_name or not embedding_result.model_revision:
        raise ChromaIngestionError("Embedding model name and revision must be present.")
    if any(SOURCE_METADATA_KEY in chunk for chunk in chunks):
        raise ChromaIngestionError(
            f"Source chunks use reserved Chroma metadata key {SOURCE_METADATA_KEY!r}."
        )

    retained_indices = [
        index
        for index, chunk in enumerate(chunks)
        if isinstance(chunk.get("text"), str) and bool(chunk["text"].strip())
    ]
    retained_index_set = set(retained_indices)
    skipped_chunks = [
        chunk for index, chunk in enumerate(chunks) if index not in retained_index_set
    ]
    retained_chunks = [chunks[index] for index in retained_indices]
    filtered_embedding_result = embeddings.EmbeddingResult(
        chunk_ids=[embedding_result.chunk_ids[index] for index in retained_indices],
        vectors=vectors[retained_indices],
        model_name=embedding_result.model_name,
        model_revision=embedding_result.model_revision,
        dimension=embedding_result.dimension,
    )

    collection_metadata = {
        "embedding_model_name": embedding_result.model_name,
        "embedding_model_revision": embedding_result.model_revision,
        "embedding_dimension": EXPECTED_DIMENSION,
        "source_chunk_count": len(chunks),
        "ingestion_count": len(retained_chunks),
        "source_chunks_sha256": source_hash,
        "embedding_vectors_sha256": _sha256_file(embeddings_root / "embeddings.npy"),
        "embedding_ids_sha256": _sha256_file(embeddings_root / "chunk_ids.json"),
        "query_prefix": str(embedding_metadata.get("query_prefix", "")),
        "distance_metric": "cosine",
        "ingestion_schema_version": INGESTION_SCHEMA_VERSION,
        "ingestion_filter_version": INGESTION_FILTER_VERSION,
    }
    return ValidatedCorpus(
        retained_chunks,
        filtered_embedding_result,
        skipped_chunks,
        collection_metadata,
    )


def _to_chroma_metadata(chunk: dict) -> dict:
    """Keep directly filterable scalar fields and losslessly retain source metadata."""
    metadata = {}
    for key, value in chunk.items():
        if key in {"text", SOURCE_METADATA_KEY} or value is None:
            continue
        if isinstance(value, (str, bool, int)):
            metadata[key] = value
        elif isinstance(value, float) and math.isfinite(value):
            metadata[key] = value

    source_metadata = {key: value for key, value in chunk.items() if key != "text"}
    try:
        metadata[SOURCE_METADATA_KEY] = json.dumps(
            source_metadata,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    except (TypeError, ValueError) as error:
        raise ChromaIngestionError(
            f"Chunk {chunk.get('chunk_id')!r} metadata is not valid JSON: {error}"
        ) from error
    return metadata


def _get_client(persist_directory: str | Path):
    try:
        import chromadb
    except ImportError as error:
        raise ChromaIngestionError(
            "ChromaDB is not installed in this Python environment. Install the project's ChromaDB dependency."
        ) from error
    return chromadb.PersistentClient(path=str(persist_directory))


def _get_or_create_compatible_collection(client, name: str, metadata: dict):
    names = {
        collection if isinstance(collection, str) else collection.name
        for collection in client.list_collections()
    }
    if name in names:
        collection = client.get_collection(name=name, embedding_function=None)
        if collection.metadata != metadata:
            raise ChromaIngestionError(
                f"Collection {name!r} exists with different corpus/model metadata; refusing to mix data."
            )
        if collection.configuration.get("hnsw", {}).get("space") != "cosine":
            raise ChromaIngestionError(
                f"Collection {name!r} does not use cosine distance; refusing to ingest."
            )
        return collection

    return client.create_collection(
        name=name,
        configuration={"hnsw": {"space": "cosine"}},
        metadata=metadata,
        embedding_function=None,
    )


def _validate_batch_size(batch_size: int) -> None:
    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size <= 0:
        raise ValueError(f"batch_size must be a positive integer, got {batch_size!r}.")


def _scan_collection_ids(collection, count: int, batch_size: int) -> list[str]:
    collection_ids = []
    for offset in range(0, count, batch_size):
        result = collection.get(limit=batch_size, offset=offset, include=["metadatas"])
        collection_ids.extend(result["ids"])
    return collection_ids


def verify_collection(
    collection,
    corpus: ValidatedCorpus,
    batch_size: int = DEFAULT_BATCH_SIZE,
    sample_size: int = 3,
) -> dict:
    """Verify collection identity, complete ID coverage, and sample contents/vectors."""
    _validate_batch_size(batch_size)
    if sample_size <= 0:
        raise ValueError("sample_size must be a positive integer.")
    if collection.metadata != corpus.collection_metadata:
        raise ChromaIngestionError("Collection metadata does not match the validated source artifacts.")
    if collection.configuration.get("hnsw", {}).get("space") != "cosine":
        raise ChromaIngestionError("Collection is not configured for cosine distance.")

    expected_count = len(corpus.chunks)
    actual_count = collection.count()
    if actual_count != expected_count:
        raise ChromaIngestionError(
            f"Collection count mismatch: expected {expected_count}, found {actual_count}."
        )

    actual_ids = _scan_collection_ids(collection, actual_count, batch_size)
    expected_ids = corpus.embedding_result.chunk_ids
    if len(actual_ids) != expected_count or set(actual_ids) != set(expected_ids):
        raise ChromaIngestionError("Collection IDs do not exactly match the source chunk IDs.")

    sample_count = min(sample_size, expected_count)
    sample_ids = expected_ids[:sample_count]
    result = collection.get(
        ids=sample_ids,
        include=["documents", "metadatas", "embeddings"],
    )
    records = {chunk_id: index for index, chunk_id in enumerate(result["ids"])}
    expected_chunk_by_id = {chunk["chunk_id"]: chunk for chunk in corpus.chunks}
    expected_index_by_id = {
        chunk_id: index for index, chunk_id in enumerate(corpus.embedding_result.chunk_ids)
    }
    for sample_index, chunk_id in enumerate(sample_ids):
        if chunk_id not in records:
            raise ChromaIngestionError(f"Verification sample {chunk_id!r} is missing from the collection.")
        record_index = records[chunk_id]
        chunk = expected_chunk_by_id[chunk_id]
        if result["documents"][record_index] != chunk["text"]:
            raise ChromaIngestionError(f"Document text mismatch for sample {chunk_id!r}.")
        if result["metadatas"][record_index] != _to_chroma_metadata(chunk):
            raise ChromaIngestionError(f"Metadata mismatch for sample {chunk_id!r}.")
        actual_vector = np.asarray(result["embeddings"][record_index])
        expected_vector = corpus.embedding_result.vectors[expected_index_by_id[chunk_id]]
        if actual_vector.shape != expected_vector.shape or not np.allclose(
            actual_vector, expected_vector, rtol=1e-6, atol=1e-7
        ):
            raise ChromaIngestionError(f"Embedding vector mismatch for sample {chunk_id!r}.")

    return {"count": actual_count, "sample_ids": sample_ids}


def ingest(
    chunks_file: str | Path = DEFAULT_CHUNKS_FILE,
    embeddings_root: str | Path = DEFAULT_EMBEDDINGS_ROOT,
    persist_directory: str | Path = DEFAULT_PERSIST_DIRECTORY,
    collection_name: str = DEFAULT_COLLECTION_NAME,
    batch_size: int = DEFAULT_BATCH_SIZE,
    client=None,
) -> dict:
    """Ingest validated existing vectors, preserving IDs and safe rerun behavior."""
    _validate_batch_size(batch_size)
    corpus = load_validated_corpus(chunks_file, embeddings_root)
    _audit_skipped_chunks(corpus.skipped_chunks, embeddings_root)
    if not corpus.chunks:
        raise ChromaIngestionError("No chunks with non-empty string text remain for ingestion.")
    persist_directory = Path(persist_directory)
    persist_directory.mkdir(parents=True, exist_ok=True)
    client = client or _get_client(persist_directory)
    collection = _get_or_create_compatible_collection(
        client, collection_name, corpus.collection_metadata
    )

    existing_count = collection.count()
    if existing_count > len(corpus.chunks):
        raise ChromaIngestionError(
            "Existing collection has more records than the source corpus; refusing to retain stale IDs."
        )
    existing_ids = _scan_collection_ids(collection, existing_count, batch_size)
    expected_ids = set(corpus.embedding_result.chunk_ids)
    if not set(existing_ids).issubset(expected_ids):
        raise ChromaIngestionError(
            "Existing collection contains IDs absent from the source corpus; refusing to mix stale data."
        )

    total = len(corpus.chunks)
    for start in range(0, total, batch_size):
        end = min(start + batch_size, total)
        batch_chunks = corpus.chunks[start:end]
        logger.info("Upserting chunks %d-%d of %d", start + 1, end, total)
        collection.upsert(
            ids=corpus.embedding_result.chunk_ids[start:end],
            embeddings=corpus.embedding_result.vectors[start:end],
            documents=[chunk["text"] for chunk in batch_chunks],
            metadatas=[_to_chroma_metadata(chunk) for chunk in batch_chunks],
        )

    verification = verify_collection(collection, corpus, batch_size=batch_size)
    logger.info("Verified %d/%d collection records", verification["count"], total)
    return {
        "collection": collection_name,
        "persist_directory": str(persist_directory),
        "count": verification["count"],
        "model_name": corpus.embedding_result.model_name,
        "model_revision": corpus.embedding_result.model_revision,
        "sample_ids": verification["sample_ids"],
    }


def verify(
    chunks_file: str | Path = DEFAULT_CHUNKS_FILE,
    embeddings_root: str | Path = DEFAULT_EMBEDDINGS_ROOT,
    persist_directory: str | Path = DEFAULT_PERSIST_DIRECTORY,
    collection_name: str = DEFAULT_COLLECTION_NAME,
    batch_size: int = DEFAULT_BATCH_SIZE,
    sample_size: int = 3,
    client=None,
) -> dict:
    """Reopen the persistent collection and verify its count and sample records."""
    corpus = load_validated_corpus(chunks_file, embeddings_root)
    if not corpus.chunks:
        raise ChromaIngestionError("No chunks with non-empty string text remain for verification.")
    client = client or _get_client(persist_directory)
    names = {
        collection if isinstance(collection, str) else collection.name
        for collection in client.list_collections()
    }
    if collection_name not in names:
        raise ChromaIngestionError(f"Collection {collection_name!r} does not exist.")
    collection = client.get_collection(name=collection_name, embedding_function=None)
    result = verify_collection(collection, corpus, batch_size=batch_size, sample_size=sample_size)
    return {"collection": collection_name, **result}


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Ingest existing financial RAG embeddings into ChromaDB.")
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ("ingest", "verify"):
        subparser = subparsers.add_parser(command)
        subparser.add_argument("--chunks-file", type=Path, default=DEFAULT_CHUNKS_FILE)
        subparser.add_argument("--embeddings-root", type=Path, default=DEFAULT_EMBEDDINGS_ROOT)
        subparser.add_argument("--persist-directory", type=Path, default=DEFAULT_PERSIST_DIRECTORY)
        subparser.add_argument("--collection-name", default=DEFAULT_COLLECTION_NAME)
        subparser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
        if command == "verify":
            subparser.add_argument("--sample-size", type=int, default=3)
    return parser


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = _build_parser().parse_args(argv)
    try:
        if args.command == "ingest":
            summary = ingest(
                args.chunks_file,
                args.embeddings_root,
                args.persist_directory,
                args.collection_name,
                args.batch_size,
            )
        else:
            summary = verify(
                args.chunks_file,
                args.embeddings_root,
                args.persist_directory,
                args.collection_name,
                args.batch_size,
                args.sample_size,
            )
    except (ChromaIngestionError, ValueError, FileNotFoundError) as error:
        logger.error("%s", error)
        return 1
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())