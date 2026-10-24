"""
src/retrieval.py

Retrieval pipeline, in three deliberately separate responsibilities:

    RetrievalLoader    -- load chunks + persisted embeddings, verify
                          they're aligned (chunk_id match, count and
                          order) before anything else touches them.
    RetrievalProcessor -- normalize a user query and embed it via
                          embeddings.embed_query() (the BGE query
                          prefix lives ONLY there -- never duplicated
                          here).
    DenseRetriever      -- brute-force NumPy dense similarity search:
                          rank the corpus by similarity to a query
                          vector, return the top_k results with
                          scores and full provenance.

This module never reimplements embedding, prefix, or validation
logic that already exists in embeddings.py -- it composes on top of
it. DenseRetriever is a deliberate brute-force baseline (not an ANN
index), appropriate at the current corpus scale and useful later as
the reference point any reranking/ANN approach gets measured against.
ChromaRetriever provides production-style retrieval from the
persistent ChromaDB collection without replacing that baseline.
"""

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# embeddings.py (and its own sibling imports) use flat imports,
# requiring src/ itself on sys.path -- not just the project root.
# Same bootstrap already used in batch_builder.py, for the same reason.
sys.path.insert(0, str(Path(__file__).resolve().parent))

import embeddings

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CHROMA_PERSIST_DIRECTORY = PROJECT_ROOT / "data" / "chroma_db"
DEFAULT_CHROMA_COLLECTION = "financial_reports_filtered"
EXPECTED_CHROMA_DIMENSION = 768


class ChromaRetrievalError(RuntimeError):
    """Raised when the configured Chroma collection cannot serve retrieval safely."""


class RetrievalLoader:
    """
    Loads the chunk corpus and its persisted embeddings, and verifies
    they actually correspond to each other before either is handed
    to a retriever.
    """

    def __init__(self, chunks_file: str | Path, embeddings_root: str | Path):
        self.chunks_file = Path(chunks_file)
        self.embeddings_root = Path(embeddings_root)

    def load_chunks(self) -> list[dict]:
        if not self.chunks_file.is_file():
            raise FileNotFoundError(f"Chunk file not found: {self.chunks_file}")
        return embeddings.load_chunks(self.chunks_file)

    def load_embeddings(self) -> embeddings.EmbeddingResult:
        if not self.embeddings_root.is_dir():
            raise FileNotFoundError(f"Embeddings directory not found: {self.embeddings_root}")
        return embeddings.load_embeddings(self.embeddings_root)

    def load(self) -> tuple[list[dict], embeddings.EmbeddingResult]:
        """
        Load both, and verify the chunk_ids in the chunk file match
        -- in count AND order -- the chunk_ids recorded in the
        embeddings. This is the practical failure mode the earlier
        embeddings.py hardening doesn't cover: chunks get
        regenerated (a chunker/build change, a source PDF update)
        but embeddings aren't re-run, so the two files individually
        look valid while no longer corresponding to each other.
        Caught here, explicitly, rather than surfacing later as a
        silently wrong retrieval result.
        """
        chunks = self.load_chunks()
        embedding_result = self.load_embeddings()

        chunk_ids_from_chunks = [c["chunk_id"] for c in chunks]
        if chunk_ids_from_chunks != embedding_result.chunk_ids:
            raise ValueError(
                "Chunk IDs in the chunk file do not match the chunk IDs "
                "recorded in the embeddings (different count or order). "
                "The embeddings are likely stale relative to the chunk "
                "file -- rerun `python -m src.embeddings` (add --force "
                "if the content hash didn't change but you know the "
                "chunks did) before retrieving."
            )

        return chunks, embedding_result


class RetrievalProcessor:
    """
    Turns a raw user query string into a query embedding vector,
    ready for similarity search against RetrievalLoader's embedding
    matrix.
    """

    def __init__(self, model=None):
        """
        model is injectable (a fake model in tests avoids downloading
        the real one); if omitted, the real model is loaded lazily,
        once, on first use.
        """
        self._model = model

    def _get_model(self):
        if self._model is None:
            self._model = embeddings.load_model()
        return self._model

    @staticmethod
    def normalize_query_text(user_input: str) -> str:
        """
        Collapse internal whitespace/newlines to single spaces and
        strip leading/trailing whitespace. Deliberately does NOT
        decide whether empty input is acceptable -- embeddings.
        embed_query() already rejects that clearly and consistently;
        duplicating the check here would just create two places that
        could disagree about what counts as "empty".
        """
        if not isinstance(user_input, str):
            raise TypeError(f"Query must be a string, got {type(user_input).__name__}.")
        return " ".join(user_input.split())

    def embed_query(self, user_input: str):
        """
        Normalize and embed a user query. Delegates entirely to
        embeddings.embed_query(), which applies the BGE QUERY_PREFIX
        (asymmetric to how corpus chunks are embedded -- see
        embeddings.py's module docstring) and rejects empty/
        whitespace-only input after normalization.
        """
        normalized = self.normalize_query_text(user_input)
        model = self._get_model()
        return embeddings.embed_query(normalized, model)


@dataclass
class RetrievalResult:
    """One ranked hit: score plus full provenance, not just an index."""
    chunk_id: str
    score: float
    rank: int
    chunk: dict  # the original chunk dict -- text, company, document,
                 # financial_year, page, chunk_type, and (for table
                 # chunks) table_id/table_heading/row_start/row_end etc.


class DenseRetriever:
    """
    Brute-force dense retrieval over a fixed corpus embedding matrix.

    Cosine similarity == a plain dot product here because both the
    corpus vectors (embeddings.embed_corpus) and the query vector
    (embeddings.embed_query) are produced with
    normalize_embeddings=True -- embeddings.py is the single source
    of truth for that; this class does not re-normalize anything,
    it just trusts and exploits it for a single vectorized matmul
    instead of a per-chunk Python loop.
    """

    def __init__(self, chunks: list[dict], embedding_result):
        """
        chunks and embedding_result are expected to already be
        aligned -- RetrievalLoader.load() is what guarantees this in
        practice. Re-validated here anyway (cheap, O(n)) rather than
        silently trusted, since DenseRetriever may be constructed
        directly by future code that bypasses RetrievalLoader.
        """
        self._validate_construction(chunks, embedding_result)
        self.chunks = chunks
        self.chunk_ids = embedding_result.chunk_ids
        self.vectors = embedding_result.vectors
        self.dimension = embedding_result.dimension

    @staticmethod
    def _validate_construction(chunks: list[dict], embedding_result) -> None:
        vectors = embedding_result.vectors
        if not isinstance(vectors, np.ndarray) or vectors.ndim != 2:
            raise ValueError(
                f"Corpus embedding matrix must be a 2-dimensional NumPy array, "
                f"got {getattr(vectors, 'ndim', type(vectors).__name__)}."
            )

        chunk_ids_from_chunks = [c["chunk_id"] for c in chunks]
        if chunk_ids_from_chunks != embedding_result.chunk_ids:
            raise ValueError(
                "chunks and embedding_result are not aligned (chunk_id count "
                "or order differs). Construct DenseRetriever from "
                "RetrievalLoader.load()'s output, which guarantees alignment."
            )

        if len(chunks) > 0 and vectors.shape[0] != len(chunks):
            raise ValueError(
                f"Corpus embedding row count ({vectors.shape[0]}) does not "
                f"match chunk count ({len(chunks)})."
            )

    def search(self, query_vector: np.ndarray, top_k: int = 5) -> list[RetrievalResult]:
        """
        Rank every corpus chunk by similarity to query_vector and
        return the top_k highest-scoring results.

        top_k larger than the corpus is not an error -- it returns
        every available chunk (fewer than top_k), matching how most
        retrieval systems behave rather than forcing the caller to
        know the corpus size in advance.

        Deterministic tie-breaking: equal scores (real with
        normalized vectors on near-duplicate/boilerplate text) are
        broken by chunk_id ascending, via np.lexsort -- a stable,
        well-defined sort, not whatever order NumPy's default sort
        happens to produce for equal keys.
        """
        if isinstance(top_k, bool) or not isinstance(top_k, int):
            raise ValueError(f"top_k must be an integer, got {type(top_k).__name__}.")
        if top_k <= 0:
            raise ValueError(f"top_k must be a positive integer, got {top_k}.")

        if len(self.chunks) == 0:
            raise ValueError("Cannot search an empty corpus -- no chunks/embeddings are loaded.")

        query_vector = np.asarray(query_vector)
        if query_vector.ndim != 1:
            raise ValueError(f"query_vector must be 1-dimensional, got {query_vector.ndim} dimension(s).")
        if query_vector.shape[0] != self.dimension:
            raise ValueError(
                f"query_vector dimension ({query_vector.shape[0]}) does not match "
                f"corpus embedding dimension ({self.dimension})."
            )

        effective_k = min(top_k, len(self.chunks))

        scores = self.vectors @ query_vector  # vectorized dot product, shape (n_chunks,)

        # lexsort sorts by the LAST key first (primary), so -scores
        # (descending similarity) is primary and chunk_ids
        # (ascending) is the deterministic tie-breaker.
        order = np.lexsort((self.chunk_ids, -scores))[:effective_k]

        return [
            RetrievalResult(
                chunk_id=self.chunk_ids[idx],
                score=float(scores[idx]),
                rank=rank,
                chunk=self.chunks[idx],
            )
            for rank, idx in enumerate(order, start=1)
        ]


class ChromaRetriever:
    """Retrieve ranked passages from an existing persistent Chroma collection."""

    FILTER_FIELDS = ("company", "financial_year", "chunk_type")

    def __init__(
        self,
        processor: RetrievalProcessor | None = None,
        persist_directory: str | Path = DEFAULT_CHROMA_PERSIST_DIRECTORY,
        collection_name: str = DEFAULT_CHROMA_COLLECTION,
        client=None,
        collection=None,
    ):
        self.processor = processor or RetrievalProcessor()
        self.persist_directory = self._resolve_persist_directory(persist_directory)
        self.collection_name = collection_name
        self._client = client

        if collection is None:
            if self._client is None:
                try:
                    import chromadb
                except ImportError as error:
                    raise ChromaRetrievalError(
                        "ChromaDB is not installed in the active Python environment."
                    ) from error
                try:
                    self._client = chromadb.PersistentClient(path=str(self.persist_directory))
                except Exception as error:
                    raise ChromaRetrievalError(
                        f"Could not open ChromaDB at {self.persist_directory}: {error}"
                    ) from error

            try:
                collection = self._client.get_collection(
                    name=collection_name,
                    embedding_function=None,
                )
            except Exception as error:
                raise ChromaRetrievalError(
                    f"Chroma collection {collection_name!r} was not found in "
                    f"{self.persist_directory}; retrieval never creates collections."
                ) from error

        self.collection = collection
        self._validate_collection()

    @staticmethod
    def _resolve_persist_directory(persist_directory: str | Path) -> Path:
        path = Path(persist_directory)
        return path if path.is_absolute() else PROJECT_ROOT / path

    def _validate_collection(self) -> None:
        try:
            count = self.collection.count()
        except Exception as error:
            raise ChromaRetrievalError(
                f"Could not read Chroma collection {self.collection_name!r}: {error}"
            ) from error
        if count <= 0:
            raise ChromaRetrievalError(
                f"Chroma collection {self.collection_name!r} is empty."
            )

        metadata = self.collection.metadata or {}
        dimension = metadata.get("embedding_dimension")
        if dimension is not None and dimension != EXPECTED_CHROMA_DIMENSION:
            raise ChromaRetrievalError(
                f"Collection embedding dimension is {dimension}; expected "
                f"{EXPECTED_CHROMA_DIMENSION}."
            )
        expected_identity = {
            "embedding_model_name": embeddings.DEFAULT_MODEL_NAME,
            "embedding_model_revision": embeddings.DEFAULT_MODEL_REVISION,
        }
        for key, expected in expected_identity.items():
            actual = metadata.get(key)
            if actual is not None and actual != expected:
                raise ChromaRetrievalError(
                    f"Collection metadata {key!r} is {actual!r}; expected {expected!r}."
                )

        metric = metadata.get("distance_metric")
        configuration = getattr(self.collection, "configuration", None)
        if configuration is not None:
            configured_metric = configuration.get("hnsw", {}).get("space")
            if configured_metric is not None and configured_metric != "cosine":
                raise ChromaRetrievalError(
                    f"Collection uses {configured_metric!r} distance; expected 'cosine'."
                )
        elif metric is not None and metric != "cosine":
            raise ChromaRetrievalError(
                f"Collection metadata declares {metric!r} distance; expected 'cosine'."
            )

        self.count = count

    @staticmethod
    def _validate_top_k(top_k: int) -> None:
        if isinstance(top_k, bool) or not isinstance(top_k, int):
            raise ValueError(f"top_k must be an integer, got {type(top_k).__name__}.")
        if top_k <= 0:
            raise ValueError(f"top_k must be a positive integer, got {top_k}.")

    def _validate_filter(self, field: str, value: str | None) -> None:
        if value is None:
            return
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{field} filter must be a non-empty string.")

        try:
            result = self.collection.get(
                where={field: value},
                limit=1,
                include=["metadatas"],
            )
        except Exception as error:
            raise ValueError(f"Invalid {field} filter {value!r}: {error}") from error
        if not isinstance(result, dict) or not isinstance(result.get("ids"), list):
            raise ChromaRetrievalError(
                f"Chroma returned a malformed response while validating {field} filter."
            )
        if not result["ids"]:
            raise ValueError(
                f"Unsupported {field} filter {value!r}; use an exact value stored in the collection."
            )

    def search(
        self,
        query: str,
        top_k: int = 5,
        company: str | None = None,
        financial_year: str | None = None,
        chunk_type: str | None = None,
    ) -> list[RetrievalResult]:
        """Embed a query and return Chroma results scored as cosine similarity.

        Chroma returns cosine distance for this collection; score is
        converted to cosine similarity with ``1 - distance``.
        Filter values must match the collection's stored values exactly.
        """
        self._validate_top_k(top_k)
        filters = {
            field: value
            for field, value in (
                ("company", company),
                ("financial_year", financial_year),
                ("chunk_type", chunk_type),
            )
            if value is not None
        }
        for field, value in filters.items():
            self._validate_filter(field, value)

        query_vector = np.asarray(self.processor.embed_query(query), dtype=np.float32)
        if query_vector.ndim != 1:
            raise ValueError(
                f"Query embedding must be 1-dimensional, got {query_vector.ndim} dimension(s)."
            )
        if query_vector.shape[0] != EXPECTED_CHROMA_DIMENSION:
            raise ValueError(
                f"Query embedding dimension ({query_vector.shape[0]}) does not match "
                f"the collection dimension ({EXPECTED_CHROMA_DIMENSION})."
            )
        if not np.isfinite(query_vector).all():
            raise ValueError("Query embedding contains non-finite values.")

        where = None
        if len(filters) == 1:
            where = dict(filters)
        elif filters:
            where = {"$and": [{field: value} for field, value in filters.items()]}

        try:
            response = self.collection.query(
                query_embeddings=[query_vector.tolist()],
                n_results=min(top_k, self.count),
                include=["distances", "documents", "metadatas"],
                **({"where": where} if where is not None else {}),
            )
        except Exception as error:
            raise ChromaRetrievalError(f"Chroma query failed: {error}") from error

        required = ("ids", "distances", "documents", "metadatas")
        if not isinstance(response, dict) or any(key not in response for key in required):
            raise ChromaRetrievalError("Chroma query returned missing result fields.")
        nested = [response[key] for key in required]
        if any(not isinstance(value, list) or len(value) != 1 for value in nested):
            raise ChromaRetrievalError("Chroma query returned malformed result arrays.")

        ids, distances, documents, metadatas = (value[0] for value in nested)
        if not all(isinstance(values, list) for values in (ids, distances, documents, metadatas)):
            raise ChromaRetrievalError("Chroma query returned malformed per-query results.")
        if not (len(ids) == len(distances) == len(documents) == len(metadatas)):
            raise ChromaRetrievalError("Chroma query result arrays have inconsistent lengths.")

        results = []
        for rank, (chunk_id, distance, document, metadata) in enumerate(
            zip(ids, distances, documents, metadatas),
            start=1,
        ):
            if not isinstance(chunk_id, str) or not chunk_id:
                raise ChromaRetrievalError("Chroma query returned an invalid chunk ID.")
            if not isinstance(document, str):
                raise ChromaRetrievalError(f"Chroma returned invalid document text for {chunk_id!r}.")
            if not isinstance(metadata, dict):
                raise ChromaRetrievalError(f"Chroma returned invalid metadata for {chunk_id!r}.")
            if "chunk_id" in metadata and metadata["chunk_id"] != chunk_id:
                raise ChromaRetrievalError(
                    f"Chroma ID {chunk_id!r} does not match metadata chunk_id {metadata['chunk_id']!r}."
                )
            if isinstance(distance, bool) or not isinstance(distance, (int, float)):
                raise ChromaRetrievalError(f"Chroma returned invalid cosine distance for {chunk_id!r}.")
            if not np.isfinite(distance):
                raise ChromaRetrievalError(f"Chroma returned non-finite cosine distance for {chunk_id!r}.")

            chunk = dict(metadata)
            chunk["chunk_id"] = chunk_id
            chunk["text"] = document
            results.append(
                RetrievalResult(
                    chunk_id=chunk_id,
                    score=1.0 - float(distance),
                    rank=rank,
                    chunk=chunk,
                )
            )
        return results