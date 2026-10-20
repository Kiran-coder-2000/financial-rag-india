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
No vector database is introduced here.
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