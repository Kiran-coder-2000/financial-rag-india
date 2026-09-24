"""
src/embeddings.py

Embedding generation for the chunk corpus, using a self-hosted
open-weight model (BAAI/bge-base-en-v1.5 by default) via
sentence-transformers -- reproducible (pinned model + revision),
free regardless of corpus size, and independent of any hosted API's
version drift.

    python -m src.embeddings
    python -m src.embeddings --force
    python -m src.embeddings --chunks-file data/chunks/all_chunks.jsonl --output-root data/embeddings

Corpus-level caching: if the input JSONL's content hash, model
name/revision, and embedding config all match what's recorded in
the existing embeddings_metadata.json, embedding is skipped and the
existing output is reused. --force bypasses this unconditionally.
Per-chunk incremental caching is a deliberate non-goal for now.

IMPORTANT -- asymmetric retrieval, read before changing anything:
bge-en models require DIFFERENT handling for documents being
indexed vs. queries being searched. Documents get NO prefix; a
query MUST be prefixed with QUERY_PREFIX below (confirmed against
BAAI's own model documentation), or retrieval quality degrades
silently, with no error or warning from the library itself.
embed_corpus() deliberately does NOT apply the prefix; embed_query()
applies it automatically. Do not "fix" embed_corpus() to also add
it -- that would break retrieval, not improve it.
"""

import argparse
import hashlib
import json
import logging
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

DEFAULT_MODEL_NAME = "BAAI/bge-base-en-v1.5"
# Pinning an exact commit (not just "main") is what actually
# guarantees reproducibility -- a model repo's default branch can
# change over time even when the model name doesn't. Verified via
# the model's own commit history as of this writing; re-pin to a
# newer commit deliberately, not by accident, if you ever upgrade.
DEFAULT_MODEL_REVISION = "08e3d2c28e3886dfbcd98caebb447d9e23614bc0"

# Confirmed against BAAI's own bge-base-en-v1.5 documentation.
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

DEFAULT_BATCH_SIZE = 32

# Bumped whenever the on-disk embedding schema itself changes (new/
# renamed metadata fields, a different file layout) -- independent
# of the model or embedding config, so an old cache is never
# silently reused with a schema this code no longer understands.
FORMAT_VERSION = "1.0"


class EmbeddingValidationError(ValueError):
    """
    Raised when chunk input or generated/persisted output fails
    validation. Carries every problem found, not just the first, so
    one run surfaces the full picture rather than forcing a
    fix-one-rerun-repeat cycle.
    """
    def __init__(self, problems: list[str]):
        self.problems = problems
        super().__init__("Embedding validation failed:\n" + "\n".join(f"  - {p}" for p in problems))


@dataclass
class EmbeddingResult:
    chunk_ids: list[str]
    vectors: np.ndarray  # shape (n_chunks, dimension)
    model_name: str
    model_revision: str
    dimension: int


def _sha256_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 16), b""):
            hasher.update(block)
    return hasher.hexdigest()


def _library_versions() -> dict:
    """Best-effort, informational only -- never blocks a run if a
    library's version can't be determined."""
    versions = {"numpy": np.__version__}
    try:
        import sentence_transformers
        versions["sentence_transformers"] = getattr(sentence_transformers, "__version__", "unknown")
    except ImportError:
        versions["sentence_transformers"] = "not installed"
    return versions


def _embedding_config_fingerprint() -> dict:
    """
    Config values that actually change the RESULTING vectors if
    altered -- used for cache-validity comparison. batch_size is
    deliberately excluded: batching is a performance grouping only,
    and a well-behaved encoder must produce identical per-text
    output regardless of how texts are grouped into batches, so a
    batch_size change alone should never force a re-embed.
    """
    return {"normalize_embeddings": True, "query_prefix": QUERY_PREFIX}


# ================================================================
# Loading
# ================================================================

def load_chunks(chunks_file: str | Path) -> list[dict]:
    """
    Load chunk dicts from a JSONL file (e.g. all_chunks.jsonl).
    Never modifies, filters, or re-validates them beyond basic JSON
    parsing -- structural validation is validate_chunks_for_embedding's
    job, and content quality auditing is audit.py's.
    """
    chunks_file = Path(chunks_file)
    chunks = []
    with open(chunks_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                chunks.append(json.loads(line))
    return chunks


# ================================================================
# Input validation
# ================================================================

def validate_chunks_for_embedding(
    chunks: list[dict],
    reject_empty_text: bool = False,
) -> list[str]:
    """
    Validate chunks BEFORE embedding. Raises EmbeddingValidationError
    (listing every problem found) for structural defects that make a
    chunk unembeddable: missing/empty/duplicate chunk_id, missing
    text. Valid chunks are never reordered, filtered, or modified as
    a side effect of this check -- a validation failure stops the
    whole run rather than silently dropping the offending chunk.

    Empty (but present) text is a deliberate special case, not
    grouped with "missing text": the chunking pipeline intentionally
    KEEPS near-empty/is_small chunks rather than dropping them (see
    chunker.py's MIN_CHUNK_TOKENS handling), so rejecting empty text
    outright here would make embeddings.py incompatible with data
    the chunker deliberately preserves. Default behavior: allow it
    through, and return it as an explicit warning string for the
    caller to log -- "handled explicitly", not silently skipped, but
    not a hard failure either. Pass reject_empty_text=True to make it
    a hard error instead, if that's ever preferred.
    """
    errors: list[str] = []
    warnings: list[str] = []
    seen_ids: dict[str, int] = {}

    for i, chunk in enumerate(chunks):
        if "chunk_id" not in chunk:
            errors.append(f"Chunk at index {i} is missing 'chunk_id'.")
            continue

        cid = chunk["chunk_id"]
        if not isinstance(cid, str) or not cid.strip():
            errors.append(f"Chunk at index {i} has an empty or invalid chunk_id: {cid!r}")
            continue

        if cid in seen_ids:
            errors.append(f"Duplicate chunk_id '{cid}' at indices {seen_ids[cid]} and {i}.")
        else:
            seen_ids[cid] = i

        if "text" not in chunk:
            errors.append(f"Chunk '{cid}' is missing 'text'.")
            continue

        text = chunk["text"]
        if not isinstance(text, str) or not text.strip():
            note = f"Chunk '{cid}' has empty text."
            if reject_empty_text:
                errors.append(note)
            else:
                warnings.append(note)

    if errors:
        raise EmbeddingValidationError(errors)

    return warnings


# ================================================================
# Embedding
# ================================================================

def load_model(model_name: str = DEFAULT_MODEL_NAME, revision: str = DEFAULT_MODEL_REVISION):
    """
    Loads the real sentence-transformers model. Imported lazily
    (inside this function, not at module top-level) so everything
    else in this file can be imported and unit-tested without
    sentence-transformers/torch installed or a model download
    succeeding.
    """
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(model_name, revision=revision)


def embed_texts(model, texts: list[str], batch_size: int = DEFAULT_BATCH_SIZE) -> np.ndarray:
    """
    Thin wrapper around model.encode -- kept as its own function so
    tests can substitute a fast fake model without needing the real
    weights. normalize_embeddings=True so downstream similarity is a
    plain dot product (equivalent to cosine similarity on unit
    vectors), which most vector DBs assume by default.
    """
    return model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )


def embed_corpus(
    chunks: list[dict],
    model,
    model_name: str = DEFAULT_MODEL_NAME,
    model_revision: str = DEFAULT_MODEL_REVISION,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> EmbeddingResult:
    """
    Embed every chunk's `text` field as a DOCUMENT -- no query
    prefix (see module docstring). Row order of the returned vectors
    matches chunk_ids exactly, index for index; this relationship is
    re-verified by validate_embedding_output() immediately after.
    """
    chunk_ids = [c["chunk_id"] for c in chunks]
    texts = [c["text"] for c in chunks]

    vectors = embed_texts(model, texts, batch_size=batch_size)

    return EmbeddingResult(
        chunk_ids=chunk_ids,
        vectors=vectors,
        model_name=model_name,
        model_revision=model_revision,
        dimension=vectors.shape[1] if len(vectors) else 0,
    )


def embed_query(query: str, model, batch_size: int = DEFAULT_BATCH_SIZE) -> np.ndarray:
    """
    Embed a QUERY string for retrieval. Applies QUERY_PREFIX exactly
    once, which embed_corpus() deliberately does not -- see module
    docstring for why mismatching this breaks retrieval silently.

    Unlike a document chunk (where empty/near-empty text is a
    legitimate, deliberately-preserved case -- see
    validate_chunks_for_embedding), an empty QUERY has no sensible
    meaning for retrieval: there is nothing to search for. Rejected
    here rather than silently embedding an empty string.
    """
    if not isinstance(query, str) or not query.strip():
        raise EmbeddingValidationError([f"Query text must be a non-empty string, got {query!r}."])
    vectors = embed_texts(model, [QUERY_PREFIX + query], batch_size=batch_size)
    return vectors[0]


# ================================================================
# Output validation
# ================================================================

def validate_embedding_output(result: EmbeddingResult, expect_normalized: bool = True) -> None:
    """
    Sanity-check an EmbeddingResult BEFORE it's persisted. Raises
    EmbeddingValidationError on any problem rather than silently
    writing inconsistent output to disk.

    Dimensionality is checked and enforced FIRST, before anything
    else touches .shape[1] -- a 1-D or 3-D array previously reached
    a raw shape[1] access here, which raises an uncontrolled
    IndexError (or a silently-wrong check) instead of a clear
    validation error.
    """
    problems: list[str] = []

    vectors = result.vectors
    if not isinstance(vectors, np.ndarray):
        try:
            vectors = np.asarray(vectors)
        except Exception as e:
            raise EmbeddingValidationError([f"Embedding output could not be interpreted as a NumPy array: {e}"])

    if vectors.ndim != 2:
        raise EmbeddingValidationError([
            f"Embedding array must be 2-dimensional, got {vectors.ndim} "
            f"dimension(s) with shape {vectors.shape}."
        ])

    n_vectors, actual_dim = vectors.shape
    n_ids = len(result.chunk_ids)

    if n_vectors != n_ids:
        problems.append(f"Vector count ({n_vectors}) does not match chunk_id count ({n_ids}).")

    if n_vectors > 0:
        if actual_dim <= 0:
            problems.append(f"Embedding dimension must be positive, got {actual_dim}.")
        elif actual_dim != result.dimension:
            problems.append(
                f"Recorded dimension ({result.dimension}) does not match "
                f"actual vector dimension ({actual_dim})."
            )

        if not np.all(np.isfinite(vectors)):
            n_bad = int(np.sum(~np.isfinite(vectors)))
            problems.append(f"Embedding array contains {n_bad} non-finite value(s) (NaN or Inf).")
        elif expect_normalized:
            norms = np.linalg.norm(vectors, axis=1)
            if not np.allclose(norms, 1.0, atol=1e-3):
                n_bad = int(np.sum(~np.isclose(norms, 1.0, atol=1e-3)))
                problems.append(
                    f"{n_bad} vector(s) are not unit-normalized (expected L2 norm "
                    f"~1.0 since normalize_embeddings=True)."
                )

    if len(set(result.chunk_ids)) != len(result.chunk_ids):
        problems.append("chunk_ids contain duplicates after embedding (should be impossible given input validation).")

    if problems:
        raise EmbeddingValidationError(problems)


def _load_from_paths(vectors_path: Path, ids_path: Path, meta_path: Path) -> EmbeddingResult:
    """
    Load and FULLY content-validate a persisted embedding set --
    this is the single choke point both load_embeddings() and the
    cache-validity check (_is_cache_valid) go through, so "the cache
    passed validation" and "the data is actually usable" can never
    silently diverge.

    Every failure mode below previously either crashed uncontrolled
    (a corrupt .npy) or was never checked at all (count/dimension
    mismatches, duplicate or non-string IDs, non-finite values) --
    metadata claiming a match was previously trusted without ever
    inspecting the files it describes.
    """
    try:
        with open(vectors_path, "rb") as f:
            vectors = np.load(f)
    except Exception as e:
        raise EmbeddingValidationError([f"Could not load embedding array from {vectors_path.name}: {e}"])

    try:
        with open(ids_path, "r", encoding="utf-8") as f:
            chunk_ids = json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        raise EmbeddingValidationError([f"Could not load chunk IDs from {ids_path.name}: {e}"])

    try:
        with open(meta_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        raise EmbeddingValidationError([f"Could not load metadata from {meta_path.name}: {e}"])

    problems: list[str] = []

    if not isinstance(chunk_ids, list):
        raise EmbeddingValidationError([f"{ids_path.name} must contain a JSON list, got {type(chunk_ids).__name__}."])
    if not all(isinstance(c, str) for c in chunk_ids):
        problems.append(f"{ids_path.name} must contain only strings.")
    if len(set(chunk_ids)) != len(chunk_ids):
        problems.append(f"{ids_path.name} contains duplicate chunk IDs.")

    if not isinstance(metadata, dict):
        raise EmbeddingValidationError([f"{meta_path.name} must contain a JSON object, got {type(metadata).__name__}."])
    for required in ("model_name", "model_revision", "dimension", "count"):
        if required not in metadata:
            problems.append(f"{meta_path.name} is missing required field '{required}'.")

    if problems:
        raise EmbeddingValidationError(problems)

    if vectors.ndim != 2:
        raise EmbeddingValidationError([
            f"{vectors_path.name} must be a 2-dimensional array, got {vectors.ndim} "
            f"dimension(s) with shape {vectors.shape}."
        ])

    n_vectors = vectors.shape[0]
    if n_vectors != len(chunk_ids):
        raise EmbeddingValidationError([
            f"Vector row count ({n_vectors}) does not match chunk_id count ({len(chunk_ids)})."
        ])
    if metadata["count"] != n_vectors:
        raise EmbeddingValidationError([
            f"metadata['count'] ({metadata['count']}) does not match actual vector row count ({n_vectors})."
        ])
    if n_vectors > 0 and vectors.shape[1] != metadata["dimension"]:
        raise EmbeddingValidationError([
            f"metadata['dimension'] ({metadata['dimension']}) does not match "
            f"actual vector dimension ({vectors.shape[1]})."
        ])
    if n_vectors > 0 and not np.all(np.isfinite(vectors)):
        raise EmbeddingValidationError([f"{vectors_path.name} contains non-finite (NaN/Inf) values."])

    return EmbeddingResult(
        chunk_ids=chunk_ids,
        vectors=vectors,
        model_name=metadata["model_name"],
        model_revision=metadata["model_revision"],
        dimension=metadata["dimension"],
    )


def load_embeddings(output_root: str | Path) -> EmbeddingResult:
    output_root = Path(output_root)
    return _load_from_paths(
        output_root / "embeddings.npy",
        output_root / "chunk_ids.json",
        output_root / "embeddings_metadata.json",
    )


def _validate_persisted_matches(reloaded: EmbeddingResult, expected: EmbeddingResult) -> list[str]:
    """Shared check used both for validating temp files before the
    atomic rename, and (optionally) for validating final output
    after a fresh write."""
    problems = []
    if reloaded.chunk_ids != expected.chunk_ids:
        problems.append("Reloaded chunk_ids do not match what was generated (order or content differs).")
    if not np.array_equal(reloaded.vectors, expected.vectors):
        problems.append("Reloaded vectors do not match what was generated.")
    if len(reloaded.chunk_ids) != len(reloaded.vectors):
        problems.append("Reloaded chunk_ids count does not match reloaded vector row count.")
    if reloaded.dimension != expected.dimension:
        problems.append("Reloaded metadata dimension does not match expected dimension.")
    return problems


# ================================================================
# Failure-safe persistence
# ================================================================

def _atomic_save_embeddings(result: EmbeddingResult, output_root: Path, metadata: dict) -> None:
    """
    Write to temp files, reload and validate FROM THOSE TEMP FILES,
    and only replace the final files if that validation passes. A
    failure at any point leaves whatever final files existed before
    this call completely untouched -- a previously valid cache
    survives a failed rebuild attempt, matching build.py's own
    failure-safety guarantee for document JSONLs.

    KNOWN LIMITATION (documented, not fixed -- see Priority 4):
    the three files below are replaced with three independent
    os.replace() calls, not a single atomic multi-file transaction.
    A crash between them (e.g. after replacing embeddings.npy but
    before chunk_ids.json) can leave a genuinely inconsistent final
    state on disk. True multi-file atomicity would need a
    journal/lock-file scheme -- excessive complexity for this
    project's scale. Instead, this is caught structurally on the
    NEXT run: _is_cache_valid() now fully content-validates the
    persisted files (not just their metadata) before ever reusing
    them, so a partially-replaced, inconsistent cache is detected
    and safely rebuilt rather than silently trusted.
    """
    output_root.mkdir(parents=True, exist_ok=True)

    vectors_path = output_root / "embeddings.npy"
    ids_path = output_root / "chunk_ids.json"
    meta_path = output_root / "embeddings_metadata.json"

    tmp_vectors = output_root / "embeddings.npy.tmp"
    tmp_ids = output_root / "chunk_ids.json.tmp"
    tmp_meta = output_root / "embeddings_metadata.json.tmp"

    try:
        # Passing an open file HANDLE (not a path string) to np.save
        # avoids numpy silently appending an extra ".npy" suffix to
        # a filename that doesn't already end in ".npy" -- which
        # "embeddings.npy.tmp" does not (it ends in ".tmp").
        with open(tmp_vectors, "wb") as f:
            np.save(f, result.vectors)
        with open(tmp_ids, "w", encoding="utf-8") as f:
            json.dump(result.chunk_ids, f, indent=2)
        with open(tmp_meta, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, sort_keys=True)

        reloaded = _load_from_paths(tmp_vectors, tmp_ids, tmp_meta)
        problems = _validate_persisted_matches(reloaded, result)
        if problems:
            raise EmbeddingValidationError(problems)

    except Exception:
        for p in (tmp_vectors, tmp_ids, tmp_meta):
            p.unlink(missing_ok=True)
        raise

    tmp_vectors.replace(vectors_path)
    tmp_ids.replace(ids_path)
    tmp_meta.replace(meta_path)


# ================================================================
# Corpus-level caching
# ================================================================

def _read_existing_metadata(output_root: Path) -> dict | None:
    """None for missing OR corrupt metadata -- either way, treated
    as 'no valid cache' rather than crashing the run."""
    meta_path = output_root / "embeddings_metadata.json"
    if not meta_path.is_file():
        return None
    try:
        data = json.loads(meta_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
    return data if isinstance(data, dict) else None


def _is_cache_valid(
    existing_metadata: dict | None,
    output_root: Path,
    input_hash: str,
    model_name: str,
    model_revision: str,
    config: dict,
) -> tuple[bool, str]:
    """
    Returns (is_valid, reason) -- reason is always a human-readable
    string: why the cache was rejected, or "valid" on success. Never
    raises; a corrupt or stale cache is reported and treated as
    absent, not a fatal error.

    Metadata-field comparison alone is NOT sufficient (the previous
    version's whole check): metadata claiming a match proves nothing
    about whether the .npy/.json files it describes are actually
    intact. The final step here loads_embeddings() the real files
    through the fully content-validating _load_from_paths(), so a
    cache is only ever reused when its ACTUAL persisted content --
    not just its metadata's claims about that content -- checks out.
    """
    if existing_metadata is None:
        return False, "no existing metadata found"

    required_files = [
        output_root / "embeddings.npy",
        output_root / "chunk_ids.json",
        output_root / "embeddings_metadata.json",
    ]
    missing = [p.name for p in required_files if not p.is_file()]
    if missing:
        return False, f"missing cache file(s): {missing}"

    metadata_checks = {
        "input_file_sha256": input_hash,
        "model_name": model_name,
        "model_revision": model_revision,
        "embedding_config": config,
        "format_version": FORMAT_VERSION,
    }
    for key, expected in metadata_checks.items():
        actual = existing_metadata.get(key)
        if actual != expected:
            return False, f"metadata field '{key}' changed (cached={actual!r}, current={expected!r})"

    try:
        load_embeddings(output_root)
    except EmbeddingValidationError as e:
        return False, f"persisted cache content failed validation: {e}"
    except Exception as e:
        return False, f"persisted cache could not be loaded: {e}"

    return True, "valid"


# ================================================================
# Orchestration
# ================================================================

def run_embedding(
    chunks_file: str | Path = "data/chunks/all_chunks.jsonl",
    output_root: str | Path = "data/embeddings",
    model_name: str = DEFAULT_MODEL_NAME,
    model_revision: str = DEFAULT_MODEL_REVISION,
    batch_size: int = DEFAULT_BATCH_SIZE,
    force: bool = False,
    reject_empty_text: bool = False,
    model=None,
) -> dict:
    """
    model is injectable (defaults to loading the real one via
    load_model()) so this orchestration function -- and by extension
    the CLI -- is fully testable without the real model ever loading.
    """
    chunks_file = Path(chunks_file)
    output_root = Path(output_root)

    chunks = load_chunks(chunks_file)

    if not chunks:
        summary = {
            "status": "empty", "chunk_count": 0, "dimension": 0,
            "model_name": model_name, "model_revision": model_revision,
            "output_root": None,
        }
        _print_summary(summary)
        return summary

    for warning in validate_chunks_for_embedding(chunks, reject_empty_text=reject_empty_text):
        logger.warning(warning)

    input_hash = _sha256_file(chunks_file)
    config = _embedding_config_fingerprint()

    if not force:
        existing_metadata = _read_existing_metadata(output_root)
        is_valid, reason = _is_cache_valid(existing_metadata, output_root, input_hash, model_name, model_revision, config)
        if is_valid:
            loaded = load_embeddings(output_root)
            summary = {
                "status": "reused", "chunk_count": len(loaded.chunk_ids), "dimension": loaded.dimension,
                "model_name": loaded.model_name, "model_revision": loaded.model_revision,
                "output_root": str(output_root),
            }
            _print_summary(summary)
            return summary
        else:
            logger.info(f"Embedding cache not reused ({output_root}): {reason}")

    if model is None:
        model = load_model(model_name, model_revision)

    result = embed_corpus(chunks, model, model_name, model_revision, batch_size)
    validate_embedding_output(result)

    metadata = {
        "input_chunks_file": str(chunks_file),
        "input_file_sha256": input_hash,
        "model_name": model_name,
        "model_revision": model_revision,
        "dimension": result.dimension,
        "count": len(result.chunk_ids),
        "batch_size": batch_size,
        "normalize_embeddings": True,
        "query_prefix": QUERY_PREFIX,
        "embedding_config": config,
        "format_version": FORMAT_VERSION,
        "library_versions": _library_versions(),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }

    _atomic_save_embeddings(result, output_root, metadata)

    summary = {
        "status": "rebuilt", "chunk_count": len(result.chunk_ids), "dimension": result.dimension,
        "model_name": model_name, "model_revision": model_revision,
        "output_root": str(output_root),
    }
    _print_summary(summary)
    return summary


def _print_summary(summary: dict) -> None:
    print()
    print("=" * 60)
    print("EMBEDDING GENERATION")
    print("=" * 60)
    print(f"Status                : {summary['status']}")
    print(f"Model                 : {summary['model_name']}")
    print(f"Model revision        : {summary['model_revision']}")
    print(f"Chunks embedded       : {summary['chunk_count']}")
    print(f"Embedding dimension   : {summary['dimension']}")
    if summary.get("output_root"):
        print(f"Output written to     : {summary['output_root']}")
    print("=" * 60)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    parser = argparse.ArgumentParser(description="Generate (or reuse cached) embeddings for the chunk corpus.")
    parser.add_argument("--chunks-file", default="data/chunks/all_chunks.jsonl", help="Default: data/chunks/all_chunks.jsonl")
    parser.add_argument("--output-root", default="data/embeddings", help="Default: data/embeddings")
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--model-revision", default=DEFAULT_MODEL_REVISION)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--force", action="store_true", help="Bypass the cache and regenerate embeddings regardless of cache validity.")
    parser.add_argument(
        "--reject-empty-text", action="store_true",
        help="Treat empty/whitespace-only chunk text as a hard validation error instead of "
             "a logged warning. Default (unset): allowed, since the chunker deliberately "
             "preserves near-empty chunks rather than dropping them.",
    )
    args = parser.parse_args()

    try:
        run_embedding(
            chunks_file=args.chunks_file,
            output_root=args.output_root,
            model_name=args.model_name,
            model_revision=args.model_revision,
            batch_size=args.batch_size,
            force=args.force,
            reject_empty_text=args.reject_empty_text,
        )
    except EmbeddingValidationError as e:
        print(f"\nEMBEDDING FAILED:\n{e}\n", file=sys.stderr)
        sys.exit(1)