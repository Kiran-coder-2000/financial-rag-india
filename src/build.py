"""
src/build.py

Top-level project build orchestrator: a lightweight caching layer on
top of the existing chunking pipeline. Reuses chunker.py,
table_chunker.py, build_dataset.py, and batch_builder.py exactly as
they are -- nothing about text chunking, table chunking, chunk IDs,
or validation is reimplemented or changed here.

    python -m src.build              # normal build, reuse valid caches
    python -m src.build --force      # rebuild every document regardless of cache

Produces, under --output-root (default data/chunks):
    <company_slug>_<fy_slug>.jsonl            persistent per-document chunk cache
    <company_slug>_<fy_slug>.manifest.json    cache-validity fingerprint
    all_chunks.jsonl                          derived master corpus

Cache validity is never "does the .jsonl file exist" alone. A
document is only reused when a fresh fingerprint of its
processed-document inputs (document_metadata.json,
page_segments.json, table_metadata.json, tables/*.csv), the current
chunking configuration, and this file's own BUILD_VERSION all match
what's recorded in its manifest.

Failure safety: a document whose rebuild fails keeps its previous
valid .jsonl/.manifest.json untouched. If ANY document ends a run
without a currently-valid cache (a fresh failure with nothing to
fall back on, or a failure with no prior cache at all), all_chunks
.jsonl is NOT regenerated this run -- whatever existed before is
left in place, and the build reports failure clearly. all_chunks
.jsonl is only ever rewritten when every discovered document has a
valid cache to draw from.
"""

import argparse
import hashlib
import json
import logging
import sys
from pathlib import Path

# build_dataset.py (and chunker.py/table_chunker.py in turn) use
# flat imports among themselves, requiring src/chunking itself to be
# on sys.path directly. batch_builder.py already carries this same
# fix at its own top, which runs as a side effect of the import
# below -- but it's harmless and kept explicit here too, so this
# file doesn't silently depend on import-order inheritance from a
# sibling module.
sys.path.insert(0, str(Path(__file__).resolve().parent / "chunking"))

from .chunking import batch_builder, chunker
from .chunking import table_chunker
from .chunking.build_dataset import (
    _slugify,
    load_table_metadata,
    validate_chunks,
)

logger = logging.getLogger(__name__)

# Bumped whenever build.py's OWN cache schema or orchestration logic
# changes in a way that should invalidate every existing cache (e.g.
# a change to what the manifest tracks) -- independent of the
# chunking-config fingerprint below, which tracks chunker.py's/
# table_chunker.py's own tunable constants instead.
BUILD_VERSION = "1.0"

REQUIRED_INPUT_FILES = ["document_metadata.json", "page_segments.json", "table_metadata.json"]


def _chunking_config_fingerprint() -> dict:
    """
    A live snapshot of the chunking pipeline's own tunable
    constants, read directly from the modules -- never hand-copied
    -- so a config change (e.g. someone edits DEFAULT_CHUNK_SIZE) is
    detected automatically, with no separate version number to
    remember to bump by hand.
    """
    return {
        "chunk_size": chunker.DEFAULT_CHUNK_SIZE,
        "chunk_overlap": chunker.DEFAULT_CHUNK_OVERLAP,
        "min_chunk_tokens": chunker.MIN_CHUNK_TOKENS,
        "table_split_threshold": table_chunker.TABLE_SPLIT_THRESHOLD,
        "overhead_margin": table_chunker.OVERHEAD_MARGIN,
        "min_row_budget": table_chunker.MIN_ROW_BUDGET,
        "wide_table_column_threshold": table_chunker.WIDE_TABLE_COLUMN_THRESHOLD,
    }


def _hash_document_inputs(document_path: Path) -> str:
    """
    Deterministic sha256 over every relevant processed-document
    input file's content: the three required JSON files plus every
    tables/*.csv, sorted by relative path so filesystem iteration
    order can never affect the result. Any single-byte change to any
    of these files changes the hash.
    """
    files: list[Path] = []

    for name in REQUIRED_INPUT_FILES:
        p = document_path / name
        if p.is_file():
            files.append(p)

    tables_dir = document_path / "tables"
    if tables_dir.is_dir():
        files.extend(tables_dir.glob("*.csv"))

    files.sort(key=lambda p: str(p.relative_to(document_path)))

    hasher = hashlib.sha256()
    for f in files:
        relative = str(f.relative_to(document_path)).encode("utf-8")
        hasher.update(len(relative).to_bytes(4, "big"))
        hasher.update(relative)
        content = f.read_bytes()
        hasher.update(len(content).to_bytes(8, "big"))
        hasher.update(content)

    return hasher.hexdigest()


def _document_slug(document_path: Path) -> tuple[str, dict]:
    """(slug, document_metadata) for a document -- slug matches the
    exact company_slug/fy_slug convention build_dataset.py already
    uses internally, reusing its own _slugify rather than
    reimplementing the naming rule a second time."""
    meta = chunker.load_document_metadata(document_path)
    company_slug = _slugify(meta["company"])
    financial_year = meta["financial_year"]
    fy_slug = _slugify(financial_year.split("-")[-1] if "-" in financial_year else financial_year)
    return f"{company_slug}_{fy_slug}", meta


def _manifest_path(output_root: Path, slug: str) -> Path:
    return output_root / f"{slug}.manifest.json"


def _jsonl_path(output_root: Path, slug: str) -> Path:
    return output_root / f"{slug}.jsonl"


def _read_manifest(path: Path) -> dict | None:
    """None for a missing OR corrupt manifest -- either way, treated
    as "no valid cache" rather than crashing the build."""
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


def _is_cache_valid(manifest: dict | None, jsonl_path: Path, input_hash: str, config: dict) -> bool:
    if manifest is None:
        return False
    if not jsonl_path.is_file() or jsonl_path.stat().st_size == 0:
        return False
    return (
        manifest.get("input_hash") == input_hash
        and manifest.get("config_fingerprint") == config
        and manifest.get("build_version") == BUILD_VERSION
    )


def _write_jsonl_atomic(path: Path, chunks: list[dict]) -> None:
    """Write to a temp file first, then atomically replace the
    target -- a crash or failure mid-write can never leave a
    corrupt/partial .jsonl in place of a previously valid one."""
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with open(tmp_path, "w", encoding="utf-8") as f:
        for chunk in chunks:
            f.write(json.dumps(chunk, ensure_ascii=False) + "\n")
    tmp_path.replace(path)


def _write_manifest_atomic(path: Path, manifest: dict) -> None:
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    tmp_path.replace(path)


def process_single_document(document_path: Path, output_root: Path, force: bool) -> dict:
    """
    Decide reuse vs. rebuild for one document, and act on it.

    Returns:
        {
            "slug": str,
            "status": "reused" | "rebuilt" | "failed",
            "text_chunks": int, "table_chunks": int, "total_chunks": int,
            "error": str | None,
            "has_valid_cache": bool,  # True if a usable .jsonl exists
                                      # RIGHT NOW, whether from this run
                                      # or a preserved previous one
        }

    A rebuild failure never destroys a previously valid .jsonl/
    manifest -- they're left exactly as they were, and "has_valid_cache"
    reflects whether that previous cache still exists.
    """
    output_root.mkdir(parents=True, exist_ok=True)

    slug, _meta = _document_slug(document_path)
    manifest_path = _manifest_path(output_root, slug)
    jsonl_path = _jsonl_path(output_root, slug)

    input_hash = _hash_document_inputs(document_path)
    config = _chunking_config_fingerprint()

    existing_manifest = _read_manifest(manifest_path)
    had_valid_cache_before = _is_cache_valid(existing_manifest, jsonl_path, input_hash, config)

    if not force and had_valid_cache_before:
        return {
            "slug": slug,
            "status": "reused",
            "text_chunks": existing_manifest["text_chunks"],
            "table_chunks": existing_manifest["table_chunks"],
            "total_chunks": existing_manifest["total_chunks"],
            "error": None,
            "has_valid_cache": True,
        }

    # Cache missing/stale, or --force: (re)run the existing,
    # unmodified single-document builder.
    result = batch_builder.process_document(document_path)

    if result["status"] == "failed":
        logger.error(f"{slug}: rebuild failed -- {result['error']}")
        # Previous valid cache (if any) is untouched -- we never
        # wrote anything new, so nothing to preserve-by-inaction.
        prior_still_valid = _is_cache_valid(_read_manifest(manifest_path), jsonl_path, input_hash, config)
        return {
            "slug": slug, "status": "failed",
            "text_chunks": 0, "table_chunks": 0, "total_chunks": 0,
            "error": result["error"], "has_valid_cache": prior_still_valid,
        }

    chunks = result["chunks"]

    # Reuse the existing validator exactly as build_dataset.py's own
    # __main__ block does -- not reimplemented here.
    page_segments = chunker.load_page_segments(document_path)
    table_metadata = load_table_metadata(document_path)
    problems = validate_chunks(chunks, page_segments, table_metadata, extra_warnings=result["warnings"])

    if problems:
        logger.error(f"{slug}: validation failed after rebuild -- {problems}")
        prior_still_valid = _is_cache_valid(_read_manifest(manifest_path), jsonl_path, input_hash, config)
        return {
            "slug": slug, "status": "failed",
            "text_chunks": 0, "table_chunks": 0, "total_chunks": 0,
            "error": f"validation failed: {problems}", "has_valid_cache": prior_still_valid,
        }

    text_chunks = sum(1 for c in chunks if c["chunk_type"] == "text")
    table_chunks = sum(1 for c in chunks if c["chunk_type"] == "table")

    # JSONL written and confirmed on disk BEFORE the manifest is
    # updated to point at it -- so a crash between the two leaves,
    # at worst, a valid-but-not-yet-marked-fresh .jsonl (which just
    # causes one harmless extra rebuild next run), never a manifest
    # that falsely vouches for a .jsonl that isn't actually there.
    _write_jsonl_atomic(jsonl_path, chunks)
    _write_manifest_atomic(manifest_path, {
        "input_hash": input_hash,
        "config_fingerprint": config,
        "build_version": BUILD_VERSION,
        "text_chunks": text_chunks,
        "table_chunks": table_chunks,
        "total_chunks": len(chunks),
    })

    return {
        "slug": slug, "status": "rebuilt",
        "text_chunks": text_chunks, "table_chunks": table_chunks, "total_chunks": len(chunks),
        "error": None, "has_valid_cache": True,
    }


def build_master_jsonl(document_slugs: list[str], output_root: Path) -> int:
    """
    Stream-concatenate each document's already-written .jsonl into
    all_chunks.jsonl, in the given (deterministic) slug order,
    without ever loading a full document's chunks into memory --
    each source file is copied through line by line, and only the
    chunk_id strings (not full chunk objects) are held in memory for
    the duplicate check.

    Returns the total line (chunk) count written. Raises ValueError
    if a duplicate chunk_id is found across documents -- this should
    be structurally impossible given how build_dataset.py assigns
    IDs, so surfacing it loudly rather than silently deduplicating
    is deliberate.
    """
    output_path = output_root / "all_chunks.jsonl"
    tmp_path = output_path.with_suffix(".jsonl.tmp")

    seen_ids: set[str] = set()
    total = 0

    with open(tmp_path, "w", encoding="utf-8") as out_f:
        for slug in document_slugs:
            jsonl_path = _jsonl_path(output_root, slug)
            with open(jsonl_path, "r", encoding="utf-8") as in_f:
                for line in in_f:
                    if not line.strip():
                        continue
                    chunk_id = json.loads(line)["chunk_id"]
                    if chunk_id in seen_ids:
                        tmp_path.unlink(missing_ok=True)
                        raise ValueError(f"Duplicate chunk_id across documents: {chunk_id}")
                    seen_ids.add(chunk_id)
                    out_f.write(line if line.endswith("\n") else line + "\n")
                    total += 1

    tmp_path.replace(output_path)
    return total


def build_project(
    processed_root: str | Path = "data/processed",
    output_root: str | Path = "data/chunks",
    force: bool = False,
) -> dict:
    """
    Full orchestration: discover -> per-document reuse/rebuild ->
    master JSONL (only if every document has a valid cache).

    Returns a summary dict (also what's printed by the CLI) so this
    is directly testable without parsing stdout.
    """
    processed_root = Path(processed_root)
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    document_paths = batch_builder.discover_processed_documents(processed_root)

    results = [process_single_document(p, output_root, force) for p in document_paths]

    reused = [r for r in results if r["status"] == "reused"]
    rebuilt = [r for r in results if r["status"] == "rebuilt"]
    failed = [r for r in results if r["status"] == "failed"]

    all_have_valid_cache = all(r["has_valid_cache"] for r in results)

    master_updated = False
    master_chunk_count = None

    if all_have_valid_cache and results:
        slugs_in_order = [r["slug"] for r in results]
        master_chunk_count = build_master_jsonl(slugs_in_order, output_root)
        master_updated = True
    elif not results:
        logger.warning("No valid processed documents discovered -- nothing to build.")

    total_text = sum(r["text_chunks"] for r in results if r["has_valid_cache"])
    total_table = sum(r["table_chunks"] for r in results if r["has_valid_cache"])
    total_chunks = total_text + total_table

    success = all_have_valid_cache and bool(results)

    summary = {
        "documents_discovered": len(results),
        "documents_reused": len(reused),
        "documents_rebuilt": len(rebuilt),
        "documents_failed": len(failed),
        "text_chunks": total_text,
        "table_chunks": total_table,
        "total_chunks": total_chunks,
        "master_updated": master_updated,
        "master_chunk_count": master_chunk_count,
        "success": success,
        "failed_documents": [{"slug": r["slug"], "error": r["error"]} for r in failed],
    }

    _print_summary(summary)

    return summary


def _print_summary(summary: dict) -> None:
    print()
    print("=" * 60)
    print("FINANCIAL RAG BUILD")
    print("=" * 60)
    print(f"Documents discovered : {summary['documents_discovered']}")
    print(f"Documents reused     : {summary['documents_reused']}")
    print(f"Documents rebuilt    : {summary['documents_rebuilt']}")
    print(f"Documents failed     : {summary['documents_failed']}")
    print()
    print(f"Text chunks          : {summary['text_chunks']}")
    print(f"Table chunks         : {summary['table_chunks']}")
    print(f"Total chunks         : {summary['total_chunks']}")
    print()
    if summary["master_updated"]:
        print(f"all_chunks.jsonl     : updated ({summary['master_chunk_count']} chunks)")
    else:
        print("all_chunks.jsonl     : NOT updated (previous version preserved, if any)")
    print()
    if summary["failed_documents"]:
        print("Failed documents:")
        for f in summary["failed_documents"]:
            print(f"  - {f['slug']}: {f['error']}")
        print()
    print("BUILD SUCCESS" if summary["success"] else "BUILD FAILED")
    print("=" * 60)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    parser = argparse.ArgumentParser(
        description="Build (or reuse cached) document-level chunk JSONLs and the master corpus."
    )
    parser.add_argument("--force", action="store_true", help="Rebuild every document regardless of cache state.")
    parser.add_argument("--processed-root", default="data/processed", help="Default: data/processed")
    parser.add_argument("--output-root", default="data/chunks", help="Default: data/chunks")
    args = parser.parse_args()

    summary = build_project(args.processed_root, args.output_root, args.force)

    sys.exit(0 if summary["success"] else 1)