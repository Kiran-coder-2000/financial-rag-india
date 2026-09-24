"""
src/chunking/batch_builder.py

Orchestrates build_dataset.py across every valid processed document
under data/processed/, producing per-document JSONL files and one
corpus-level all_chunks.jsonl.

This file currently implements document discovery and per-document
processing (Steps 4-5 of the agreed incremental plan). JSONL
writing, corpus validation, and statistics are added in later steps.

Responsibility boundary: this module discovers documents and
orchestrates calls to build_dataset.py -- it never duplicates
chunking logic, and never assigns chunk_ids (that stays inside
build_dataset.py, scoped per-document, so adding a new document can
never change an existing document's IDs).
"""

import argparse
import logging
import os
from pathlib import Path

from .build_dataset import build_dataset
from .chunker import load_document_metadata

logger = logging.getLogger(__name__)

# A directory must contain all of these to be considered a valid
# processed document. "tables" is checked as a directory, the rest
# as files.
REQUIRED_FILES = ["document_metadata.json", "page_segments.json", "table_metadata.json"]
REQUIRED_DIRS = ["tables"]


def _is_valid_processed_document(path: Path) -> bool:
    """
    True if `path` contains every expected processed-document
    artifact. Checked by presence, not by directory depth or naming
    convention -- so data/processed/hdfcbank/2025/ is discovered
    automatically the same way data/processed/relianceindustries/2025/
    is, with no code change and no hardcoded company/path pattern.
    """
    if not path.is_dir():
        return False
    if not all((path / f).is_file() for f in REQUIRED_FILES):
        return False
    if not all((path / d).is_dir() for d in REQUIRED_DIRS):
        return False
    return True


def discover_processed_documents(processed_root: str | Path) -> list[Path]:
    """
    Recursively find every valid processed-document directory under
    processed_root, in deterministic order.

    Order is determined by each candidate's OWN document_metadata.json
    -- (company, financial_year, document, str(path)) -- not by
    filesystem iteration order, which isn't guaranteed stable across
    OSes or reruns. The path is included only as a final tiebreaker
    for two documents that otherwise sort identically.

    A directory that looks like a processed document but is missing
    a required artifact, or whose document_metadata.json can't be
    read, is skipped and logged -- never silently ignored, and never
    allowed to abort discovery of the other valid documents.
    """
    processed_root = Path(processed_root)

    if not processed_root.is_dir():
        logger.warning(f"processed_root does not exist or is not a directory: {processed_root}")
        return []

    candidates: list[Path] = []
    incomplete_count = 0

    # os.walk with pruning, rather than Path.rglob("*"): once a
    # directory is confirmed valid, there's no need to descend into
    # its own subdirectories (notably tables/, which can hold
    # hundreds of per-table CSVs per document) -- pruning avoids
    # scanning all of that for nothing.
    for dirpath, dirnames, _filenames in os.walk(processed_root):
        current = Path(dirpath)

        if _is_valid_processed_document(current):
            candidates.append(current)
            dirnames.clear()  # don't descend further into this document
            continue

        if any((current / f).exists() for f in REQUIRED_FILES + REQUIRED_DIRS):
            # Looks like it was INTENDED to be a processed document
            # (has at least one expected artifact) but is missing
            # others -- worth flagging distinctly from an unrelated
            # directory that has none of the artifacts at all.
            incomplete_count += 1
            logger.warning(f"Incomplete processed-document directory, skipping: {current}")

    if incomplete_count:
        logger.info(f"Skipped {incomplete_count} incomplete director(y/ies) during discovery.")

    # Read each candidate's metadata once, up front, so a candidate
    # whose document_metadata.json can't be read is EXCLUDED here --
    # not kept in the list with a placeholder sort key. Sorting and
    # filtering are deliberately separate passes: a sort key's
    # exception handler is the wrong place to decide inclusion,
    # since a key must always be returned for every element sort()
    # is given, which is exactly how the previous version's
    # placeholder key leaked an unreadable document back in.
    readable: list[tuple[Path, dict]] = []
    for doc_path in candidates:
        try:
            meta = load_document_metadata(doc_path)
        except (FileNotFoundError, ValueError, OSError) as e:
            logger.warning(
                f"Skipping {doc_path}: document_metadata.json exists "
                f"but could not be read ({e})."
            )
            continue
        readable.append((doc_path, meta))

    readable.sort(
        key=lambda item: (
            item[1].get("company", ""),
            item[1].get("financial_year", ""),
            item[1].get("document", ""),
            str(item[0]),
        )
    )

    result = [doc_path for doc_path, _meta in readable]

    logger.info(f"Discovered {len(result)} valid processed document(s) under {processed_root}.")

    return result


def process_document(document_path: Path) -> dict:
    """
    Run the existing single-document builder (build_dataset.py,
    untouched) against one discovered document directory.

    Never raises -- any exception from build_dataset() (missing
    file, malformed metadata, etc.) is caught here and reported in
    the returned dict's "error" field, so one bad document can never
    abort processing of the rest of the corpus.

    Returns:
        {
            "path": Path,
            "status": "success" | "failed",
            "chunks": list[dict] | None,   # None on failure
            "warnings": list[str],         # from build_dataset itself
            "error": str | None,
        }
    """
    try:
        chunks, warnings = build_dataset(document_path)
    except Exception as e:
        logger.error(f"Failed to process {document_path}: {e}")
        return {
            "path": document_path,
            "status": "failed",
            "chunks": None,
            "warnings": [],
            "error": str(e),
        }

    return {
        "path": document_path,
        "status": "success",
        "chunks": chunks,
        "warnings": warnings,
        "error": None,
    }


def process_all_documents(processed_root: str | Path) -> tuple[list[dict], list[dict]]:
    """
    Discover every valid processed document under processed_root
    and run process_document() on each, in the deterministic order
    discover_processed_documents() returns.

    A failure in one document is reported clearly and excludes only
    that document -- it never aborts processing of the others, and
    it's never silently dropped from the summary.

    Returns (succeeded, failed) -- both lists of the per-document
    result dicts process_document() produces. Chunks are not yet
    combined or written anywhere; that's the next step.
    """
    processed_root = Path(processed_root)

    document_paths = discover_processed_documents(processed_root)

    succeeded: list[dict] = []
    failed: list[dict] = []

    for document_path in document_paths:
        result = process_document(document_path)
        if result["status"] == "success":
            succeeded.append(result)
        else:
            failed.append(result)

    logger.info(
        f"Documents discovered: {len(document_paths)}. "
        f"Succeeded: {len(succeeded)}. Failed: {len(failed)}."
    )

    if failed:
        for result in failed:
            logger.warning(f"  Failed: {result['path']} -- {result['error']}")

    return succeeded, failed

def main() -> None:
    """Run batch processing from the command line."""
    parser = argparse.ArgumentParser(
        description="Process all valid documents under a processed-data directory."
    )
    parser.add_argument(
        "processed_root",
        nargs="?",
        default="data/processed",
        help="Root directory containing processed documents "
             "(default: data/processed)",
    )

    args = parser.parse_args()

    succeeded, failed = process_all_documents(args.processed_root)

    total_text_chunks = 0
    total_table_chunks = 0

    for result in succeeded:
        chunks = result["chunks"] or []

        for chunk in chunks:
            if chunk.get("chunk_type") == "text":
                total_text_chunks += 1
            elif chunk.get("chunk_type") == "table":
                total_table_chunks += 1

    total_chunks = total_text_chunks + total_table_chunks

    print("\n" + "=" * 60)
    print("BATCH CHUNKING SUMMARY")
    print("=" * 60)
    print(f"Documents discovered : {len(succeeded) + len(failed)}")
    print(f"Documents succeeded  : {len(succeeded)}")
    print(f"Documents failed     : {len(failed)}")
    print(f"Text chunks          : {total_text_chunks}")
    print(f"Table chunks         : {total_table_chunks}")
    print(f"Total chunks         : {total_chunks}")
    print("=" * 60)


if __name__ == "__main__":
    main()