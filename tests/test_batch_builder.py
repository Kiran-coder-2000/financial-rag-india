"""
tests/test_batch_builder.py

Tests for discover_processed_documents() only -- no other
batch_builder.py functionality exists yet, per the incremental plan.

All cases use tmp_path-built fake processed-document trees, never
the real corpus.
"""

import json
import sys
from pathlib import Path

import pytest

# batch_builder.py imports its sibling modules with a relative
# import ("from .chunker import ...") and so must be imported as
# part of the src.chunking package -- not as a flat top-level
# module the way the other test files import chunker/table_chunker/
# build_dataset (that flat style relies on conftest.py adding
# src/chunking itself to sys.path). Adding the PROJECT ROOT here
# instead lets `from src.chunking import batch_builder` resolve.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.chunking import batch_builder


def _make_valid_document(
    processed_root: Path,
    company_dir: str,
    fy_dir: str,
    company: str,
    financial_year: str,
    document: str = "Annual Report",
) -> Path:
    """Build a minimal-but-complete fake processed-document directory."""
    doc_dir = processed_root / company_dir / fy_dir
    (doc_dir / "tables").mkdir(parents=True)
    (doc_dir / "document_metadata.json").write_text(json.dumps({
        "company": company, "document": document,
        "financial_year": financial_year, "parser_version": "1.0",
    }))
    (doc_dir / "page_segments.json").write_text(json.dumps([{"page": 1, "text": "content"}]))
    (doc_dir / "table_metadata.json").write_text(json.dumps([]))
    return doc_dir


class TestDiscoversValidDocuments:
    def test_finds_all_valid_documents(self, tmp_path):
        _make_valid_document(tmp_path, "reliance", "2025", "RelianceIndustries", "2024-2025")
        _make_valid_document(tmp_path, "infosys", "2026", "Infosys", "2025-2026")

        result = batch_builder.discover_processed_documents(tmp_path)

        assert len(result) == 2

    def test_returns_path_objects(self, tmp_path):
        doc_dir = _make_valid_document(tmp_path, "reliance", "2025", "RelianceIndustries", "2024-2025")

        result = batch_builder.discover_processed_documents(tmp_path)

        assert result == [doc_dir]
        assert isinstance(result[0], Path)


class TestIgnoresIncompleteAndUnrelatedDirectories:
    def test_missing_table_metadata_is_excluded(self, tmp_path):
        incomplete = tmp_path / "hdfcbank" / "2025"
        (incomplete / "tables").mkdir(parents=True)
        (incomplete / "document_metadata.json").write_text(json.dumps({"company": "HDFCBank"}))
        (incomplete / "page_segments.json").write_text("[]")
        # table_metadata.json deliberately missing

        result = batch_builder.discover_processed_documents(tmp_path)

        assert result == []

    def test_missing_tables_directory_is_excluded(self, tmp_path):
        incomplete = tmp_path / "hdfcbank" / "2025"
        incomplete.mkdir(parents=True)
        (incomplete / "document_metadata.json").write_text(json.dumps({"company": "HDFCBank"}))
        (incomplete / "page_segments.json").write_text("[]")
        (incomplete / "table_metadata.json").write_text("[]")
        # tables/ directory deliberately missing

        result = batch_builder.discover_processed_documents(tmp_path)

        assert result == []

    def test_unrelated_directory_with_no_artifacts_ignored(self, tmp_path):
        unrelated = tmp_path / "random_notes"
        unrelated.mkdir()
        (unrelated / "readme.txt").write_text("not a processed document")
        valid = _make_valid_document(tmp_path, "reliance", "2025", "RelianceIndustries", "2024-2025")

        result = batch_builder.discover_processed_documents(tmp_path)

        assert result == [valid]

    def test_valid_documents_still_found_alongside_incomplete_one(self, tmp_path):
        incomplete = tmp_path / "hdfcbank" / "2025"
        (incomplete / "tables").mkdir(parents=True)
        (incomplete / "document_metadata.json").write_text(json.dumps({"company": "HDFCBank"}))
        # page_segments.json and table_metadata.json missing

        _make_valid_document(tmp_path, "reliance", "2025", "RelianceIndustries", "2024-2025")
        _make_valid_document(tmp_path, "infosys", "2026", "Infosys", "2025-2026")

        result = batch_builder.discover_processed_documents(tmp_path)

        assert len(result) == 2


class TestExcludesUnreadableMetadata:
    def test_corrupt_document_metadata_json_is_excluded(self, tmp_path):
        corrupt = tmp_path / "corruptco" / "2025"
        (corrupt / "tables").mkdir(parents=True)
        (corrupt / "document_metadata.json").write_text("{not valid json")
        (corrupt / "page_segments.json").write_text("[]")
        (corrupt / "table_metadata.json").write_text("[]")

        valid = _make_valid_document(tmp_path, "reliance", "2025", "RelianceIndustries", "2024-2025")

        result = batch_builder.discover_processed_documents(tmp_path)

        assert result == [valid]

    def test_corrupt_metadata_logs_a_warning(self, tmp_path, caplog):
        corrupt = tmp_path / "corruptco" / "2025"
        (corrupt / "tables").mkdir(parents=True)
        (corrupt / "document_metadata.json").write_text("{not valid json")
        (corrupt / "page_segments.json").write_text("[]")
        (corrupt / "table_metadata.json").write_text("[]")

        with caplog.at_level("WARNING"):
            batch_builder.discover_processed_documents(tmp_path)

        assert any("corruptco" in record.message for record in caplog.records)

    def test_other_documents_unaffected_by_one_corrupt_document(self, tmp_path):
        corrupt = tmp_path / "corruptco" / "2025"
        (corrupt / "tables").mkdir(parents=True)
        (corrupt / "document_metadata.json").write_text("{not valid json")
        (corrupt / "page_segments.json").write_text("[]")
        (corrupt / "table_metadata.json").write_text("[]")

        _make_valid_document(tmp_path, "reliance", "2025", "RelianceIndustries", "2024-2025")
        _make_valid_document(tmp_path, "infosys", "2026", "Infosys", "2025-2026")
        _make_valid_document(tmp_path, "tcs", "2025", "TataConsultancyServices", "2024-2025")

        result = batch_builder.discover_processed_documents(tmp_path)

        assert len(result) == 3


class TestDeterministicOrder:
    def test_order_is_alphabetical_by_company_not_filesystem_order(self, tmp_path):
        # Deliberately created in NON-alphabetical order, and with
        # directory names that don't match company name casing/order,
        # to prove the sort key comes from metadata content, not the
        # filesystem.
        _make_valid_document(tmp_path, "zzz_tcs_dir", "2025", "TataConsultancyServices", "2024-2025")
        _make_valid_document(tmp_path, "aaa_infosys_dir", "2026", "Infosys", "2025-2026")
        _make_valid_document(tmp_path, "mmm_reliance_dir", "2025", "RelianceIndustries", "2024-2025")

        result = batch_builder.discover_processed_documents(tmp_path)
        companies = [json.loads((d / "document_metadata.json").read_text())["company"] for d in result]

        assert companies == ["Infosys", "RelianceIndustries", "TataConsultancyServices"]

    def test_repeated_calls_produce_identical_order(self, tmp_path):
        _make_valid_document(tmp_path, "tcs", "2025", "TataConsultancyServices", "2024-2025")
        _make_valid_document(tmp_path, "infosys", "2026", "Infosys", "2025-2026")
        _make_valid_document(tmp_path, "reliance", "2025", "RelianceIndustries", "2024-2025")

        runs = [batch_builder.discover_processed_documents(tmp_path) for _ in range(5)]

        assert all(r == runs[0] for r in runs)

    def test_same_company_sorted_by_financial_year(self, tmp_path):
        _make_valid_document(tmp_path, "reliance", "2026", "RelianceIndustries", "2025-2026")
        _make_valid_document(tmp_path, "reliance_prior", "2025", "RelianceIndustries", "2024-2025")

        result = batch_builder.discover_processed_documents(tmp_path)
        years = [json.loads((d / "document_metadata.json").read_text())["financial_year"] for d in result]

        assert years == ["2024-2025", "2025-2026"]


class TestEmptyAndMissingInput:
    def test_empty_processed_root_returns_empty_list(self, tmp_path):
        result = batch_builder.discover_processed_documents(tmp_path)
        assert result == []

    def test_nonexistent_processed_root_returns_empty_list(self, tmp_path):
        result = batch_builder.discover_processed_documents(tmp_path / "does_not_exist")
        assert result == []

    def test_processed_root_that_is_a_file_returns_empty_list(self, tmp_path):
        not_a_dir = tmp_path / "not_a_directory.txt"
        not_a_dir.write_text("x")
        result = batch_builder.discover_processed_documents(not_a_dir)
        assert result == []


class TestPruning:
    def test_does_not_treat_nested_content_inside_tables_as_a_candidate(self, tmp_path):
        """Once a document directory is confirmed valid, discovery
        must not descend into its own tables/ folder looking for
        further (spurious) candidates -- verified here by planting
        files inside tables/ that would themselves look like a
        second valid document if traversal weren't pruned."""
        doc_dir = _make_valid_document(tmp_path, "reliance", "2025", "RelianceIndustries", "2024-2025")

        # Plant a nested structure inside tables/ that mimics the
        # required artifacts, to confirm it's never picked up as a
        # second candidate.
        nested = doc_dir / "tables" / "nested_lookalike"
        (nested / "tables").mkdir(parents=True)
        (nested / "document_metadata.json").write_text(json.dumps({
            "company": "ShouldNotAppear", "document": "X", "financial_year": "2025",
        }))
        (nested / "page_segments.json").write_text("[]")
        (nested / "table_metadata.json").write_text("[]")

        result = batch_builder.discover_processed_documents(tmp_path)

        assert result == [doc_dir]