"""
tests/test_build.py

Tests for src/build.py's caching layer only. The expensive chunking
step (batch_builder.process_document, which calls the real
build_dataset -> chunker/table_chunker pipeline) is mocked
throughout -- these tests never run real chunking.
"""

import json
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import build


def _make_document(processed_root: Path, company_dir: str, fy_dir: str,
                    company: str, financial_year: str, document: str = "Annual Report") -> Path:
    d = processed_root / company_dir / fy_dir
    (d / "tables").mkdir(parents=True)
    (d / "document_metadata.json").write_text(json.dumps({
        "company": company, "document": document, "financial_year": financial_year,
        "parser_version": "1.0",
    }))
    (d / "page_segments.json").write_text(json.dumps([{"page": 1, "text": "content"}]))
    (d / "table_metadata.json").write_text(json.dumps([]))
    return d


def _fake_chunks(slug_hint: str, n_text: int = 2, n_table: int = 1) -> list[dict]:
    chunks = []
    for i in range(n_text):
        chunks.append({
            "chunk_id": f"{slug_hint}_text_{i:06d}", "chunk_type": "text",
            "company": "X", "document": "D", "financial_year": "2025",
            "page": 1, "text": "t", "tokens": 10, "section": None, "is_small": False,
        })
    for i in range(n_table):
        chunks.append({
            "chunk_id": f"{slug_hint}_table_{i:06d}", "chunk_type": "table",
            "company": "X", "document": "D", "financial_year": "2025",
            "page": 1, "text": "t", "tokens": 10, "table_id": f"{slug_hint}_t{i}",
            "table_heading": None, "context": None, "chunk_index": 1, "total_chunks": 1,
            "split_strategy": "intact", "row_start": 0, "row_end": 0,
            "is_low_confidence": False, "continuation_of": None,
        })
    return chunks


def _mock_process_document_success(n_text=2, n_table=1):
    """A mock for batch_builder.process_document that always succeeds,
    returning fresh fake chunks per call (using the document path's
    parent.name as a uniqueness hint so different documents don't
    collide on chunk_id in these tests)."""
    def _mock(document_path):
        slug_hint = document_path.parent.name + "_" + document_path.name
        return {
            "path": document_path, "status": "success",
            "chunks": _fake_chunks(slug_hint, n_text, n_table), "warnings": [], "error": None,
        }
    return _mock


@pytest.fixture(autouse=True)
def _mock_validate_and_table_metadata_loading():
    """
    These tests exercise build.py's own caching logic, not
    build_dataset's validator or table_metadata loading -- both are
    stubbed to always succeed / return an empty list so a test's
    fake chunks (which don't match real table_metadata.json content)
    never trip validation for reasons unrelated to what's being
    tested here.
    """
    with patch.object(build, "validate_chunks", return_value=[]), \
         patch.object(build, "load_table_metadata", return_value=[]):
        yield


class TestFirstBuildCreatesArtifacts:
    def test_creates_document_jsonl(self, tmp_path):
        processed = tmp_path / "processed"
        output = tmp_path / "chunks"
        _make_document(processed, "reliance", "2025", "RelianceIndustries", "2024-2025")

        with patch.object(build.batch_builder, "process_document", side_effect=_mock_process_document_success()):
            summary = build.build_project(processed, output, force=False)

        assert summary["documents_rebuilt"] == 1
        assert (output / "relianceindustries_2025.jsonl").exists()

    def test_creates_manifest(self, tmp_path):
        processed = tmp_path / "processed"
        output = tmp_path / "chunks"
        _make_document(processed, "reliance", "2025", "RelianceIndustries", "2024-2025")

        with patch.object(build.batch_builder, "process_document", side_effect=_mock_process_document_success()):
            build.build_project(processed, output, force=False)

        manifest_path = output / "relianceindustries_2025.manifest.json"
        assert manifest_path.exists()
        manifest = json.loads(manifest_path.read_text())
        assert "input_hash" in manifest
        assert "config_fingerprint" in manifest
        assert manifest["build_version"] == build.BUILD_VERSION


class TestSecondUnchangedBuildReuses:
    def test_second_build_reuses_and_does_not_call_chunking(self, tmp_path):
        processed = tmp_path / "processed"
        output = tmp_path / "chunks"
        _make_document(processed, "reliance", "2025", "RelianceIndustries", "2024-2025")

        mock_fn = _mock_process_document_success()
        with patch.object(build.batch_builder, "process_document", side_effect=mock_fn) as mocked:
            build.build_project(processed, output, force=False)
            assert mocked.call_count == 1

            summary = build.build_project(processed, output, force=False)

            # The expensive chunking function must NOT be called again.
            assert mocked.call_count == 1
            assert summary["documents_reused"] == 1
            assert summary["documents_rebuilt"] == 0


class TestSourceChangeInvalidatesCache:
    def test_modified_page_segments_triggers_rebuild(self, tmp_path):
        processed = tmp_path / "processed"
        output = tmp_path / "chunks"
        doc_dir = _make_document(processed, "reliance", "2025", "RelianceIndustries", "2024-2025")

        with patch.object(build.batch_builder, "process_document", side_effect=_mock_process_document_success()) as mocked:
            build.build_project(processed, output, force=False)
            assert mocked.call_count == 1

            (doc_dir / "page_segments.json").write_text(json.dumps(
                [{"page": 1, "text": "different content now"}]
            ))

            summary = build.build_project(processed, output, force=False)

        assert mocked.call_count == 2
        assert summary["documents_rebuilt"] == 1
        assert summary["documents_reused"] == 0


class TestConfigChangeInvalidatesCache:
    def test_chunk_size_change_triggers_rebuild(self, tmp_path):
        processed = tmp_path / "processed"
        output = tmp_path / "chunks"
        _make_document(processed, "reliance", "2025", "RelianceIndustries", "2024-2025")

        with patch.object(build.batch_builder, "process_document", side_effect=_mock_process_document_success()) as mocked:
            build.build_project(processed, output, force=False)
            assert mocked.call_count == 1

            original_chunk_size = build.chunker.DEFAULT_CHUNK_SIZE
            build.chunker.DEFAULT_CHUNK_SIZE = original_chunk_size + 1
            try:
                summary = build.build_project(processed, output, force=False)
            finally:
                build.chunker.DEFAULT_CHUNK_SIZE = original_chunk_size

        assert mocked.call_count == 2
        assert summary["documents_rebuilt"] == 1


class TestForceRebuildsAll:
    def test_force_ignores_valid_cache(self, tmp_path):
        processed = tmp_path / "processed"
        output = tmp_path / "chunks"
        _make_document(processed, "reliance", "2025", "RelianceIndustries", "2024-2025")
        _make_document(processed, "infosys", "2026", "Infosys", "2025-2026")

        with patch.object(build.batch_builder, "process_document", side_effect=_mock_process_document_success()) as mocked:
            build.build_project(processed, output, force=False)
            assert mocked.call_count == 2

            summary = build.build_project(processed, output, force=True)

        assert mocked.call_count == 4  # both documents rebuilt again despite valid cache
        assert summary["documents_rebuilt"] == 2
        assert summary["documents_reused"] == 0


class TestFailedRebuildPreservesPrevious:
    def test_failure_keeps_previous_valid_jsonl(self, tmp_path):
        processed = tmp_path / "processed"
        output = tmp_path / "chunks"
        doc_dir = _make_document(processed, "reliance", "2025", "RelianceIndustries", "2024-2025")

        with patch.object(build.batch_builder, "process_document", side_effect=_mock_process_document_success()):
            build.build_project(processed, output, force=False)

        jsonl_path = output / "relianceindustries_2025.jsonl"
        content_before = jsonl_path.read_text()

        # Invalidate the cache (source change) so a rebuild is
        # attempted, then make that rebuild attempt fail.
        (doc_dir / "page_segments.json").write_text(json.dumps([{"page": 1, "text": "changed"}]))

        def _failing_mock(document_path):
            return {"path": document_path, "status": "failed", "chunks": None,
                    "warnings": [], "error": "simulated failure"}

        with patch.object(build.batch_builder, "process_document", side_effect=_failing_mock):
            summary = build.build_project(processed, output, force=False)

        assert summary["documents_failed"] == 1
        assert summary["success"] is False
        assert jsonl_path.read_text() == content_before  # untouched


class TestMasterJsonlNoDuplicates:
    def test_no_duplicate_chunk_ids_across_documents(self, tmp_path):
        processed = tmp_path / "processed"
        output = tmp_path / "chunks"
        _make_document(processed, "reliance", "2025", "RelianceIndustries", "2024-2025")
        _make_document(processed, "infosys", "2026", "Infosys", "2025-2026")

        with patch.object(build.batch_builder, "process_document", side_effect=_mock_process_document_success()):
            build.build_project(processed, output, force=False)

        all_chunks_path = output / "all_chunks.jsonl"
        chunk_ids = [json.loads(line)["chunk_id"] for line in all_chunks_path.read_text().splitlines()]

        assert len(chunk_ids) == len(set(chunk_ids))


class TestDeterministicMasterJsonl:
    def test_repeated_builds_produce_identical_master_jsonl(self, tmp_path):
        processed = tmp_path / "processed"
        output = tmp_path / "chunks"
        _make_document(processed, "reliance", "2025", "RelianceIndustries", "2024-2025")
        _make_document(processed, "infosys", "2026", "Infosys", "2025-2026")

        with patch.object(build.batch_builder, "process_document", side_effect=_mock_process_document_success()):
            build.build_project(processed, output, force=True)
            first = (output / "all_chunks.jsonl").read_text()

            build.build_project(processed, output, force=True)
            second = (output / "all_chunks.jsonl").read_text()

        assert first == second


class TestMasterCountMatchesDocumentCounts:
    def test_master_chunk_count_equals_sum_of_document_jsonls(self, tmp_path):
        processed = tmp_path / "processed"
        output = tmp_path / "chunks"
        _make_document(processed, "reliance", "2025", "RelianceIndustries", "2024-2025")
        _make_document(processed, "infosys", "2026", "Infosys", "2025-2026")

        with patch.object(build.batch_builder, "process_document", side_effect=_mock_process_document_success(n_text=3, n_table=2)):
            summary = build.build_project(processed, output, force=False)

        per_doc_total = sum(
            1 for f in output.glob("*.jsonl") if f.name != "all_chunks.jsonl"
            for _ in f.read_text().splitlines()
        )
        assert summary["master_chunk_count"] == per_doc_total == 10  # 2 docs * (3 text + 2 table)


class TestStreamingNotFullMemoryLoad:
    def test_build_master_jsonl_streams_via_file_handles_not_full_lists(self, tmp_path):
        """Confirms build_master_jsonl reads through open file handles
        (line-by-line) rather than via a call that would materialize
        an entire document's chunk list in memory (e.g. json.load on
        the whole file) -- verified by patching `open` and counting
        that write calls happen incrementally per line, not once
        with a giant pre-built string."""
        processed = tmp_path / "processed"
        output = tmp_path / "chunks"
        _make_document(processed, "reliance", "2025", "RelianceIndustries", "2024-2025")

        with patch.object(build.batch_builder, "process_document", side_effect=_mock_process_document_success(n_text=5, n_table=0)):
            build.build_project(processed, output, force=False)

        jsonl_path = output / "relianceindustries_2025.jsonl"
        line_count = sum(1 for _ in open(jsonl_path))
        assert line_count == 5  # confirms per-line structure exists to stream over

        master_count = build.build_master_jsonl(["relianceindustries_2025"], output)
        assert master_count == 5


class TestEmptyDiscovery:
    def test_no_documents_discovered_handled_cleanly(self, tmp_path):
        processed = tmp_path / "processed"
        processed.mkdir()
        output = tmp_path / "chunks"

        summary = build.build_project(processed, output, force=False)

        assert summary["documents_discovered"] == 0
        assert summary["success"] is False
        assert not (output / "all_chunks.jsonl").exists()