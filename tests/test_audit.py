"""
tests/test_audit.py

Tests for src/audit.py, using temporary synthetic JSONL files only.
The audit module has no dependency on chunker.py/table_chunker.py/
build_dataset.py/tiktoken, so nothing needs mocking here -- these
tests exercise the real audit code end to end.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import audit


def _text_chunk(chunk_id, company="RelianceIndustries", document="Annual Report",
                 financial_year="2024-2025", page=1, text="Some narrative content here.",
                 tokens=50, section=None, is_small=False, **overrides):
    chunk = {
        "chunk_id": chunk_id, "company": company, "document": document,
        "financial_year": financial_year, "page": page, "chunk_type": "text",
        "text": text, "tokens": tokens, "section": section, "is_small": is_small,
    }
    chunk.update(overrides)
    return chunk


def _table_chunk(chunk_id, company="RelianceIndustries", document="Annual Report",
                  financial_year="2024-2025", page=57, text="| a | b |",
                  tokens=200, table_id="t1", table_heading="Balance Sheet",
                  context="As at 31 March 2025", split_strategy="intact",
                  row_start=0, row_end=10, is_low_confidence=False,
                  continuation_of=None, chunk_index=1, total_chunks=1, **overrides):
    chunk = {
        "chunk_id": chunk_id, "company": company, "document": document,
        "financial_year": financial_year, "page": page, "chunk_type": "table",
        "text": text, "tokens": tokens, "table_id": table_id,
        "table_heading": table_heading, "context": context,
        "split_strategy": split_strategy, "row_start": row_start, "row_end": row_end,
        "is_low_confidence": is_low_confidence, "continuation_of": continuation_of,
        "chunk_index": chunk_index, "total_chunks": total_chunks,
    }
    chunk.update(overrides)
    return chunk


def _write_jsonl(path: Path, chunks: list[dict]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(c) + "\n")


class TestValidInventory:
    def test_counts_text_and_table_chunks_correctly(self, tmp_path):
        chunks_root = tmp_path / "chunks"
        chunks_root.mkdir()
        _write_jsonl(chunks_root / "reliance_2025.jsonl", [
            _text_chunk("r1"), _text_chunk("r2"), _table_chunk("r3"),
        ])
        _write_jsonl(chunks_root / "infosys_2026.jsonl", [
            _text_chunk("i1", company="Infosys", financial_year="2025-2026"),
        ])

        report = audit.run_audit(chunks_root, tmp_path / "audit")

        inv = report["inventory"]
        assert inv["total_chunks"] == 4
        assert inv["text_chunks"] == 3
        assert inv["table_chunks"] == 1
        assert inv["by_document"] == {"infosys_2026.jsonl": 1, "reliance_2025.jsonl": 3}
        assert inv["by_company"] == {"Infosys": 1, "RelianceIndustries": 3}

    def test_excludes_all_chunks_jsonl(self, tmp_path):
        chunks_root = tmp_path / "chunks"
        chunks_root.mkdir()
        _write_jsonl(chunks_root / "reliance_2025.jsonl", [_text_chunk("r1")])
        _write_jsonl(chunks_root / "all_chunks.jsonl", [_text_chunk("r1"), _text_chunk("r1")])

        report = audit.run_audit(chunks_root, tmp_path / "audit")

        assert report["inventory"]["total_chunks"] == 1
        assert "all_chunks.jsonl" not in report["documents_audited"]


class TestTokenStatistics:
    def test_min_max_mean_median_percentiles(self, tmp_path):
        chunks_root = tmp_path / "chunks"
        chunks_root.mkdir()
        tokens = [10, 20, 30, 40, 50, 100, 500, 1500]
        _write_jsonl(chunks_root / "doc.jsonl", [
            _text_chunk(f"c{i}", tokens=t) for i, t in enumerate(tokens)
        ])

        report = audit.run_audit(chunks_root, tmp_path / "audit")
        stats = report["token_statistics"]

        assert stats["count"] == 8
        assert stats["min"] == 10
        assert stats["max"] == 1500
        assert stats["mean"] == sum(tokens) / len(tokens)
        assert stats["over_400_tokens"] == 2
        assert stats["over_1000_tokens"] == 1

    def test_excludes_non_numeric_tokens_field(self, tmp_path):
        chunks_root = tmp_path / "chunks"
        chunks_root.mkdir()
        _write_jsonl(chunks_root / "doc.jsonl", [
            _text_chunk("c1", tokens=50),
            _text_chunk("c2", tokens="not a number"),
        ])

        report = audit.run_audit(chunks_root, tmp_path / "audit")
        stats = report["token_statistics"]

        assert stats["count"] == 1
        assert stats["excluded_count"] == 1


class TestDuplicateChunkIds:
    def test_duplicate_chunk_id_is_error(self, tmp_path):
        chunks_root = tmp_path / "chunks"
        chunks_root.mkdir()
        _write_jsonl(chunks_root / "doc.jsonl", [
            _text_chunk("dup_id"), _text_chunk("dup_id"),
        ])

        report = audit.run_audit(chunks_root, tmp_path / "audit")

        dup_findings = [f for f in report["findings"] if f["category"] == "duplicate_chunk_id"]
        assert len(dup_findings) == 1
        assert dup_findings[0]["severity"] == "ERROR"
        assert report["summary"]["overall_status"] == "FAIL"


class TestMissingRequiredMetadata:
    def test_missing_field_reported_as_error(self, tmp_path):
        chunks_root = tmp_path / "chunks"
        chunks_root.mkdir()
        chunk = _text_chunk("c1")
        del chunk["company"]
        _write_jsonl(chunks_root / "doc.jsonl", [chunk])

        report = audit.run_audit(chunks_root, tmp_path / "audit")

        missing = [f for f in report["findings"] if f["category"] == "missing_field"]
        assert any("company" in f["message"] for f in missing)
        assert report["summary"]["overall_status"] == "FAIL"

    def test_missing_table_specific_field(self, tmp_path):
        chunks_root = tmp_path / "chunks"
        chunks_root.mkdir()
        chunk = _table_chunk("t1")
        del chunk["table_id"]
        _write_jsonl(chunks_root / "doc.jsonl", [chunk])

        report = audit.run_audit(chunks_root, tmp_path / "audit")

        missing = [f for f in report["findings"] if f["category"] == "missing_table_field"]
        assert any("table_id" in f["message"] for f in missing)


class TestAllowedNoneValues:
    def test_section_none_is_not_an_error(self, tmp_path):
        chunks_root = tmp_path / "chunks"
        chunks_root.mkdir()
        _write_jsonl(chunks_root / "doc.jsonl", [_text_chunk("c1", section=None)])

        report = audit.run_audit(chunks_root, tmp_path / "audit")

        assert report["summary"]["errors"] == 0

    def test_context_and_table_heading_and_continuation_of_none_are_not_errors(self, tmp_path):
        chunks_root = tmp_path / "chunks"
        chunks_root.mkdir()
        _write_jsonl(chunks_root / "doc.jsonl", [
            _table_chunk("t1", context=None, table_heading=None, continuation_of=None)
        ])

        report = audit.run_audit(chunks_root, tmp_path / "audit")

        assert report["summary"]["errors"] == 0

    def test_company_none_is_still_an_error(self, tmp_path):
        """None is only acceptable for the explicitly nullable
        fields -- a core identifying field being None is still
        wrong, distinct from it being absent."""
        chunks_root = tmp_path / "chunks"
        chunks_root.mkdir()
        chunk = _text_chunk("c1")
        chunk["company"] = None
        _write_jsonl(chunks_root / "doc.jsonl", [chunk])

        report = audit.run_audit(chunks_root, tmp_path / "audit")

        assert any(f["category"] == "invalid_null_field" for f in report["findings"])
        assert report["summary"]["overall_status"] == "FAIL"


class TestEmptyAndSmallChunks:
    def test_empty_text_chunk_reported(self, tmp_path):
        chunks_root = tmp_path / "chunks"
        chunks_root.mkdir()
        _write_jsonl(chunks_root / "doc.jsonl", [_text_chunk("c1", text="   ", tokens=0)])

        report = audit.run_audit(chunks_root, tmp_path / "audit")

        assert report["small_and_empty_chunks"]["empty_chunks"]["count"] == 1

    def test_chunks_kept_not_dropped(self, tmp_path):
        """Empty/small chunks must still appear in inventory totals
        -- audit reports, never removes."""
        chunks_root = tmp_path / "chunks"
        chunks_root.mkdir()
        _write_jsonl(chunks_root / "doc.jsonl", [_text_chunk("c1", text="", tokens=0)])

        report = audit.run_audit(chunks_root, tmp_path / "audit")

        assert report["inventory"]["total_chunks"] == 1

    def test_near_empty_and_small_text_and_is_small_flag(self, tmp_path):
        chunks_root = tmp_path / "chunks"
        chunks_root.mkdir()
        _write_jsonl(chunks_root / "doc.jsonl", [
            _text_chunk("c1", tokens=3),                    # near-empty (<=5) and small text (<10)
            _text_chunk("c2", tokens=8),                     # small text only
            _text_chunk("c3", tokens=50, is_small=True),      # flagged, not actually tiny
        ])

        report = audit.run_audit(chunks_root, tmp_path / "audit")
        small = report["small_and_empty_chunks"]

        assert small["near_empty_chunks_le_5_tokens"]["count"] == 1
        assert small["text_chunks_under_10_tokens"]["count"] == 2
        assert small["flagged_is_small"]["count"] == 1


class TestDuplicateContent:
    def test_duplicate_text_is_warning_not_error(self, tmp_path):
        chunks_root = tmp_path / "chunks"
        chunks_root.mkdir()
        _write_jsonl(chunks_root / "doc.jsonl", [
            _text_chunk("c1", text="Exact same content"),
            _text_chunk("c2", text="Exact same content"),
        ])

        report = audit.run_audit(chunks_root, tmp_path / "audit")

        dup = [f for f in report["findings"] if f["category"] == "duplicate_text"]
        assert len(dup) == 1
        assert dup[0]["severity"] == "WARNING"
        # A warning alone must not fail the audit.
        assert report["summary"]["overall_status"] == "PASS_WITH_WARNINGS"


class TestLowConfidenceTableChunks:
    def test_low_confidence_count_reported(self, tmp_path):
        chunks_root = tmp_path / "chunks"
        chunks_root.mkdir()
        _write_jsonl(chunks_root / "doc.jsonl", [
            _table_chunk("t1", is_low_confidence=True),
            _table_chunk("t2", is_low_confidence=False, table_id="t2"),
        ])

        report = audit.run_audit(chunks_root, tmp_path / "audit")

        assert report["table_inspection"]["low_confidence_count"] == 1

    def test_large_table_chunk_not_flagged_as_error(self, tmp_path):
        """A table chunk over 400 tokens is an expected, correct
        outcome -- must never be reported as an error/warning here."""
        chunks_root = tmp_path / "chunks"
        chunks_root.mkdir()
        _write_jsonl(chunks_root / "doc.jsonl", [_table_chunk("t1", tokens=5000)])

        report = audit.run_audit(chunks_root, tmp_path / "audit")

        assert report["summary"]["errors"] == 0
        assert report["summary"]["warnings"] == 0


class TestInvalidJsonl:
    def test_malformed_line_reported_not_silently_skipped(self, tmp_path):
        chunks_root = tmp_path / "chunks"
        chunks_root.mkdir()
        path = chunks_root / "doc.jsonl"
        with open(path, "w") as f:
            f.write(json.dumps(_text_chunk("c1")) + "\n")
            f.write("{this is not valid json\n")
            f.write(json.dumps(_text_chunk("c2")) + "\n")

        report = audit.run_audit(chunks_root, tmp_path / "audit")

        malformed = [f for f in report["findings"] if f["category"] == "malformed_jsonl"]
        assert len(malformed) == 1
        assert report["inventory"]["total_chunks"] == 2  # the two valid lines still counted
        assert report["summary"]["overall_status"] == "FAIL"

    def test_json_array_instead_of_object_reported(self, tmp_path):
        chunks_root = tmp_path / "chunks"
        chunks_root.mkdir()
        path = chunks_root / "doc.jsonl"
        with open(path, "w") as f:
            f.write(json.dumps([1, 2, 3]) + "\n")

        report = audit.run_audit(chunks_root, tmp_path / "audit")

        assert any(f["category"] == "malformed_jsonl" for f in report["findings"])


class TestEmptyDataset:
    def test_no_jsonl_files_at_all(self, tmp_path):
        chunks_root = tmp_path / "chunks"
        chunks_root.mkdir()

        report = audit.run_audit(chunks_root, tmp_path / "audit")

        assert report["inventory"]["total_chunks"] == 0
        assert report["summary"]["overall_status"] == "PASS"
        assert report["token_statistics"]["count"] == 0

    def test_nonexistent_chunks_root(self, tmp_path):
        report = audit.run_audit(tmp_path / "does_not_exist", tmp_path / "audit")
        assert report["inventory"]["total_chunks"] == 0


class TestReportWrittenToDisk:
    def test_report_file_created_and_valid_json(self, tmp_path):
        chunks_root = tmp_path / "chunks"
        chunks_root.mkdir()
        _write_jsonl(chunks_root / "doc.jsonl", [_text_chunk("c1")])

        output_root = tmp_path / "audit"
        audit.run_audit(chunks_root, output_root)

        report_path = output_root / "audit_report.json"
        assert report_path.exists()
        json.loads(report_path.read_text())  # must parse cleanly

    def test_deterministic_repeated_runs(self, tmp_path):
        chunks_root = tmp_path / "chunks"
        chunks_root.mkdir()
        _write_jsonl(chunks_root / "doc.jsonl", [_text_chunk("c1"), _table_chunk("t1")])

        r1 = audit.run_audit(chunks_root, tmp_path / "audit")
        r2 = audit.run_audit(chunks_root, tmp_path / "audit")

        # generated_at will differ; everything substantive must not.
        r1.pop("generated_at"); r2.pop("generated_at")
        assert r1 == r2