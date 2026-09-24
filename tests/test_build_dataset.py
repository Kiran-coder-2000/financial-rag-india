"""
tests/test_build_dataset.py

Tests for Part C (build_dataset.py): chunk_id uniqueness/prefixing,
fuzzy continuation detection, Counter-based validation, and
missing-CSV warning propagation.
"""

import json

from src.chunking import build_dataset as bd

class TestChunkIdPrefixing:
    def test_document_slug_prevents_collision_across_documents(self):
        """Same company, year, page, index -- but two DIFFERENT
        documents -- must not produce the same chunk_id."""
        id_annual = bd.build_text_chunk_id("annualreport", "reliance", "2025", 5, 1)
        id_sustainability = bd.build_text_chunk_id("sustainabilityreport", "reliance", "2025", 5, 1)
        assert id_annual != id_sustainability

    def test_table_chunk_id_single_chunk_has_no_suffix(self):
        chunk_id = bd.build_table_chunk_id("annualreport", "reliance_2025_p057_t065", 1, 1)
        assert chunk_id == "annualreport_reliance_2025_p057_t065"

    def test_table_chunk_id_multi_chunk_has_suffix(self):
        chunk_id = bd.build_table_chunk_id("annualreport", "reliance_2025_p057_t065", 2, 5)
        assert chunk_id == "annualreport_reliance_2025_p057_t065_c02"


class TestFuzzyContinuationDetection:
    def test_chain_correctly_linked_despite_heading_variation(self, continuation_table_metadata):
        """Quote style, case, and a '(contd.)' suffix should all be
        treated as the same heading for continuation purposes."""
        result = bd.detect_continuations(continuation_table_metadata)
        assert result == {"annex_t2": "annex_t1", "annex_t3": "annex_t2"}

    def test_unrelated_table_not_linked(self, continuation_table_metadata):
        result = bd.detect_continuations(continuation_table_metadata)
        assert "unrelated_t1" not in result

    def test_different_column_count_not_linked_despite_matching_heading(self):
        """Same heading, adjacent pages, but DIFFERENT column counts
        -- must NOT be flagged as a continuation."""
        metadata = [
            {"table_id": "a", "table_number": 1, "page": 1, "table_heading": "Annexure A", "columns": 26},
            {"table_id": "b", "table_number": 2, "page": 2, "table_heading": "Annexure A", "columns": 18},
        ]
        result = bd.detect_continuations(metadata)
        assert result == {}

    def test_non_adjacent_pages_not_linked(self):
        metadata = [
            {"table_id": "a", "table_number": 1, "page": 1, "table_heading": "Annexure A", "columns": 10},
            {"table_id": "b", "table_number": 2, "page": 50, "table_heading": "Annexure A", "columns": 10},
        ]
        result = bd.detect_continuations(metadata)
        assert result == {}

    def test_headings_match_exact_and_fuzzy(self):
        assert bd._headings_match("Annexure 'A'", "Annexure 'A'")
        assert bd._headings_match("Annexure 'A'", "ANNEXURE 'A'")
        assert bd._headings_match("Annexure A", "Annexure A (contd.)")
        assert not bd._headings_match("Balance Sheet", "Cash Flow Statement")


class TestValidateChunksWithCounter:
    def _complete_chunk(self, chunk_id, page=1):
        return {
            "chunk_id": chunk_id, "company": "X", "document": "D",
            "financial_year": "2025", "page": page, "chunk_type": "text",
            "text": "t", "tokens": 10, "section": None, "is_small": False,
        }

    def test_duplicate_ids_detected(self):
        chunks = [self._complete_chunk("a"), self._complete_chunk("b"), self._complete_chunk("a")]
        problems = bd.validate_chunks(chunks, [{"page": 1}], [])
        assert any("Duplicate" in p for p in problems)

    def test_no_duplicates_no_problem(self):
        chunks = [self._complete_chunk("a"), self._complete_chunk("b")]
        problems = bd.validate_chunks(chunks, [{"page": 1}], [])
        assert not any("Duplicate" in p for p in problems)

    def test_validator_does_not_crash_on_malformed_chunk(self):
        """A chunk missing required fields must be REPORTED, not
        cause validate_chunks itself to raise."""
        bad_chunks = [{"chunk_id": "z", "chunk_type": "text"}]
        problems = bd.validate_chunks(bad_chunks, [], [])
        assert len(problems) > 0  # reported, no exception raised

    def test_unknown_page_flagged(self):
        chunks = [self._complete_chunk("a", page=999)]
        problems = bd.validate_chunks(chunks, [{"page": 1}], [])
        assert any("unknown page" in p for p in problems)


class TestMissingCsvWarningPropagation:
    def test_extra_warnings_appear_in_problems(self):
        problems = bd.validate_chunks(
            [], [], [], extra_warnings=["Missing CSV for t1: /fake/path.csv"]
        )
        assert "Missing CSV for t1: /fake/path.csv" in problems

    def test_missing_csv_during_build_is_collected_not_silent(self, tmp_path):
        """End-to-end: a table_metadata entry whose CSV doesn't
        exist must surface as a warning from build_dataset, not just
        a log line the caller has to remember to check."""
        (tmp_path / "tables").mkdir()
        (tmp_path / "document_metadata.json").write_text(json.dumps({
            "company": "TestCo", "document": "Annual Report",
            "financial_year": "2024-2025", "parser_version": "1.0",
        }))
        (tmp_path / "page_segments.json").write_text(json.dumps([
            {"page": 1, "text": "Some content here that is long enough to survive chunking."}
        ]))
        (tmp_path / "table_metadata.json").write_text(json.dumps([
            {"table_id": "missing_t1", "table_number": 1, "page": 5,
             "filename": "does_not_exist.csv", "company": "TestCo",
             "document": "Annual Report", "financial_year": "2024-2025",
             "table_heading": "Ghost Table", "context": None, "rows": 5, "columns": 3}
        ]))

        chunks, warnings = bd.build_dataset(tmp_path)
        assert any("missing_t1" in w for w in warnings)
        assert not any(c.get("table_id") == "missing_t1" for c in chunks)
