"""
tests/test_table_chunker.py

Tests for Part B (table_chunker.py): intact/structural/wide-table
splitting, Indian currency parsing, edge cases, and the
header/context repetition invariant.
"""

import csv

import pandas as pd
import pytest

from src.chunking import table_chunker as tc


def _meta(table_id="t1", filename="t.csv", heading=None, context=None,
          company="TestCo", document="Annual Report", financial_year="2024-2025"):
    return {
        "table_id": table_id, "table_number": 1, "page": 1, "filename": filename,
        "company": company, "document": document, "financial_year": financial_year,
        "table_heading": heading, "context": context, "rows": 0, "columns": 0,
    }


class TestFinancialNumberParsing:
    @pytest.mark.parametrize("raw,expected", [
        ("1,234", 1234.0),
        ("(1,234)", -1234.0),
        ("Rs. (1,234)", -1234.0),          # currency prefix before parens
        ("\u20b9(1,234)", -1234.0),         # rupee symbol immediately before parens
        ("` 1,234", 1234.0),                # backtick-as-rupee artifact seen in real corpus
        ("12,34,567", 1234567.0),           # Indian lakh/crore grouping
        ("1,234,567", 1234567.0),           # Western grouping -- same numeric result
        ("1,234.56", 1234.56),
        ("12.5%", 12.5),
        ("-", None),                        # lone dash = nil, not negative
        ("\u2013", None),                   # en-dash = nil
        ("", None),
        (None, None),
    ])
    def test_parses_indian_and_western_formats(self, raw, expected):
        assert tc._parse_financial_number(raw) == expected

    def test_stray_period_in_prefix_does_not_corrupt_result(self):
        """Regression test: 'Rs.' has a literal period that a naive
        keep-digits-and-dots filter would leak into the number,
        turning '(1,234)' into '.1234' -> 0.1234 instead of 1234."""
        assert tc._parse_financial_number("Rs. (1,234)") == -1234.0


class TestWideTableDetection:
    def test_wide_table_identified(self, wide_table_csv):
        df = pd.read_csv(wide_table_csv, dtype=str)
        assert tc.is_wide_table(df)

    def test_normal_table_not_wide(self, balance_sheet_csv):
        df = pd.read_csv(balance_sheet_csv, dtype=str)
        assert not tc.is_wide_table(df)


class TestWideTableChunking:
    def test_wide_table_intact_when_it_fits(self, wide_table_csv, wide_table_meta):
        chunks = tc.chunk_table(wide_table_csv, wide_table_meta, split_threshold=2000)
        assert len(chunks) == 1
        assert chunks[0]["split_strategy"] == "intact"

    def test_wide_table_uses_field_value_serialization(self, wide_table_csv, wide_table_meta):
        """Every value in a wide table's serialization must be
        paired with its column name -- never a bare wide pipe-table
        that could lose column identity when split."""
        chunks = tc.chunk_table(wide_table_csv, wide_table_meta, split_threshold=2000)
        text = chunks[0]["text"]
        assert "Record:" in text
        assert "2016:" in text and "2020:" in text  # column names present as labels

    def test_wide_table_splits_into_record_batches(self, wide_table_csv, wide_table_meta):
        chunks = tc.chunk_table(wide_table_csv, wide_table_meta, split_threshold=15, min_row_budget=15)
        assert len(chunks) > 1
        for c in chunks:
            assert c["split_strategy"] in ("wide_record_batch", "wide_column_group")

    def test_wide_table_row_coverage_is_complete_and_non_overlapping(self, wide_table_csv, wide_table_meta):
        """Every original row must be covered by exactly the
        expected chunks -- no row silently dropped or duplicated."""
        chunks = tc.chunk_table(wide_table_csv, wide_table_meta, split_threshold=15, min_row_budget=15)
        covered_rows = set()
        for c in chunks:
            covered_rows.update(range(c["row_start"], c["row_end"] + 1))
        assert covered_rows == {0, 1}  # wide_table_csv fixture has 2 data rows

    def test_single_oversized_record_splits_into_column_groups(self):
        """Unit test for the deepest fallback: a SINGLE record alone
        exceeds budget, so its own fields must split into groups,
        each still tagged with the record's identity."""
        rows = [["Metric0"] + [str(i) for i in range(12)]]
        df = pd.DataFrame(rows, columns=["Metric"] + [str(2016 + i) for i in range(12)])
        pieces = tc._split_wide_table(df, budget=8)
        assert len(pieces) > 1
        for body, strategy, row_range in pieces:
            assert strategy == "wide_column_group"
            assert "Record: Metric0" in body  # identity preserved in every fragment


class TestStructuralSegmentDetection:
    def test_label_row_detected(self, balance_sheet_csv):
        df = pd.read_csv(balance_sheet_csv, dtype=str)
        row = df.iloc[0]  # "Assets" row: first cell has text, rest blank
        assert tc._looks_like_label_row(row)

    def test_keyword_total_row_detected(self, balance_sheet_csv):
        df = pd.read_csv(balance_sheet_csv, dtype=str)
        total_row = df[df.iloc[:, 0] == "Total Assets"].iloc[0]
        assert tc._looks_like_keyword_total_row(total_row)

    def test_detail_row_not_flagged_as_boundary(self, balance_sheet_csv):
        df = pd.read_csv(balance_sheet_csv, dtype=str)
        detail_row = df[df.iloc[:, 0] == "Property, plant and equipment"].iloc[0]
        assert not tc._looks_like_label_row(detail_row)
        assert not tc._looks_like_keyword_total_row(detail_row)

    def test_segments_cover_every_row_exactly_once(self, balance_sheet_csv):
        df = pd.read_csv(balance_sheet_csv, dtype=str)
        segments = tc.build_logical_segments(df)
        covered = []
        for seg in segments:
            covered.extend(range(seg["start"], seg["end"] + 1))
        assert covered == list(range(len(df)))  # exactly once, in order


class TestTableChunkingHierarchy:
    def test_small_table_kept_intact(self, balance_sheet_csv, balance_sheet_meta):
        chunks = tc.chunk_table(balance_sheet_csv, balance_sheet_meta, split_threshold=1000)
        assert len(chunks) == 1
        assert chunks[0]["split_strategy"] == "intact"

    def test_oversized_table_splits_on_structure(self, balance_sheet_csv, balance_sheet_meta):
        chunks = tc.chunk_table(balance_sheet_csv, balance_sheet_meta, split_threshold=40)
        assert len(chunks) > 1

    def test_chunk_index_and_total_chunks_consistent(self, balance_sheet_csv, balance_sheet_meta):
        chunks = tc.chunk_table(balance_sheet_csv, balance_sheet_meta, split_threshold=5)
        total = len(chunks)
        for i, c in enumerate(chunks, start=1):
            assert c["chunk_index"] == i
            assert c["total_chunks"] == total


class TestHeaderAndContextAlwaysRepeated:
    def test_every_subchunk_repeats_heading_and_context(self, balance_sheet_csv, balance_sheet_meta):
        chunks = tc.chunk_table(balance_sheet_csv, balance_sheet_meta, split_threshold=5)
        assert len(chunks) > 1  # must actually exercise multi-chunk splitting
        for c in chunks:
            assert f"Table Heading: {balance_sheet_meta['table_heading']}" in c["text"]
            assert f"Context: {balance_sheet_meta['context']}" in c["text"]

    def test_missing_heading_never_invented(self, balance_sheet_csv):
        meta = _meta(filename="balance_sheet.csv", heading=None, context=None)
        meta["filename"] = balance_sheet_csv.name
        chunks = tc.chunk_table(balance_sheet_csv, meta)
        assert "Table Heading:" not in chunks[0]["text"]


class TestEdgeCases:
    def test_empty_table_dropped(self, tmp_path):
        path = tmp_path / "empty.csv"
        with open(path, "w", newline="") as f:
            csv.writer(f).writerow(["Col1", "Col2"])
        chunks = tc.chunk_table(path, _meta(filename="empty.csv"))
        assert chunks == []

    def test_one_column_table_kept_and_flagged_low_confidence(self, tmp_path):
        path = tmp_path / "degenerate.csv"
        with open(path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["Notes"])
            w.writerow(["A misdetected list item"])
        chunks = tc.chunk_table(path, _meta(filename="degenerate.csv"))
        assert len(chunks) == 1  # kept, not dropped -- avoids silent data loss
        assert chunks[0]["is_low_confidence"] is True

    def test_blank_rows_dropped_duplicate_data_rows_preserved(self, tmp_path):
        path = tmp_path / "malformed.csv"
        with open(path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["Particulars", "Value"])
            w.writerow(["Revenue", "1000"])
            w.writerow(["", ""])                    # fully blank -- dropped
            w.writerow(["Particulars", "Value"])     # header-as-data -- dropped
            w.writerow(["Revenue", "1000"])          # genuine duplicate data row -- KEPT
        chunks = tc.chunk_table(path, _meta(filename="malformed.csv"))
        assert len(chunks) == 1
        assert chunks[0]["text"].count("Revenue") == 2  # duplicate preserved, not deduplicated

    def test_missing_csv_returns_empty_list(self, tmp_path):
        chunks = tc.chunk_table(tmp_path / "does_not_exist.csv", _meta())
        assert chunks == []
