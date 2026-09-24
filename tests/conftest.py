"""
tests/conftest.py

Shared fixtures for the chunking module's test suite.
"""

import csv
import sys
from pathlib import Path

import pytest

# tests/ is a sibling of src/, so the module folder is
# <project_root>/src/chunking -- not this file's own parent.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src" / "chunking"))


@pytest.fixture
def sample_page_segments():
    """
    A small set of realistic page_segments.json entries: one page
    with ordinary narrative text under an ATX heading, one page
    that's almost entirely a table (mimicking a real balance-sheet
    page — should produce a small is_small=True leftover once the
    table rows are stripped), and one page with a bold-line heading
    to exercise the non-ATX detection path.
    """
    return [
        {
            "page": 1,
            "text": (
                "## Overview\n\n"
                "The company delivered strong performance this year, "
                "with revenue growth driven by continued expansion "
                "across all business segments and a robust demand "
                "environment in key markets."
            ),
        },
        {
            "page": 2,
            "text": (
                "## Balance Sheet\n\n"
                "N/A\n\n"
                "| Particulars | 2025 | 2024 |\n"
                "|---|---|---|\n"
                "| Assets | 100 | 90 |\n"
                "| Liabilities | 40 | 35 |\n"
            ),
        },
        {
            "page": 3,
            "text": (
                "**Segment Performance**\n\n"
                "The retail segment continued to outperform "
                "expectations, contributing significantly to overall "
                "growth for the reporting period."
            ),
        },
    ]


@pytest.fixture
def wide_table_csv(tmp_path):
    """
    A wide table (few rows, many columns) written to a real CSV file
    on disk, matching the shape of a real 10-year-financial-highlights
    table -- few metrics as rows, many years as columns.
    """
    path = tmp_path / "wide_table.csv"
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        years = [str(2016 + i) for i in range(11)]
        writer.writerow(["Metric"] + years)
        writer.writerow(["Revenue"] + [str(100000 + i * 15000) for i in range(11)])
        writer.writerow(["Net Profit"] + [str(8000 + i * 1200) for i in range(11)])
    return path


@pytest.fixture
def wide_table_meta():
    return {
        "table_id": "test_wide_t1", "table_number": 1, "page": 5,
        "filename": "wide_table.csv", "company": "TestCo",
        "document": "Annual Report", "financial_year": "2024-2025",
        "table_heading": "10-Year Financial Highlights", "context": None,
        "rows": 2, "columns": 12,
    }


@pytest.fixture
def continuation_table_metadata():
    """
    A genuine 3-table continuation chain (same heading with minor
    variation, same column count, adjacent pages) plus one unrelated
    table -- exercises both the positive fuzzy-match case and the
    "don't over-match" negative case in one fixture.
    """
    return [
        {"table_id": "annex_t1", "table_number": 1, "page": 138,
         "table_heading": "Annexure 'A'", "columns": 26},
        {"table_id": "annex_t2", "table_number": 2, "page": 139,
         "table_heading": "Annexure A (contd.)", "columns": 26},
        {"table_id": "annex_t3", "table_number": 3, "page": 140,
         "table_heading": "ANNEXURE 'A'", "columns": 26},
        {"table_id": "unrelated_t1", "table_number": 4, "page": 200,
         "table_heading": "Segment Revenue", "columns": 5},
    ]


@pytest.fixture
def balance_sheet_csv(tmp_path):
    """A balance-sheet-style table with several label + subtotal/
    total segments, for exercising structural segment detection and
    multi-chunk splitting (needs enough rows/segments that a tight
    token budget genuinely forces more than one output chunk)."""
    path = tmp_path / "balance_sheet.csv"
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Particulars", "Notes", "2025", "2024"])
        writer.writerow(["Assets", "", "", ""])
        writer.writerow(["Non-current assets", "", "", ""])
        writer.writerow(["Property, plant and equipment", "3", "450000", "420000"])
        writer.writerow(["Capital work-in-progress", "3", "25000", "18000"])
        writer.writerow(["Total non-current assets", "", "475000", "438000"])
        writer.writerow(["Current assets", "", "", ""])
        writer.writerow(["Inventories", "5", "80000", "75000"])
        writer.writerow(["Cash and cash equivalents", "6", "30000", "28000"])
        writer.writerow(["Total current assets", "", "110000", "103000"])
        writer.writerow(["Total Assets", "", "585000", "541000"])
        writer.writerow(["Equity and Liabilities", "", "", ""])
        writer.writerow(["Equity share capital", "7", "6350", "6350"])
        writer.writerow(["Other equity", "8", "400000", "370000"])
        writer.writerow(["Total equity", "", "406350", "376350"])
    return path


@pytest.fixture
def balance_sheet_meta():
    return {
        "table_id": "test_bs_t1", "table_number": 1, "page": 57,
        "filename": "balance_sheet.csv", "company": "TestCo",
        "document": "Annual Report", "financial_year": "2024-2025",
        "table_heading": "Balance Sheet", "context": "As at 31st March 2025",
        "rows": 3, "columns": 4,
    }