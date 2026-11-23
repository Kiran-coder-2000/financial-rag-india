import hashlib
import json
import zipfile
from pathlib import Path
from xml.etree import ElementTree

import pytest

from scripts import build_question_level_retrieval_audit as audit


def _fixture_data():
    benchmark = {
        "cases": [
            {
                "id": "q1",
                "question": "What was revenue?",
                "company": "Infosys",
                "category": "financial_metric",
                "financial_periods": ["2023-2024"],
            },
            {
                "id": "q2",
                "question": "What is current?",
                "company": "Infosys",
                "category": "temporal_ambiguity_or_current_information",
                "financial_periods": ["current"],
            },
        ]
    }
    strategy_names = ["baseline", "keyword_lexical"]
    raw_cases = []
    for case in benchmark["cases"]:
        qid = case["id"]
        raw_cases.append({
            "question_id": qid,
            "question": case["question"],
            "company": case["company"],
            "category": case["category"],
            "relevant_chunk_ids": ["support-a", "missing-gold"] if qid == "q1" else [],
            "strategies": {
                "baseline": {
                    "status": "completed",
                    "retrieved_count": 1,
                    "retrieved": [{
                        "chunk_id": "support-a" if qid == "q1" else "missing-retrieved",
                        "rank": 1,
                        "score": 0.8,
                        "text": "Revenue passage." if qid == "q1" else "Should not substitute this.",
                    }] if qid == "q1" or qid == "q2" else [],
                    "error": None,
                },
                "keyword_lexical": {
                    "status": "failed" if qid == "q2" else "completed",
                    "retrieved_count": 0,
                    "retrieved": [],
                    "error": "index unavailable" if qid == "q2" else None,
                },
            },
        })
    raw = {"cases": raw_cases}
    metric_rows = []
    for case in benchmark["cases"]:
        for strategy in strategy_names:
            failed = case["id"] == "q2" and strategy == "keyword_lexical"
            not_evaluable = case["id"] == "q2" and strategy == "baseline"
            metric_rows.append({
                "question_id": case["id"],
                "company": case["company"],
                "category": case["category"],
                "strategy": strategy,
                "retrieval_status": "failed" if failed else "completed",
                "metric_status": "failed" if failed else "not_evaluable" if not_evaluable else "scored",
                "relevant_count": "0" if case["id"] == "q2" else "2",
                "retrieved_count": (
                    "1" if case["id"] == "q2" and strategy == "baseline"
                    else "0" if case["id"] == "q2" or strategy == "keyword_lexical"
                    else "1"
                ),
                "precision@5": "" if failed or not_evaluable else "0.2",
                "recall@5": "" if failed or not_evaluable else "1.0",
                "recall@10": "" if failed or not_evaluable else "1.0",
                "mrr": "" if failed or not_evaluable else "1.0",
                "common_subset": "True" if case["id"] == "q1" else "False",
            })
    annotations = [{
        "cases": [{
            "id": "q1",
            "question": "What was revenue?",
            "proposed_gold": {
                "expected_assertions": [{
                    "assertion_id": "q1_a1",
                    "evidence": [
                        {
                            "chunk_id": "support-a",
                            "verification_status": "proposed_unverified",
                            "verified_by": None,
                            "relation": "supports",
                            "rationale": "Candidate evidence.",
                        },
                        {
                            "chunk_id": "context-a",
                            "verification_status": "proposed_unverified",
                            "verified_by": None,
                            "relation": "context_only",
                            "rationale": "Context only.",
                        },
                        {
                            "chunk_id": "missing-gold",
                            "verification_status": "proposed_unverified",
                            "verified_by": None,
                            "relation": "supports",
                            "rationale": "Preserve missing evidence reference.",
                        },
                    ],
                }]
            },
        }]
    }]
    chunks = {
        "support-a": {
            "chunk_id": "support-a",
            "company": "Infosys",
            "document": "Annual Report",
            "financial_year": "2023-2024",
            "page": 11,
            "text": "Revenue passage.",
        },
        "context-a": {
            "chunk_id": "context-a",
            "company": "Infosys",
            "document": "Annual Report",
            "financial_year": "2023-2024",
            "page": 12,
            "text": "Context passage.",
        },
    }
    return benchmark, raw, metric_rows, annotations, chunks


def test_question_strategy_gold_and_metric_mapping():
    benchmark, raw, metric_rows, annotations, chunks = _fixture_data()
    model = audit.prepare_audit_data(benchmark, raw, metric_rows, annotations, chunks)

    assert list(model["benchmark_by_id"]) == ["q1", "q2"]
    assert model["strategy_names"] == ["baseline", "keyword_lexical"]
    assert len(model["metrics_by_key"]) == 4
    assert [row["chunk_id"] for row in model["gold_by_question"]["q1"]] == [
        "support-a", "context-a", "missing-gold"
    ]
    assert model["gold_by_question"]["q1"][0]["relation"] == "supports"
    assert model["gold_by_question"]["q1"][1]["relation"] == "context_only"
    assert model["metrics_by_key"][("q1", "baseline")]["precision@5"] == 0.2


def test_exact_id_comparison_and_full_passage_mapping():
    benchmark, raw, metric_rows, annotations, chunks = _fixture_data()
    model = audit.prepare_audit_data(benchmark, raw, metric_rows, annotations, chunks)

    rows = list(audit._comparison_rows(model))
    support_rows = [
        row for row in rows
        if row["Question ID"] == "q1"
        and row["Retrieval strategy"] == "baseline"
        and row["Comparison item"] == "Expected supporting evidence"
        and row["Chunk ID"] == "support-a"
    ]
    context_rows = [
        row for row in rows
        if row["Question ID"] == "q1"
        and row["Retrieval strategy"] == "baseline"
        and row["Comparison item"] == "Expected context-only evidence"
    ]

    assert len(support_rows) == 1
    assert support_rows[0]["Exact chunk ID match"] == "Yes"
    assert support_rows[0]["Expected supporting retrieval rank"] == 1
    assert support_rows[0]["Full original chunk text"] == "Revenue passage."
    assert context_rows[0]["Supporting expected chunk retrieved?"] == "Not a supporting reference"
    assert context_rows[0]["Full original chunk text"] == "Context passage."


def test_missing_retrieved_and_expected_chunks_are_flagged_without_substitution():
    benchmark, raw, metric_rows, annotations, chunks = _fixture_data()
    model = audit.prepare_audit_data(benchmark, raw, metric_rows, annotations, chunks)

    retrieved_rows = list(audit._retrieved_rows(model))
    missing_result = next(
        row for row in retrieved_rows
        if row["Question ID"] == "q2" and row["Retrieval strategy"] == "baseline"
    )
    expected_rows = list(audit._expected_rows(model))
    missing_gold = next(row for row in expected_rows if row["Expected chunk ID"] == "missing-gold")

    assert missing_result["Chunk ID"] == "missing-retrieved"
    assert missing_result["Chunk lookup status"] == "missing"
    assert missing_result["Full original chunk text"] == ""
    assert missing_gold["Chunk lookup status"] == "missing"
    assert missing_gold["Full original chunk text"] == ""
    assert model["missing_retrieved_chunks"] == [("q2", "baseline", "missing-retrieved")]
    assert model["missing_gold_chunks"] == [
        ("q1", "q1::q1_a1::evidence-03", "missing-gold")
    ]


def test_null_metrics_remain_unavailable_and_saved_metric_values_are_retained():
    benchmark, raw, metric_rows, annotations, chunks = _fixture_data()
    model = audit.prepare_audit_data(benchmark, raw, metric_rows, annotations, chunks)

    rows = {
        (row["Question ID"], row["Retrieval strategy"]): row
        for row in audit._performance_rows(model)
    }
    assert rows[("q2", "baseline")]["Precision@5"] is None
    assert rows[("q2", "keyword_lexical")]["Recall@10"] is None
    assert rows[("q1", "baseline")]["Precision@5"] == 0.2


def test_invalid_metric_grid_and_rank_mismatch_are_rejected():
    benchmark, raw, metric_rows, annotations, chunks = _fixture_data()
    bad_ranks = json.loads(json.dumps(raw))
    bad_ranks["cases"][0]["strategies"]["baseline"]["retrieved"][0]["rank"] = 2

    with pytest.raises(ValueError, match="ranks are not consecutive"):
        audit.prepare_audit_data(benchmark, bad_ranks, metric_rows, annotations, chunks)

    with pytest.raises(ValueError, match="do not match question/strategy combinations"):
        audit.prepare_audit_data(benchmark, raw, metric_rows[:-1], annotations, chunks)


def test_workbook_has_requested_sheets_and_does_not_modify_source_files(tmp_path):
    benchmark, raw, metric_rows, annotations, chunks = _fixture_data()
    source_path = tmp_path / "source.json"
    source_path.write_text(json.dumps(raw), encoding="utf-8")
    before_hash = hashlib.sha256(source_path.read_bytes()).hexdigest()
    model = audit.prepare_audit_data(benchmark, raw, metric_rows, annotations, chunks)
    workbook_path = tmp_path / "question_level_retrieval_audit.xlsx"

    audit.write_workbook(model, workbook_path)

    assert hashlib.sha256(source_path.read_bytes()).hexdigest() == before_hash
    with zipfile.ZipFile(workbook_path) as workbook:
        xml = ElementTree.fromstring(workbook.read("xl/workbook.xml"))
    namespaces = {"main": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
    sheet_names = {
        sheet.attrib["name"]
        for sheet in xml.findall(".//main:sheets/main:sheet", namespaces)
    }
    assert sheet_names == {
        "Instructions",
        "Question Performance Summary",
        "Retrieved Chunks",
        "Expected Gold Chunks",
        "Side-by-Side Comparison",
        "Error Analysis",
        "Investigation Priorities",
    }


def test_workbook_writer_rejects_excel_cell_text_overflow():
    class FakeSheet:
        def write_string(self, *args):
            raise AssertionError("Should reject before writing.")

    with pytest.raises(ValueError, match="Excel's 32767-character limit"):
        audit._write_value(FakeSheet(), 0, 0, "x" * 32768)
