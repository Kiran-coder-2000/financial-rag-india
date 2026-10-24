import importlib.util
import json
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

MODULE_PATH = PROJECT_ROOT / "scripts" / "build_retrieval_gold_dataset.py"
MODULE_SPEC = importlib.util.spec_from_file_location("build_retrieval_gold_dataset", MODULE_PATH)
builder = importlib.util.module_from_spec(MODULE_SPEC)
sys.modules[MODULE_SPEC.name] = builder
MODULE_SPEC.loader.exec_module(builder)


def _chunk(chunk_id, text, **metadata):
    return {
        "chunk_id": chunk_id,
        "text": text,
        "company": "Infosys",
        "document": "Annual Report",
        "financial_year": "2025-2026",
        "page": 77,
        "chunk_type": "text",
        "section": "Revenue",
        "table_heading": None,
        **metadata,
    }


def _spec(targets=None):
    return {
        "id": "infosys_revenue_fy2026_test",
        "question": "What was Infosys's consolidated revenue in FY2026?",
        "company": "Infosys",
        "financial_year": "2025-2026",
        "financial_years": ["2025-2026"],
        "category": "financial_metric",
        "targets": targets or [builder._target(
            "primary", "Find consolidated revenue.", ("consolidated revenue",), "2025-2026"
        )],
    }


def test_build_keeps_candidates_pending_and_expected_answer_empty():
    chunks = [
        _chunk("source-text", "Our consolidated revenue for fiscal 2026 was reported."),
        _chunk("other-company", "Consolidated revenue", company="RelianceIndustries"),
    ]

    dataset = builder.build_dataset(chunks, "source-hash", question_specs=[_spec()])
    case = dataset["cases"][0]

    assert case["expected_answer"] is None
    assert case["verification_status"] == "pending"
    assert case["relevant_chunk_ids"] == []
    assert case["candidate_relevant_chunk_ids"] == ["source-text"]
    assert case["candidate_chunks"][0]["evidence_role"] == "primary"
    assert case["candidate_chunks"][0]["relevance_explanations"]


def test_candidate_search_uses_table_heading_and_context():
    chunks = [_chunk(
        "table-source",
        "2026: 1,78,650",
        chunk_type="table",
        section=None,
        table_heading="Consolidated Revenue",
        context="Reported results for fiscal 2026",
    )]

    dataset = builder.build_dataset(chunks, "source-hash", question_specs=[_spec()])

    assert dataset["cases"][0]["candidate_relevant_chunk_ids"] == ["table-source"]
    candidate = dataset["cases"][0]["candidate_chunks"][0]
    assert "consolidated revenue" in candidate["matched_terms"]


def test_candidate_id_validation_rejects_unknown_ids():
    dataset = builder.build_dataset(
        [_chunk("real", "consolidated revenue")],
        "source-hash",
        question_specs=[_spec()],
    )
    dataset["cases"][0]["candidate_relevant_chunk_ids"] = ["invented"]
    dataset["cases"][0]["candidate_chunks"][0]["chunk_id"] = "invented"

    with pytest.raises(ValueError, match="unknown chunk ID"):
        builder.validate_dataset(dataset, {"real": _chunk("real", "consolidated revenue")})


def test_duplicate_source_and_question_ids_are_rejected():
    chunk = _chunk("same", "consolidated revenue")
    with pytest.raises(ValueError, match="Duplicate source chunk_id"):
        builder.build_dataset([chunk, chunk], "source-hash", question_specs=[_spec()])

    dataset = builder.build_dataset([chunk], "source-hash", question_specs=[_spec()])
    duplicate_case = dict(dataset["cases"][0])
    dataset["cases"].append(duplicate_case)
    with pytest.raises(ValueError, match="question ID is missing or duplicated"):
        builder.validate_dataset(dataset, {"same": chunk})


def test_output_refuses_to_overwrite_without_force(tmp_path):
    output = tmp_path / "gold.json"
    output.write_text("existing", encoding="utf-8")

    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        builder.write_dataset({"cases": []}, output)

    builder.write_dataset({"cases": []}, output, force=True)
    assert json.loads(output.read_text(encoding="utf-8")) == {"cases": []}