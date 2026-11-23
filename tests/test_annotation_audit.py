import hashlib
import json

import pytest

from scripts import build_annotation_audit as audit


def _inputs():
    benchmark = {
        "cases": [
            {
                "id": "q1",
                "company": "Infosys",
                "financial_periods": ["2023-2024"],
                "question": "What was revenue?",
            },
            {
                "id": "q2",
                "company": "TataConsultancyServices",
                "financial_periods": ["current"],
                "question": "What is the current price?",
            },
        ]
    }
    annotations = [{
        "cases": [{
            "id": "q1",
            "question": "What was revenue?",
            "proposed_gold": {
                "expected_assertions": [{
                    "assertion_id": "q1_a1",
                    "evidence": [
                        {
                            "chunk_id": "chunk-a",
                            "verification_status": "proposed_unverified",
                            "relation": "supports",
                        },
                        {
                            "chunk_id": "chunk-a",
                            "verification_status": "proposed_unverified",
                            "relation": "context_only",
                        },
                    ],
                }]
            },
        }]
    }]
    chunks = {
        "chunk-a": {
            "chunk_id": "chunk-a",
            "company": "Infosys",
            "document": "Annual Report",
            "financial_year": "2023-2024",
            "page": 12,
            "text": "Revenue passage.",
        }
    }
    return benchmark, annotations, chunks


def test_build_maps_source_chunks_and_preserves_duplicate_references():
    benchmark, annotations, chunks = _inputs()

    review_rows, question_rows = audit.build_review_rows(benchmark, annotations, chunks)

    assert len(review_rows) == 2
    assert [row["chunk_id"] for row in review_rows] == ["chunk-a", "chunk-a"]
    assert [row["evidence_reference_id"] for row in review_rows] == [
        "q1::q1_a1::evidence-01",
        "q1::q1_a1::evidence-02",
    ]
    assert review_rows[0]["document_name"] == "Annual Report"
    assert review_rows[0]["page_number"] == "12"
    assert review_rows[0]["original_evidence_passage"] == "Revenue passage."
    assert review_rows[0]["current_evidence_label"] == "supports"
    assert all(not row["reviewer_decision"] for row in review_rows)
    assert question_rows[1]["question_id"] == "q2"
    assert question_rows[1]["completeness_status"] == "pending"


def test_missing_chunk_retains_reference_and_marks_lookup_failure():
    benchmark, annotations, _ = _inputs()

    review_rows, _ = audit.build_review_rows(benchmark, annotations, {})

    assert review_rows[0]["chunk_id"] == "chunk-a"
    assert review_rows[0]["chunk_lookup_status"] == "missing"
    assert review_rows[0]["document_name"] == ""
    assert review_rows[0]["original_evidence_passage"] == ""


def test_summary_rejects_invalid_reviewer_decision():
    benchmark, annotations, chunks = _inputs()
    review_rows, question_rows = audit.build_review_rows(benchmark, annotations, chunks)
    review_rows[0]["reviewer_decision"] = "verified"

    with pytest.raises(ValueError, match="Invalid reviewer decision"):
        audit.summarize_reviews(review_rows, question_rows, chunks)


def test_summary_counts_decisions_and_additional_evidence():
    benchmark, annotations, chunks = _inputs()
    chunks["new-chunk"] = {
        "chunk_id": "new-chunk",
        "company": "Infosys",
        "document": "Annual Report",
        "financial_year": "2023-2024",
        "page": 13,
        "text": "Additional supporting passage.",
    }
    review_rows, question_rows = audit.build_review_rows(benchmark, annotations, chunks)
    review_rows[0]["reviewer_decision"] = "supports"
    review_rows[1]["reviewer_decision"] = "needs_review"
    question_rows[0]["additional_supporting_evidence_discovered"] = "new-chunk"
    question_rows[1]["completeness_status"] = "uncertain"

    summary = audit.summarize_reviews(review_rows, question_rows, chunks)

    assert summary["total_references"] == 2
    assert summary["reviewed_references"] == 2
    assert summary["pending_references"] == 0
    assert summary["decision_counts"] == {
        "context_only": 0,
        "needs_review": 1,
        "not_relevant": 0,
        "supports": 1,
    }
    assert summary["questions_with_new_evidence"] == ["q1"]
    assert summary["new_evidence_chunk_count"] == 1
    assert summary["questions_with_uncertain_completeness"] == ["q1", "q2"]
    assert summary["questions_explicitly_marked_uncertain"] == ["q2"]
    assert "References reviewed: **2**" in audit.render_summary(summary)


def test_summary_rejects_unknown_additional_chunk():
    benchmark, annotations, chunks = _inputs()
    review_rows, question_rows = audit.build_review_rows(benchmark, annotations, chunks)
    question_rows[0]["additional_supporting_evidence_discovered"] = "unknown-chunk"

    with pytest.raises(ValueError, match="unknown additional chunk ID"):
        audit.summarize_reviews(review_rows, question_rows, chunks)


def test_generation_does_not_modify_original_annotation_files(tmp_path, monkeypatch):
    benchmark, annotations, chunks = _inputs()
    annotation_path = tmp_path / "source_annotations.json"
    annotation_path.write_text(json.dumps(annotations[0]), encoding="utf-8")
    before = hashlib.sha256(annotation_path.read_bytes()).digest()
    monkeypatch.setattr(audit, "BENCHMARK_PATH", tmp_path / "benchmark.json")
    monkeypatch.setattr(audit, "ANNOTATION_PATHS", (annotation_path,))
    monkeypatch.setattr(audit, "CHUNKS_PATH", tmp_path / "chunks.jsonl")
    audit.BENCHMARK_PATH.write_text(json.dumps(benchmark), encoding="utf-8")
    audit.CHUNKS_PATH.write_text(
        "".join(json.dumps(chunk) + "\n" for chunk in chunks.values()),
        encoding="utf-8",
    )

    audit.create_template(tmp_path / "audit")

    assert hashlib.sha256(annotation_path.read_bytes()).digest() == before
    assert (tmp_path / "audit" / audit.REVIEW_CSV).exists()
    assert (tmp_path / "audit" / audit.QUESTIONS_CSV).exists()


def test_template_refuses_to_overwrite_reviewer_data(tmp_path):
    review_path = tmp_path / audit.REVIEW_CSV
    review_path.write_text("reviewer edit", encoding="utf-8")

    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        audit.create_template(tmp_path)

    assert review_path.read_text(encoding="utf-8") == "reviewer edit"


def test_duplicate_review_reference_ids_are_rejected():
    benchmark, annotations, chunks = _inputs()
    review_rows, question_rows = audit.build_review_rows(benchmark, annotations, chunks)
    review_rows[1]["evidence_reference_id"] = review_rows[0]["evidence_reference_id"]

    with pytest.raises(ValueError, match="duplicate evidence reference IDs"):
        audit.summarize_reviews(review_rows, question_rows, chunks)
