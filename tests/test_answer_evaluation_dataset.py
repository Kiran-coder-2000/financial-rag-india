import copy
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts import validate_answer_evaluation_dataset as validation


DATASET_PATH = (
    PROJECT_ROOT
    / "benchmarks"
    / "answer_evaluation"
    / "answer_evaluation_dataset.json"
)


@pytest.fixture
def dataset():
    return validation.load_json(DATASET_PATH)


def _verified_case(dataset):
    result = copy.deepcopy(dataset)
    case = result["cases"][0]
    case["review_status"] = "verified"
    case["gold"] = {
        "expected_action": "answer",
        "action_rationale": "The reviewed annual-report chunk states the figure.",
        "expected_assertions": [{
            "assertion_id": "expected-revenue",
            "proposition": "Infosys reported consolidated revenue for FY2026.",
            "evidence": [{
                "chunk_id": "reviewed-chunk",
                "verification_status": "human_verified",
                "verified_by": "reviewer-1",
                "relation": "supports",
                "rationale": "The table heading and FY2026 column identify the figure.",
            }],
            "numerical_reference": {
                "metric": "revenue",
                "value": "178650",
                "currency": "INR",
                "unit": "currency",
                "scale": "crore",
                "entity_scope": "consolidated",
                "period": "2025-2026",
            },
            "temporal_reference": {
                "periods": ["2025-2026"],
                "comparison_basis": "single_period",
                "temporal_scope": "historical",
            },
        }],
    }
    return result


def test_seed_dataset_preserves_15_pending_cases_without_promoting_candidates(dataset):
    validation.validate_dataset(dataset)

    assert len(dataset["cases"]) == 15
    assert {case["id"] for case in dataset["cases"]} == {
        "infosys_consolidated_revenue_fy2026",
        "infosys_revenue_yoy_fy2025_fy2026",
        "infosys_segment_performance_fy2024",
        "infosys_profit_after_tax_fy2023",
        "infosys_revenue_and_profit_yoy_fy2025_fy2026",
        "tataconsultancyservices_consolidated_revenue_fy2026",
        "tataconsultancyservices_revenue_yoy_fy2025_fy2026",
        "tataconsultancyservices_segment_performance_fy2024",
        "tataconsultancyservices_profit_after_tax_fy2023",
        "tataconsultancyservices_revenue_and_profit_yoy_fy2025_fy2026",
        "relianceindustries_consolidated_revenue_fy2026",
        "relianceindustries_revenue_yoy_fy2025_fy2026",
        "relianceindustries_segment_performance_fy2024",
        "relianceindustries_profit_after_tax_fy2023",
        "relianceindustries_revenue_and_profit_yoy_fy2025_fy2026",
    }
    assert {case["company"] for case in dataset["cases"]} == {
        "Infosys",
        "TataConsultancyServices",
        "RelianceIndustries",
    }
    assert all(case["review_status"] == "pending" for case in dataset["cases"])
    assert all(case["gold"] is None for case in dataset["cases"])
    assert all(not case["unverified_candidate_chunk_ids"] for case in dataset["cases"])
    cases_by_id = {case["id"]: case for case in dataset["cases"]}
    assert cases_by_id["infosys_consolidated_revenue_fy2026"]["question"] == (
        "What was Infosys's consolidated revenue in FY2026?"
    )
    assert cases_by_id["tataconsultancyservices_revenue_yoy_fy2025_fy2026"][
        "question"
    ] == "How did TCS's revenue change from FY2025 to FY2026?"


def test_verified_case_accepts_reviewed_evidence_and_expected_dimensions(dataset):
    validation.validate_dataset(_verified_case(dataset))


def test_duplicate_case_ids_are_rejected(dataset):
    duplicate = copy.deepcopy(dataset["cases"][0])
    dataset["cases"].append(duplicate)

    with pytest.raises(validation.DatasetValidationError, match="Case IDs must be unique"):
        validation.validate_dataset(dataset)


def test_verified_case_requires_gold_and_pending_case_rejects_gold(dataset):
    missing_gold = copy.deepcopy(dataset)
    missing_gold["cases"][0]["review_status"] = "verified"
    with pytest.raises(validation.DatasetValidationError, match="Invalid dataset"):
        validation.validate_dataset(missing_gold)

    unreviewed_gold = copy.deepcopy(dataset)
    unreviewed_gold["cases"][0]["gold"] = _verified_case(dataset)["cases"][0]["gold"]
    with pytest.raises(validation.DatasetValidationError, match="Invalid dataset"):
        validation.validate_dataset(unreviewed_gold)


def test_answerable_gold_assertion_requires_human_verified_support(dataset):
    case = _verified_case(dataset)["cases"][0]
    case["gold"]["expected_assertions"][0]["evidence"][0]["relation"] = "contradicts"

    with pytest.raises(validation.DatasetValidationError, match="supporting evidence"):
        validation.validate_dataset({"cases": [case], **{
            key: value for key, value in dataset.items() if key != "cases"
        }})


def test_unverified_candidate_reference_cannot_be_used_as_gold_evidence(dataset):
    case = _verified_case(dataset)["cases"][0]
    case["unverified_candidate_chunk_ids"] = ["suggested-chunk"]
    case["gold"]["expected_assertions"][0]["evidence"][0].update({
        "chunk_id": "suggested-chunk",
        "verification_status": "unverified",
    })

    with pytest.raises(validation.DatasetValidationError, match="Invalid dataset"):
        validation.validate_dataset({"cases": [case], **{
            key: value for key, value in dataset.items() if key != "cases"
        }})


def test_non_answer_action_cannot_contain_expected_factual_assertions(dataset):
    case = _verified_case(dataset)["cases"][0]
    case["gold"]["expected_action"] = "abstain"

    with pytest.raises(validation.DatasetValidationError, match="must not include"):
        validation.validate_dataset({"cases": [case], **{
            key: value for key, value in dataset.items() if key != "cases"
        }})


def test_completed_answer_review_assesses_each_claim_citation(dataset):
    case = _verified_case(dataset)["cases"][0]
    case["answer_reviews"] = [{
        "review_id": "review-1",
        "run_id": "run-1",
        "reviewer_id": "reviewer-1",
        "review_status": "reviewed",
        "claims": [{
            "claim_id": "claim-1",
            "text": "Revenue was INR 178650 crore.",
            "cited_chunk_ids": ["reviewed-chunk"],
            "citation_assessments": [{
                "chunk_id": "reviewed-chunk",
                "support_status": "supported",
                "rationale": "The cited table states this amount and scope.",
            }],
            "atomic_assertions": [{
                "assertion_id": "claim-1-a1",
                "text": "Revenue was INR 178650 crore.",
                "support_status": "supported",
                "rationale": "Value, currency, scale, period, and scope match.",
                "numerical_assessment": {
                    "status": "correct",
                    "rationale": "The value and units match the verified figure.",
                    "expected": case["gold"]["expected_assertions"][0]["numerical_reference"],
                    "observed": case["gold"]["expected_assertions"][0]["numerical_reference"],
                },
                "temporal_assessment": {
                    "status": "correct",
                    "rationale": "The claim refers to the requested fiscal year.",
                    "expected": case["gold"]["expected_assertions"][0]["temporal_reference"],
                    "observed": case["gold"]["expected_assertions"][0]["temporal_reference"],
                },
            }],
        }],
    }]

    validation.validate_dataset({"cases": [case], **{
        key: value for key, value in dataset.items() if key != "cases"
    }})


def test_completed_answer_review_must_assess_all_cited_chunks(dataset):
    case = _verified_case(dataset)["cases"][0]
    case["answer_reviews"] = [{
        "review_id": "review-1",
        "run_id": "run-1",
        "reviewer_id": "reviewer-1",
        "review_status": "reviewed",
        "claims": [{
            "claim_id": "claim-1",
            "text": "Revenue was reported.",
            "cited_chunk_ids": ["chunk-a", "chunk-b"],
            "citation_assessments": [{
                "chunk_id": "chunk-a",
                "support_status": "supported",
                "rationale": "This chunk states the reported amount.",
            }],
            "atomic_assertions": [{
                "assertion_id": "claim-1-a1",
                "text": "Revenue was reported.",
                "support_status": "supported",
                "rationale": "The assertion matches the source.",
                "numerical_assessment": None,
                "temporal_assessment": None,
            }],
        }],
    }]

    with pytest.raises(validation.DatasetValidationError, match="every cited chunk"):
        validation.validate_dataset({"cases": [case], **{
            key: value for key, value in dataset.items() if key != "cases"
        }})


def test_validation_rejects_unknown_labels_and_duplicate_assertion_ids(dataset):
    case = _verified_case(dataset)["cases"][0]
    case["gold"]["expected_assertions"].append(
        copy.deepcopy(case["gold"]["expected_assertions"][0])
    )

    with pytest.raises(validation.DatasetValidationError, match="expected assertion IDs"):
        validation.validate_dataset({"cases": [case], **{
            key: value for key, value in dataset.items() if key != "cases"
        }})

    invalid_label = _verified_case(dataset)["cases"][0]
    invalid_label["gold"]["expected_assertions"][0]["evidence"][0][
        "relation"
    ] = "maybe"
    with pytest.raises(validation.DatasetValidationError, match="Invalid dataset"):
        validation.validate_dataset({"cases": [invalid_label], **{
            key: value for key, value in dataset.items() if key != "cases"
        }})


def test_not_applicable_accuracy_assessment_has_no_expected_or_observed_values(dataset):
    case = _verified_case(dataset)["cases"][0]
    case["answer_reviews"] = [{
        "review_id": "review-1",
        "run_id": "run-1",
        "reviewer_id": "reviewer-1",
        "review_status": "draft",
        "claims": [{
            "claim_id": "claim-1",
            "text": "Revenue was reported.",
            "cited_chunk_ids": [],
            "citation_assessments": [],
            "atomic_assertions": [{
                "assertion_id": "claim-1-a1",
                "text": "Revenue was reported.",
                "support_status": "not_supported",
                "rationale": "The captured text does not state the claim.",
                "numerical_assessment": {
                    "status": "not_applicable",
                    "rationale": "This assertion contains no number.",
                    "expected": None,
                    "observed": None,
                },
                "temporal_assessment": None,
            }],
        }],
    }]

    validation.validate_dataset({"cases": [case], **{
        key: value for key, value in dataset.items() if key != "cases"
    }})

    assessment = case["answer_reviews"][0]["claims"][0]["atomic_assertions"][0][
        "numerical_assessment"
    ]
    assessment["observed"] = {}
    with pytest.raises(validation.DatasetValidationError, match="not_applicable"):
        validation.validate_dataset({"cases": [case], **{
            key: value for key, value in dataset.items() if key != "cases"
        }})


def test_correct_accuracy_assessment_requires_expected_and_observed(dataset):
    case = _verified_case(dataset)["cases"][0]
    case["answer_reviews"] = [{
        "review_id": "review-1",
        "run_id": "run-1",
        "reviewer_id": "reviewer-1",
        "review_status": "draft",
        "claims": [{
            "claim_id": "claim-1",
            "text": "Revenue was reported.",
            "cited_chunk_ids": [],
            "citation_assessments": [],
            "atomic_assertions": [{
                "assertion_id": "claim-1-a1",
                "text": "Revenue was reported.",
                "support_status": "supported",
                "rationale": "The source supports this statement.",
                "numerical_assessment": {
                    "status": "correct",
                    "rationale": "The values match.",
                    "expected": None,
                    "observed": None,
                },
                "temporal_assessment": None,
            }],
        }],
    }]

    with pytest.raises(validation.DatasetValidationError, match="must include expected"):
        validation.validate_dataset({"cases": [case], **{
            key: value for key, value in dataset.items() if key != "cases"
        }})


def test_loader_reports_invalid_json(tmp_path):
    path = tmp_path / "invalid.json"
    path.write_text("{", encoding="utf-8")

    with pytest.raises(validation.DatasetValidationError, match="Could not read JSON"):
        validation.load_json(path)
