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
    / "financial_rag_60_question_benchmark.json"
)


@pytest.fixture
def dataset():
    return validation.load_json(DATASET_PATH)


def _case(dataset, case_id):
    return next(case for case in dataset["cases"] if case["id"] == case_id)


def _verified_case(dataset, case_id="tcs_q01"):
    result = copy.deepcopy(dataset)
    case = _case(result, case_id)
    case["review_status"] = "verified"
    case["gold"] = {
        "expected_action": "answer",
        "action_rationale": "A human reviewer verified the source evidence.",
        "expected_assertions": [{
            "assertion_id": "verified-revenue",
            "proposition": "The company reported the stated consolidated revenue.",
            "evidence": [{
                "chunk_id": "verified-source-chunk",
                "verification_status": "human_verified",
                "verified_by": "reviewer-1",
                "relation": "supports",
                "rationale": "The reviewed source states the metric and period.",
            }],
            "numerical_reference": None,
            "temporal_reference": None,
        }],
    }
    return result


def test_refined_benchmark_has_60_pending_cases_and_preserves_category_distribution(
    dataset,
):
    validation.validate_dataset(dataset)

    assert len(dataset["cases"]) == 60
    assert dataset["total_cases"] == 60
    assert all(case["review_status"] == "pending" for case in dataset["cases"])
    assert all(case["gold"] is None for case in dataset["cases"])
    assert all(
        case["candidate_expected_action_status"] == "unverified"
        for case in dataset["cases"]
    )
    assert {company: sum(c["company"] == company for c in dataset["cases"])
            for company in dataset["category_distribution_per_company"]} == {
        "Infosys": 20,
        "RelianceIndustries": 20,
        "TataConsultancyServices": 20,
    }


def test_refined_benchmark_records_question_replacements_and_predecessors(dataset):
    replacements = {
        "tcs_q07",
        "ril_q41",
        "ril_q42",
        "ril_q45",
        "ril_q46",
    }

    for case_id in replacements:
        case = _case(dataset, case_id)
        assert case["revision"]["status"] == "replaced"
        assert case["revision"]["predecessor_id"] == case_id
        assert case["revision"]["original_question"]

    assert _case(dataset, "tcs_q17")["question"].startswith("How should TCS")
    assert _case(dataset, "tcs_q07")["financial_periods"] == [
        "2022-2023",
        "2023-2024",
    ]
    assert _case(dataset, "tcs_q09")["financial_periods"] == ["2023-2024"]
    assert len([case for case in dataset["cases"] if "revision" in case]) == 13


def test_current_information_cases_remain_unverified_current_data_tests(dataset):
    cases = [
        case
        for case in dataset["cases"]
        if case["category"] == "temporal_ambiguity_or_current_information"
    ]

    assert len(cases) == 6
    assert all(case["financial_periods"] == ["current"] for case in cases)
    expected_actions = {
        "tcs_q19": "abstain",
        "tcs_q20": "abstain",
        "infosys_q39": "current_information_unavailable",
        "infosys_q40": "current_information_unavailable",
        "ril_q59": "current_information_unavailable",
        "ril_q60": "current_information_unavailable",
    }
    assert {
        case["id"]: case["candidate_expected_action"] for case in cases
    } == expected_actions
    assert all(case["candidate_expected_action_status"] == "unverified" for case in cases)
    assert all(case["gold"] is None for case in cases)


def test_refined_benchmark_rejects_invalid_category(dataset):
    _case(dataset, "tcs_q01")["category"] = "financial_fact"

    with pytest.raises(validation.DatasetValidationError, match="Invalid dataset"):
        validation.validate_dataset(dataset)


@pytest.mark.parametrize("period", ["FY2024", "FY2024-Q5", "2023-2025"])
def test_refined_benchmark_rejects_invalid_periods(dataset, period):
    _case(dataset, "tcs_q01")["financial_periods"] = [period]

    with pytest.raises(validation.DatasetValidationError):
        validation.validate_dataset(dataset)


def test_refined_benchmark_rejects_duplicate_case_ids(dataset):
    dataset["cases"][1]["id"] = dataset["cases"][0]["id"]

    with pytest.raises(validation.DatasetValidationError, match="Case IDs must be unique"):
        validation.validate_dataset(dataset)


def test_refined_benchmark_rejects_invalid_company_filter(dataset):
    _case(dataset, "tcs_q01")["filters"]["company"] = "TCS"

    with pytest.raises(validation.DatasetValidationError):
        validation.validate_dataset(dataset)


def test_refined_benchmark_rejects_unverified_candidate_as_gold_evidence(dataset):
    dataset = _verified_case(dataset)
    case = dataset["cases"][0]
    case["unverified_candidate_chunk_ids"] = ["verified-source-chunk"]

    with pytest.raises(
        validation.DatasetValidationError,
        match="cannot use unverified candidate chunks",
    ):
        validation.validate_dataset(dataset)


def test_refined_benchmark_rejects_verified_candidate_annotation_while_pending(dataset):
    _case(dataset, "tcs_q01")["candidate_expected_action_status"] = "human_verified"

    with pytest.raises(
        validation.DatasetValidationError,
        match="keep candidate annotations unverified",
    ):
        validation.validate_dataset(dataset)


def test_pending_case_rejects_completed_answer_review(dataset):
    case = _case(dataset, "tcs_q01")
    case["answer_reviews"] = [{
        "review_id": "review-1",
        "run_id": "run-1",
        "reviewer_id": "reviewer-1",
        "review_status": "reviewed",
        "claims": [],
    }]

    with pytest.raises(
        validation.DatasetValidationError,
        match="cannot contain a completed answer review",
    ):
        validation.validate_dataset(dataset)


def test_refined_benchmark_rejects_unknown_revision_predecessor(dataset):
    _case(dataset, "ril_q41")["revision"]["predecessor_id"] = "unknown-question"

    with pytest.raises(
        validation.DatasetValidationError,
        match="revision predecessor must reference a case ID",
    ):
        validation.validate_dataset(dataset)


def test_refined_benchmark_rejects_invalid_answer_review_chunk_reference(dataset):
    dataset = _verified_case(dataset)
    case = dataset["cases"][0]
    case["answer_reviews"] = [{
        "review_id": "review-1",
        "run_id": "run-1",
        "reviewer_id": "reviewer-1",
        "review_status": "draft",
        "claims": [{
            "claim_id": "claim-1",
            "text": "Revenue is reported.",
            "cited_chunk_ids": ["cited-chunk"],
            "citation_assessments": [{
                "chunk_id": "uncited-chunk",
                "support_status": "not_supported",
                "rationale": "This chunk was not cited by the claim.",
            }],
            "atomic_assertions": [{
                "assertion_id": "claim-1-assertion-1",
                "text": "Revenue is reported.",
                "support_status": "not_supported",
                "rationale": "The cited evidence does not establish the claim.",
                "numerical_assessment": None,
                "temporal_assessment": None,
            }],
        }],
    }]

    with pytest.raises(
        validation.DatasetValidationError,
        match="assesses a chunk it did not cite",
    ):
        validation.validate_dataset(dataset)


def test_refined_benchmark_verified_gold_requires_supporting_evidence(dataset):
    dataset = _verified_case(dataset)
    case = dataset["cases"][0]
    case["gold"]["expected_assertions"][0]["evidence"][0]["relation"] = "context_only"

    with pytest.raises(
        validation.DatasetValidationError,
        match="supporting evidence",
    ):
        validation.validate_dataset(dataset)
