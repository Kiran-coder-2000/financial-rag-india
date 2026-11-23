"""Validate the human-reviewed answer-evaluation dataset without scoring it."""

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator
from jsonschema.exceptions import SchemaError

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATASET = (
    PROJECT_ROOT
    / "benchmarks"
    / "answer_evaluation"
    / "answer_evaluation_dataset.json"
)
DEFAULT_SCHEMA = (
    PROJECT_ROOT
    / "benchmarks"
    / "answer_evaluation"
    / "answer_evaluation.schema.json"
)
COMPANIES = {
    "Infosys",
    "RelianceIndustries",
    "TataConsultancyServices",
}
CATEGORIES_BY_COMPANY = {
    company: {
        "financial_metric",
        "year_over_year_comparison",
        "segment_performance",
        "financial_statement_figure",
        "multi_evidence_comparison",
        "insufficient_evidence",
        "conflicting_or_reconcilable_evidence",
        "temporal_ambiguity_or_current_information",
    }
    for company in COMPANIES
}
REFINED_60_CATEGORIES = {
    "financial_metric",
    "year_over_year_comparison",
    "segment_performance",
    "multi_evidence_comparison",
    "insufficient_evidence",
    "conflicting_or_reconcilable_evidence",
    "temporal_ambiguity_or_current_information",
}
FINANCIAL_YEAR_PATTERN = re.compile(r"^((?:19|20)\d{2})-((?:19|20)\d{2})$")
QUARTER_PATTERN = re.compile(r"^FY((?:19|20)\d{2})-Q[1-4]$")


class DatasetValidationError(ValueError):
    """Raised when an answer-evaluation dataset violates its contract."""


def load_json(path: str | Path) -> Any:
    path = Path(path)
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise DatasetValidationError(f"Could not read JSON at {path}: {error}") from error


def _require_unique(values: list[str], description: str) -> None:
    if len(values) != len(set(values)):
        raise DatasetValidationError(f"{description} must be unique.")


def _financial_year_for_period(period: str) -> str | None:
    if period == "current":
        return None

    match = FINANCIAL_YEAR_PATTERN.fullmatch(period)
    if match:
        start_year, end_year = (int(value) for value in match.groups())
        if end_year != start_year + 1:
            raise DatasetValidationError(
                f"Financial period {period!r} must span consecutive fiscal years."
            )
        return period

    quarter_match = QUARTER_PATTERN.fullmatch(period)
    if quarter_match:
        end_year = int(quarter_match.group(1))
        return f"{end_year - 1}-{end_year}"

    raise DatasetValidationError(f"Invalid financial period {period!r}.")


def _validate_case_semantics(
    case: dict[str, Any],
    case_ids: set[str],
) -> None:
    company = case["company"]
    category = case["category"]
    if company not in CATEGORIES_BY_COMPANY or category not in CATEGORIES_BY_COMPANY[company]:
        raise DatasetValidationError(
            f"Case {case['id']!r} has an invalid company/category combination."
        )

    period_years = {
        year
        for period in case["financial_periods"]
        if (year := _financial_year_for_period(period)) is not None
    }
    contains_current = "current" in case["financial_periods"]
    if (
        contains_current
        and category != "temporal_ambiguity_or_current_information"
    ):
        raise DatasetValidationError(
            f"Case {case['id']!r} can use a current period only for "
            "current-information questions."
        )

    if case["filters"]["company"] != case["company"]:
        raise DatasetValidationError(
            f"Case {case['id']!r} filter company must match the case company."
        )
    filtered_companies = case["filters"].get("companies", [case["company"]])
    if case["company"] not in filtered_companies:
        raise DatasetValidationError(
            f"Case {case['id']!r} company must be included in its company filters."
        )
    filtered_year = case["filters"].get("financial_year")
    if filtered_year is not None and filtered_year not in period_years:
        raise DatasetValidationError(
            f"Case {case['id']!r} filter financial_year must match one of its periods."
        )

    revision = case.get("revision")
    if revision is not None and revision["predecessor_id"] not in case_ids:
        raise DatasetValidationError(
            f"Case {case['id']!r} revision predecessor must reference a case ID."
        )

    candidate_ids = set(case["unverified_candidate_chunk_ids"])
    gold = case["gold"]
    if gold is not None:
        assertions = gold["expected_assertions"]
        _require_unique(
            [assertion["assertion_id"] for assertion in assertions],
            f"Case {case['id']!r} expected assertion IDs",
        )
        if gold["expected_action"] in {"answer", "partial"}:
            if not assertions:
                raise DatasetValidationError(
                    f"Answer action for case {case['id']!r} needs expected assertions."
                )
            for assertion in assertions:
                if not any(
                    reference["relation"] == "supports"
                    for reference in assertion["evidence"]
                ):
                    raise DatasetValidationError(
                        f"Answerable assertion {assertion['assertion_id']!r} "
                        "needs a human-verified supporting evidence reference."
                    )
        elif assertions:
            raise DatasetValidationError(
                f"Non-answer action in case {case['id']!r} must not include "
                "expected factual assertions."
            )
        verified_ids = {
            reference["chunk_id"]
            for assertion in assertions
            for reference in assertion["evidence"]
        }
        if candidate_ids & verified_ids:
            raise DatasetValidationError(
                f"Case {case['id']!r} cannot use unverified candidate chunks "
                "as verified gold evidence."
            )

    candidate_action_status = case.get("candidate_expected_action_status")
    if (
        case["review_status"] == "pending"
        and candidate_action_status is not None
        and candidate_action_status != "unverified"
    ):
        raise DatasetValidationError(
            f"Pending case {case['id']!r} must keep candidate annotations unverified."
        )

    if case["review_status"] == "pending" and gold is not None:
        raise DatasetValidationError(
            f"Pending case {case['id']!r} cannot contain gold annotations."
        )
    if case["review_status"] == "verified" and gold is None:
        raise DatasetValidationError(
            f"Verified case {case['id']!r} must contain gold annotations."
        )

    review_ids = [review["review_id"] for review in case["answer_reviews"]]
    _require_unique(review_ids, f"Case {case['id']!r} review IDs")
    for review in case["answer_reviews"]:
        if (
            case["review_status"] == "pending"
            and review["review_status"] in {"reviewed", "adjudicated"}
        ):
            raise DatasetValidationError(
                f"Pending case {case['id']!r} cannot contain a completed answer review."
            )
        claims = review["claims"]
        _require_unique(
            [claim["claim_id"] for claim in claims],
            f"Review {review['review_id']!r} claim IDs",
        )
        for claim in claims:
            cited_ids = set(claim["cited_chunk_ids"])
            assessment_ids = [
                assessment["chunk_id"]
                for assessment in claim["citation_assessments"]
            ]
            _require_unique(
                assessment_ids,
                f"Claim {claim['claim_id']!r} citation assessment IDs",
            )
            if not set(assessment_ids) <= cited_ids:
                raise DatasetValidationError(
                    f"Claim {claim['claim_id']!r} assesses a chunk it did not cite."
                )
            if (
                review["review_status"] in {"reviewed", "adjudicated"}
                and set(assessment_ids) != cited_ids
            ):
                raise DatasetValidationError(
                    f"Completed review for claim {claim['claim_id']!r} must assess "
                    "every cited chunk exactly once."
                )
            assertion_ids = [
                assertion["assertion_id"]
                for assertion in claim["atomic_assertions"]
            ]
            _require_unique(
                assertion_ids,
                f"Claim {claim['claim_id']!r} atomic assertion IDs",
            )
            for assertion in claim["atomic_assertions"]:
                for field in ("numerical_assessment", "temporal_assessment"):
                    assessment = assertion[field]
                    if assessment is None:
                        continue
                    if assessment["status"] == "not_applicable":
                        if assessment["expected"] is not None or assessment["observed"] is not None:
                            raise DatasetValidationError(
                                f"{field} marked not_applicable must have null expected "
                                "and observed values."
                            )
                    elif assessment["status"] in {"correct", "incorrect"}:
                        if assessment["expected"] is None or assessment["observed"] is None:
                            raise DatasetValidationError(
                                f"{field} marked {assessment['status']} must include "
                                "expected and observed values."
                            )


def validate_dataset(
    dataset: Any,
    schema: dict[str, Any] | None = None,
) -> None:
    """Validate JSON Schema constraints and cross-field semantic invariants."""
    if schema is None:
        schema = load_json(DEFAULT_SCHEMA)
    try:
        Draft202012Validator.check_schema(schema)
    except (SchemaError, TypeError, ValueError) as error:
        raise DatasetValidationError(f"Invalid dataset JSON Schema: {error}") from error

    errors = sorted(
        Draft202012Validator(schema).iter_errors(dataset),
        key=lambda error: tuple(str(part) for part in error.absolute_path),
    )
    if errors:
        error = errors[0]
        location = "/".join(str(part) for part in error.absolute_path) or "<root>"
        raise DatasetValidationError(f"Invalid dataset at {location}: {error.message}")

    case_ids = [case["id"] for case in dataset["cases"]]
    _require_unique(case_ids, "Case IDs")
    case_id_set = set(case_ids)
    for case in dataset["cases"]:
        _validate_case_semantics(case, case_id_set)

    if dataset.get("benchmark_profile") == "refined_60":
        if dataset.get("total_cases") != 60 or len(dataset["cases"]) != 60:
            raise DatasetValidationError(
                "The refined_60 benchmark must contain exactly 60 cases."
            )
        company_counts = {company: 0 for company in COMPANIES}
        actual_distribution = {
            company: {category: 0 for category in REFINED_60_CATEGORIES}
            for company in COMPANIES
        }
        for case in dataset["cases"]:
            company_counts[case["company"]] += 1
            if case["category"] not in REFINED_60_CATEGORIES:
                raise DatasetValidationError(
                    f"Refined benchmark category {case['category']!r} is unsupported."
                )
            actual_distribution[case["company"]][case["category"]] += 1
        if set(company_counts.values()) != {20}:
            raise DatasetValidationError(
                "The refined_60 benchmark must contain 20 cases per company."
            )
        if dataset["category_distribution_per_company"] != actual_distribution:
            raise DatasetValidationError(
                "Declared per-company category distribution does not match the cases."
            )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Validate the answer-evaluation dataset; this does not score answers."
    )
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--schema", type=Path, default=DEFAULT_SCHEMA)
    args = parser.parse_args(argv)

    try:
        validate_dataset(load_json(args.dataset), load_json(args.schema))
    except DatasetValidationError as error:
        print(f"Dataset validation failed: {error}", file=sys.stderr)
        return 1

    print(f"Dataset is valid: {args.dataset}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
