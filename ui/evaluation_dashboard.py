"""Read-only presentation helpers for saved retrieval evaluation outputs."""

import csv
import html
import json
import logging
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import streamlit as st

logger = logging.getLogger(__name__)

_CSV_FILES = {
    "overall": ("metrics_overall.csv", {"strategy", "query_count", "scored_queries"}),
    "company": (
        "metrics_by_company.csv",
        {"strategy", "company", "query_count", "scored_queries"},
    ),
    "category": (
        "metrics_by_category.csv",
        {"strategy", "category", "query_count", "scored_queries"},
    ),
    "common": ("metrics_common_subset.csv", {"strategy", "query_count", "scored_queries"}),
    "per_query": (
        "per_query_metrics.csv",
        {
            "question_id",
            "company",
            "category",
            "strategy",
            "retrieval_status",
            "metric_status",
            "common_subset",
        },
    ),
    "common_questions": (
        "common_evaluation_subset.csv",
        {
            "question_id",
            "company",
            "category",
            "common_subset",
            "supporting_evidence_label_count",
        },
    ),
}
_ANNOTATION_FILES = (
    "tcs_q01_q20_gold_annotations.json",
    "infosys_gold_annotations.json",
    "reliance_gold_annotations.json",
)
_COMPANY_LABELS = {
    "TataConsultancyServices": "TCS",
    "Infosys": "Infosys",
    "RelianceIndustries": "Reliance Industries",
}
_METRIC_PATTERNS = {
    "MRR": ("macro_mrr", "mrr"),
    "Precision": ("precision", "precision"),
    "Recall": ("recall", "recall"),
    "F1": ("f1", "f1"),
    "nDCG": ("ndcg", "ndcg"),
}
_K_COLUMN = re.compile(r"(?:macro|micro)_(?:precision|recall|f1|ndcg)@(\d+)$")
_STRATEGY_DEFINITIONS = {
    "baseline": "Direct Chroma semantic retrieval.",
    "chunk_semantic": (
        "Direct Chroma semantic retrieval; shares the baseline retrieval path."
    ),
    "keyword_semantic": "Keyword-assisted semantic retrieval using a pilot index.",
    "keyword_lexical": "Lexical retrieval using a pilot index.",
    "hybrid_chunk_keyword": (
        "Combines chunk-semantic and keyword-semantic retrieval using Reciprocal Rank Fusion (RRF)."
    ),
    "hybrid_chunk_lexical": (
        "Combines chunk-semantic and keyword-lexical retrieval using Reciprocal Rank Fusion (RRF)."
    ),
    "hybrid_all": (
        "Combines configured chunk-semantic and keyword retrieval sources using Reciprocal Rank Fusion (RRF)."
    ),
}


@dataclass(frozen=True)
class EvaluationData:
    tables: dict[str, list[dict[str, str]]]
    errors: dict[str, str]
    quality: dict[str, Any]
    failure_details: dict[str, list[dict[str, str]]]
    metadata: dict[str, Any]


def _read_csv(path: Path, required: set[str]) -> tuple[list[dict[str, str]], str | None]:
    try:
        with path.open(encoding="utf-8-sig", newline="") as file:
            reader = csv.DictReader(file)
            if not reader.fieldnames:
                return [], "The file has no CSV header."
            if len(reader.fieldnames) != len(set(reader.fieldnames)):
                return [], "The file has duplicate column names."
            missing = required.difference(reader.fieldnames)
            if missing:
                return [], "Missing required columns: " + ", ".join(sorted(missing)) + "."
            rows = list(reader)
    except (OSError, UnicodeError, csv.Error) as error:
        return [], f"Could not read the file: {error}"
    if any(None in row for row in rows):
        return [], "One or more CSV records contain more fields than the header."
    if any(any(value is None for value in row.values()) for row in rows):
        return [], "One or more CSV records have fewer fields than the header."
    return rows, None


def _read_json(path: Path) -> tuple[dict[str, Any] | None, str | None]:
    try:
        with path.open(encoding="utf-8") as file:
            value = json.load(file)
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        return None, f"Could not read the file: {error}"
    if not isinstance(value, dict):
        return None, "Expected a JSON object."
    return value, None


def _relative_name(path: Path, project_root: Path) -> str:
    return path.relative_to(project_root).as_posix()


def _annotation_quality(
    annotation_documents: list[dict[str, Any]],
    metadata: dict[str, Any] | None,
    complete: bool,
) -> dict[str, Any]:
    status_counts: dict[str, int] = {}
    supporting_questions = 0
    annotation_questions = 0
    evidence_references = 0
    supporting_references = 0

    for document in annotation_documents:
        for case in document.get("cases", []):
            if not isinstance(case, dict):
                continue
            annotation_questions += 1
            proposed_gold = case.get("proposed_gold", {})
            if not isinstance(proposed_gold, dict):
                continue
            assertions = proposed_gold.get("expected_assertions", [])
            case_supporting_ids: set[str] = set()
            for assertion in assertions if isinstance(assertions, list) else []:
                if not isinstance(assertion, dict):
                    continue
                references = assertion.get("evidence", [])
                for reference in references if isinstance(references, list) else []:
                    if not isinstance(reference, dict):
                        continue
                    evidence_references += 1
                    status = str(reference.get("verification_status") or "unspecified")
                    status_counts[status] = status_counts.get(status, 0) + 1
                    if reference.get("relation") == "supports":
                        supporting_references += 1
                        chunk_id = reference.get("chunk_id")
                        if isinstance(chunk_id, str) and chunk_id:
                            case_supporting_ids.add(chunk_id)
            supporting_questions += bool(case_supporting_ids)

    human_verified = sum(
        count
        for status, count in status_counts.items()
        if status.casefold() in {"human_verified", "verified"}
    )
    if metadata is not None:
        raw_human_verified = metadata.get("human_verified_evidence_reference_count")
        if isinstance(raw_human_verified, int) and not isinstance(raw_human_verified, bool):
            human_verified = raw_human_verified

    status = "Unavailable"
    if complete:
        observed_statuses = set(status_counts)
        if len(observed_statuses) == 1:
            status = next(iter(observed_statuses))
        elif observed_statuses:
            status = "Mixed annotation statuses"

    return {
        "annotation_inputs_complete": complete,
        "annotation_question_count": annotation_questions if complete else None,
        "evidence_reference_count": evidence_references if complete else None,
        "supporting_evidence_reference_count": supporting_references if complete else None,
        "evidence_status_counts": status_counts if complete else {},
        "human_verified_evidence_reference_count": human_verified if complete else None,
        "human_verification_performed": (
            metadata.get("human_evidence_verification_performed")
            if metadata is not None
            and isinstance(metadata.get("human_evidence_verification_performed"), bool)
            else human_verified > 0 if complete else None
        ),
        "questions_with_supporting_evidence_labels": (
            supporting_questions if complete else None
        ),
        "questions_without_supporting_evidence_labels": (
            annotation_questions - supporting_questions if complete else None
        ),
        "annotation_status": status,
    }


def load_evaluation_data(project_root: Path) -> EvaluationData:
    """Load saved result files without recalculating or filling in metrics."""
    project_root = Path(project_root)
    results_dir = project_root / "evaluation_results"
    tables: dict[str, list[dict[str, str]]] = {}
    errors: dict[str, str] = {}
    for key, (filename, required) in _CSV_FILES.items():
        path = results_dir / filename
        rows, error = _read_csv(path, required)
        if error:
            errors[_relative_name(path, project_root)] = error
        else:
            tables[key] = rows

    benchmark_path = (
        project_root
        / "benchmarks"
        / "answer_evaluation"
        / "financial_rag_60_question_benchmark.json"
    )
    benchmark, error = _read_json(benchmark_path)
    benchmark_cases = benchmark.get("cases") if benchmark else None
    benchmark_count = (
        len(benchmark_cases)
        if isinstance(benchmark_cases, list)
        and all(isinstance(case, dict) for case in benchmark_cases)
        else None
    )
    if error or benchmark_count is None:
        errors[_relative_name(benchmark_path, project_root)] = (
            error or "The file must contain a list of benchmark cases."
        )

    annotation_documents: list[dict[str, Any]] = []
    annotation_complete = True
    annotations_dir = benchmark_path.parent
    for filename in _ANNOTATION_FILES:
        path = annotations_dir / filename
        document, error = _read_json(path)
        cases = document.get("cases") if document else None
        if error or not isinstance(cases, list):
            annotation_complete = False
            errors[_relative_name(path, project_root)] = (
                error or "The file must contain a list of annotation cases."
            )
        else:
            annotation_documents.append(document)

    raw_path = results_dir / "raw_retrieval_results.json"
    raw_results, raw_error = _read_json(raw_path)
    metadata = raw_results.get("metadata") if raw_results else None
    if raw_error:
        errors[_relative_name(raw_path, project_root)] = raw_error
    if not isinstance(metadata, dict):
        metadata = None
        if raw_results is not None:
            errors[_relative_name(raw_path, project_root)] = (
                "The file is missing its evaluation metadata object."
            )

    failure_details: dict[str, list[dict[str, str]]] = {}
    raw_cases = raw_results.get("cases", []) if raw_results else []
    if isinstance(raw_cases, list):
        for case in raw_cases:
            if not isinstance(case, dict):
                continue
            question_id = case.get("question_id")
            strategies = case.get("strategies", {})
            if not isinstance(question_id, str) or not isinstance(strategies, dict):
                continue
            details = []
            for strategy, result in strategies.items():
                if not isinstance(result, dict) or result.get("status") == "completed":
                    continue
                details.append(
                    {
                        "strategy": str(strategy),
                        "status": str(result.get("status", "unknown")),
                        "reason": str(result.get("error", "")),
                    }
                )
            if details:
                failure_details[question_id] = details

    quality = _annotation_quality(
        annotation_documents,
        metadata,
        annotation_complete,
    )
    quality["benchmark_question_count"] = benchmark_count
    common_rows = tables.get("common_questions", [])
    common_label_counts = [
        _parse_integer(row.get("supporting_evidence_label_count"))
        for row in common_rows
    ]
    quality["common_subset_question_count"] = (
        sum(_is_true(row.get("common_subset")) for row in common_rows)
        if "common_questions" in tables
        else None
    )
    quality["common_subset_scored_question_count"] = (
        sum(
            _is_true(row.get("common_subset")) and label_count > 0
            for row, label_count in zip(common_rows, common_label_counts)
        )
        if "common_questions" in tables
        and all(label_count is not None for label_count in common_label_counts)
        else None
    )
    quality["common_subset_excluded_question_count"] = (
        sum(not _is_true(row.get("common_subset")) for row in common_rows)
        if "common_questions" in tables
        else None
    )
    quality["evaluation_record_count"] = (
        len(tables["per_query"]) if "per_query" in tables else None
    )
    return EvaluationData(tables, errors, quality, failure_details, metadata or {})


def _is_true(value: str | None) -> bool:
    return (value or "").strip().casefold() in {"true", "1", "yes"}


def _parse_integer(value: str | None) -> int | None:
    try:
        return int(value) if value is not None and value.strip() else None
    except ValueError:
        return None


def discover_strategies(rows: list[dict[str, str]]) -> list[str]:
    return list(dict.fromkeys(row["strategy"] for row in rows if row.get("strategy")))


def available_metric_cutoffs(rows: list[dict[str, str]]) -> list[int]:
    columns = {column for row in rows for column in row}
    return sorted(
        {
            int(match.group(1))
            for column in columns
            if (match := _K_COLUMN.fullmatch(column))
        }
    )


def available_metric_choices(rows: list[dict[str, str]]) -> list[tuple[str, str, str]]:
    columns = {column for row in rows for column in row}
    choices: list[tuple[str, str, str]] = []
    if "macro_mrr" in columns:
        choices.append(("Macro MRR", "macro", "MRR"))
    cutoffs = available_metric_cutoffs(rows)
    for family in ("macro", "micro"):
        for metric in ("Precision", "Recall", "F1", "nDCG"):
            if family == "micro" and metric == "nDCG":
                continue
            prefix, _ = _METRIC_PATTERNS[metric]
            if any(f"{family}_{prefix}@{cutoff}" in columns for cutoff in cutoffs):
                choices.append((f"{family.title()} {metric}", family, metric))
    return choices


def metric_column(family: str, metric: str, cutoff: int) -> str | None:
    if metric == "MRR":
        return "macro_mrr" if family == "macro" else None
    pattern = _METRIC_PATTERNS.get(metric)
    if pattern is None or (family == "micro" and metric == "nDCG"):
        return None
    return f"{family}_{pattern[0]}@{cutoff}"


def record_metric_column(metric: str, cutoff: int) -> str | None:
    if metric == "MRR":
        return "mrr"
    pattern = _METRIC_PATTERNS.get(metric)
    if pattern is None:
        return None
    return f"{pattern[1]}@{cutoff}"


def parse_number(value: str | None) -> float | None:
    if value is None or not value.strip():
        return None
    try:
        number = float(value)
    except ValueError:
        return None
    return number if math.isfinite(number) else None


def filter_records(
    rows: list[dict[str, str]],
    *,
    strategies: set[str] | None = None,
    company: str | None = None,
    category: str | None = None,
    subset: str = "all",
) -> list[dict[str, str]]:
    """Filter original per-query records without aggregating their metrics."""
    filtered = []
    for row in rows:
        if strategies is not None and row.get("strategy") not in strategies:
            continue
        if company and row.get("company") != company:
            continue
        if category and row.get("category") != category:
            continue
        is_common = _is_true(row.get("common_subset"))
        if subset == "common" and not is_common:
            continue
        if subset == "excluded" and is_common:
            continue
        filtered.append(row)
    return filtered


def _format_value(value: str | None, column: str) -> str:
    if column.endswith("_rate"):
        number = parse_number(value)
        return f"{number:.1%}" if number is not None else "Unavailable"
    if column.endswith("_seconds"):
        number = parse_number(value)
        return f"{number:.4f} s" if number is not None else "Unavailable"
    if column in {"query_count", "successful_queries", "failed_queries",
                  "unavailable_queries", "scored_queries", "empty_retrieval_count"}:
        try:
            return f"{int(value):,}" if value else "Unavailable"
        except ValueError:
            return "Unavailable"
    number = parse_number(value)
    if number is None:
        return "Unavailable"
    value_format = metric_value_format(column)
    if value_format == "percentage":
        return f"{number:.2%}"
    if value_format == "count":
        return f"{number:,.0f}"
    return f"{number:.4f}"


def metric_value_format(column: str) -> str:
    """Return display formatting only; the saved numeric value is unchanged."""
    normalized = column.casefold()
    if normalized in {"macro_mrr", "mrr"} or re.fullmatch(
        r"(?:macro_)?ndcg@\d+", normalized
    ):
        return "decimal"
    if re.fullmatch(r"(?:macro_|micro_)?(?:precision|recall|f1)@\d+", normalized):
        return "percentage"
    if re.fullmatch(r"hits@\d+", normalized):
        return "count"
    return "decimal"


def strategy_definitions(
    strategies: list[str],
    metadata: dict[str, Any],
) -> list[dict[str, str]]:
    availability = metadata.get("strategy_availability", {})
    if not isinstance(availability, dict):
        availability = {}
    definitions = []
    for strategy in strategies:
        description = _STRATEGY_DEFINITIONS.get(
            strategy,
            "Description requires verification; no strategy details are available.",
        )
        if strategy in {"keyword_semantic", "keyword_lexical"}:
            details = availability.get(strategy, {})
            chunk_count = (
                details.get("indexed_chunks") if isinstance(details, dict) else None
            )
            size = (
                f" Pilot index contains {chunk_count:,} indexed chunks in this evaluation."
                if isinstance(chunk_count, int) and not isinstance(chunk_count, bool)
                else " Pilot-index size is unavailable in the saved metadata."
            )
            description += size
        definitions.append({"strategy": strategy, "description": description})
    return definitions


def strategy_scope_warnings(strategies: list[str]) -> list[str]:
    present = set(strategies)
    warnings = []
    if {"baseline", "chunk_semantic"}.issubset(present):
        warnings.append(
            "Comparison limitation: `baseline` and `chunk_semantic` share the same "
            "direct Chroma semantic retrieval path. Their comparison is not evidence "
            "of two independent retrieval algorithms."
        )
    pilot_strategies = present.intersection({"keyword_semantic", "keyword_lexical"})
    if pilot_strategies:
        names = ", ".join(sorted(pilot_strategies))
        warnings.append(
            f"Comparison limitation: {names} use pilot indexes, not the complete "
            "production corpus. Their results are not directly equivalent to "
            "full-corpus comparisons."
        )
    return warnings


def _render_strategy_scope(strategies: list[str], metadata: dict[str, Any]) -> None:
    st.subheader("Retrieval Strategy Definitions & Evaluation Scope")
    definitions = strategy_definitions(strategies, metadata)
    if definitions:
        _render_table(
            definitions,
            [
                ("Strategy", "strategy", "raw"),
                ("Definition / limitation", "description", "raw"),
            ],
        )
    else:
        st.info("No retrieval strategies are present in the saved evaluation outputs.")
    for warning in strategy_scope_warnings(strategies):
        st.warning(warning)


def _render_table(
    rows: list[dict[str, str]],
    columns: list[tuple[str, str, str]],
) -> None:
    if not rows:
        st.info("No saved records are available for this view.")
        return
    header = "".join(f"<th>{html.escape(label)}</th>" for label, _, _ in columns)
    body = []
    for row in rows:
        cells = []
        for _, key, display in columns:
            value = row.get(key, "")
            if display == "company":
                value = _COMPANY_LABELS.get(value, value)
            elif display == "metric":
                value = _format_value(value, key)
            elif display == "raw":
                value = value or "Unavailable"
            cells.append(f"<td>{html.escape(value)}</td>")
        body.append("<tr>" + "".join(cells) + "</tr>")
    st.markdown(
        '<div style="overflow-x:auto"><table style="width:100%;border-collapse:collapse;'
        'font-size:.86rem"><thead><tr style="color:#9aabbe;text-align:left">'
        + header
        + "</tr></thead><tbody>"
        + "".join(body)
        + "</tbody></table></div>"
        "<style>table td,table th{padding:10px 12px;border-bottom:1px solid #263648}"
        "table td{color:#dce6f0}table tbody tr:hover{background:#142235}</style>",
        unsafe_allow_html=True,
    )


def _render_bar_chart(
    rows: list[dict[str, str]],
    value_column: str,
    label_builder,
    *,
    title: str,
    value_format: str,
) -> None:
    values = [
        (label_builder(row), parse_number(row.get(value_column)))
        for row in rows
    ]
    values = [(label, value) for label, value in values if value is not None]
    if not values:
        st.info("No comparable values are available in the saved results for this metric.")
        return

    bar_height = 30
    chart_height = max(112, 62 + len(values) * bar_height)
    label_width = 220
    bar_width = 430
    max_value = max(max(value for _, value in values), 0.000001)
    unit_label = {
        "percentage": "Percentage",
        "count": "Count",
        "decimal": "Decimal score",
    }.get(value_format, "Value")
    elements = [
        f'<svg role="img" aria-label="{html.escape(title)} ({unit_label})" '
        f'viewBox="0 0 760 '
        f'{chart_height}" width="100%" xmlns="http://www.w3.org/2000/svg">'
    ]
    for index, (label, value) in enumerate(values):
        y = 28 + index * bar_height
        width = max(0, min(bar_width, value / max_value * bar_width))
        if value_format == "percentage":
            display = f"{value:.2%}"
        elif value_format == "count":
            display = f"{value:,.0f}"
        else:
            display = f"{value:.4f}"
        safe_label = html.escape(label)
        elements.append(
            f"<g><title>{safe_label}: {display} {unit_label}</title>"
            f'<text x="0" y="{y + 15}" fill="#b9c7d7" font-size="12">'
            f"{safe_label}</text>"
            f'<rect x="{label_width}" y="{y}" width="{bar_width}" height="17" '
            'rx="4" fill="#1b2c3e"/>'
            f'<rect x="{label_width}" y="{y}" width="{width:.2f}" height="17" '
            'rx="4" fill="#25b6a5"/>'
            f'<text x="{label_width + bar_width + 10}" y="{y + 14}" '
            f'fill="#e7edf5" font-size="12">{display}</text>'
            "</g>"
        )
    axis_y = 32 + len(values) * bar_height
    for fraction in (0, 0.25, 0.5, 0.75, 1):
        x = label_width + fraction * bar_width
        axis_value = max_value * fraction
        if value_format == "percentage":
            tick_label = f"{axis_value:.0%}"
        elif value_format == "count":
            tick_label = f"{axis_value:,.0f}"
        else:
            tick_label = f"{axis_value:.3f}"
        elements.append(
            f'<line x1="{x:.2f}" y1="22" x2="{x:.2f}" y2="{axis_y - 4}" '
            'stroke="#263648" stroke-width="1"/>'
            f'<text x="{x:.2f}" y="{axis_y + 12}" text-anchor="middle" '
            f'fill="#8ea1b6" font-size="10">{tick_label}</text>'
        )
    elements.append(
        f'<text x="{label_width + bar_width / 2}" y="{axis_y + 28}" '
        f'text-anchor="middle" fill="#9aabbe" font-size="11">{unit_label}</text>'
    )
    elements.append("</svg>")
    st.markdown("".join(elements), unsafe_allow_html=True)


def _metric_rows(
    rows: list[dict[str, str]],
    strategies: set[str],
    metric_key: str | None,
) -> list[dict[str, str]]:
    return [
        row
        for row in rows
        if row.get("strategy") in strategies
        and (metric_key is None or metric_key in row)
    ]


def _render_metric_comparison(
    rows: list[dict[str, str]],
    strategies: set[str],
    metric_key: str | None,
    metric_label: str,
    *,
    dimension: str | None = None,
) -> None:
    if not rows:
        st.info("No saved aggregate results are available for this comparison.")
        return
    if metric_key is None:
        st.info(
            f"{metric_label} is not available as a saved aggregate. "
            "The dashboard does not calculate replacement metrics."
        )
        return
    selected_rows = _metric_rows(rows, strategies, metric_key)
    if not selected_rows:
        st.info("No saved aggregate rows match the selected strategy filters.")
        return
    columns = [
        ("Strategy", "strategy", "raw"),
    ]
    if dimension:
        columns.append(
            (
                "Company" if dimension == "company" else "Question category",
                dimension,
                "company" if dimension == "company" else "raw",
            )
        )
    columns.extend(
        [
            ("Questions", "query_count", "metric"),
            ("Scored questions", "scored_queries", "metric"),
            (metric_label, metric_key, "metric"),
        ]
    )
    _render_table(selected_rows, columns)
    label_builder = (
        (lambda row: f"{row.get(dimension, '')} · {row.get('strategy', '')}")
        if dimension
        else (lambda row: row.get("strategy", ""))
    )
    _render_bar_chart(
        selected_rows,
        metric_key,
        label_builder,
        title=metric_label,
        value_format=metric_value_format(metric_key),
    )


def _render_operations(
    rows: list[dict[str, str]],
    strategies: set[str],
    *,
    subset_label: str,
) -> None:
    if not rows:
        st.info("Saved operational metrics are unavailable.")
        return
    selected_rows = [row for row in rows if row.get("strategy") in strategies]
    st.caption(
        f"Operational results use the saved {subset_label.lower()} aggregate. "
        "Latency summarizes all measured strategy calls, including failed and "
        "unavailable calls; it is not success-only latency."
    )
    columns = [
        ("Strategy", "strategy", "raw"),
        ("Queries", "query_count", "metric"),
        ("Successful", "successful_queries", "metric"),
        ("Failed", "failed_queries", "metric"),
        ("Unavailable", "unavailable_queries", "metric"),
        ("Empty", "empty_retrieval_count", "metric"),
        ("Success rate", "successful_retrieval_rate", "metric"),
        ("Failure rate", "failed_retrieval_rate", "metric"),
        ("Unavailable rate", "unavailable_retrieval_rate", "metric"),
        ("Empty rate", "empty_retrieval_rate", "metric"),
        ("Mean latency", "mean_latency_seconds", "metric"),
        ("Median latency", "median_latency_seconds", "metric"),
        ("P95 latency", "p95_latency_seconds", "metric"),
        ("Minimum", "min_latency_seconds", "metric"),
        ("Maximum", "max_latency_seconds", "metric"),
    ]
    _render_table(selected_rows, columns)
    stat = st.selectbox(
        "Latency statistic",
        ["mean_latency_seconds", "median_latency_seconds", "p95_latency_seconds"],
        format_func=lambda value: {
            "mean_latency_seconds": "Mean",
            "median_latency_seconds": "Median",
            "p95_latency_seconds": "P95",
        }[value],
        key="evaluation_latency_stat",
    )
    _render_bar_chart(
        selected_rows,
        stat,
        lambda row: row.get("strategy", ""),
        title=f"Retrieval latency: {stat}",
        value_format="decimal",
    )


def _render_records(
    data: EvaluationData,
    strategies: set[str],
    metric: str,
    cutoff: int,
    subset: str,
) -> None:
    rows = data.tables.get("per_query")
    if rows is None:
        st.info("Per-query records are unavailable.")
        return
    companies = sorted({row.get("company", "") for row in rows if row.get("company")})
    categories = sorted({row.get("category", "") for row in rows if row.get("category")})
    company_options = ["All companies", *companies]
    category_options = ["All categories", *categories]
    first, second, third = st.columns(3)
    with first:
        company = st.selectbox("Record company", company_options, key="evaluation_record_company")
    with second:
        category = st.selectbox(
            "Record category", category_options, key="evaluation_record_category"
        )
    with third:
        record_subset = st.selectbox(
            "Record subset",
            ["Selected evaluation subset", "All records", "Common only", "Excluded only"],
            key="evaluation_record_subset",
        )

    subset_filter = {
        "Selected evaluation subset": subset,
        "All records": "all",
        "Common only": "common",
        "Excluded only": "excluded",
    }[record_subset]
    filtered = filter_records(
        rows,
        strategies=strategies,
        company=None if company == "All companies" else company,
        category=None if category == "All categories" else category,
        subset=subset_filter,
    )
    st.caption(f"{len(filtered):,} matching evaluation records · original per-query values")
    if not filtered:
        st.info("No evaluation records match these filters.")
        return

    page_size = 25
    page_count = (len(filtered) + page_size - 1) // page_size
    page = st.number_input(
        "Record page",
        min_value=1,
        max_value=page_count,
        step=1,
        key="evaluation_record_page",
    )
    record_column = record_metric_column(metric, cutoff)
    start = (int(page) - 1) * page_size
    for row in filtered[start : start + page_size]:
        company_label = _COMPANY_LABELS.get(row.get("company", ""), row.get("company", ""))
        title = (
            f"{row.get('question_id', 'Unknown question')} · {company_label} · "
            f"{row.get('strategy', 'Unknown strategy')}"
        )
        with st.expander(title):
            st.markdown(
                f"**Category:** {html.escape(row.get('category', 'Unavailable'))}  \n"
                f"**Retrieval:** {html.escape(row.get('retrieval_status', 'Unavailable'))} · "
                f"**Relevance labels:** {html.escape(row.get('metric_status', 'Unavailable'))}  \n"
                f"**Retrieved:** {html.escape(row.get('retrieved_count', 'Unavailable'))} · "
                f"**Supporting labels:** {html.escape(row.get('relevant_count', 'Unavailable'))} · "
                f"**Latency:** {_format_value(row.get('latency_seconds'), 'latency_seconds')}  \n"
                f"**Common subset:** {'Yes' if _is_true(row.get('common_subset')) else 'No'}",
                unsafe_allow_html=True,
            )
            if record_column:
                raw_value = row.get(record_column)
                display = _format_value(raw_value, record_column)
                st.caption(f"Per-question {metric} value: {display}")
            hit_columns = sorted(
                (column for column in row if re.fullmatch(r"hits@\d+", column)),
                key=lambda column: int(column.split("@", 1)[1]),
            )
            if hit_columns:
                hit_values = " · ".join(
                    f"{column}: {row.get(column) or 'Unavailable'}"
                    for column in hit_columns
                )
                st.caption(f"Saved per-query hit counts: {hit_values}")


def _render_quality(data: EvaluationData) -> None:
    quality = data.quality
    if quality.get("annotation_inputs_complete"):
        references = quality.get("evidence_reference_count")
        human_verified = quality.get("human_verified_evidence_reference_count")
        status_counts = quality.get("evidence_status_counts", {})
        proposed = sum(
            count
            for status, count in status_counts.items()
            if "proposed" in status.casefold() or "unverified" in status.casefold()
        )
        if proposed or human_verified == 0:
            st.warning(
                "Provisional evaluation — evidence annotations have not yet been "
                "independently human verified. Relevance metrics are not final research findings."
            )
        if quality.get("human_verification_performed") is False:
            st.caption("The saved evaluator metadata states that human evidence verification was not performed.")

        st.subheader("Evidence annotation status")
        st.write(
            f"Evidence references: **{references:,}** · "
            f"Human-verified references: **{human_verified:,}**"
            if references is not None and human_verified is not None
            else "Evidence annotation counts are unavailable."
        )
        if status_counts:
            status_rows = [
                {"status": status, "count": str(count)}
                for status, count in sorted(status_counts.items())
            ]
            _render_table(
                status_rows,
                [("Evidence status", "status", "raw"), ("References", "count", "raw")],
            )
        st.write(
            f"Questions with supporting evidence labels: "
            f"**{_display_count(quality.get('questions_with_supporting_evidence_labels'))}** · "
            f"Without supporting labels: "
            f"**{_display_count(quality.get('questions_without_supporting_evidence_labels'))}**"
        )
    else:
        st.info("Evidence annotation quality is unavailable because one or more annotation files could not be read.")

    st.subheader("Evaluation subset coverage")
    st.write(
        f"Benchmark questions: **{_display_count(quality.get('benchmark_question_count'))}** · "
        f"Common-subset questions: **{_display_count(quality.get('common_subset_question_count'))}** · "
        f"Common-subset questions with supporting labels: "
        f"**{_display_count(quality.get('common_subset_scored_question_count'))}** · "
        f"Excluded from common subset: "
        f"**{_display_count(quality.get('common_subset_excluded_question_count'))}**"
    )
    st.write(
        f"Full-evaluation query records: **{_display_count(quality.get('evaluation_record_count'))}**"
    )

    common_rows = data.tables.get("common_questions", [])
    excluded = [row for row in common_rows if not _is_true(row.get("common_subset"))]
    if excluded:
        st.subheader("Questions excluded from the common subset")
        exclusion_rows = []
        per_query = data.tables.get("per_query", [])
        for question in excluded:
            question_id = question.get("question_id", "")
            details = data.failure_details.get(question_id, [])
            if details:
                reason = "; ".join(
                    f"{detail['strategy']}: {detail['reason'] or detail['status']}"
                    for detail in details
                )
            else:
                statuses = {
                    row.get("retrieval_status", "")
                    for row in per_query
                    if row.get("question_id") == question_id
                    and row.get("retrieval_status") != "completed"
                }
                reason = ", ".join(sorted(statuses)) or "No exclusion reason is recorded."
            exclusion_rows.append(
                {
                    "question_id": question_id,
                    "company": question.get("company", ""),
                    "category": question.get("category", ""),
                    "reason": reason,
                }
            )
        _render_table(
            exclusion_rows,
            [
                ("Question ID", "question_id", "raw"),
                ("Company", "company", "company"),
                ("Category", "category", "raw"),
                ("Recorded exclusion reason", "reason", "raw"),
            ],
        )


def _display_count(value: Any) -> str:
    return f"{value:,}" if isinstance(value, int) and not isinstance(value, bool) else "Unavailable"


def _strategy_display_name(strategy: str) -> str:
    labels = {
        "baseline": "Baseline",
        "chunk_semantic": "Chunk Semantic",
        "keyword_semantic": "Keyword Semantic",
        "keyword_lexical": "Keyword Lexical",
        "hybrid_chunk_keyword": "Hybrid Chunk + Keyword",
        "hybrid_chunk_lexical": "Hybrid Chunk + Lexical",
        "hybrid_all": "Hybrid All",
    }
    return labels.get(strategy, strategy.replace("_", " ").title())


def _render_default_metric_cards(data: EvaluationData, strategy: str) -> None:
    overall_rows = data.tables.get("overall") or []
    if strategy == "All":
        rows = overall_rows
        metric_specs = [
            ("Precision@5", metric_column("macro", "Precision", 5), "Proportion of the top five retrieved chunks that are relevant."),
            ("Recall@5", metric_column("macro", "Recall", 5), "Proportion of all known relevant chunks retrieved within the top five."),
            ("Recall@10", metric_column("macro", "Recall", 10), "Proportion of all known relevant chunks retrieved within the top ten."),
            ("MRR", metric_column("macro", "MRR", 5), "How highly the first relevant chunk is ranked."),
        ]
        st.subheader("All strategies")
        table_rows = []
        for row in rows:
            strategy_name = row.get("strategy", "")
            metric_values = {
                "Retrieval Strategy": _strategy_display_name(strategy_name),
                "Precision@5": _format_value(row.get(metric_specs[0][1]), metric_specs[0][1]),
                "Recall@5": _format_value(row.get(metric_specs[1][1]), metric_specs[1][1]),
                "Recall@10": _format_value(row.get(metric_specs[2][1]), metric_specs[2][1]),
                "MRR": _format_value(row.get(metric_specs[3][1]), metric_specs[3][1]),
            }
            table_rows.append(metric_values)
        if table_rows:
            _render_table(
                table_rows,
                [
                    ("Retrieval Strategy", "Retrieval Strategy", "raw"),
                    ("Precision@5", "Precision@5", "raw"),
                    ("Recall@5", "Recall@5", "raw"),
                    ("Recall@10", "Recall@10", "raw"),
                    ("MRR", "MRR", "raw"),
                ],
            )
        return

    row = next((item for item in overall_rows if item.get("strategy") == strategy), None)
    if row is None:
        st.info("No saved strategy metrics are available for the selected retrieval strategy.")
        return

    metric_specs = [
        (
            "Precision@5",
            metric_column("macro", "Precision", 5),
            "Proportion of the top five retrieved chunks that are relevant.",
        ),
        (
            "Recall@5",
            metric_column("macro", "Recall", 5),
            "Proportion of all known relevant chunks retrieved within the top five.",
        ),
        (
            "Recall@10",
            metric_column("macro", "Recall", 10),
            "Proportion of all known relevant chunks retrieved within the top ten.",
        ),
        (
            "MRR",
            metric_column("macro", "MRR", 5),
            "How highly the first relevant chunk is ranked.",
        ),
    ]

    first_row = st.columns(2)
    for column, (label, metric_key, description) in zip(first_row, metric_specs[:2]):
        with column:
            value = _format_value(row.get(metric_key), metric_key) if metric_key else "N/A"
            st.metric(label, value, help=description)
            st.caption(description)

    second_row = st.columns(2)
    for column, (label, metric_key, description) in zip(second_row, metric_specs[2:]):
        with column:
            value = _format_value(row.get(metric_key), metric_key) if metric_key else "N/A"
            st.metric(label, value, help=description)
            st.caption(description)


def render_evaluation_dashboard(data: EvaluationData) -> None:
    for path, error in data.errors.items():
        st.warning(f"Evaluation source unavailable — `{path}`: {error}")

    st.markdown("<h2>Retrieval Performance</h2>", unsafe_allow_html=True)
    st.caption("Evaluate how effectively the system retrieves relevant financial evidence.")

    quality = data.quality
    status_counts = quality.get("evidence_status_counts", {})
    proposed = sum(
        count
        for status, count in status_counts.items()
        if "proposed" in status.casefold() or "unverified" in status.casefold()
    )
    if proposed or (
        quality.get("human_verification_performed") is False
        and quality.get("human_verified_evidence_reference_count") == 0
    ):
        st.warning(
            "Provisional evaluation — evidence annotations have not yet been independently "
            "human verified. Relevance metrics are not final research findings."
        )

    overall_rows = data.tables.get("overall")
    strategies = discover_strategies(overall_rows or [])
    if not strategies:
        st.info("Strategy-level metrics are unavailable until metrics_overall.csv can be read.")
        return

    preferred_order = [
        "All",
        "baseline",
        "chunk_semantic",
        "hybrid_all",
        "hybrid_chunk_keyword",
        "hybrid_chunk_lexical",
        "keyword_lexical",
        "keyword_semantic",
    ]
    strategy_options = [
        option for option in preferred_order if option == "All" or option in strategies
    ] + [strategy for strategy in strategies if strategy not in preferred_order]
    strategy_default = "All"
    selected_strategy = st.selectbox(
        "Retrieval Strategy",
        strategy_options,
        index=strategy_options.index(strategy_default),
        format_func=lambda value: "All" if value == "All" else _strategy_display_name(value),
        key="evaluation_strategy_selector",
    )

    _render_default_metric_cards(data, selected_strategy)

    expanded = st.session_state.get("evaluation_more_statistics", False)
    button_label = "Hide Statistics" if expanded else "More Statistics"
    if st.button(button_label, key="evaluation_more_statistics_toggle"):
        st.session_state.evaluation_more_statistics = not expanded
        st.rerun()

    if expanded:
        common_available = "common" in data.tables
        subset_options = ["Common successful subset", "Full evaluation"]
        subset_choice = st.selectbox(
            "Evaluation subset",
            subset_options if common_available else ["Full evaluation"],
            index=0,
            key="evaluation_subset",
        )
        subset = "common" if subset_choice == "Common successful subset" else "full"
        metric_choices = available_metric_choices(overall_rows or [])
        if metric_choices:
            choice_labels = [label for label, _, _ in metric_choices]
            default_choice = (
                "Macro F1"
                if "Macro F1" in choice_labels
                else choice_labels[0]
            )
            selected_label = st.selectbox(
                "Metric",
                choice_labels,
                index=choice_labels.index(default_choice),
                key="evaluation_metric_choice",
            )
            _, family, metric = next(item for item in metric_choices if item[0] == selected_label)
        else:
            selected_label, family, metric = "Macro F1@5", "macro", "F1"
            st.info("No supported aggregate metric columns were found in the saved output.")
        cutoffs = available_metric_cutoffs(overall_rows or [])
        if cutoffs:
            default_k = 5 if 5 in cutoffs else cutoffs[0]
            cutoff = st.selectbox(
                "K cutoff",
                cutoffs,
                index=cutoffs.index(default_k),
                key="evaluation_k_cutoff",
            )
        else:
            cutoff = 5
        metric_label = (
            selected_label
            if metric == "MRR"
            else f"{selected_label}@{cutoff}"
        )
        st.caption(
            "Macro metrics average eligible per-question scores; micro metrics are pooled "
            "aggregates where saved. MRR and nDCG are macro-only. Hit@K is present only "
            "in per-query records, not as a saved aggregate."
        )
        selected_strategies = st.multiselect(
            "Retrieval strategies",
            strategies,
            default=strategies,
            key="evaluation_strategies",
        )
        strategy_set = set(selected_strategies)
        metric_key = metric_column(family, metric, cutoff)
        _render_strategy_scope(strategies, data.metadata)

        overview_tab, company_tab, category_tab, operations_tab, records_tab, quality_tab = st.tabs(
            ["Strategy comparison", "By company", "By category", "Operations", "Records", "Dataset quality"]
        )
        with overview_tab:
            source_key = "common" if subset == "common" else "overall"
            source_rows = data.tables.get(source_key)
            if source_rows is None:
                source_filename = _CSV_FILES[source_key][0]
                st.info(f"Saved aggregate `{source_filename}` is unavailable.")
            else:
                st.caption(
                    "Comparisons use the same common successful question subset across strategies."
                    if subset == "common"
                    else "Comparisons use the full saved strategy-level evaluation aggregates."
                )
                _render_metric_comparison(
                    source_rows,
                    strategy_set,
                    metric_key,
                    metric_label,
                )
                if strategies:
                    st.caption(
                        "Strategy outputs are shown as saved; no winner is declared because "
                        "relevance labels remain provisional."
                    )
        with company_tab:
            if subset == "common":
                st.info(
                    "The saved outputs do not include common-subset aggregates by company. "
                    "No company-level common-subset scores are recomputed here."
                )
            else:
                _render_metric_comparison(
                    data.tables.get("company", []),
                    strategy_set,
                    metric_key,
                    metric_label,
                    dimension="company",
                )

        with category_tab:
            if subset == "common":
                st.info(
                    "The saved outputs do not include common-subset aggregates by category. "
                    "No category-level common-subset scores are recomputed here."
                )
            else:
                _render_metric_comparison(
                    data.tables.get("category", []),
                    strategy_set,
                    metric_key,
                    metric_label,
                    dimension="category",
                )

        with operations_tab:
            operation_key = "common" if subset == "common" else "overall"
            _render_operations(
                data.tables.get(operation_key, []),
                strategy_set,
                subset_label=subset_choice,
            )

        with records_tab:
            _render_records(
                data,
                strategy_set,
                metric,
                cutoff,
                subset,
            )

        with quality_tab:
            _render_quality(data)
