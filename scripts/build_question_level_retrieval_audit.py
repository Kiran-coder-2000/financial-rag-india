"""Build a question-level retrieval comparison workbook from saved evaluation data."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

import xlsxwriter


PROJECT_ROOT = Path(__file__).resolve().parent.parent
BENCHMARK_PATH = (
    PROJECT_ROOT
    / "benchmarks"
    / "answer_evaluation"
    / "financial_rag_60_question_benchmark.json"
)
ANNOTATION_PATHS = (
    PROJECT_ROOT / "benchmarks" / "answer_evaluation" / "tcs_q01_q20_gold_annotations.json",
    PROJECT_ROOT / "benchmarks" / "answer_evaluation" / "infosys_gold_annotations.json",
    PROJECT_ROOT / "benchmarks" / "answer_evaluation" / "reliance_gold_annotations.json",
)
RAW_RESULTS_PATH = PROJECT_ROOT / "evaluation_results" / "raw_retrieval_results.json"
PER_QUERY_METRICS_PATH = PROJECT_ROOT / "evaluation_results" / "per_query_metrics.csv"
CHUNKS_PATH = PROJECT_ROOT / "data" / "chunks" / "all_chunks.jsonl"
DEFAULT_OUTPUT_PATH = PROJECT_ROOT / "evaluation_results" / "question_level_retrieval_audit.xlsx"
DEFAULT_REPORT_PATH = (
    PROJECT_ROOT / "evaluation_results" / "question_level_retrieval_audit_validation.md"
)

METRIC_FIELDS = (
    ("Precision@5", "precision@5"),
    ("Recall@5", "recall@5"),
    ("Recall@10", "recall@10"),
    ("MRR", "mrr"),
)
ERROR_CLASSIFICATIONS = (
    "Relevant chunk not retrieved",
    "Relevant chunk retrieved but ranked too low",
    "Wrong financial year",
    "Wrong company",
    "Relevant information split across chunks",
    "Metadata filtering issue",
    "Possible chunking issue",
    "Gold annotation incomplete",
    "Other",
    "Needs review",
)
REVIEW_STATUSES = ("Pending", "In progress", "Reviewed")
MANUAL_RELEVANCE_DECISIONS = ("supports", "context_only", "not_relevant", "needs_review")
EXCEL_CELL_TEXT_LIMIT = 32767
def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"Could not read JSON input {path}: {error}") from error
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}.")
    return value


def _read_csv(path: Path) -> list[dict[str, str]]:
    try:
        with path.open(encoding="utf-8-sig", newline="") as source:
            return list(csv.DictReader(source))
    except OSError as error:
        raise ValueError(f"Could not read CSV input {path}: {error}") from error


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _source_paths() -> list[Path]:
    return [
        BENCHMARK_PATH,
        *ANNOTATION_PATHS,
        RAW_RESULTS_PATH,
        PER_QUERY_METRICS_PATH,
        CHUNKS_PATH,
    ]


def load_chunks(path: Path = CHUNKS_PATH) -> dict[str, dict[str, Any]]:
    chunks: dict[str, dict[str, Any]] = {}
    try:
        with path.open(encoding="utf-8") as source:
            for line_number, line in enumerate(source, start=1):
                if not line.strip():
                    continue
                try:
                    chunk = json.loads(line)
                except json.JSONDecodeError as error:
                    raise ValueError(
                        f"Invalid JSON in {path} at line {line_number}: {error}"
                    ) from error
                if not isinstance(chunk, dict):
                    raise ValueError(f"Chunk at line {line_number} is not a JSON object.")
                chunk_id = chunk.get("chunk_id")
                if not isinstance(chunk_id, str) or not chunk_id:
                    raise ValueError(f"Chunk at line {line_number} has no valid chunk_id.")
                if chunk_id in chunks:
                    raise ValueError(f"Duplicate chunk ID {chunk_id!r} in {path}.")
                chunks[chunk_id] = chunk
    except OSError as error:
        raise ValueError(f"Could not read chunk corpus {path}: {error}") from error
    return chunks


def _periods_text(case: dict[str, Any]) -> str:
    periods = case.get("financial_periods")
    if not isinstance(periods, list):
        return ""
    return "; ".join(str(period) for period in periods)


def _optional_number(value: Any) -> float | int | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"Expected a saved numeric value or blank, received {value!r}.") from error


def _optional_int(value: Any) -> int | None:
    if value is None or value == "":
        return None
    try:
        return int(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"Expected a saved integer value or blank, received {value!r}.") from error


def prepare_audit_data(
    benchmark: dict[str, Any],
    raw_results: dict[str, Any],
    metric_rows: list[dict[str, str]],
    annotation_datasets: list[dict[str, Any]],
    chunks_by_id: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    benchmark_cases = benchmark.get("cases")
    raw_cases_list = raw_results.get("cases")
    if not isinstance(benchmark_cases, list) or not isinstance(raw_cases_list, list):
        raise ValueError("Benchmark and raw results must each contain a cases list.")

    benchmark_by_id: dict[str, dict[str, Any]] = {}
    for case in benchmark_cases:
        question_id = case.get("id")
        if not isinstance(question_id, str) or not question_id:
            raise ValueError("Benchmark contains a case with no valid question ID.")
        if question_id in benchmark_by_id:
            raise ValueError(f"Duplicate benchmark question ID {question_id!r}.")
        benchmark_by_id[question_id] = case

    raw_by_id: dict[str, dict[str, Any]] = {}
    for case in raw_cases_list:
        question_id = case.get("question_id")
        if question_id not in benchmark_by_id:
            raise ValueError(f"Raw results contain unknown question ID {question_id!r}.")
        if question_id in raw_by_id:
            raise ValueError(f"Duplicate raw result for question {question_id!r}.")
        benchmark_case = benchmark_by_id[question_id]
        for key in ("question", "company", "category"):
            if case.get(key) != benchmark_case.get(key):
                raise ValueError(
                    f"Raw result {question_id!r} does not match benchmark field {key!r}."
                )
        raw_by_id[question_id] = case
    if set(raw_by_id) != set(benchmark_by_id):
        missing = sorted(set(benchmark_by_id) - set(raw_by_id))
        raise ValueError(f"Raw results are missing benchmark questions: {missing}")

    strategy_names = list(raw_by_id[next(iter(benchmark_by_id))].get("strategies", {}))
    if not strategy_names:
        raise ValueError("Raw results do not contain any retrieval strategies.")

    metrics_by_key: dict[tuple[str, str], dict[str, Any]] = {}
    for row in metric_rows:
        key = (row.get("question_id", ""), row.get("strategy", ""))
        if key in metrics_by_key:
            raise ValueError(f"Duplicate per-query metrics row for {key!r}.")
        if key[0] not in benchmark_by_id or key[1] not in strategy_names:
            raise ValueError(f"Per-query metrics contain an unknown question or strategy: {key!r}.")
        benchmark_case = benchmark_by_id[key[0]]
        for field in ("company", "category"):
            if row.get(field) != benchmark_case.get(field):
                raise ValueError(f"Metrics row {key!r} does not match benchmark {field!r}.")
        parsed = dict(row)
        parsed["relevant_count"] = _optional_int(row.get("relevant_count"))
        parsed["retrieved_count"] = _optional_int(row.get("retrieved_count"))
        parsed["common_subset"] = row.get("common_subset", "").strip().lower() == "true"
        for _, metric_key in METRIC_FIELDS:
            parsed[metric_key] = _optional_number(row.get(metric_key))
        metrics_by_key[key] = parsed
    expected_metric_keys = {
        (question_id, strategy)
        for question_id in benchmark_by_id
        for strategy in strategy_names
    }
    if set(metrics_by_key) != expected_metric_keys:
        missing = sorted(expected_metric_keys - set(metrics_by_key))
        extra = sorted(set(metrics_by_key) - expected_metric_keys)
        raise ValueError(f"Per-query metric rows do not match question/strategy combinations; missing={missing}, extra={extra}.")

    gold_by_question: dict[str, list[dict[str, Any]]] = defaultdict(list)
    annotation_case_ids: set[str] = set()
    for dataset in annotation_datasets:
        cases = dataset.get("cases")
        if not isinstance(cases, list):
            raise ValueError("Gold annotation file must contain a cases list.")
        for case in cases:
            question_id = case.get("id")
            benchmark_case = benchmark_by_id.get(question_id)
            if benchmark_case is None:
                raise ValueError(f"Gold annotation references unknown question {question_id!r}.")
            if question_id in annotation_case_ids:
                raise ValueError(f"Gold annotations duplicate question {question_id!r}.")
            annotation_case_ids.add(question_id)
            if case.get("question") != benchmark_case.get("question"):
                raise ValueError(f"Gold annotation question text differs for {question_id!r}.")
            proposed_gold = case.get("proposed_gold")
            assertions = proposed_gold.get("expected_assertions") if isinstance(proposed_gold, dict) else None
            if not isinstance(assertions, list):
                raise ValueError(f"Gold annotation {question_id!r} has no expected_assertions list.")
            reference_number = 0
            for assertion in assertions:
                assertion_id = assertion.get("assertion_id", "")
                evidence_list = assertion.get("evidence")
                if not isinstance(evidence_list, list):
                    raise ValueError(f"Gold assertion {assertion_id!r} has no evidence list.")
                for evidence_index, evidence in enumerate(evidence_list, start=1):
                    reference_number += 1
                    chunk_id = evidence.get("chunk_id")
                    if not isinstance(chunk_id, str) or not chunk_id:
                        chunk_id = ""
                    gold_by_question[question_id].append({
                        "evidence_reference_id": (
                            f"{question_id}::{assertion_id}::evidence-{evidence_index:02d}"
                        ),
                        "assertion_id": assertion_id,
                        "reference_number": reference_number,
                        "chunk_id": chunk_id,
                        "relation": evidence.get("relation", ""),
                        "verification_status": evidence.get("verification_status", ""),
                        "verified_by": evidence.get("verified_by"),
                        "rationale": evidence.get("rationale", ""),
                    })

    missing_retrieved_chunks: list[tuple[str, str, str]] = []
    retrieved_text_mismatches: list[tuple[str, str, str]] = []
    retrieval_record_count = 0
    retrieval_status_counts: Counter[str] = Counter()
    for question_id, raw_case in raw_by_id.items():
        expected_support_ids = {
            entry["chunk_id"]
            for entry in gold_by_question.get(question_id, [])
            if entry["relation"] == "supports" and entry["chunk_id"]
        }
        raw_support_ids = set(raw_case.get("relevant_chunk_ids", []))
        if raw_support_ids != expected_support_ids:
            raise ValueError(
                f"Raw support IDs do not match gold supports for {question_id!r}: "
                f"raw={sorted(raw_support_ids)}, gold={sorted(expected_support_ids)}"
            )
        strategies = raw_case.get("strategies")
        if not isinstance(strategies, dict) or list(strategies) != strategy_names:
            raise ValueError(f"Strategy mapping differs for question {question_id!r}.")
        for strategy in strategy_names:
            result = strategies[strategy]
            retrieval_status_counts[result.get("status", "missing")] += 1
            retrieved = result.get("retrieved")
            if not isinstance(retrieved, list):
                raise ValueError(f"Retrieved chunk list is missing for {question_id!r}/{strategy}.")
            if result.get("retrieved_count") != len(retrieved):
                raise ValueError(f"Retrieved count does not match records for {question_id!r}/{strategy}.")
            metric_row = metrics_by_key[(question_id, strategy)]
            if metric_row["retrieved_count"] != len(retrieved):
                raise ValueError(f"Saved retrieved_count mismatch for {question_id!r}/{strategy}.")
            if metric_row["retrieval_status"] != result.get("status"):
                raise ValueError(f"Retrieval status mismatch for {question_id!r}/{strategy}.")
            ranks = [record.get("rank") for record in retrieved]
            if ranks != list(range(1, len(retrieved) + 1)):
                raise ValueError(f"Saved retrieval ranks are not consecutive for {question_id!r}/{strategy}.")
            for record in retrieved:
                retrieval_record_count += 1
                chunk_id = record.get("chunk_id")
                if not isinstance(chunk_id, str) or not chunk_id:
                    missing_retrieved_chunks.append((question_id, strategy, str(chunk_id)))
                    continue
                chunk = chunks_by_id.get(chunk_id)
                if chunk is None:
                    missing_retrieved_chunks.append((question_id, strategy, chunk_id))
                elif record.get("text", "") != chunk.get("text", ""):
                    retrieved_text_mismatches.append((question_id, strategy, chunk_id))

    missing_gold_chunks = [
        (question_id, evidence["evidence_reference_id"], evidence["chunk_id"])
        for question_id, evidence_rows in gold_by_question.items()
        for evidence in evidence_rows
        if not evidence["chunk_id"] or evidence["chunk_id"] not in chunks_by_id
    ]
    return {
        "benchmark": benchmark,
        "benchmark_by_id": benchmark_by_id,
        "raw_by_id": raw_by_id,
        "metrics_by_key": metrics_by_key,
        "strategy_names": strategy_names,
        "gold_by_question": dict(gold_by_question),
        "chunks_by_id": chunks_by_id,
        "missing_retrieved_chunks": missing_retrieved_chunks,
        "missing_gold_chunks": missing_gold_chunks,
        "retrieved_text_mismatches": retrieved_text_mismatches,
        "retrieval_record_count": retrieval_record_count,
        "retrieval_status_counts": retrieval_status_counts,
    }


def _metric_values(metric_row: dict[str, Any]) -> dict[str, float | int | None]:
    return {display: metric_row.get(key) for display, key in METRIC_FIELDS}


def _evaluation_status(raw_status: str, metric_status: str) -> str:
    return f"retrieval={raw_status}; metrics={metric_status}"


def _supporting_ids(gold_rows: list[dict[str, Any]]) -> list[str]:
    return list(dict.fromkeys(
        row["chunk_id"] for row in gold_rows
        if row["relation"] == "supports" and row["chunk_id"]
    ))


def _gold_by_chunk(gold_rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    result: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in gold_rows:
        if row["chunk_id"]:
            result[row["chunk_id"]].append(row)
    return dict(result)


def _base_case_fields(case: dict[str, Any]) -> dict[str, str]:
    return {
        "Question ID": case["id"],
        "Category": str(case.get("category", "")),
        "Company": str(case.get("company", "")),
        "Financial year": _periods_text(case),
        "Benchmark question": str(case.get("question", "")),
    }


def _performance_rows(model: dict[str, Any]) -> Iterable[dict[str, Any]]:
    for question_id, case in model["benchmark_by_id"].items():
        raw_case = model["raw_by_id"][question_id]
        base = _base_case_fields(case)
        for strategy in model["strategy_names"]:
            metric_row = model["metrics_by_key"][(question_id, strategy)]
            result = raw_case["strategies"][strategy]
            yield {
                **base,
                "Retrieval strategy": strategy,
                "Number of retrieved chunks": result.get("retrieved_count"),
                "Number of known supporting chunks": metric_row.get("relevant_count"),
                "Precision@5": metric_row.get("precision@5"),
                "Recall@5": metric_row.get("recall@5"),
                "Recall@10": metric_row.get("recall@10"),
                "MRR": metric_row.get("mrr"),
                "Evaluation status": _evaluation_status(
                    result.get("status", "missing"),
                    metric_row.get("metric_status", "missing"),
                ),
                "Common evaluation subset": metric_row.get("common_subset"),
                "Retrieval error": result.get("error", ""),
            }


def _retrieved_rows(model: dict[str, Any]) -> Iterable[dict[str, Any]]:
    for question_id, case in model["benchmark_by_id"].items():
        raw_case = model["raw_by_id"][question_id]
        base = _base_case_fields(case)
        gold_chunk_ids = _gold_by_chunk(model["gold_by_question"].get(question_id, []))
        for strategy in model["strategy_names"]:
            result = raw_case["strategies"][strategy]
            retrieved = result.get("retrieved", [])
            if not retrieved:
                status = "failed" if result.get("status") == "failed" else "completed_empty"
                yield {
                    **base,
                    "Retrieval strategy": strategy,
                    "Retrieval rank": None,
                    "Chunk ID": "",
                    "Document name": "",
                    "Financial year": "",
                    "Question financial year": base["Financial year"],
                    "Chunk company": "",
                    "Page number": None,
                    "Full original chunk text": "",
                    "Manual relevance decision": "",
                    "Reviewer comments": "",
                    "Retrieval record status": status,
                    "Retrieval error": result.get("error", ""),
                    "Chunk lookup status": "not_applicable",
                    "Gold evidence type by exact ID": "",
                }
                continue
            for record in retrieved:
                chunk_id = record["chunk_id"]
                chunk = model["chunks_by_id"].get(chunk_id)
                relation_values = list(dict.fromkeys(
                    str(item["relation"]) for item in gold_chunk_ids.get(chunk_id, [])
                ))
                yield {
                    **base,
                    "Retrieval strategy": strategy,
                    "Retrieval rank": record.get("rank"),
                    "Chunk ID": chunk_id,
                    "Document name": str(chunk.get("document", "")) if chunk else "",
                    "Financial year": str(chunk.get("financial_year", "")) if chunk else "",
                    "Question financial year": base["Financial year"],
                    "Chunk company": str(chunk.get("company", "")) if chunk else "",
                    "Page number": chunk.get("page") if chunk else None,
                    "Full original chunk text": str(chunk.get("text", "")) if chunk else "",
                    "Manual relevance decision": "",
                    "Reviewer comments": "",
                    "Retrieval record status": result.get("status", "missing"),
                    "Retrieval error": result.get("error", ""),
                    "Chunk lookup status": "found" if chunk else "missing",
                    "Gold evidence type by exact ID": "; ".join(relation_values),
                }


def _expected_rows(model: dict[str, Any]) -> Iterable[dict[str, Any]]:
    for question_id, case in model["benchmark_by_id"].items():
        base = _base_case_fields(case)
        for evidence in model["gold_by_question"].get(question_id, []):
            chunk = model["chunks_by_id"].get(evidence["chunk_id"])
            yield {
                **base,
                "Expected chunk ID": evidence["chunk_id"],
                "Document": str(chunk.get("document", "")) if chunk else "",
                "Financial year": str(chunk.get("financial_year", "")) if chunk else "",
                "Question financial year": base["Financial year"],
                "Chunk company": str(chunk.get("company", "")) if chunk else "",
                "Page number": chunk.get("page") if chunk else None,
                "Full original chunk text": str(chunk.get("text", "")) if chunk else "",
                "Evidence type": evidence["relation"],
                "Annotation verification status": evidence["verification_status"],
                "Reviewer comments": "",
                "Assertion ID": evidence["assertion_id"],
                "Evidence reference ID": evidence["evidence_reference_id"],
                "Annotation rationale (unverified source text)": evidence["rationale"],
                "Chunk lookup status": "found" if chunk else "missing",
            }


def _comparison_rows(model: dict[str, Any]) -> Iterable[dict[str, Any]]:
    for question_id, case in model["benchmark_by_id"].items():
        raw_case = model["raw_by_id"][question_id]
        base = _base_case_fields(case)
        gold_rows = model["gold_by_question"].get(question_id, [])
        by_chunk = _gold_by_chunk(gold_rows)
        supporting = [row for row in gold_rows if row["relation"] == "supports"]
        context_only = [row for row in gold_rows if row["relation"] == "context_only"]
        supporting_ids = set(_supporting_ids(gold_rows))
        for strategy in model["strategy_names"]:
            result = raw_case["strategies"][strategy]
            metric_row = model["metrics_by_key"][(question_id, strategy)]
            ranks = {
                record["chunk_id"]: record["rank"]
                for record in result.get("retrieved", [])
            }
            common = {
                **base,
                "Retrieval strategy": strategy,
                **_metric_values(metric_row),
                "Evaluation status": _evaluation_status(
                    result.get("status", "missing"),
                    metric_row.get("metric_status", "missing"),
                ),
                "Common evaluation subset": metric_row.get("common_subset"),
            }
            if result.get("retrieved"):
                for record in result["retrieved"]:
                    chunk_id = record["chunk_id"]
                    chunk = model["chunks_by_id"].get(chunk_id)
                    relations = list(dict.fromkeys(
                        str(row["relation"]) for row in by_chunk.get(chunk_id, [])
                    ))
                    yield {
                        **common,
                        "Comparison item": "Retrieved result",
                        "Chunk ID": chunk_id,
                        "Retrieval rank": record.get("rank"),
                        "Expected evidence type": "; ".join(relations),
                        "Exact chunk ID match": "Yes" if relations else "No",
                        "Supporting expected chunk retrieved?": (
                            "Yes" if chunk_id in supporting_ids else "No"
                        ) if supporting_ids else "No supporting gold IDs",
                        "Expected supporting retrieval rank": record.get("rank") if chunk_id in supporting_ids else None,
                        "Document": str(chunk.get("document", "")) if chunk else "",
                        "Chunk financial year": str(chunk.get("financial_year", "")) if chunk else "",
                        "Chunk company": str(chunk.get("company", "")) if chunk else "",
                        "Page number": chunk.get("page") if chunk else None,
                        "Full original chunk text": str(chunk.get("text", "")) if chunk else "",
                        "Chunk lookup status": "found" if chunk else "missing",
                        "Retrieval record status": result.get("status", "missing"),
                        "Evidence reference ID": "",
                    }
            else:
                yield {
                    **common,
                    "Comparison item": (
                        "Retrieval failed"
                        if result.get("status") == "failed"
                        else "No retrieved chunks"
                    ),
                    "Chunk ID": "",
                    "Retrieval rank": None,
                    "Expected evidence type": "",
                    "Exact chunk ID match": "",
                    "Supporting expected chunk retrieved?": (
                        "No" if supporting_ids else "No supporting gold IDs"
                    ),
                    "Expected supporting retrieval rank": None,
                    "Document": "",
                    "Chunk financial year": "",
                    "Chunk company": "",
                    "Page number": None,
                    "Full original chunk text": "",
                    "Chunk lookup status": "not_applicable",
                    "Retrieval record status": result.get("status", "missing"),
                    "Evidence reference ID": "",
                }

            for section, evidence_rows in (
                ("Expected supporting evidence", supporting),
                ("Expected context-only evidence", context_only),
            ):
                if not evidence_rows:
                    if section == "Expected supporting evidence" and not supporting_ids:
                        yield {
                            **common,
                            "Comparison item": "No expected supporting evidence references",
                            "Chunk ID": "",
                            "Retrieval rank": None,
                            "Expected evidence type": "supports",
                            "Exact chunk ID match": "",
                            "Supporting expected chunk retrieved?": "Not evaluable",
                            "Expected supporting retrieval rank": None,
                            "Document": "",
                            "Chunk financial year": "",
                            "Chunk company": "",
                            "Page number": None,
                            "Full original chunk text": "",
                            "Chunk lookup status": "not_applicable",
                            "Retrieval record status": result.get("status", "missing"),
                            "Evidence reference ID": "",
                        }
                    continue
                for evidence in evidence_rows:
                    chunk_id = evidence["chunk_id"]
                    chunk = model["chunks_by_id"].get(chunk_id)
                    found_rank = ranks.get(chunk_id)
                    yield {
                        **common,
                        "Comparison item": section,
                        "Chunk ID": chunk_id,
                        "Retrieval rank": found_rank,
                        "Expected evidence type": evidence["relation"],
                        "Exact chunk ID match": "Yes" if found_rank is not None else "No",
                        "Supporting expected chunk retrieved?": (
                            "Yes" if found_rank is not None else "No"
                        ) if evidence["relation"] == "supports" else "Not a supporting reference",
                        "Expected supporting retrieval rank": (
                            found_rank if evidence["relation"] == "supports" else None
                        ),
                        "Document": str(chunk.get("document", "")) if chunk else "",
                        "Chunk financial year": str(chunk.get("financial_year", "")) if chunk else "",
                        "Chunk company": str(chunk.get("company", "")) if chunk else "",
                        "Page number": chunk.get("page") if chunk else None,
                        "Full original chunk text": str(chunk.get("text", "")) if chunk else "",
                        "Chunk lookup status": "found" if chunk else "missing",
                        "Retrieval record status": result.get("status", "missing"),
                        "Evidence reference ID": evidence["evidence_reference_id"],
                    }


def _error_analysis_rows(model: dict[str, Any]) -> Iterable[dict[str, Any]]:
    for row in _performance_rows(model):
        metrics = []
        for display, key in METRIC_FIELDS:
            value = row[display]
            metrics.append(f"{key}={value if value is not None else 'unavailable'}")
        yield {
            "Question ID": row["Question ID"],
            "Category": row["Category"],
            "Company": row["Company"],
            "Retrieval strategy": row["Retrieval strategy"],
            "Relevant metric values (saved)": "; ".join(metrics),
            "Failure classification": "",
            "Manual observation": "",
            "Possible cause": "",
            "Proposed improvement": "",
            "Review status": "Pending",
            "Evaluation status": row["Evaluation status"],
            "Common evaluation subset": row["Common evaluation subset"],
        }


def _investigation_rows(model: dict[str, Any]) -> Iterable[dict[str, Any]]:
    for question_id, case in model["benchmark_by_id"].items():
        gold_rows = model["gold_by_question"].get(question_id, [])
        supporting_ids = _supporting_ids(gold_rows)
        q_metric_rows = [
            model["metrics_by_key"][(question_id, strategy)]
            for strategy in model["strategy_names"]
        ]
        scored_rows = [row for row in q_metric_rows if row.get("metric_status") == "scored"]
        values_by_metric: dict[str, list[float]] = {
            display: [
                float(row[key]) for row in scored_rows
                if row.get(key) is not None
            ]
            for display, key in METRIC_FIELDS
        }
        zero_counts = {
            display: sum(value == 0 for value in values)
            for display, values in values_by_metric.items()
        }
        multiple_zero_rows = sum(
            sum(row.get(key) == 0 for _, key in METRIC_FIELDS) >= 2
            for row in scored_rows
        )
        spreads = {
            display: max(values) - min(values) if values else None
            for display, values in values_by_metric.items()
        }
        reasons: list[str] = []
        if not supporting_ids:
            reasons.append("No supporting gold chunk IDs; retrieval metrics are not evaluable")
        for display, count in zero_counts.items():
            if count:
                reasons.append(f"{display}=0 in {count} scored strategy row(s)")
        if multiple_zero_rows:
            reasons.append(f"At least two zero metrics in {multiple_zero_rows} scored strategy row(s)")
        varying = [display for display, spread in spreads.items() if spread is not None and spread > 0]
        if varying:
            reasons.append("Saved values vary across strategies; see metric ranges")
        yield {
            **_base_case_fields(case),
            "Scored strategy rows": len(scored_rows),
            "Known supporting chunk IDs": len(supporting_ids),
            "Precision@5 zero count": zero_counts["Precision@5"],
            "Recall@5 zero count": zero_counts["Recall@5"],
            "Recall@10 zero count": zero_counts["Recall@10"],
            "MRR zero count": zero_counts["MRR"],
            "Strategy range Precision@5": spreads["Precision@5"],
            "Strategy range Recall@5": spreads["Recall@5"],
            "Strategy range Recall@10": spreads["Recall@10"],
            "Strategy range MRR": spreads["MRR"],
            "Investigation reasons": "; ".join(reasons) if reasons else "No zero metric or metric range detected",
        }


def _write_value(
    worksheet: Any,
    row_num: int,
    col_num: int,
    value: Any,
    cell_format: Any = None,
) -> None:
    if value is None:
        worksheet.write_blank(row_num, col_num, None, cell_format)
    elif isinstance(value, bool):
        worksheet.write_boolean(row_num, col_num, value, cell_format)
    elif isinstance(value, (int, float)):
        worksheet.write_number(row_num, col_num, value, cell_format)
    else:
        text = str(value)
        if len(text) > EXCEL_CELL_TEXT_LIMIT:
            raise ValueError(
                f"Workbook cell exceeds Excel's {EXCEL_CELL_TEXT_LIMIT}-character limit."
            )
        worksheet.write_string(row_num, col_num, text, cell_format)


def _write_data_sheet(
    workbook: Any,
    name: str,
    headers: list[str],
    rows: Iterable[dict[str, Any]],
    widths: dict[str, int],
    header_format: Any,
    default_format: Any,
    field_formats: dict[str, Any] | None = None,
    row_formats: dict[str, Any] | None = None,
    row_style_field: str | None = None,
    freeze_first_column: bool = False,
) -> tuple[Any, int]:
    worksheet = workbook.add_worksheet(name)
    worksheet.freeze_panes(1, 1 if freeze_first_column else 0)
    worksheet.set_default_row(48)
    for col_num, header in enumerate(headers):
        worksheet.write_string(0, col_num, header, header_format)
        worksheet.set_column(
            col_num,
            col_num,
            widths.get(header, 18),
            (field_formats or {}).get(header, default_format),
        )
    row_count = 0
    for row_count, row in enumerate(rows, start=1):
        style = (row_formats or {}).get(str(row.get(row_style_field, ""))) if row_style_field else None
        for col_num, header in enumerate(headers):
            fmt = style or (field_formats or {}).get(header, default_format)
            _write_value(worksheet, row_count, col_num, row.get(header), fmt)
    if row_count:
        worksheet.autofilter(0, 0, row_count, len(headers) - 1)
    worksheet.set_landscape()
    worksheet.fit_to_pages(1, 0)
    worksheet.repeat_rows(0)
    return worksheet, row_count


def _write_instructions(workbook: Any, model: dict[str, Any]) -> None:
    sheet = workbook.add_worksheet("Instructions")
    title = workbook.add_format({
        "bold": True, "font_size": 18, "font_color": "#FFFFFF",
        "bg_color": "#17365D", "valign": "vcenter",
    })
    heading = workbook.add_format({
        "bold": True, "font_size": 12, "font_color": "#17365D",
        "bottom": 1, "bottom_color": "#9FBAD0",
    })
    body = workbook.add_format({"text_wrap": True, "valign": "top"})
    warning = workbook.add_format({
        "text_wrap": True, "valign": "top", "bg_color": "#FFF2CC",
        "font_color": "#7F6000",
    })
    sheet.set_column("A:A", 31)
    sheet.set_column("B:B", 112)
    sheet.set_default_row(34)
    sheet.merge_range("A1:B1", "Question-Level Retrieval Audit Workbook", title)
    sheet.set_row(0, 34)
    content = [
        ("Scope", "This workbook joins the 60-case benchmark, saved raw retrieval results, saved per-query metrics, proposed gold annotations, and the canonical chunk corpus. It is an analysis aid; it does not change retrieval behavior or evaluation calculations."),
        ("Question categories", "The benchmark currently contains seven category values. They are preserved exactly as stored rather than collapsed into the three-category description in the task context."),
        ("Question Performance Summary", "One row per question and strategy. Metric cells are copied from per_query_metrics.csv. Blank metric cells mean unavailable/not evaluable or failed; they are not zero. Use Common evaluation subset to keep cross-strategy comparisons within the saved 58-question common scope."),
        ("Retrieved Chunks", "Contains the actual result records in their saved rank order. Fewer than ten results remain fewer than ten. Failed and completed-empty searches have explicit placeholder rows; no substitute chunk is inserted. Passage text is looked up by exact chunk ID in all_chunks.jsonl."),
        ("Expected Gold Chunks", "One row per evidence reference occurrence. `supports` and `context_only` remain separate. Only `supports` IDs are used in primary exact-ID comparisons. Annotation verification status is copied from the original gold JSON; proposed references are not treated as verified."),
        ("Side-by-Side Comparison", "Uses exact chunk-ID equality only. Retrieved rows and expected support/context rows are separate, with one chunk passage per row. Supporting evidence rows show whether and at what rank that exact ID appears. A context-only match is not a supporting hit."),
        ("Error Analysis", "Use the dropdown classification only after manual inspection. Observation, possible cause, and improvement fields are intentionally blank; no cause is inferred from metrics. Review status starts Pending."),
        ("Investigation Priorities", "Shows zero-metric counts, exact metric ranges across scored strategies, and questions without known supporting evidence. Ranges are descriptive and do not select a strategy winner. Zero values are drawn only from saved scored metrics."),
        ("Manual relevance review", "In Retrieved Chunks, choose supports, context_only, not_relevant, or needs_review after inspecting the source and context. Add reviewer comments; this does not change expected gold labels or saved metrics."),
        ("Data limitations", "All 134 proposed evidence references remain proposed_unverified. Eleven questions have no supporting gold chunk IDs, so their retrieval metrics are not evaluable. Retrieval failures and unavailable metrics are preserved."),
        ("Source integrity", "Input paths, record counts, checksums, mapping checks, missing IDs, and post-build integrity status are recorded in question_level_retrieval_audit_validation.md."),
        ("Rebuild", "Install the scoped writer dependency with `.venv\\Scripts\\python.exe -m pip install -r requirements-audit.txt`, then run `.venv\\Scripts\\python.exe scripts\\build_question_level_retrieval_audit.py`. Existing workbook output is protected unless `--force` is provided."),
    ]
    current_row = 2
    for label, description in content:
        sheet.write_string(current_row, 0, label, heading)
        fmt = warning if label in ("Data limitations",) else body
        sheet.write_string(current_row, 1, description, fmt)
        current_row += 1
    sheet.freeze_panes(2, 0)


def _apply_metric_formatting(
    worksheet: Any,
    headers: list[str],
    first_data_row: int,
    last_data_row: int,
    workbook: Any,
) -> None:
    if last_data_row < first_data_row:
        return
    zero_format = workbook.add_format({"bg_color": "#FCE4D6", "font_color": "#9C0006"})
    for metric_name, _ in METRIC_FIELDS:
        col_num = headers.index(metric_name)
        worksheet.conditional_format(
            first_data_row,
            col_num,
            last_data_row,
            col_num,
            {
                "type": "3_color_scale",
                "min_color": "#F8696B",
                "mid_color": "#FFEB84",
                "max_color": "#63BE7B",
            },
        )
        worksheet.conditional_format(
            first_data_row,
            col_num,
            last_data_row,
            col_num,
            {"type": "cell", "criteria": "==", "value": 0, "format": zero_format},
        )


def _add_validation_lists(
    worksheet: Any,
    headers: list[str],
    last_row: int,
    field: str,
    values: tuple[str, ...],
) -> None:
    if last_row < 1:
        return
    col = headers.index(field)
    worksheet.data_validation(1, col, last_row, col, {
        "validate": "list",
        "source": list(values),
        "ignore_blank": True,
        "input_title": "Manual review",
        "input_message": "Select a value only after human review.",
        "error_title": "Invalid value",
        "error_message": "Choose a value from the list.",
        "error_type": "stop",
    })


def write_workbook(model: dict[str, Any], output_path: Path, force: bool = False) -> dict[str, int]:
    if output_path.exists() and not force:
        raise FileExistsError(
            f"Refusing to overwrite {output_path}. Pass --force to intentionally replace it."
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    workbook = xlsxwriter.Workbook(str(output_path), {
        "constant_memory": True,
        "strings_to_urls": False,
        "strings_to_formulas": False,
    })
    workbook.set_properties({
        "title": "Question-Level Retrieval Audit",
        "subject": "Manual question-level comparison of retrieval results and proposed gold evidence",
        "author": "Financial RAG Research",
        "comments": "Generated from saved benchmark, retrieval, metrics, annotations, and chunk corpus inputs.",
    })
    header_format = workbook.add_format({
        "bold": True, "font_color": "#FFFFFF", "bg_color": "#17365D",
        "text_wrap": True, "valign": "vcenter", "border": 1,
        "border_color": "#FFFFFF",
    })
    default_format = workbook.add_format({"valign": "top"})
    wrap_format = workbook.add_format({"text_wrap": True, "valign": "top"})
    retrieved_passage_format = workbook.add_format({
        "text_wrap": True, "valign": "top", "bg_color": "#EAF3F8",
    })
    gold_passage_format = workbook.add_format({
        "text_wrap": True, "valign": "top", "bg_color": "#E2F0D9",
    })
    context_passage_format = workbook.add_format({
        "text_wrap": True, "valign": "top", "bg_color": "#FFF2CC",
    })
    retrieved_text_field = {"Full original chunk text": retrieved_passage_format}
    gold_text_field = {"Full original chunk text": gold_passage_format}

    _write_instructions(workbook, model)
    performance_headers = [
        "Question ID", "Category", "Company", "Financial year", "Benchmark question",
        "Retrieval strategy", "Number of retrieved chunks", "Number of known supporting chunks",
        "Precision@5", "Recall@5", "Recall@10", "MRR", "Evaluation status",
        "Common evaluation subset", "Retrieval error",
    ]
    performance_widths = {
        "Question ID": 18, "Category": 34, "Company": 30, "Financial year": 23,
        "Benchmark question": 56, "Retrieval strategy": 26, "Number of retrieved chunks": 18,
        "Number of known supporting chunks": 22, "Precision@5": 14, "Recall@5": 14,
        "Recall@10": 14, "MRR": 14, "Evaluation status": 28,
        "Common evaluation subset": 22, "Retrieval error": 70,
    }
    performance_sheet, performance_count = _write_data_sheet(
        workbook, "Question Performance Summary", performance_headers,
        _performance_rows(model), performance_widths, header_format, default_format,
        field_formats={"Benchmark question": wrap_format, "Retrieval error": wrap_format},
    )
    _apply_metric_formatting(
        performance_sheet, performance_headers, 1, performance_count, workbook
    )

    retrieved_headers = [
        "Question ID", "Category", "Company", "Retrieval strategy", "Retrieval rank",
        "Chunk ID", "Document name", "Financial year", "Question financial year",
        "Chunk company", "Page number",
        "Full original chunk text", "Manual relevance decision", "Reviewer comments",
        "Retrieval record status", "Retrieval error", "Chunk lookup status",
        "Gold evidence type by exact ID",
    ]
    retrieved_widths = {
        "Question ID": 18, "Category": 34, "Company": 30, "Retrieval strategy": 26,
        "Retrieval rank": 14, "Chunk ID": 54, "Document name": 24, "Financial year": 20,
        "Question financial year": 23, "Chunk company": 30,
        "Page number": 13, "Full original chunk text": 76, "Manual relevance decision": 23,
        "Reviewer comments": 44, "Retrieval record status": 22, "Retrieval error": 70,
        "Chunk lookup status": 20, "Gold evidence type by exact ID": 30,
    }
    retrieved_sheet, retrieved_count = _write_data_sheet(
        workbook, "Retrieved Chunks", retrieved_headers, _retrieved_rows(model),
        retrieved_widths, header_format, default_format,
        field_formats={"Full original chunk text": retrieved_passage_format,
                       "Reviewer comments": wrap_format, "Retrieval error": wrap_format},
    )
    _add_validation_lists(
        retrieved_sheet, retrieved_headers, retrieved_count,
        "Manual relevance decision", MANUAL_RELEVANCE_DECISIONS,
    )

    expected_headers = [
        "Question ID", "Category", "Company", "Benchmark question", "Expected chunk ID",
        "Document", "Financial year", "Question financial year", "Chunk company",
        "Page number", "Full original chunk text",
        "Evidence type", "Annotation verification status", "Reviewer comments",
        "Assertion ID", "Evidence reference ID", "Annotation rationale (unverified source text)",
        "Chunk lookup status",
    ]
    expected_widths = {
        "Question ID": 18, "Category": 34, "Company": 30, "Benchmark question": 56,
        "Expected chunk ID": 54, "Document": 24, "Financial year": 20, "Page number": 13,
        "Question financial year": 23, "Chunk company": 30,
        "Full original chunk text": 76, "Evidence type": 20, "Annotation verification status": 31,
        "Reviewer comments": 44, "Assertion ID": 28, "Evidence reference ID": 44,
        "Annotation rationale (unverified source text)": 74, "Chunk lookup status": 20,
    }
    expected_sheet, expected_count = _write_data_sheet(
        workbook, "Expected Gold Chunks", expected_headers, _expected_rows(model),
        expected_widths, header_format, default_format,
        field_formats={
            "Full original chunk text": gold_passage_format,
            "Benchmark question": wrap_format,
            "Reviewer comments": wrap_format,
            "Annotation rationale (unverified source text)": wrap_format,
        },
    )
    expected_sheet.set_tab_color("#70AD47")

    comparison_headers = [
        "Question ID", "Category", "Company", "Financial year", "Benchmark question",
        "Retrieval strategy", "Precision@5", "Recall@5", "Recall@10", "MRR",
        "Evaluation status", "Common evaluation subset", "Comparison item", "Chunk ID",
        "Retrieval rank", "Expected evidence type", "Exact chunk ID match",
        "Supporting expected chunk retrieved?", "Expected supporting retrieval rank",
        "Document", "Chunk financial year", "Chunk company", "Page number",
        "Full original chunk text", "Chunk lookup status", "Retrieval record status",
        "Evidence reference ID",
    ]
    comparison_widths = {
        "Question ID": 18, "Category": 34, "Company": 30, "Financial year": 22,
        "Benchmark question": 56, "Retrieval strategy": 26, "Precision@5": 14,
        "Recall@5": 14, "Recall@10": 14, "MRR": 14, "Evaluation status": 29,
        "Common evaluation subset": 22, "Comparison item": 38, "Chunk ID": 54,
        "Retrieval rank": 14, "Expected evidence type": 30, "Exact chunk ID match": 21,
        "Supporting expected chunk retrieved?": 31, "Expected supporting retrieval rank": 28,
        "Document": 24, "Chunk financial year": 23, "Chunk company": 30,
        "Page number": 13,
        "Full original chunk text": 76, "Chunk lookup status": 20,
        "Retrieval record status": 22, "Evidence reference ID": 44,
    }
    comparison_row_formats = {
        "Retrieved result": retrieved_passage_format,
        "Expected supporting evidence": gold_passage_format,
        "Expected context-only evidence": context_passage_format,
    }
    comparison_sheet, comparison_count = _write_data_sheet(
        workbook, "Side-by-Side Comparison", comparison_headers,
        _comparison_rows(model), comparison_widths, header_format, default_format,
        field_formats={"Benchmark question": wrap_format},
        row_formats=comparison_row_formats,
        row_style_field="Comparison item",
    )
    _apply_metric_formatting(
        comparison_sheet, comparison_headers, 1, comparison_count, workbook
    )

    error_headers = [
        "Question ID", "Category", "Company", "Retrieval strategy",
        "Relevant metric values (saved)", "Failure classification", "Manual observation",
        "Possible cause", "Proposed improvement", "Review status", "Evaluation status",
        "Common evaluation subset",
    ]
    error_widths = {
        "Question ID": 18, "Category": 34, "Company": 30, "Retrieval strategy": 26,
        "Relevant metric values (saved)": 65, "Failure classification": 48,
        "Manual observation": 64, "Possible cause": 52, "Proposed improvement": 52,
        "Review status": 20, "Evaluation status": 29, "Common evaluation subset": 22,
    }
    error_sheet, error_count = _write_data_sheet(
        workbook, "Error Analysis", error_headers, _error_analysis_rows(model),
        error_widths, header_format, default_format,
        field_formats={
            "Relevant metric values (saved)": wrap_format,
            "Manual observation": wrap_format,
            "Possible cause": wrap_format,
            "Proposed improvement": wrap_format,
        },
    )
    _add_validation_lists(
        error_sheet, error_headers, error_count,
        "Failure classification", ERROR_CLASSIFICATIONS,
    )
    _add_validation_lists(error_sheet, error_headers, error_count, "Review status", REVIEW_STATUSES)

    priority_headers = [
        "Question ID", "Category", "Company", "Financial year", "Benchmark question",
        "Scored strategy rows", "Known supporting chunk IDs", "Precision@5 zero count",
        "Recall@5 zero count", "Recall@10 zero count", "MRR zero count",
        "Strategy range Precision@5", "Strategy range Recall@5",
        "Strategy range Recall@10", "Strategy range MRR", "Investigation reasons",
    ]
    priority_widths = {
        "Question ID": 18, "Category": 34, "Company": 30, "Financial year": 23,
        "Benchmark question": 56, "Scored strategy rows": 18, "Known supporting chunk IDs": 23,
        "Precision@5 zero count": 20, "Recall@5 zero count": 20, "Recall@10 zero count": 20,
        "MRR zero count": 17, "Strategy range Precision@5": 24,
        "Strategy range Recall@5": 24, "Strategy range Recall@10": 25,
        "Strategy range MRR": 22, "Investigation reasons": 86,
    }
    priority_sheet, priority_count = _write_data_sheet(
        workbook, "Investigation Priorities", priority_headers,
        _investigation_rows(model), priority_widths, header_format, default_format,
        field_formats={
            "Benchmark question": wrap_format,
            "Investigation reasons": wrap_format,
        },
    )
    priority_sheet.set_tab_color("#ED7D31")
    workbook.close()
    return {
        "Question Performance Summary": performance_count,
        "Retrieved Chunks": retrieved_count,
        "Expected Gold Chunks": expected_count,
        "Side-by-Side Comparison": comparison_count,
        "Error Analysis": error_count,
        "Investigation Priorities": priority_count,
    }


def _hashes_by_relative_path(paths: list[Path]) -> dict[str, str]:
    return {
        str(path.relative_to(PROJECT_ROOT)): _sha256(path)
        for path in paths
    }


def write_validation_report(
    model: dict[str, Any],
    source_hashes: dict[str, str],
    final_hashes: dict[str, str],
    workbook_rows: dict[str, int],
    report_path: Path,
    workbook_path: Path,
) -> None:
    benchmark_cases = model["benchmark_by_id"]
    metrics = list(model["metrics_by_key"].values())
    raw_statuses = Counter(
        result.get("status", "missing")
        for case in model["raw_by_id"].values()
        for result in case["strategies"].values()
    )
    metric_statuses = Counter(row.get("metric_status", "missing") for row in metrics)
    categories = Counter(case.get("category", "") for case in benchmark_cases.values())
    companies = Counter(case.get("company", "") for case in benchmark_cases.values())
    gold_rows = [
        row for entries in model["gold_by_question"].values()
        for row in entries
    ]
    relation_counts = Counter(str(row["relation"]) for row in gold_rows)
    verification_counts = Counter(str(row["verification_status"]) for row in gold_rows)
    null_counts = {
        display: sum(row.get(key) is None for row in metrics)
        for display, key in METRIC_FIELDS
    }
    no_support_questions = sorted(
        question_id for question_id in benchmark_cases
        if not _supporting_ids(model["gold_by_question"].get(question_id, []))
    )
    common_questions = sorted({
        key[0] for key, row in model["metrics_by_key"].items()
        if row.get("common_subset")
    })
    unchanged = source_hashes == final_hashes

    lines = [
        "# Question-Level Retrieval Audit: Data Integrity Validation",
        "",
        f"- Workbook output: `{workbook_path.name}`",
        f"- Source files unchanged after build: **{'yes' if unchanged else 'NO'}**",
        f"- Benchmark questions mapped: **{len(benchmark_cases)}**",
        f"- Retrieval strategies: **{len(model['strategy_names'])}**",
        f"- Question/strategy metric rows mapped: **{len(metrics)}**",
        f"- Actual retrieved chunk records: **{model['retrieval_record_count']}**",
        f"- Expected evidence reference occurrences: **{len(gold_rows)}**",
        "",
        "## Mapping and statuses",
        "",
        f"- Categories preserved: {', '.join(f'`{key}` ({value})' for key, value in sorted(categories.items()))}",
        f"- Companies preserved: {', '.join(f'`{key}` ({value} question/strategy rows)' for key, value in sorted(companies.items()))}",
        f"- Strategies: {', '.join(f'`{strategy}`' for strategy in model['strategy_names'])}",
        f"- Retrieval statuses: {', '.join(f'`{key}`={value}' for key, value in sorted(raw_statuses.items()))}",
        f"- Metric statuses: {', '.join(f'`{key}`={value}' for key, value in sorted(metric_statuses.items()))}",
        f"- Expected evidence types: {', '.join(f'`{key}`={value}' for key, value in sorted(relation_counts.items()))}",
        f"- Annotation verification statuses: {', '.join(f'`{key}`={value}' for key, value in sorted(verification_counts.items()))}",
        f"- Questions without supporting gold IDs (metrics not evaluable): {', '.join(f'`{qid}`' for qid in no_support_questions) or 'none'}",
        f"- Questions in saved common evaluation subset: **{len(common_questions)}**",
        "",
        "## Integrity checks",
        "",
        f"- Raw result question IDs match benchmark IDs: **{'pass' if len(model['raw_by_id']) == len(benchmark_cases) else 'fail'}**",
        f"- Per-query metric rows match every question/strategy pair: **{'pass' if len(metrics) == len(benchmark_cases) * len(model['strategy_names']) else 'fail'}**",
        f"- Retrieved IDs absent from all_chunks.jsonl: **{len(model['missing_retrieved_chunks'])}**",
        f"- Expected evidence IDs absent from all_chunks.jsonl: **{len(model['missing_gold_chunks'])}**",
        f"- Retrieved passages differing from canonical chunk text: **{len(model['retrieved_text_mismatches'])}**",
        "- Retrieved ranks are consecutive and retrieved counts match both saved result and metric records: **pass**",
        "- Raw relevance-ID sets equal exact `supports` chunk IDs from gold annotations: **pass**",
        f"- Blank metric values preserved as unavailable: {', '.join(f'{key}={value}' for key, value in null_counts.items())}",
        f"- Full passage lengths exceeding Excel cell limit in included records: **0**",
        "",
        "Workbook worksheet row counts (header rows excluded):",
        "",
    ]
    lines.extend(f"- `{name}`: {count}" for name, count in workbook_rows.items())
    lines.extend(["", "## Input SHA-256", ""])
    lines.extend(f"- `{path}`: `{digest}`" for path, digest in source_hashes.items())
    lines.extend([
        "",
        "The evidence labels are provisional (`proposed_unverified`). This report validates source relationships and workbook construction; it does not certify annotation correctness or infer retrieval failure causes.",
        "",
    ])
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines), encoding="utf-8")
    if not unchanged:
        raise ValueError("An input source changed while the workbook was being generated.")


def load_default_data() -> tuple[dict[str, Any], dict[str, str]]:
    source_paths = _source_paths()
    before_hashes = _hashes_by_relative_path(source_paths)
    benchmark = _read_json(BENCHMARK_PATH)
    raw_results = _read_json(RAW_RESULTS_PATH)
    metric_rows = _read_csv(PER_QUERY_METRICS_PATH)
    annotation_datasets = [_read_json(path) for path in ANNOTATION_PATHS]
    chunks = load_chunks()
    model = prepare_audit_data(
        benchmark, raw_results, metric_rows, annotation_datasets, chunks
    )
    model["source_paths"] = source_paths
    return model, before_hashes


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT_PATH)
    parser.add_argument(
        "--force",
        action="store_true",
        help="Replace an existing workbook and validation report.",
    )
    args = parser.parse_args(argv)
    if not args.force and (args.output.exists() or args.report.exists()):
        print(
            "Refusing to overwrite existing audit outputs; use --force to replace them.",
            file=sys.stderr,
        )
        return 1

    try:
        model, source_hashes = load_default_data()
        workbook_rows = write_workbook(model, args.output, force=args.force)
        final_hashes = _hashes_by_relative_path(model["source_paths"])
        write_validation_report(
            model, source_hashes, final_hashes, workbook_rows, args.report, args.output
        )
    except (FileExistsError, OSError, ValueError) as error:
        print(f"Question-level retrieval audit failed: {error}", file=sys.stderr)
        return 1

    print(f"Wrote question-level retrieval workbook to {args.output}")
    print(f"Wrote data integrity report to {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
