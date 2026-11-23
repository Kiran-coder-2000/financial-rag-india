"""Build and summarize a separate human review workflow for gold evidence."""

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parent.parent
ANNOTATION_DIR = PROJECT_ROOT / "benchmarks" / "answer_evaluation"
OUTPUT_DIR = PROJECT_ROOT / "benchmarks" / "annotation_audit"
BENCHMARK_PATH = ANNOTATION_DIR / "financial_rag_60_question_benchmark.json"
ANNOTATION_PATHS = (
    ANNOTATION_DIR / "tcs_q01_q20_gold_annotations.json",
    ANNOTATION_DIR / "infosys_gold_annotations.json",
    ANNOTATION_DIR / "reliance_gold_annotations.json",
)
CHUNKS_PATH = PROJECT_ROOT / "data" / "chunks" / "all_chunks.jsonl"
REVIEW_CSV = "annotation_review.csv"
QUESTIONS_CSV = "question_completeness.csv"
SUMMARY_MD = "annotation_audit_summary.md"
DECISIONS = {"supports", "context_only", "not_relevant", "needs_review"}
COMPLETENESS_STATUSES = {"pending", "complete", "uncertain"}

REVIEW_FIELDS = (
    "question_id",
    "company",
    "financial_year",
    "benchmark_question",
    "evidence_reference_id",
    "assertion_id",
    "evidence_ordinal",
    "chunk_id",
    "document_name",
    "page_number",
    "current_evidence_label",
    "original_evidence_passage",
    "reviewer_decision",
    "reviewer_comments",
    "additional_supporting_evidence_discovered",
    "chunk_lookup_status",
)
QUESTION_FIELDS = (
    "question_id",
    "company",
    "financial_year",
    "benchmark_question",
    "completeness_status",
    "reviewer_comments",
    "additional_supporting_evidence_discovered",
)


def _read_json(path: Path) -> dict[str, Any]:
    try:
        with path.open(encoding="utf-8") as source:
            value = json.load(source)
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"Could not load JSON input {path}: {error}") from error
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}.")
    return value


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
                    raise ValueError(f"Duplicate chunk_id {chunk_id!r} in {path}.")
                chunks[chunk_id] = chunk
    except OSError as error:
        raise ValueError(f"Could not read chunks from {path}: {error}") from error
    return chunks


def _financial_year(case: dict[str, Any]) -> str:
    periods = case.get("financial_periods", [])
    if not isinstance(periods, list):
        raise ValueError(f"Question {case.get('id')!r} has invalid financial_periods.")
    return "; ".join(str(period) for period in periods)


def build_review_rows(
    benchmark: dict[str, Any],
    annotation_datasets: list[dict[str, Any]],
    chunks_by_id: dict[str, dict[str, Any]],
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    benchmark_cases = benchmark.get("cases")
    if not isinstance(benchmark_cases, list):
        raise ValueError("Benchmark must contain a cases list.")
    questions: dict[str, dict[str, Any]] = {}
    for case in benchmark_cases:
        question_id = case.get("id")
        if not isinstance(question_id, str) or not question_id:
            raise ValueError("Benchmark case has a missing or invalid ID.")
        if question_id in questions:
            raise ValueError(f"Duplicate benchmark question ID {question_id!r}.")
        questions[question_id] = case

    review_rows: list[dict[str, str]] = []
    seen_reference_ids: set[str] = set()
    seen_annotation_cases: set[str] = set()
    for dataset in annotation_datasets:
        cases = dataset.get("cases")
        if not isinstance(cases, list):
            raise ValueError("Annotation dataset must contain a cases list.")
        for annotation_case in cases:
            question_id = annotation_case.get("id")
            benchmark_case = questions.get(question_id)
            if benchmark_case is None:
                raise ValueError(f"Annotation references unknown question {question_id!r}.")
            if question_id in seen_annotation_cases:
                raise ValueError(f"Duplicate annotation case for question {question_id!r}.")
            seen_annotation_cases.add(question_id)
            if annotation_case.get("question") != benchmark_case.get("question"):
                raise ValueError(f"Question text differs between benchmark and annotation for {question_id!r}.")

            proposed_gold = annotation_case.get("proposed_gold")
            assertions = proposed_gold.get("expected_assertions") if isinstance(proposed_gold, dict) else None
            if not isinstance(assertions, list):
                raise ValueError(f"Question {question_id!r} has no expected_assertions list.")
            for assertion in assertions:
                assertion_id = assertion.get("assertion_id")
                evidence = assertion.get("evidence")
                if not isinstance(assertion_id, str) or not assertion_id:
                    raise ValueError(f"Question {question_id!r} has an invalid assertion_id.")
                if not isinstance(evidence, list):
                    raise ValueError(f"Assertion {assertion_id!r} has no evidence list.")
                for ordinal, reference in enumerate(evidence, start=1):
                    chunk_id = reference.get("chunk_id")
                    if not isinstance(chunk_id, str) or not chunk_id:
                        raise ValueError(f"Assertion {assertion_id!r} has an invalid chunk_id.")
                    reference_id = f"{question_id}::{assertion_id}::evidence-{ordinal:02d}"
                    if reference_id in seen_reference_ids:
                        raise ValueError(f"Duplicate evidence reference ID {reference_id!r}.")
                    seen_reference_ids.add(reference_id)
                    chunk = chunks_by_id.get(chunk_id)
                    review_rows.append({
                        "question_id": question_id,
                        "company": str(benchmark_case.get("company", "")),
                        "financial_year": _financial_year(benchmark_case),
                        "benchmark_question": str(benchmark_case.get("question", "")),
                        "evidence_reference_id": reference_id,
                        "assertion_id": assertion_id,
                        "evidence_ordinal": str(ordinal),
                        "chunk_id": chunk_id,
                        "document_name": str(chunk.get("document", "")) if chunk else "",
                        "page_number": str(chunk.get("page", "")) if chunk else "",
                        "current_evidence_label": str(reference.get("relation", "")),
                        "original_evidence_passage": str(chunk.get("text", "")) if chunk else "",
                        "reviewer_decision": "",
                        "reviewer_comments": "",
                        "additional_supporting_evidence_discovered": "",
                        "chunk_lookup_status": "found" if chunk else "missing",
                    })

    question_rows = [
        {
            "question_id": question_id,
            "company": str(case.get("company", "")),
            "financial_year": _financial_year(case),
            "benchmark_question": str(case.get("question", "")),
            "completeness_status": "pending",
            "reviewer_comments": "",
            "additional_supporting_evidence_discovered": "",
        }
        for question_id, case in questions.items()
    ]
    return review_rows, question_rows


def _read_csv(path: Path, fields: tuple[str, ...]) -> list[dict[str, str]]:
    try:
        with path.open(encoding="utf-8-sig", newline="") as source:
            reader = csv.DictReader(source)
            if reader.fieldnames is None or not set(fields).issubset(reader.fieldnames):
                raise ValueError(f"{path} is missing required columns: {', '.join(fields)}")
            rows = list(reader)
    except OSError as error:
        raise ValueError(f"Could not read review file {path}: {error}") from error
    return [{field: row.get(field, "") or "" for field in fields} for row in rows]


def _parse_chunk_ids(value: str, question_id: str, chunks_by_id: dict[str, dict[str, Any]]) -> list[str]:
    chunk_ids = [item.strip() for item in value.split(";") if item.strip()]
    if len(chunk_ids) != len(set(chunk_ids)):
        raise ValueError(f"Question {question_id!r} lists additional evidence chunk IDs more than once.")
    for chunk_id in chunk_ids:
        if chunk_id not in chunks_by_id:
            raise ValueError(f"Question {question_id!r} lists unknown additional chunk ID {chunk_id!r}.")
    return chunk_ids


def summarize_reviews(
    review_rows: list[dict[str, str]],
    question_rows: list[dict[str, str]],
    chunks_by_id: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    reference_ids = [row.get("evidence_reference_id", "") for row in review_rows]
    if any(not item for item in reference_ids) or len(reference_ids) != len(set(reference_ids)):
        raise ValueError("Review sheet has missing or duplicate evidence reference IDs.")

    decisions: list[str] = []
    discovered_by_question: dict[str, set[str]] = defaultdict(set)
    questions_by_id: set[str] = set()
    for row in review_rows:
        question_id = row.get("question_id", "")
        questions_by_id.add(question_id)
        decision = row.get("reviewer_decision", "").strip()
        if decision and decision not in DECISIONS:
            raise ValueError(
                f"Invalid reviewer decision {decision!r} for {row.get('evidence_reference_id')!r}."
            )
        decisions.append(decision)
        discovered_by_question[question_id].update(_parse_chunk_ids(
            row.get("additional_supporting_evidence_discovered", ""),
            question_id,
            chunks_by_id,
        ))

    completeness_by_question: dict[str, str] = {}
    for row in question_rows:
        question_id = row.get("question_id", "")
        if not question_id or question_id in completeness_by_question:
            raise ValueError(f"Question completeness sheet has missing or duplicate ID {question_id!r}.")
        status = row.get("completeness_status", "").strip()
        if status not in COMPLETENESS_STATUSES:
            raise ValueError(
                f"Invalid completeness status {status!r} for question {question_id!r}."
            )
        completeness_by_question[question_id] = status
        discovered_by_question[question_id].update(_parse_chunk_ids(
            row.get("additional_supporting_evidence_discovered", ""),
            question_id,
            chunks_by_id,
        ))
    if questions_by_id - set(completeness_by_question):
        raise ValueError("Question completeness sheet is missing questions present in the reference sheet.")

    counts = Counter(decision for decision in decisions if decision)
    discovered_questions = sorted(
        question_id for question_id, chunk_ids in discovered_by_question.items() if chunk_ids
    )
    explicitly_uncertain_questions = sorted(
        question_id for question_id, status in completeness_by_question.items()
        if status == "uncertain"
    )
    pending_completeness = sorted(
        question_id for question_id, status in completeness_by_question.items()
        if status == "pending"
    )
    return {
        "total_references": len(review_rows),
        "reviewed_references": sum(bool(decision) for decision in decisions),
        "pending_references": sum(not decision for decision in decisions),
        "decision_counts": {decision: counts[decision] for decision in sorted(DECISIONS)},
        "questions_with_new_evidence": discovered_questions,
        "new_evidence_chunk_count": sum(len(ids) for ids in discovered_by_question.values()),
        "questions_with_uncertain_completeness": sorted(
            question_id for question_id, status in completeness_by_question.items()
            if status != "complete"
        ),
        "questions_explicitly_marked_uncertain": explicitly_uncertain_questions,
        "questions_with_pending_completeness": pending_completeness,
    }


def render_summary(summary: dict[str, Any]) -> str:
    counts = summary["decision_counts"]
    lines = [
        "# Annotation Audit Summary",
        "",
        "This report is calculated from the separate review CSVs. Reviewer decisions are not inferred.",
        "",
        "## Evidence reference review",
        "",
        f"- Total references: **{summary['total_references']}**",
        f"- References reviewed: **{summary['reviewed_references']}**",
        f"- References still pending: **{summary['pending_references']}**",
        f"- Classified as supports: **{counts['supports']}**",
        f"- Classified as context-only: **{counts['context_only']}**",
        f"- Classified as not relevant: **{counts['not_relevant']}**",
        f"- Requiring further review: **{counts['needs_review']}**",
        "",
        "## Evidence completeness",
        "",
        f"- Questions with newly discovered evidence: **{len(summary['questions_with_new_evidence'])}**",
        f"- Additional unique chunks recorded: **{summary['new_evidence_chunk_count']}**",
        f"- Questions whose completeness is uncertain or pending: **{len(summary['questions_with_uncertain_completeness'])}**",
        f"- Questions explicitly marked uncertain: **{len(summary['questions_explicitly_marked_uncertain'])}**",
        f"- Questions still pending completeness review: **{len(summary['questions_with_pending_completeness'])}**",
        "",
    ]
    for heading, key in (
        ("Questions with newly discovered evidence", "questions_with_new_evidence"),
        ("Questions whose completeness is uncertain or pending review", "questions_with_uncertain_completeness"),
        ("Questions pending completeness review", "questions_with_pending_completeness"),
    ):
        lines.extend([f"### {heading}", ""])
        ids = summary[key]
        lines.extend([f"- `{question_id}`" for question_id in ids] if ids else ["- None"])
        lines.append("")
    lines.extend([
        "A blank reviewer decision remains pending. `needs_review` is a recorded human decision and is counted as reviewed. "
        "A question is only listed as explicitly uncertain when its completeness status is `uncertain`; pending questions "
        "are listed separately and are not treated as complete.",
        "",
    ])
    return "\n".join(lines)


def _write_csv(path: Path, fields: tuple[str, ...], rows: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=fields, extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)


def create_template(output_dir: Path, force: bool = False) -> None:
    output_paths = [output_dir / REVIEW_CSV, output_dir / QUESTIONS_CSV]
    existing = [path for path in output_paths if path.exists()]
    if existing and not force:
        names = ", ".join(str(path) for path in existing)
        raise FileExistsError(
            f"Refusing to overwrite review data: {names}. Use --force only to intentionally reset it."
        )
    benchmark = _read_json(BENCHMARK_PATH)
    annotations = [_read_json(path) for path in ANNOTATION_PATHS]
    chunks = load_chunks()
    review_rows, question_rows = build_review_rows(benchmark, annotations, chunks)
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(output_dir / REVIEW_CSV, REVIEW_FIELDS, review_rows)
    _write_csv(output_dir / QUESTIONS_CSV, QUESTION_FIELDS, question_rows)
    summary = summarize_reviews(review_rows, question_rows, chunks)
    (output_dir / SUMMARY_MD).write_text(render_summary(summary), encoding="utf-8")
    print(f"Wrote {len(review_rows)} evidence references and {len(question_rows)} questions to {output_dir}")


def refresh_summary(output_dir: Path) -> None:
    chunks = load_chunks()
    review_rows = _read_csv(output_dir / REVIEW_CSV, REVIEW_FIELDS)
    question_rows = _read_csv(output_dir / QUESTIONS_CSV, QUESTION_FIELDS)
    summary = summarize_reviews(review_rows, question_rows, chunks)
    (output_dir / SUMMARY_MD).write_text(render_summary(summary), encoding="utf-8")
    print(f"Updated audit summary at {output_dir / SUMMARY_MD}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument(
        "--summary-only",
        action="store_true",
        help="Recalculate the summary from the existing review CSVs without changing them.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Regenerate templates even if review CSVs already exist, discarding their edits.",
    )
    args = parser.parse_args(argv)
    try:
        if args.summary_only:
            if args.force:
                parser.error("--force cannot be combined with --summary-only.")
            refresh_summary(args.output_dir)
        else:
            create_template(args.output_dir, force=args.force)
    except (FileExistsError, OSError, ValueError) as error:
        print(f"Annotation audit failed: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
