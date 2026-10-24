"""Build human-review candidates for the financial retrieval gold set.

The deterministic keyword matcher suggests source-backed candidate chunks; it
does not verify relevance or populate expected answers. Every generated case
stays pending, and ``relevant_chunk_ids`` remains empty until a human confirms
the candidate IDs. Existing output is protected unless --force is supplied.
"""

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

SOURCE_CHUNKS = PROJECT_ROOT / "data" / "chunks" / "all_chunks.jsonl"
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "benchmark" / "retrieval_gold_dataset.json"
DATASET_VERSION = "1.0"
EVIDENCE_ROLES = {"primary", "supporting"}

COMPANIES = (
    ("Infosys", "Infosys"),
    ("TataConsultancyServices", "TCS"),
    ("RelianceIndustries", "Reliance Industries"),
)

REVENUE_TERMS = (
    "consolidated revenue",
    "consolidated revenues",
    "revenue from operations",
    "revenue",
)
PROFIT_TERMS = (
    "consolidated profit after tax",
    "profit after tax",
    "profit for the year",
    "net profit",
    "profit attributable",
)
SEGMENT_TERMS = (
    "segment revenue",
    "segment performance",
    "segment results",
    "reportable segments",
    "operating segments",
)


def _target(role: str, explanation: str, terms: tuple[str, ...], financial_year: str | None = None) -> dict:
    return {
        "role": role,
        "explanation": explanation,
        "terms": list(terms),
        "financial_year": financial_year,
    }


def curated_questions() -> list[dict]:
    """Return the fixed question catalog; figures are intentionally not supplied."""
    questions = []
    for company, display_name in COMPANIES:
        safe_name = company.casefold()
        questions.extend([
            {
                "id": f"{safe_name}_consolidated_revenue_fy2026",
                "question": f"What was {display_name}'s consolidated revenue in FY2026?",
                "company": company,
                "financial_year": "2025-2026",
                "financial_years": ["2025-2026"],
                "category": "financial_metric",
                "targets": [
                    _target("primary", "Find the consolidated revenue figure for FY2026.", REVENUE_TERMS, "2025-2026"),
                    _target("supporting", "Find narrative or table context explaining the FY2026 revenue figure.", ("revenue growth", "reported currency", "revenue"), "2025-2026"),
                ],
            },
            {
                "id": f"{safe_name}_revenue_yoy_fy2025_fy2026",
                "question": f"How did {display_name}'s revenue change from FY2025 to FY2026?",
                "company": company,
                "financial_year": None,
                "financial_years": ["2024-2025", "2025-2026"],
                "category": "year_over_year_comparison",
                "targets": [
                    _target("primary", "Find FY2025 revenue evidence for the comparison.", REVENUE_TERMS, "2024-2025"),
                    _target("primary", "Find FY2026 revenue evidence for the comparison.", REVENUE_TERMS, "2025-2026"),
                ],
            },
            {
                "id": f"{safe_name}_segment_performance_fy2024",
                "question": f"What were the main business-segment performance results reported by {display_name} in FY2024?",
                "company": company,
                "financial_year": "2023-2024",
                "financial_years": ["2023-2024"],
                "category": "segment_performance",
                "targets": [
                    _target("primary", "Find segment-level performance or results for FY2024.", SEGMENT_TERMS, "2023-2024"),
                    _target("supporting", "Find segment revenue or contribution details for FY2024.", ("segment revenue", "revenue by segment", "segment results"), "2023-2024"),
                ],
            },
            {
                "id": f"{safe_name}_profit_after_tax_fy2023",
                "question": f"What consolidated profit-after-tax figure did {display_name} report for FY2023?",
                "company": company,
                "financial_year": "2022-2023",
                "financial_years": ["2022-2023"],
                "category": "financial_statement_figure",
                "targets": [
                    _target("primary", "Find the consolidated profit-after-tax figure for FY2023.", PROFIT_TERMS, "2022-2023"),
                    _target("supporting", "Find the corresponding consolidated profit-and-loss statement or note.", ("consolidated statement of profit and loss", "profit after tax", "net profit"), "2022-2023"),
                ],
            },
            {
                "id": f"{safe_name}_revenue_and_profit_yoy_fy2025_fy2026",
                "question": f"What evidence compares {display_name}'s revenue and profit between FY2025 and FY2026?",
                "company": company,
                "financial_year": None,
                "financial_years": ["2024-2025", "2025-2026"],
                "category": "multi_evidence_comparison",
                "targets": [
                    _target("primary", "Find FY2025 revenue evidence.", REVENUE_TERMS, "2024-2025"),
                    _target("primary", "Find FY2026 revenue evidence.", REVENUE_TERMS, "2025-2026"),
                    _target("supporting", "Find FY2025 profit evidence.", PROFIT_TERMS, "2024-2025"),
                    _target("supporting", "Find FY2026 profit evidence.", PROFIT_TERMS, "2025-2026"),
                ],
            },
        ])
    return questions


def _normalize(value: object) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(value or "").casefold()).strip()


def _searchable_fields(chunk: dict) -> tuple[str, str]:
    text_fields = ("text", "context", "document")
    metadata_fields = ("section", "table_heading", "context")
    return (
        _normalize(" ".join(str(chunk.get(field) or "") for field in text_fields)),
        _normalize(" ".join(str(chunk.get(field) or "") for field in metadata_fields)),
    )


def _target_matches(chunk: dict, target: dict) -> tuple[list[str], int] | None:
    target_year = target.get("financial_year")
    if target_year is not None and chunk.get("financial_year") != target_year:
        return None

    searchable_text, searchable_metadata = _searchable_fields(chunk)
    matched_terms = []
    score = 0
    for raw_term in target["terms"]:
        term = _normalize(raw_term)
        if not term:
            continue
        in_text = term in searchable_text
        in_metadata = term in searchable_metadata
        if in_text or in_metadata:
            matched_terms.append(raw_term)
            score = max(score, len(term.split()) * 10 + (3 if in_metadata else 0))
    if not matched_terms:
        return None
    if chunk.get("chunk_type") == "table":
        score += 2
    return matched_terms, score


def _select_candidates(
    chunks: list[dict],
    spec: dict,
    max_candidates_per_target: int,
) -> list[dict]:
    selected: dict[str, dict] = {}
    chunks_by_target = {}
    for target_index, target in enumerate(spec["targets"]):
        scored = []
        for chunk in chunks:
            if chunk.get("company") != spec["company"]:
                continue
            match = _target_matches(chunk, target)
            if match is not None:
                matched_terms, score = match
                scored.append((score, chunk["chunk_id"], chunk, matched_terms))
        scored.sort(key=lambda item: (-item[0], item[1]))
        chosen = scored[:max_candidates_per_target]
        chunks_by_target[target_index] = len(chosen)
        for score, chunk_id, chunk, matched_terms in chosen:
            candidate = selected.setdefault(chunk_id, {
                "chunk_id": chunk_id,
                "company": chunk.get("company"),
                "document": chunk.get("document"),
                "financial_year": chunk.get("financial_year"),
                "page": chunk.get("page"),
                "section": chunk.get("section"),
                "chunk_type": chunk.get("chunk_type"),
                "table_id": chunk.get("table_id"),
                "table_heading": chunk.get("table_heading"),
                "evidence_role": target["role"],
                "relevance_explanations": [],
                "matched_terms": [],
                "candidate_match_score": score,
            })
            if target["role"] == "primary":
                candidate["evidence_role"] = "primary"
            candidate["candidate_match_score"] = max(candidate["candidate_match_score"], score)
            explanation = target["explanation"]
            if explanation not in candidate["relevance_explanations"]:
                candidate["relevance_explanations"].append(explanation)
            for term in matched_terms:
                if term not in candidate["matched_terms"]:
                    candidate["matched_terms"].append(term)

    return sorted(
        selected.values(),
        key=lambda candidate: (
            candidate["evidence_role"] != "primary",
            -candidate["candidate_match_score"],
            candidate["chunk_id"],
        ),
    )


def _validate_source_chunks(chunks: list[dict]) -> dict[str, dict]:
    by_id = {}
    for index, chunk in enumerate(chunks):
        if not isinstance(chunk, dict):
            raise ValueError(f"Source chunk at line index {index} is not a JSON object.")
        chunk_id = chunk.get("chunk_id")
        if not isinstance(chunk_id, str) or not chunk_id.strip():
            raise ValueError(f"Source chunk at line index {index} has an invalid chunk_id.")
        if chunk_id in by_id:
            raise ValueError(f"Duplicate source chunk_id {chunk_id!r}.")
        by_id[chunk_id] = chunk
    if not by_id:
        raise ValueError("Source chunk JSONL contains no chunks.")
    return by_id


def validate_dataset(dataset: dict, source_chunks_by_id: dict[str, dict]) -> None:
    cases = dataset.get("cases")
    if not isinstance(cases, list) or not cases:
        raise ValueError("Generated dataset must contain at least one case.")
    seen_question_ids = set()
    for case in cases:
        question_id = case.get("id")
        if not isinstance(question_id, str) or not question_id.strip() or question_id in seen_question_ids:
            raise ValueError(f"Generated question ID is missing or duplicated: {question_id!r}.")
        seen_question_ids.add(question_id)
        for required in ("question", "company", "financial_year", "expected_answer", "verification_status"):
            if required not in case:
                raise ValueError(f"Question {question_id!r} is missing required field {required!r}.")
        if case["verification_status"] != "pending":
            raise ValueError(f"Question {question_id!r} must remain pending human review.")
        if case["relevant_chunk_ids"]:
            raise ValueError(
                f"Question {question_id!r} has unverified relevant_chunk_ids; keep them empty until human review."
            )
        candidate_ids = case.get("candidate_relevant_chunk_ids")
        candidates = case.get("candidate_chunks")
        if not isinstance(candidate_ids, list) or not isinstance(candidates, list):
            raise ValueError(f"Question {question_id!r} is missing candidate evidence fields.")
        if len(candidate_ids) != len(set(candidate_ids)):
            raise ValueError(f"Question {question_id!r} contains duplicate candidate chunk IDs.")
        if candidate_ids != [candidate.get("chunk_id") for candidate in candidates]:
            raise ValueError(f"Question {question_id!r} candidate IDs do not match candidate records.")
        for candidate in candidates:
            chunk_id = candidate.get("chunk_id")
            if chunk_id not in source_chunks_by_id:
                raise ValueError(f"Question {question_id!r} references unknown chunk ID {chunk_id!r}.")
            if candidate.get("evidence_role") not in EVIDENCE_ROLES:
                raise ValueError(f"Candidate {chunk_id!r} has invalid evidence_role.")
            if not candidate.get("relevance_explanations"):
                raise ValueError(f"Candidate {chunk_id!r} has no relevance explanation.")


def build_dataset(
    chunks: list[dict],
    source_sha256: str,
    max_candidates_per_target: int = 5,
    question_specs: list[dict] | None = None,
) -> dict:
    if isinstance(max_candidates_per_target, bool) or not isinstance(max_candidates_per_target, int) or max_candidates_per_target <= 0:
        raise ValueError("max_candidates_per_target must be a positive integer.")
    chunks_by_id = _validate_source_chunks(chunks)
    specs = question_specs if question_specs is not None else curated_questions()
    questions = []
    for spec in specs:
        candidate_chunks = _select_candidates(chunks, spec, max_candidates_per_target)
        missing_targets = []
        for target in spec["targets"]:
            if not any(
                candidate["evidence_role"] == target["role"]
                and any(term in candidate["matched_terms"] for term in target["terms"])
                for candidate in candidate_chunks
            ):
                missing_targets.append(target["explanation"])
        questions.append({
            "id": spec["id"],
            "question": spec["question"],
            "expected_answer": None,
            "company": spec["company"],
            "financial_year": spec["financial_year"],
            "financial_years": spec["financial_years"],
            "filters": (
                {"company": spec["company"], "financial_year": spec["financial_year"]}
                if spec["financial_year"] is not None
                else {"company": spec["company"]}
            ),
            "category": spec["category"],
            "relevant_chunk_ids": [],
            "candidate_relevant_chunk_ids": [candidate["chunk_id"] for candidate in candidate_chunks],
            "candidate_chunks": candidate_chunks,
            "candidate_selection_warnings": missing_targets,
            "verification_status": "pending",
        })

    dataset = {
        "dataset_version": DATASET_VERSION,
        "source": {
            "path": "data/chunks/all_chunks.jsonl",
            "sha256": source_sha256,
            "chunk_count": len(chunks_by_id),
        },
        "cases": questions,
    }
    validate_dataset(dataset, chunks_by_id)
    return dataset


def load_chunks_jsonl(path: str | Path) -> list[dict]:
    chunks = []
    try:
        with Path(path).open("r", encoding="utf-8") as source:
            for line_number, line in enumerate(source, start=1):
                if not line.strip():
                    continue
                try:
                    chunks.append(json.loads(line))
                except json.JSONDecodeError as error:
                    raise ValueError(f"Invalid JSON in {path} at line {line_number}: {error}") from error
    except OSError as error:
        raise ValueError(f"Could not read source chunks at {path}: {error}") from error
    _validate_source_chunks(chunks)
    return chunks


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1 << 16), b""):
            digest.update(block)
    return digest.hexdigest()


def write_dataset(dataset: dict, output_path: str | Path, force: bool = False) -> None:
    output_path = Path(output_path)
    if output_path.exists() and not force:
        raise FileExistsError(
            f"Refusing to overwrite {output_path}; inspect it first or pass --force."
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(dataset, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Generate pending, source-backed candidate evidence for the retrieval gold dataset."
    )
    parser.add_argument("--chunks-file", type=Path, default=SOURCE_CHUNKS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--max-candidates-per-target", type=int, default=5)
    parser.add_argument("--force", action="store_true", help="Overwrite an existing output explicitly.")
    args = parser.parse_args(argv)

    try:
        chunks = load_chunks_jsonl(args.chunks_file)
        dataset = build_dataset(
            chunks,
            source_sha256=_sha256_file(args.chunks_file),
            max_candidates_per_target=args.max_candidates_per_target,
        )
        write_dataset(dataset, args.output, force=args.force)
    except (OSError, ValueError) as error:
        print(f"Gold dataset build failed: {error}", file=sys.stderr)
        return 1

    print(f"Wrote {len(dataset['cases'])} pending questions to {args.output}")
    for case in dataset["cases"]:
        warning_count = len(case["candidate_selection_warnings"])
        print(
            f"{case['id']}: {len(case['candidate_relevant_chunk_ids'])} candidate(s), "
            f"{warning_count} evidence target(s) without candidates"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())