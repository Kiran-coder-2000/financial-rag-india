"""Read-only dashboard views over existing corpus and evaluation artifacts."""

import csv
import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path

from src.evaluation_config import STRATEGIES

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CHUNKS_FILE = PROJECT_ROOT / "data" / "chunks" / "all_chunks.jsonl"
RAW_DOCUMENTS_DIR = PROJECT_ROOT / "data" / "raw"
BENCHMARK_FILE = (
    PROJECT_ROOT
    / "benchmarks"
    / "answer_evaluation"
    / "financial_rag_60_question_benchmark.json"
)
EVALUATION_METRICS_FILE = PROJECT_ROOT / "evaluation_results" / "metrics_overall.csv"

COMPANIES = (
    ("TCS", "TataConsultancyServices"),
    ("Infosys", "Infosys"),
    ("Reliance Industries", "RelianceIndustries"),
)


@dataclass(frozen=True)
class OverviewData:
    companies_covered: int | None
    benchmark_questions: int | None
    retrieval_strategies: int
    corpus_documents: int | None
    corpus_chunks: int | None
    company_documents: dict[str, int] | None


def _count_source_documents(raw_documents_dir: Path) -> dict[str, int] | None:
    if not raw_documents_dir.is_dir():
        logger.warning("Annual-report source directory is unavailable: %s", raw_documents_dir)
        return None

    counts = {canonical_name: 0 for _, canonical_name in COMPANIES}
    for path in raw_documents_dir.rglob("*.pdf"):
        filename = re.sub(r"[^a-z0-9]", "", path.stem.casefold())
        for canonical_name in counts:
            company_key = re.sub(r"[^a-z0-9]", "", canonical_name.casefold())
            if filename.startswith(company_key):
                counts[canonical_name] += 1
                break
    return counts


def _count_benchmark_questions(benchmark_file: Path) -> int | None:
    try:
        benchmark = json.loads(benchmark_file.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        logger.warning("Could not read benchmark questions from %s: %s", benchmark_file, error)
        return None

    cases = benchmark.get("cases") if isinstance(benchmark, dict) else None
    if not isinstance(cases, list):
        logger.warning("Benchmark file has no cases list: %s", benchmark_file)
        return None
    return len(cases)


def _count_indexed_chunks(project_root: Path) -> int | None:
    try:
        from src.retrieval import ChromaRetrievalError, ChromaRetriever
    except ImportError as error:
        logger.warning("Could not import the retrieval index client: %s", error)
        return None
    try:
        return ChromaRetriever(
            persist_directory=project_root / "data" / "chroma_db",
        ).count
    except (OSError, ChromaRetrievalError) as error:
        logger.warning("Could not read the persisted retrieval index: %s", error)
        return None


def load_overview_data(project_root: Path = PROJECT_ROOT) -> OverviewData:
    """Return metrics derived from files and the configured persisted index."""
    company_documents = _count_source_documents(project_root / "data" / "raw")
    benchmark_questions = _count_benchmark_questions(
        project_root
        / "benchmarks"
        / "answer_evaluation"
        / "financial_rag_60_question_benchmark.json"
    )
    corpus_documents = (
        sum(company_documents.values()) if company_documents is not None else None
    )
    companies_covered = (
        sum(count > 0 for count in company_documents.values())
        if company_documents is not None
        else None
    )

    return OverviewData(
        companies_covered=companies_covered,
        benchmark_questions=benchmark_questions,
        retrieval_strategies=len(STRATEGIES),
        corpus_documents=corpus_documents,
        corpus_chunks=_count_indexed_chunks(project_root),
        company_documents=company_documents,
    )


def load_evaluation_metrics(
    metrics_file: Path = EVALUATION_METRICS_FILE,
) -> list[dict[str, str]] | None:
    """Read existing aggregate evaluation output without recomputing metrics."""
    try:
        with metrics_file.open(encoding="utf-8-sig", newline="") as file:
            rows = list(csv.DictReader(file))
    except (OSError, UnicodeError, csv.Error) as error:
        logger.warning("Could not read saved evaluation metrics from %s: %s", metrics_file, error)
        return None
    return rows or None
