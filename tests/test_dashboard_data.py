import json

from ui.dashboard_data import (
    COMPANIES,
    _count_benchmark_questions,
    _count_source_documents,
    load_evaluation_metrics,
)


def test_count_source_documents_groups_known_company_reports(tmp_path):
    for _, canonical_name in COMPANIES:
        (tmp_path / f"{canonical_name}_AnnualReport_2025.pdf").touch()
        (tmp_path / f"{canonical_name}_AnnualReport_2024.pdf").touch()
    (tmp_path / "unrelated_document.pdf").touch()

    counts = _count_source_documents(tmp_path)

    assert counts == {canonical_name: 2 for _, canonical_name in COMPANIES}


def test_count_benchmark_questions_uses_actual_cases(tmp_path):
    benchmark = tmp_path / "benchmark.json"
    benchmark.write_text(
        json.dumps({"total_cases": 100, "cases": [{"id": "q1"}, {"id": "q2"}]}),
        encoding="utf-8",
    )

    assert _count_benchmark_questions(benchmark) == 2


def test_load_evaluation_metrics_reads_existing_csv(tmp_path):
    metrics = tmp_path / "metrics.csv"
    metrics.write_text("strategy,query_count\nbaseline,60\n", encoding="utf-8")

    assert load_evaluation_metrics(metrics) == [
        {"strategy": "baseline", "query_count": "60"}
    ]
