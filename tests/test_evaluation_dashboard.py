import csv
import json
from pathlib import Path

from streamlit.testing.v1 import AppTest

from ui.evaluation_dashboard import (
    _format_value,
    _render_metric_comparison,
    available_metric_choices,
    available_metric_cutoffs,
    discover_strategies,
    filter_records,
    load_evaluation_data,
    metric_column,
    metric_value_format,
    parse_number,
    strategy_definitions,
    strategy_scope_warnings,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _write_csv(path, headers, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=headers)
        writer.writeheader()
        writer.writerows(rows)


def _fixture_project(root):
    results = root / "evaluation_results"
    benchmark_dir = root / "benchmarks" / "answer_evaluation"
    (benchmark_dir).mkdir(parents=True, exist_ok=True)
    metric_headers = [
        "strategy",
        "query_count",
        "scored_queries",
        "macro_mrr",
        "macro_precision@1",
        "micro_precision@1",
        "macro_f1@5",
        "mean_latency_seconds",
        "median_latency_seconds",
        "p95_latency_seconds",
        "min_latency_seconds",
        "max_latency_seconds",
        "successful_queries",
        "failed_queries",
        "unavailable_queries",
        "empty_retrieval_count",
        "successful_retrieval_rate",
        "failed_retrieval_rate",
        "unavailable_retrieval_rate",
        "empty_retrieval_rate",
    ]
    overall_rows = [
        {
            "strategy": "baseline",
            "query_count": "3",
            "scored_queries": "2",
            "macro_mrr": "0.5",
            "macro_precision@1": "0.25",
            "micro_precision@1": "",
            "macro_f1@5": "0.1",
            "mean_latency_seconds": "0.2",
            "median_latency_seconds": "0.15",
            "p95_latency_seconds": "0.3",
            "min_latency_seconds": "0.1",
            "max_latency_seconds": "0.3",
            "successful_queries": "2",
            "failed_queries": "1",
            "unavailable_queries": "0",
            "empty_retrieval_count": "0",
            "successful_retrieval_rate": "0.6667",
            "failed_retrieval_rate": "0.3333",
            "unavailable_retrieval_rate": "0",
            "empty_retrieval_rate": "0",
        }
    ]
    overall_rows.append({**overall_rows[0], "strategy": "chunk_semantic"})
    overall_rows.append({**overall_rows[0], "strategy": "keyword_semantic"})
    _write_csv(results / "metrics_overall.csv", metric_headers, overall_rows)
    for filename, dimension, value in (
        ("metrics_by_company.csv", "company", "TataConsultancyServices"),
        ("metrics_by_category.csv", "category", "financial_metric"),
    ):
        _write_csv(
            results / filename,
            ["strategy", dimension, "query_count", "scored_queries", "macro_f1@5"],
            [{"strategy": "baseline", dimension: value, "query_count": "1",
              "scored_queries": "1", "macro_f1@5": "0.1"}],
        )
    _write_csv(
        results / "metrics_common_subset.csv",
        metric_headers,
        [{**overall_rows[0], "query_count": "2", "failed_queries": "0"}],
    )
    common_rows = [
        {
            "question_id": "tcs_q1",
            "company": "TataConsultancyServices",
            "category": "financial_metric",
            "common_subset": "True",
            "supporting_evidence_label_count": "1",
        },
        {
            "question_id": "infosys_q1",
            "company": "Infosys",
            "category": "financial_metric",
            "common_subset": "False",
            "supporting_evidence_label_count": "1",
        },
    ]
    _write_csv(
        results / "common_evaluation_subset.csv",
        [
            "question_id",
            "company",
            "category",
            "common_subset",
            "supporting_evidence_label_count",
        ],
        common_rows,
    )
    per_query_headers = [
        "question_id",
        "company",
        "category",
        "strategy",
        "retrieval_status",
        "metric_status",
        "common_subset",
        "precision@5",
        "latency_seconds",
        "retrieved_count",
        "relevant_count",
    ]
    per_query_rows = [
        {
            "question_id": "tcs_q1",
            "company": "TataConsultancyServices",
            "category": "financial_metric",
            "strategy": "baseline",
            "retrieval_status": "completed",
            "metric_status": "scored",
            "common_subset": "True",
            "precision@5": "0.1",
            "latency_seconds": "0.2",
            "retrieved_count": "5",
            "relevant_count": "1",
        },
        {
            "question_id": "infosys_q1",
            "company": "Infosys",
            "category": "financial_metric",
            "strategy": "baseline",
            "retrieval_status": "failed",
            "metric_status": "failed",
            "common_subset": "False",
            "precision@5": "",
            "latency_seconds": "0.3",
            "retrieved_count": "0",
            "relevant_count": "0",
        },
        {
            "question_id": "ril_q1",
            "company": "RelianceIndustries",
            "category": "risk",
            "strategy": "baseline",
            "retrieval_status": "completed",
            "metric_status": "not_evaluable",
            "common_subset": "True",
            "precision@5": "",
            "latency_seconds": "0.1",
            "retrieved_count": "0",
            "relevant_count": "0",
        },
    ]
    _write_csv(results / "per_query_metrics.csv", per_query_headers, per_query_rows)
    (benchmark_dir / "financial_rag_60_question_benchmark.json").write_text(
        json.dumps({"cases": [{"id": "tcs_q1"}, {"id": "infosys_q1"}, {"id": "ril_q1"}]}),
        encoding="utf-8",
    )
    annotation = {
        "status": "proposed_pending_human_review",
        "cases": [
            {
                "id": "q_with_labels",
                "proposed_gold": {
                    "expected_assertions": [
                        {
                            "evidence": [
                                {
                                    "chunk_id": "chunk-1",
                                    "relation": "supports",
                                    "verification_status": "proposed_unverified",
                                }
                            ]
                        }
                    ]
                },
            },
            {"id": "q_without_labels", "proposed_gold": {"expected_assertions": []}},
        ],
    }
    for filename in (
        "tcs_q01_q20_gold_annotations.json",
        "infosys_gold_annotations.json",
        "reliance_gold_annotations.json",
    ):
        (benchmark_dir / filename).write_text(json.dumps(annotation), encoding="utf-8")
    (results / "raw_retrieval_results.json").write_text(
        json.dumps(
            {
                "metadata": {
                    "human_evidence_verification_performed": False,
                    "human_verified_evidence_reference_count": 0,
                    "strategy_availability": {
                        "keyword_semantic": {"indexed_chunks": 150},
                        "keyword_lexical": {"indexed_chunks": 150},
                    },
                },
                "cases": [
                    {
                        "question_id": "infosys_q1",
                        "strategies": {
                            "baseline": {
                                "status": "failed",
                                "error": "Unsupported financial year.",
                            }
                        },
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    return root


def test_load_evaluation_data_preserves_saved_counts_and_annotation_status(tmp_path):
    data = load_evaluation_data(_fixture_project(tmp_path))

    assert data.errors == {}
    assert data.quality["benchmark_question_count"] == 3
    assert data.quality["common_subset_question_count"] == 1
    assert data.quality["common_subset_scored_question_count"] == 1
    assert data.quality["common_subset_excluded_question_count"] == 1
    assert data.quality["evaluation_record_count"] == 3
    assert data.quality["questions_with_supporting_evidence_labels"] == 3
    assert data.quality["questions_without_supporting_evidence_labels"] == 3
    assert data.quality["evidence_status_counts"] == {"proposed_unverified": 3}
    assert data.quality["human_verified_evidence_reference_count"] == 0
    assert data.quality["human_verification_performed"] is False
    assert data.failure_details["infosys_q1"][0]["reason"] == "Unsupported financial year."


def test_missing_and_malformed_evaluation_files_are_reported(tmp_path):
    missing = load_evaluation_data(tmp_path)
    assert "evaluation_results/metrics_overall.csv" in missing.errors
    assert "benchmarks/answer_evaluation/financial_rag_60_question_benchmark.json" in missing.errors

    root = _fixture_project(tmp_path / "malformed")
    (root / "evaluation_results" / "metrics_overall.csv").write_text(
        "strategy,unrelated_column\nbaseline,1\n", encoding="utf-8"
    )
    malformed = load_evaluation_data(root)
    assert "evaluation_results/metrics_overall.csv" in malformed.errors
    assert "Missing required columns" in malformed.errors["evaluation_results/metrics_overall.csv"]


def test_load_reports_csv_records_with_missing_fields(tmp_path):
    root = _fixture_project(tmp_path)
    (root / "evaluation_results" / "metrics_overall.csv").write_text(
        "strategy,query_count,scored_queries\nbaseline,3\n", encoding="utf-8"
    )

    data = load_evaluation_data(root)

    assert "fewer fields" in data.errors["evaluation_results/metrics_overall.csv"]


def test_saved_operational_metrics_keep_status_and_latency_values(tmp_path):
    data = load_evaluation_data(_fixture_project(tmp_path))

    row = data.tables["overall"][0]
    assert row["successful_queries"] == "2"
    assert row["failed_queries"] == "1"
    assert row["unavailable_queries"] == "0"
    assert row["mean_latency_seconds"] == "0.2"
    assert row["p95_latency_seconds"] == "0.3"


def test_strategy_and_metric_discovery_preserves_macro_micro_and_k():
    rows = [
        {
            "strategy": "baseline",
            "macro_mrr": "0.2",
            "macro_precision@1": "0.1",
            "micro_precision@1": "",
            "macro_f1@5": "0.3",
        },
        {"strategy": "hybrid_all", "macro_f1@5": "0.4"},
    ]

    assert discover_strategies(rows) == ["baseline", "hybrid_all"]
    assert available_metric_cutoffs(rows) == [1, 5]
    choices = available_metric_choices(rows)
    assert ("Macro MRR", "macro", "MRR") in choices
    assert ("Micro Precision", "micro", "Precision") in choices
    assert ("Macro F1", "macro", "F1") in choices
    assert metric_column("macro", "F1", 5) == "macro_f1@5"
    assert metric_column("micro", "Precision", 1) == "micro_precision@1"
    assert metric_column("micro", "MRR", 5) is None
    assert parse_number("") is None
    assert parse_number(None) is None
    assert parse_number("not-a-number") is None
    assert parse_number("0.0") == 0.0


def test_metric_formatting_distinguishes_decimal_percentages_and_hits():
    assert _format_value("0.1655", "macro_mrr") == "0.1655"
    assert _format_value("0.284", "macro_ndcg@10") == "0.2840"
    assert _format_value("0.1655", "mrr") == "0.1655"
    assert _format_value("0.284", "ndcg@10") == "0.2840"
    assert _format_value("0.1655", "macro_precision@5") == "16.55%"
    assert _format_value("0.284", "micro_recall@10") == "28.40%"
    assert _format_value("0.5", "macro_f1@3") == "50.00%"
    assert _format_value("2", "hits@5") == "2"
    assert _format_value("", "macro_mrr") == "Unavailable"
    assert _format_value(None, "macro_ndcg@5") == "Unavailable"
    assert metric_value_format("macro_mrr") == "decimal"
    assert metric_value_format("macro_ndcg@5") == "decimal"
    assert metric_value_format("micro_precision@5") == "percentage"
    assert metric_value_format("hits@5") == "count"


def test_metric_comparison_table_and_chart_use_metric_aware_units():
    app = AppTest.from_string(
        "import streamlit as st\n"
        "from ui.evaluation_dashboard import _render_metric_comparison\n"
        "rows = [{'strategy': 'baseline', 'query_count': '10', "
        "'scored_queries': '8', 'macro_mrr': '0.1655', "
        "'macro_ndcg@5': '0.284', 'macro_precision@5': '0.1655', "
        "'macro_recall@5': '0.284', 'macro_f1@5': '0.5'}]\n"
        "for column, label in [('macro_mrr', 'Macro MRR'), "
        "('macro_ndcg@5', 'Macro nDCG@5'), "
        "('macro_precision@5', 'Macro Precision@5'), "
        "('macro_recall@5', 'Macro Recall@5'), "
        "('macro_f1@5', 'Macro F1@5')]:\n"
        "    _render_metric_comparison(rows, {'baseline'}, column, label)\n"
    ).run()

    assert not app.exception
    rendered = "\n".join(item.value for item in app.markdown)
    assert "0.1655" in rendered
    assert "0.2840" in rendered
    assert "16.55%" in rendered
    assert "28.40%" in rendered
    assert "50.00%" in rendered
    assert "Decimal score" in rendered
    assert "Percentage" in rendered


def test_strategy_descriptions_and_scope_warnings_are_verified_and_filtered():
    known = [
        "baseline",
        "chunk_semantic",
        "keyword_semantic",
        "keyword_lexical",
        "hybrid_chunk_keyword",
        "hybrid_chunk_lexical",
        "hybrid_all",
    ]
    definitions = strategy_definitions(
        known,
        {
            "strategy_availability": {
                "keyword_semantic": {"indexed_chunks": 150},
                "keyword_lexical": {"indexed_chunks": 147},
            }
        },
    )
    by_strategy = {row["strategy"]: row["description"] for row in definitions}

    assert len(by_strategy) == len(known)
    assert by_strategy["baseline"] == "Direct Chroma semantic retrieval."
    assert "shares the baseline retrieval path" in by_strategy["chunk_semantic"]
    assert "150 indexed chunks" in by_strategy["keyword_semantic"]
    assert "147 indexed chunks" in by_strategy["keyword_lexical"]
    assert "Reciprocal Rank Fusion (RRF)" in by_strategy["hybrid_all"]
    assert strategy_definitions(
        ["baseline", "unknown_strategy"], {}
    )[1]["description"] == (
        "Description requires verification; no strategy details are available."
    )
    warnings = strategy_scope_warnings(known)
    assert any("not evidence of two independent retrieval algorithms" in item for item in warnings)
    assert any("not directly equivalent to full-corpus comparisons" in item for item in warnings)
    assert len(strategy_scope_warnings(["baseline", "unknown_strategy"])) == 0


def test_combined_record_filters_apply_without_aggregating():
    rows = [
        {
            "strategy": "baseline",
            "company": "TataConsultancyServices",
            "category": "financial_metric",
            "common_subset": "True",
        },
        {
            "strategy": "hybrid_all",
            "company": "TataConsultancyServices",
            "category": "financial_metric",
            "common_subset": "True",
        },
        {
            "strategy": "baseline",
            "company": "Infosys",
            "category": "risk",
            "common_subset": "False",
        },
    ]

    result = filter_records(
        rows,
        strategies={"baseline"},
        company="TataConsultancyServices",
        category="financial_metric",
        subset="common",
    )

    assert result == [rows[0]]
    assert filter_records(rows, subset="excluded") == [rows[2]]


def test_actual_saved_evaluation_files_have_provisional_annotations():
    data = load_evaluation_data(PROJECT_ROOT)

    assert not data.errors
    assert data.quality["benchmark_question_count"] == 60
    assert data.quality["common_subset_question_count"] == 58
    assert data.quality["evaluation_record_count"] == 420
    assert data.quality["evidence_reference_count"] == 134
    assert data.quality["questions_with_supporting_evidence_labels"] == 49
    assert data.quality["questions_without_supporting_evidence_labels"] == 11
    assert data.quality["human_verified_evidence_reference_count"] == 0
    assert data.quality["human_verification_performed"] is False
    strategies = discover_strategies(data.tables["overall"])
    assert len(strategies) == 7
    actual_definitions = strategy_definitions(strategies, data.metadata)
    assert {row["strategy"] for row in actual_definitions} == set(strategies)
    assert "150 indexed chunks" in next(
        row["description"]
        for row in actual_definitions
        if row["strategy"] == "keyword_semantic"
    )


def test_default_dashboard_uses_all_strategy_view_and_hides_advanced_sections(tmp_path):
    root = _fixture_project(tmp_path)
    app = AppTest.from_string(
        "from pathlib import Path\n"
        "from ui.evaluation_dashboard import load_evaluation_data, render_evaluation_dashboard\n"
        f"render_evaluation_dashboard(load_evaluation_data(Path({str(root)!r})))"
    ).run()

    assert not app.exception
    assert app.selectbox[0].value == "All"
    assert len(app.metric) == 0
    rendered = "\n".join(item.value for item in app.markdown)
    assert any("All strategies" in item.value for item in app.subheader)
    assert "Baseline" in rendered
    assert "Chunk Semantic" in rendered
    assert "Keyword Semantic" in rendered
    assert [button.label for button in app.button] == ["More Statistics"]
    assert len(app.tabs) == 0
    assert any("Provisional evaluation" in warning.value for warning in app.warning)


def test_more_statistics_toggle_expands_details_and_preserves_selected_strategy(tmp_path):
    root = _fixture_project(tmp_path)
    app = AppTest.from_string(
        "from pathlib import Path\n"
        "from ui.evaluation_dashboard import load_evaluation_data, render_evaluation_dashboard\n"
        f"render_evaluation_dashboard(load_evaluation_data(Path({str(root)!r})))"
    ).run()

    app.button[0].click().run()

    assert len(app.tabs) == 6
    assert app.button[0].label == "Hide Statistics"
    assert app.selectbox[0].value == "All"
    assert any("Retrieval Strategy Definitions & Evaluation Scope" in item.value for item in app.subheader)

    app.button[0].click().run()
    assert [button.label for button in app.button] == ["More Statistics"]
    assert len(app.tabs) == 0
    assert app.selectbox[0].value == "All"


def test_individual_strategy_selection_restores_card_layout(tmp_path):
    root = _fixture_project(tmp_path)
    app = AppTest.from_string(
        "from pathlib import Path\n"
        "from ui.evaluation_dashboard import load_evaluation_data, render_evaluation_dashboard\n"
        f"render_evaluation_dashboard(load_evaluation_data(Path({str(root)!r})))"
    ).run()

    app.selectbox[0].select("baseline").run()

    assert not app.exception
    assert app.selectbox[0].value == "baseline"
    labels = {metric.label for metric in app.metric}
    assert labels == {"Precision@5", "Recall@5", "Recall@10", "MRR"}


def test_evaluation_dashboard_renders_saved_data_without_dataframe(tmp_path):
    root = _fixture_project(tmp_path)
    app = AppTest.from_string(
        "from pathlib import Path\n"
        "from ui.evaluation_dashboard import load_evaluation_data, render_evaluation_dashboard\n"
        f"render_evaluation_dashboard(load_evaluation_data(Path({str(root)!r})))"
    ).run()
    app.button[0].click().run()

    assert not app.exception
    assert any("Provisional evaluation" in warning.value for warning in app.warning)
    assert any("independent retrieval algorithms" in warning.value for warning in app.warning)
    assert any("full-corpus comparisons" in warning.value for warning in app.warning)
    assert any(
        "Retrieval Strategy Definitions & Evaluation Scope" in item.value
        for item in app.subheader
    )


def test_evaluation_dashboard_subset_filter_interaction(tmp_path):
    root = _fixture_project(tmp_path)
    app = AppTest.from_string(
        "from pathlib import Path\n"
        "from ui.evaluation_dashboard import load_evaluation_data, render_evaluation_dashboard\n"
        f"render_evaluation_dashboard(load_evaluation_data(Path({str(root)!r})))"
    ).run()
    app.button[0].click().run()
    app.selectbox(key="evaluation_subset").select("Full evaluation").run()

    assert not app.exception
    assert app.selectbox(key="evaluation_subset").value == "Full evaluation"


def test_evaluation_dashboard_metric_and_k_filters_are_independent(tmp_path):
    root = _fixture_project(tmp_path)
    app = AppTest.from_string(
        "from pathlib import Path\n"
        "from ui.evaluation_dashboard import load_evaluation_data, render_evaluation_dashboard\n"
        f"render_evaluation_dashboard(load_evaluation_data(Path({str(root)!r})))"
    ).run()
    app.button[0].click().run()
    app.selectbox(key="evaluation_metric_choice").select("Micro Precision").run()
    app.selectbox(key="evaluation_k_cutoff").select(1).run()

    assert not app.exception
    assert app.selectbox(key="evaluation_metric_choice").value == "Micro Precision"
    assert app.selectbox(key="evaluation_k_cutoff").value == 1
    assert any("Micro Precision@1" in item.value for item in app.markdown)


def test_evaluation_dashboard_renders_actual_data():
    app = AppTest.from_string(
        "from pathlib import Path\n"
        "from ui.evaluation_dashboard import load_evaluation_data, render_evaluation_dashboard\n"
        f"render_evaluation_dashboard(load_evaluation_data(Path({str(PROJECT_ROOT)!r})))"
    ).run()

    assert not app.exception
    assert any("Provisional evaluation" in warning.value for warning in app.warning)
