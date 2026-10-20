"""
tests/test_query_planner.py

Query planner tests. Gemini is fully mocked (FakeClient): no network,
no API key, no cost. NOTE what these tests can and cannot prove: they
verify the schema, the deterministic post-validation, error handling
and the request we send. They do NOT measure how well the real model
plans -- that is what test_query_planner_manual.py is for.
"""

import json
import logging
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from google.genai import errors as genai_errors
from pydantic import ValidationError

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import query_planner as qp

KEY = "SECRET-KEY-123"
REG = qp.CompanyRegistry(
    ["RelianceIndustries", "TataConsultancyServices", "Infosys"],
    {"RelianceIndustries": ["RIL"], "Infosys": ["INFY"]},
)


# ---------------- helpers ----------------

def task(tid, query="q", mention=None, company=None, fy=None, metric=None, chunk="table"):
    return {"task_id": tid, "query": query, "company_mention": mention, "company": company,
            "financial_year": fy, "metric": metric, "chunk_type": chunk}


def plan_json(**over):
    base = {"normalized_query": "q", "intents": ["lookup"], "tasks": [task("t1")], "calculations": [],
            "comparison": None, "visualization_required": False, "visualization": None,
            "needs_clarification": False, "clarification_question": None}
    base.update(over)
    return json.dumps(base)


class FakeModels:
    def __init__(self, script):
        self.script, self.calls = list(script), []

    def generate_content(self, model, contents, config):
        self.calls.append({"model": model, "contents": contents, "config": config})
        item = self.script.pop(0)
        if isinstance(item, Exception):
            raise item
        return SimpleNamespace(text=item)


class FakeClient:
    def __init__(self, script):
        self.models = FakeModels(script)


def make_planner(script, **settings_over):
    sleeps = []
    settings = qp.PlannerSettings(api_key=KEY, model="test-model", backoff_s=1.0, **settings_over)
    client = FakeClient(script)
    planner = qp.QueryPlanner(REG, settings, client=client, sleep=sleeps.append)
    return planner, client, sleeps


def intent_values(plan):
    return {i.value for i in plan.intents}


# ================= the four specification examples =================
# The mocked JSON is what a good model would return; the assertions
# check that the pipeline keeps it correct and fills in the guarantees.

class TestSpecExamples:
    def test_example1_compare_pe_ratio_tcs_and_ri(self):
        q = "Compare the P/e ratio of TCS and RI"
        raw = plan_json(
            normalized_query=q, intents=["comparison"],
            tasks=[task("t1", "Tata Consultancy Services price to earnings ratio", "TCS", "TCS", None, "P/E"),
                   task("t2", "Reliance Industries price to earnings ratio", "RI", "RI", None, "price_to_earnings_ratio")],
            comparison={"dimensions": ["company"]})
        planner, _, _ = make_planner([raw])
        plan = planner.plan(q)

        assert plan.original_query == q
        assert [t.company for t in plan.tasks] == ["TataConsultancyServices", "RelianceIndustries"]
        assert all(t.metric == "price_to_earnings_ratio" for t in plan.tasks)
        assert all(t.financial_year is None for t in plan.tasks)        # unspecified year stays null
        assert "comparison" in intent_values(plan)
        assert plan.comparison.dimensions == [qp.Dimension.company]
        assert plan.unresolved_companies == []

    def test_example2_calculation_subtraction_not_replaced(self):
        q = "What is the total income of TCS subtracting its total operating cost?"
        raw = plan_json(
            normalized_query=q, intents=["calculation"],
            tasks=[task("t1", "TCS total income", "TCS", "TataConsultancyServices", None, "total_income"),
                   task("t2", "TCS total operating cost", "TCS", "TataConsultancyServices", None, "total_operating_cost")],
            calculations=[{"operation": "subtract", "operands": ["t1", "t2"], "description": "total income minus total operating cost"}])
        plan = make_planner([raw])[0].plan(q)

        assert [t.metric for t in plan.tasks] == ["total_income", "total_operating_cost"]
        assert plan.calculations[0].operation is qp.Operation.subtract
        assert plan.calculations[0].operands == ["t1", "t2"]            # order = a - b
        assert "operating_profit" not in [t.metric for t in plan.tasks]
        assert "calculation" in intent_values(plan)

    def test_example3_decomposition_and_visualization(self):
        q = "Compare TCS and Infosys revenue and PAT for FY2025 and FY2026, and show a graph."
        tasks = []
        for co in ("TataConsultancyServices", "Infosys"):
            for metric in ("revenue", "profit_after_tax"):
                for fy in ("FY2025", "FY2026"):
                    tasks.append(task(f"t{len(tasks) + 1}", f"{co} {metric} {fy}", None, co, fy, metric))
        raw = plan_json(
            normalized_query=q, intents=["comparison"], tasks=tasks,
            comparison={"dimensions": ["company"]}, visualization_required=True,
            visualization={"chart_type": "grouped_bar", "x_axis": "financial_year", "y_axis": "value", "group_by": "company"})
        plan = make_planner([raw])[0].plan(q)

        assert len(plan.tasks) == 8
        assert {t.financial_year for t in plan.tasks} == {"FY2025", "FY2026"}
        assert plan.visualization_required is True
        assert plan.visualization.chart_type is qp.ChartType.grouped_bar
        assert plan.calculations == []                                  # growth was NOT requested -> none invented

    def test_example4_typo_corrected_original_preserved(self):
        q = "What was Infosys revnue in FY2026?"
        raw = plan_json(normalized_query="What was Infosys revenue in FY2026?",
                        tasks=[task("t1", "Infosys revenue FY2026", "Infosys", "Infosys", "FY2026", "revenue")])
        plan = make_planner([raw])[0].plan(q)

        assert plan.original_query == "What was Infosys revnue in FY2026?"   # exact, untouched
        assert plan.normalized_query == "What was Infosys revenue in FY2026?"
        assert plan.tasks[0].financial_year == "FY2026"


# ================= deterministic safety guards =================

class TestPostValidation:
    def test_original_query_is_stamped_by_code_not_llm(self):
        raw = json.loads(plan_json())
        raw["original_query"] = "SOMETHING THE LLM MADE UP"           # extra key must be ignored
        plan = make_planner([json.dumps(raw)])[0].plan("real user text")
        assert plan.original_query == "real user text"

    def test_normalized_query_that_changes_a_number_is_rejected(self):
        q = "Infosys revenue in FY2026"
        raw = plan_json(normalized_query="Infosys revenue in FY2025", tasks=[task("t1", "x", None, "Infosys", "FY2026")])
        plan = make_planner([raw])[0].plan(q)
        assert plan.normalized_query == q
        assert any("number" in w for w in plan.warnings)

    def test_hallucinated_financial_year_set_to_null(self):
        raw = plan_json(tasks=[task("t1", "x", None, "Infosys", "FY2025")])
        plan = make_planner([raw])[0].plan("What is Infosys revenue?")   # user never gave a year
        assert plan.tasks[0].financial_year is None
        assert any("does not appear" in w for w in plan.warnings)

    @pytest.mark.parametrize("llm_value,query", [("FY25", "Infosys FY25 revenue"), ("2024-25", "Infosys 2024-25 revenue"),
                                                 ("FY 2025", "Infosys FY 2025 revenue")])
    def test_year_formats_normalised_when_user_stated_them(self, llm_value, query):
        plan = make_planner([plan_json(tasks=[task("t1", "x", None, "Infosys", llm_value)])])[0].plan(query)
        assert plan.tasks[0].financial_year == "FY2025"

    def test_unparseable_year_nulled_with_warning(self):
        plan = make_planner([plan_json(tasks=[task("t1", "x", None, "Infosys", "next year")])])[0].plan("Infosys revenue next year")
        assert plan.tasks[0].financial_year is None
        assert any("could not parse" in w for w in plan.warnings)

    def test_only_unsupported_company_asks_for_clarification(self):
        raw = plan_json(tasks=[task("t1", "HDFC net profit", "HDFC Bank", "HDFC Bank", None, "net_profit")])
        plan = make_planner([raw])[0].plan("What is HDFC Bank net profit?")
        assert plan.tasks[0].company is None and plan.tasks[0].is_unresolved_company
        assert plan.unresolved_companies == ["HDFC Bank"]
        assert plan.needs_clarification is True
        assert "Infosys" in plan.clarification_question                  # lists what IS supported

    def test_mixed_supported_and_unsupported_keeps_plan(self):
        raw = plan_json(tasks=[task("t1", "x", "TCS", "TCS"), task("t2", "y", "HDFC", "HDFC")])
        plan = make_planner([raw])[0].plan("Compare TCS and HDFC")
        assert plan.needs_clarification is False
        assert [t.is_unresolved_company for t in plan.tasks] == [False, True]

    def test_visualization_dropped_when_not_required(self):
        viz = {"chart_type": "bar", "x_axis": "a", "y_axis": "b", "group_by": None}
        plan = make_planner([plan_json(visualization_required=False, visualization=viz)])[0].plan("q")
        assert plan.visualization is None

    def test_intents_made_consistent_with_structure(self):
        raw = plan_json(
            intents=["lookup"], tasks=[task("t1"), task("t2")],
            calculations=[{"operation": "percent_change", "operands": ["t1", "t2"], "description": "growth"}],
            comparison={"dimensions": ["financial_year"]})
        plan = make_planner([raw])[0].plan("q")
        assert {"growth", "comparison"} <= intent_values(plan)


# ================= schema / structural validation =================

class TestSchemaValidation:
    def _load(self, **over):
        return qp.LLMPlan.model_validate_json(plan_json(**over))

    def test_valid_plan_loads(self):
        assert self._load().tasks[0].task_id == "t1"

    def test_missing_required_field_rejected(self):
        raw = json.loads(plan_json()); del raw["comparison"]
        with pytest.raises(ValidationError):
            qp.LLMPlan.model_validate_json(json.dumps(raw))

    def test_unknown_operand_rejected(self):
        with pytest.raises(ValidationError, match="unknown task_id"):
            self._load(calculations=[{"operation": "subtract", "operands": ["t1", "t9"], "description": "d"}])

    def test_subtract_needs_exactly_two_operands(self):
        with pytest.raises(ValidationError, match="exactly 2"):
            self._load(tasks=[task("t1"), task("t2"), task("t3")],
                       calculations=[{"operation": "subtract", "operands": ["t1", "t2", "t3"], "description": "d"}])

    def test_duplicate_task_ids_rejected(self):
        with pytest.raises(ValidationError, match="unique"):
            self._load(tasks=[task("t1"), task("t1")])

    def test_empty_tasks_need_clarification_or_out_of_scope(self):
        with pytest.raises(ValidationError, match="tasks may be empty"):
            self._load(tasks=[])
        assert self._load(tasks=[], intents=["out_of_scope"]).tasks == []
        assert self._load(tasks=[], needs_clarification=True, clarification_question="Which company?")

    def test_clarification_requires_question(self):
        with pytest.raises(ValidationError, match="clarification_question"):
            self._load(needs_clarification=True, clarification_question=None)

    def test_task_cap(self):
        with pytest.raises(ValidationError, match="too many tasks"):
            self._load(tasks=[task(f"t{i}") for i in range(qp.MAX_TASKS + 1)])

    def test_invalid_enum_value_rejected(self):
        with pytest.raises(ValidationError):
            self._load(tasks=[task("t1", chunk="spreadsheet")])


# ================= planner: request, retries, errors =================

class TestRequestSent:
    def test_request_shape(self):
        planner, client, _ = make_planner([plan_json()])
        planner.plan("  Infosys revenue  ")
        call = client.models.calls[0]
        assert call["model"] == "test-model"
        assert call["config"].temperature == 0.0
        assert call["config"].response_mime_type == "application/json"
        assert call["config"].response_schema is qp.LLMPlan
        assert "<<<\nInfosys revenue\n>>>" in call["contents"]            # stripped, delimited as data
        for name in REG.canonical_names:
            assert name in call["config"].system_instruction              # registry is in the prompt

    def test_api_key_never_in_prompt(self):
        planner, client, _ = make_planner([plan_json()])
        planner.plan("q")
        call = client.models.calls[0]
        assert KEY not in call["contents"] and KEY not in call["config"].system_instruction


class TestMalformedResponses:
    def test_bad_json_then_valid_retries_with_feedback(self):
        planner, client, _ = make_planner(["{not json", plan_json()])
        plan = planner.plan("q")
        assert len(client.models.calls) == 2 and plan.tasks
        assert "rejected" in client.models.calls[1]["contents"]

    def test_missing_field_then_valid(self):
        broken = json.loads(plan_json()); del broken["tasks"]
        planner, client, _ = make_planner([json.dumps(broken), plan_json()])
        planner.plan("q")
        assert "tasks" in client.models.calls[1]["contents"]              # error names the field

    def test_invalid_twice_raises_response_error(self):
        planner, client, _ = make_planner(["nope", "still nope"])
        with pytest.raises(qp.PlannerResponseError):
            planner.plan("q")
        assert len(client.models.calls) == 2                              # bounded, not infinite

    def test_empty_reply_treated_as_invalid(self):
        planner, _, _ = make_planner(["", "   "])
        with pytest.raises(qp.PlannerResponseError):
            planner.plan("q")

    def test_markdown_fences_tolerated(self):
        planner, _, _ = make_planner(["```json\n" + plan_json() + "\n```"])
        assert planner.plan("q").tasks


class TestApiErrors:
    def test_transient_429_retried_with_backoff(self):
        err = genai_errors.ClientError(429, {"error": {"message": "slow down"}})
        planner, client, sleeps = make_planner([err, plan_json()])
        assert planner.plan("q").tasks
        assert len(client.models.calls) == 2 and sleeps == [1.0]

    def test_transient_errors_exhausted_raises_api_error(self):
        err = genai_errors.ServerError(503, {"error": {"message": "down"}})
        planner, client, sleeps = make_planner([err, err, err])
        with pytest.raises(qp.PlannerAPIError, match="503"):
            planner.plan("q")
        assert sleeps == [1.0, 2.0]                                       # exponential backoff

    def test_permanent_400_not_retried(self):
        err = genai_errors.ClientError(400, {"error": {"message": "bad request"}})
        planner, client, sleeps = make_planner([err])
        with pytest.raises(qp.PlannerAPIError):
            planner.plan("q")
        assert len(client.models.calls) == 1 and sleeps == []

    def test_network_timeout_is_transient(self):
        planner, _, sleeps = make_planner([TimeoutError("t/o"), plan_json()])
        assert planner.plan("q").tasks and sleeps == [1.0]


class TestCredentialSafety:
    def test_key_redacted_from_api_error_message(self):
        err = genai_errors.ClientError(403, {"error": {"message": f"invalid key {KEY}"}})
        planner, _, _ = make_planner([err])
        with pytest.raises(qp.PlannerAPIError) as exc:
            planner.plan("q")
        assert KEY not in str(exc.value) and "***" in str(exc.value)
        assert exc.value.__cause__ is None and exc.value.__suppress_context__

    def test_key_not_in_settings_repr_or_logs(self, caplog):
        settings = qp.PlannerSettings(api_key=KEY, model="m")
        assert KEY not in repr(settings)
        err = genai_errors.ClientError(403, {"error": {"message": f"bad {KEY}"}})
        planner, _, _ = make_planner([err])
        with caplog.at_level(logging.DEBUG), pytest.raises(qp.PlannerAPIError):
            planner.plan("q")
        assert KEY not in caplog.text


class TestInputValidation:
    @pytest.mark.parametrize("bad", ["", "   \n", None, 123])
    def test_unusable_input_rejected_without_api_call(self, bad):
        planner, client, _ = make_planner([])
        with pytest.raises(qp.PlannerInputError):
            planner.plan(bad)
        assert client.models.calls == []

    def test_too_long_rejected(self):
        planner, client, _ = make_planner([])
        with pytest.raises(qp.PlannerInputError, match="too long"):
            planner.plan("x" * (qp.MAX_QUERY_CHARS + 1))
        assert client.models.calls == []


class TestSettings:
    def test_missing_key_reported_by_name(self):
        with pytest.raises(qp.PlannerConfigError, match="GEMINI_API_KEY"):
            qp.load_settings({"GEMINI_MODEL": "m"})

    def test_missing_model_reported_by_name(self):
        with pytest.raises(qp.PlannerConfigError, match="GEMINI_MODEL"):
            qp.load_settings({"GEMINI_API_KEY": KEY})

    def test_valid_env(self):
        s = qp.load_settings({"GEMINI_API_KEY": f" {KEY} ", "GEMINI_MODEL": "m"})
        assert s.api_key == KEY and s.model == "m"


class TestFallbackPlan:
    def test_fallback_is_explicit_and_unfiltered(self):
        plan = qp.fallback_plan("What is revenue?", "API down")
        assert plan.planner_status == "fallback"
        assert len(plan.tasks) == 1 and plan.tasks[0].company is None
        assert "API down" in plan.warnings[0]


# ================= registry & helpers =================

class TestCompanyRegistry:
    def test_aliases_derived_from_canonical_names(self):
        assert REG.resolve("RI") == "RelianceIndustries"                  # auto-initialism
        assert REG.resolve("TCS") == "TataConsultancyServices"
        assert REG.resolve("Tata Consultancy Services Ltd.") == "TataConsultancyServices"
        assert REG.resolve("INFY") == "Infosys"                           # explicit extra alias
        assert REG.resolve("HDFC") is None and REG.resolve(None) is None

    def test_colliding_alias_is_dropped_not_guessed(self):
        reg = qp.CompanyRegistry(["AlphaBeta", "AaronBrown"])             # both -> initialism "AB"
        assert reg.resolve("AB") is None and "ab" in reg.ambiguous_aliases
        assert reg.resolve("AlphaBeta") == "AlphaBeta"

    def test_from_chunks_file_uses_corpus_company_values(self, tmp_path):
        f = tmp_path / "all_chunks.jsonl"
        f.write_text("\n".join(json.dumps({"chunk_id": str(i), "text": "x", "company": c})
                               for i, c in enumerate(["Infosys", "Infosys", "RelianceIndustries"])))
        reg = qp.registry_from_chunks_file(f)
        assert reg.canonical_names == ["Infosys", "RelianceIndustries"]

    def test_empty_corpus_rejected(self, tmp_path):
        f = tmp_path / "all_chunks.jsonl"; f.write_text(json.dumps({"chunk_id": "1", "text": "x"}))
        with pytest.raises(qp.PlannerConfigError):
            qp.registry_from_chunks_file(f)


class TestHelpers:
    @pytest.mark.parametrize("value,expected", [
        ("FY2025", "FY2025"), ("fy25", "FY2025"), ("2024-25", "FY2025"), ("2024-2025", "FY2025"),
        ("FY'26", "FY2026"), ("fiscal year 2026", "FY2026"), ("latest", None), ("FY1", None), (None, None)])
    def test_normalize_financial_year(self, value, expected):
        assert qp.normalize_financial_year(value) == expected

    def test_metadata_year_matches_chunk_format(self):
        assert qp.metadata_financial_year("FY2025") == "2024-2025"

    @pytest.mark.parametrize("value,expected", [
        ("P/E", "price_to_earnings_ratio"), ("PAT", "profit_after_tax"),
        ("Total Operating Cost", "total_operating_cost"), ("revenue", "revenue"), (None, None)])
    def test_normalize_metric(self, value, expected):
        assert qp.normalize_metric(value) == expected

    def test_total_income_is_not_rewritten_to_revenue(self):
        assert qp.normalize_metric("total_income") == "total_income"