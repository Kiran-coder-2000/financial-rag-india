"""
test_query_planner_manual.py

Manual check of the query planner against the REAL Gemini API. NOT a
pytest test despite the filename (it contributes zero tests and does
nothing on import), and it makes real, billable API calls -- run it
deliberately:

    python test_query_planner_manual.py                  # the 4 spec examples + 2 extra probes
    python test_query_planner_manual.py --query "..."    # plan one ad-hoc question

Needs GEMINI_API_KEY and GEMINI_MODEL (from .env or the environment)
and a built corpus at data/chunks/all_chunks.jsonl (used only to learn
the canonical company names; nothing is retrieved).

How to read the output: every plan is printed in full -- read it.
The PASS/FAIL lines are cheap sanity checks on the properties the
spec cares about, not a quality score. A FAIL on metric naming (e.g.
"revenue" vs "revenue_from_operations") may be a legitimate variation;
a FAIL on company resolution, calculation order or an invented
calculation is a real prompt/pipeline problem.
"""

import argparse
import sys
import time

from src.query_planner import (
    QueryPlanner,
    QueryPlannerError,
    load_settings,
    registry_from_chunks_file,
)

CHUNKS_FILE = "data/chunks/all_chunks.jsonl"


# ---------------- per-example checks: (label, passed) ----------------

def checks_example1(plan, reg):
    companies = {t.company for t in plan.tasks}
    return [
        ("one task per company (TCS, Reliance)", companies == {reg.resolve("TCS"), reg.resolve("RI")}),
        ("metric normalised to price_to_earnings_ratio", all(t.metric == "price_to_earnings_ratio" for t in plan.tasks)),
        ("financial year left null", all(t.financial_year is None for t in plan.tasks)),
        ("comparison intent identified", "comparison" in {i.value for i in plan.intents}),
        ("no unresolved companies", plan.unresolved_companies == []),
    ]


def checks_example2(plan, reg):
    ops = [c for c in plan.calculations if c.operation.value == "subtract"]
    by_id = {t.task_id: t for t in plan.tasks}
    ordered = [by_id[o].metric for o in ops[0].operands] if ops and all(o in by_id for o in ops[0].operands) else []
    return [
        ("a subtract calculation exists", bool(ops)),
        ("operands are [total_income, total_operating_cost] in that order", ordered == ["total_income", "total_operating_cost"]),
        ("not silently replaced by operating profit", not any("operating_profit" in (t.metric or "") for t in plan.tasks)),
        ("all tasks resolve to TCS", {t.company for t in plan.tasks} == {reg.resolve("TCS")}),
    ]


def checks_example3(plan, reg):
    return [
        ("8 tasks (2 companies x 2 metrics x 2 years)", len(plan.tasks) == 8),
        ("both financial years present", {t.financial_year for t in plan.tasks} == {"FY2025", "FY2026"}),
        ("both companies present", {t.company for t in plan.tasks} == {reg.resolve("TCS"), reg.resolve("Infosys")}),
        ("visualization_required is true", plan.visualization_required is True),
        ("chart type suggested", plan.visualization is not None),
        ("no growth/calculation invented (not requested)", plan.calculations == []),
    ]


def checks_example4(plan, reg):
    return [
        ("typo corrected in normalized_query", "revenue" in plan.normalized_query.lower() and "revnue" not in plan.normalized_query.lower()),
        ("original query preserved exactly", plan.original_query == "What was Infosys revnue in FY2026?"),
        ("FY2026 kept", all(t.financial_year == "FY2026" for t in plan.tasks)),
        ("company resolves to Infosys", {t.company for t in plan.tasks} == {reg.resolve("Infosys")}),
    ]


def checks_unsupported(plan, reg):
    supported = set(reg.canonical_names)
    return [
        ("unsupported company flagged", bool(plan.unresolved_companies)),
        ("never mapped to a supported company", not any(t.company in supported for t in plan.tasks)),
        ("asks for clarification", plan.needs_clarification),
    ]


def checks_out_of_scope(plan, reg):
    return [("no retrieval tasks for an unrelated question", plan.tasks == [] or plan.needs_clarification)]


EXAMPLES = [
    ("Example 1 - alias + terminology", "Compare the P/e ratio of TCS and RI", checks_example1),
    ("Example 2 - calculation", "What is the total income of TCS subtracting its total operating cost?", checks_example2),
    ("Example 3 - decomposition + chart", "Compare TCS and Infosys revenue and PAT for FY2025 and FY2026, and show a graph.", checks_example3),
    ("Example 4 - typo", "What was Infosys revnue in FY2026?", checks_example4),
    ("Probe - unsupported company", "What is HDFC Bank's net profit in FY2025?", checks_unsupported),
    ("Probe - out of scope", "What is the weather in Mumbai today?", checks_out_of_scope),
]


def main() -> int:
    parser = argparse.ArgumentParser(description="Manual real-Gemini check of the query planner.")
    parser.add_argument("--query", help="plan a single ad-hoc question instead of the built-in examples")
    parser.add_argument("--chunks-file", default=CHUNKS_FILE)
    args = parser.parse_args()

    try:
        settings = load_settings()
        registry = registry_from_chunks_file(args.chunks_file)
        planner = QueryPlanner(registry, settings)
    except (QueryPlannerError, FileNotFoundError) as e:
        print(f"Setup failed: {e}")
        return 1

    print(f"Model: {settings.model}")                       # never print the API key
    print(f"Companies in corpus: {', '.join(registry.canonical_names)}")
    if registry.ambiguous_aliases:
        print(f"Ambiguous aliases dropped: {registry.ambiguous_aliases}")

    jobs = [("Ad-hoc query", args.query, None)] if args.query else EXAMPLES
    passed = failed = errors = 0

    for title, query, checker in jobs:
        print("\n" + "=" * 70)
        print(f"{title}\nQuery: {query}")
        print("=" * 70)
        started = time.perf_counter()
        try:
            plan = planner.plan(query)
        except QueryPlannerError as e:
            errors += 1
            print(f"PLANNER ERROR ({type(e).__name__}): {e}")
            continue
        print(f"({time.perf_counter() - started:.1f}s)")
        print(plan.model_dump_json(indent=2))
        if plan.warnings:
            print("\nWarnings raised by post-validation:")
            for w in plan.warnings:
                print(f"  - {w}")
        if checker:
            print("\nChecks:")
            for label, ok in checker(plan, registry):
                print(f"  [{'PASS' if ok else 'FAIL'}] {label}")
                passed += ok
                failed += (not ok)

    if not args.query:
        print("\n" + "-" * 70)
        print(f"Checks passed: {passed}   failed: {failed}   planner errors: {errors}")
    return 1 if (failed or errors) else 0


if __name__ == "__main__":
    sys.exit(main())