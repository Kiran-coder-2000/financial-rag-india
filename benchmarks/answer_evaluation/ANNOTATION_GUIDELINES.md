# Financial Answer Evaluation Annotation Guidelines

Version: 1.0.0

## Scope and evidence

These guidelines apply to the human-reviewed gold cases and answer reviews in
`answer_evaluation_dataset.json` and
`financial_rag_60_question_benchmark.json`. A candidate chunk ID is only a
search lead. It is not a relevance judgment, a citation, or ground truth.

The refined 60-case benchmark is a separate dataset; the 15-case seed remains
unchanged. Its candidate expected actions remain unverified, its gold fields
remain null while cases are pending, and any revised prompt records its
predecessor ID and prior wording.

Use canonical company metadata values: `Infosys`, `TataConsultancyServices`,
and `RelianceIndustries`. Represent a fiscal year as `YYYY-YYYY` (for example,
`2023-2024`), a quarter as `FYyyyy-Qn` (for example, `FY2024-Q4`), and a
live/current request as `current`. Periods in questions and gold assertions
must use the same normalized representation.

Before marking a case verified, inspect the original source chunk and enough
adjacent table context to establish its meaning. Record the chunk ID, reviewer,
relationship to the assertion, and a concise rationale. Verify the company,
document, reporting period, table heading, row/column labels, units, and
consolidated/standalone or segment scope whenever applicable. Do not copy
candidate metadata as a substitute for checking the source.

Gold assertions describe the facts that an answerable response should convey.
Keep each assertion atomic: one entity, metric or proposition, and reporting
period per assertion where practical. Compound generated claims must be split
into atomic assertions in their answer review.

## Support labels

Use the following labels for each atomic assertion and each individual
claim-to-citation relationship:

- `supported`: the cited source directly supports the complete material
  assertion, including required period, scope, units, and surrounding context.
- `partially_supported`: some material components are supported, but another
  component is absent, ambiguous, or insufficiently specified. Explain which
  components are supported and which are not.
- `contradicted`: the source affirmatively conflicts with a material part of
  the assertion. Preserve this label when other parts are supported; a material
  contradiction takes precedence over partial support.
- `not_supported`: the cited source neither supports nor affirmatively
  contradicts the assertion.

An existing chunk ID proves only that the chunk was retrieved; it does not
prove that it supports the claim. Assess the exact text of every cited chunk,
not the answer as a whole. A citation that is irrelevant to a claim is
`not_supported`. Record separate judgments for separate citations.

For expected gold assertions, evidence references use `supports`,
`contradicts`, or `context_only`. Each reference must have
`verification_status: human_verified`, a reviewer identity, and a rationale.
The authoritative location for human-verified gold evidence is
`gold.expected_assertions[].evidence`. Each evidence reference must include
its `chunk_id`, `verification_status`, `verified_by` reviewer identity,
`relation`, and `rationale`. Do not duplicate these references in a
top-level `verified_evidence` field. Include a supporting source for expected
assertions. Use abstention or clarification actions when evidence does not
establish a safe assertion.

## Numerical accuracy

Keep reported values as decimal strings in the dataset to avoid binary
floating-point ambiguity. Record the metric, currency when applicable, unit,
scale, entity scope, and period. Distinguish currency from percentages, ratios,
counts, and basis points. Do not treat `5` as interchangeable with `5%`, or
crores as interchangeable with millions without an explicit conversion.

Compare normalized values only under a documented conversion and rounding
policy. Preserve source precision; do not mark a rounded value wrong when the
source supports it under the agreed precision. For computed growth or
comparisons, record the inputs, comparison basis, and calculation rationale in
the assertion rationale. The dataset schema records the expected and observed
facts; this phase does not compute numerical scores.

Use `correct`, `incorrect`, `not_supported`, or `not_applicable` for numerical
assessments. Use `not_supported` when the evidence cannot establish whether a
number is right; absence of evidence is not proof that the number is false.

## Temporal and reporting-scope accuracy

Record each financial period explicitly, using the source's fiscal-year
convention (for example `2025-2026`), and distinguish fiscal years from
calendar years, quarters, trailing periods, and point-in-time balances.
For comparisons, identify both periods and the basis (year-over-year,
quarter-over-quarter, cumulative, or other).

Check that the answer does not present historical data as current, use the
wrong comparative period, or confuse consolidated, standalone, and segment
figures. Use the same four assessment statuses as numerical accuracy. A period
or scope omitted from the source should be marked unsupported/unspecified, not
filled in by assumption.

## Expected answer actions

Choose exactly one action after reviewing the question and evidence:

- `answer`: evidence supports a complete answer to the requested material
  facts.
- `partial`: evidence supports at least one material assertion, but the
  question cannot be fully answered; identify the supported assertions and
  explain what is missing.
- `abstain`: evidence is insufficient or materially contradictory, so making
  factual claims would be unsafe. Record the reason.
- `clarify`: the question lacks necessary scope, entity, period, or other
  information that cannot be safely inferred.
- `current_information_unavailable`: the question requires live/current
  information not present in the corpus or available to the system.

Do not use `abstain` merely because retrieval missed a source if human review
can establish an answer from the corpus: this dataset captures the expected
behavior for the available reviewed evidence/run, and the rationale should
state which evidence was considered. Do not label a current-information request
answerable from historical reports.

For `answer` and `partial`, add the expected atomic assertions and verified
supporting evidence references. For non-answer actions, an empty assertion list
is valid; explain why the action is appropriate. Record material conflicting
evidence rather than silently selecting one source.

## Answer-run reviews

Create one `answer_reviews` entry per reviewed generated run. Record the run
identifier and reviewer. For every generated claim:

1. Copy the claim ID and text and list its cited chunk IDs exactly.
2. Decompose the text into atomic factual assertions.
3. Assign each assertion a support label and rationale against the retrieved
   evidence.
4. Assess every cited chunk for that claim separately in
   `citation_assessments`; do not infer support from ID validity.
5. Add numerical and temporal assessments where relevant. Compare the observed
   claim against the verified gold fact, including unit, scale, period, and
   reporting scope.

Keep an entry `draft` until every claim and citation relationship is reviewed.
Use `reviewed` when one reviewer has completed it, and `adjudicated` only after
the required independent review/disagreement resolution process has completed.
Never report unreviewed labels as ground truth. LLM-generated judgments may be
used later to prioritize human review, but cannot replace or override these
labels.
