# Question-Level Retrieval Audit Workbook

The workbook is generated as
`evaluation_results/question_level_retrieval_audit.xlsx`. The workbook's
`Instructions` sheet contains an abbreviated version of this guide.

## Generate the workbook

From the repository root, install the scoped Excel writer dependency if it is
not already available, then run the builder:

```powershell
.venv\Scripts\python.exe -m pip install -r requirements-audit.txt
.venv\Scripts\python.exe scripts\build_question_level_retrieval_audit.py
```

The builder refuses to replace an existing workbook or validation report.
Pass `--force` only when intentionally regenerating both outputs:

```powershell
.venv\Scripts\python.exe scripts\build_question_level_retrieval_audit.py --force
```

## Worksheets

- **Instructions** explains data scope, matching, metric limitations, and
  manual review.
- **Question Performance Summary** has one row for each of the 60 questions
  and seven saved retrieval strategies. Metrics are copied from
  `per_query_metrics.csv`. Blank metric cells are unavailable, not zero.
  `Common evaluation subset` identifies the saved scope for strategy
  comparisons.
- **Retrieved Chunks** has one row for each actual returned chunk, in its
  original rank order. Searches that failed or returned no chunks are clearly
  represented with a status row. Use `Manual relevance decision` and
  `Reviewer comments` for human review; these fields start blank.
- **Expected Gold Chunks** preserves every evidence-reference occurrence,
  including repeated chunk IDs. `supports` and `context_only` remain distinct,
  and annotation verification status is copied from the original annotation
  JSON. Full chunk text is looked up only by the exact chunk ID.
- **Side-by-Side Comparison** repeats one chunk passage per row so retrieved
  and expected evidence can be compared without placing many passages in one
  cell. Only exact chunk-ID equality is a match. Expected support rows show
  whether and at what rank that ID was retrieved; context-only rows are
  separate and are not supporting hits. Question financial period and chunk
  company/year provenance are shown separately to make metadata mismatches
  inspectable.
- **Error Analysis** is a manual worksheet. Select a failure classification
  only after inspection; record observations, possible causes, and proposed
  improvements yourself. The workbook does not infer a root cause.
- **Investigation Priorities** lists every question, the counts of exact zero
  metrics among scored strategy rows, and the observed ranges across saved
  strategies. Metric ranges are descriptive and do not designate a winner.
  Questions without supporting gold IDs are flagged separately because their
  metrics are not evaluable.

All worksheets have frozen header rows, filters, wrapped text, and distinct
retrieved/expected evidence styling. Long passages are held in separate
chunk-level rows; select a passage cell to inspect the full text in Excel's
formula bar. Original retrieval ordering and chunk IDs are not changed.

## Interpretation and limitations

The source benchmark contains seven actual category values; the workbook
preserves these as written. Eleven questions have no proposed supporting
chunk IDs. Their metric values remain unavailable rather than being set to
zero. The saved per-query data also contains failed retrieval records; these
are shown as failed, not as empty successful retrievals.

All 134 annotation references remain `proposed_unverified`. The workbook is
not a verification of the benchmark's ground truth and does not change gold
annotations, retrieval outputs, metric calculations, or algorithms. The
annotation review workflow is separate. Manual observations in this workbook
do not update the source data.

See
[`question_level_retrieval_audit_validation.md`](./question_level_retrieval_audit_validation.md)
for source hashes, mapping counts, and integrity checks.
