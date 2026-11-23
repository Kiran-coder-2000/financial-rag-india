# Financial RAG Evidence Annotation Audit

This is a separate manual audit workflow for the 134 proposed evidence
references in the 60-question benchmark. The files here do not replace or
modify the benchmark or gold annotation JSON files. No reviewer decisions are
assigned when templates are generated.

## Files

- `annotation_review.csv`: one row per original evidence-reference occurrence,
  including the source passage, chunk provenance, current proposed relation,
  and blank reviewer fields.
- `question_completeness.csv`: one row per benchmark question, including
  questions that have no proposed evidence references. Use this sheet to decide
  whether the known evidence set is complete or whether completeness remains
  uncertain.
- `annotation_audit_summary.md`: counts recalculated from the two CSV files.

The generated evidence reference ID identifies the question, assertion, and
evidence position in its original annotation. It is an audit identifier, not
an ID from the source annotation. Repeated chunk IDs are intentionally retained
when a chunk is cited for multiple assertions or references.

## Review process

1. Review each row in `annotation_review.csv` against the displayed passage
   and its source context. Follow
   `../answer_evaluation/ANNOTATION_GUIDELINES.md` for checks such as company,
   reporting period, table heading, scope, units, and adjacent context. A
   candidate ID or rationale is not itself proof of support.
2. Set `reviewer_decision` to exactly one of `supports`, `context_only`,
   `not_relevant`, or `needs_review`. Leave it blank until an actual reviewer
   has assessed the reference. Record concise reasoning in `reviewer_comments`.
   `needs_review` is a reviewer decision, not an automatic fallback.
3. If relevant chunks are missing from the proposed references, record their
   exact IDs in `additional_supporting_evidence_discovered`, separated by
   semicolons. Do not copy passages or make up chunk IDs; use
   `data/chunks/all_chunks.jsonl` to inspect and confirm each ID and provenance.
   The summary checks that additional IDs exist in that corpus.
4. In `question_completeness.csv`, set `completeness_status` to `complete`
   only when the question's relevant evidence set has been checked and no
   additional support is missing. Use `uncertain` if completeness cannot be
   established, and leave it `pending` until reviewed. Record newly found
   chunk IDs here when the question has no reference row or to capture
   question-level completeness. IDs listed on either sheet are combined in
   the summary.
5. After editing and saving the CSV files, recalculate the summary:

   ```powershell
   .venv\Scripts\python.exe scripts\build_annotation_audit.py --summary-only
   ```

The summary counts any nonblank valid decision as reviewed, including
`needs_review`. It lists every question not marked `complete` as uncertain or
pending, and separately reports pending and explicitly uncertain statuses;
neither is treated as complete.

## Generating fresh templates

From the repository root, generate the initial review sheets and summary with:

```powershell
.venv\Scripts\python.exe scripts\build_annotation_audit.py
```

The script refuses to replace existing review CSVs. Use `--force` only when
you intentionally want to discard all edits and recreate the templates. Do not
use `--force` to update the summary; use `--summary-only`.

The workflow reads the benchmark, proposed annotation drafts, and chunk corpus
only. It does not update original annotation files, benchmark questions,
retrieval code, or evaluation calculations. After humans complete the audit,
any changes to authoritative gold labels and metric recalculation are a
separate, explicit step.
