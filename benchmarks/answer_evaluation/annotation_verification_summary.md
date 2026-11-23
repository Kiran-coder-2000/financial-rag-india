# Infosys and Reliance Annotation Verification Summary

## Status

Prepared separate proposed annotation drafts for the existing Infosys Q21–Q40 and Reliance Industries Q41–Q60 benchmark cases. All 40 IDs and original question wordings were retained. No benchmark or TCS records were edited.

These are drafts pending human review, not authoritative gold annotations. The actual corpus chunks and source reports were inspected, but evidence references remain `proposed_unverified`; no reviewer identity was supplied or invented.

## Annotation counts

| Company | Cases | Answer | Partial | Abstain | Current information unavailable |
|---|---:|---:|---:|---:|---:|
| Infosys | 20 | 15 | 1 | 2 | 2 |
| Reliance Industries | 20 | 14 | 2 | 2 | 2 |
| **Total** | **40** | **29** | **3** | **4** | **4** |

- Cases with inspected evidence references, still unverified: **32**
- Cases with verified evidence: **0**
- Cases with no evidence references because the expected action is abstention/current-information-unavailable: **8**
- Cases requiring human review: **40**
- Unique cited chunk IDs checked against `data/chunks/all_chunks.jsonl`: **28**

## Cases with missing or ambiguous evidence

| Case(s) | Limitation to preserve in review |
|---|---|
| `infosys_q31` | The report groups core and digital offerings as software-related services. No separate comparable FY2024 digital-versus-core revenue split was found. |
| `infosys_q34` | TCV and revenue-recognition policy are available, but the corpus does not establish a guaranteed amount or timing of future TCV conversion. |
| `infosys_q35` | No FY2024 space-research-project spending amount was located. This does not establish whether such spending occurred or was disclosed elsewhere. |
| `infosys_q36` | FY1990 predates the indexed Infosys annual-report coverage, which begins at FY2014–15; the dividend amount cannot be established from this corpus. |
| `infosys_q38` | Management's “debt-free” wording is not defined as including or excluding lease liabilities. Standalone and consolidated lease balances must remain distinct. |
| `ril_q53` | Aggregate capex is reported as covered by cash profits, but funding is not allocated to individual projects. |
| `ril_q54` | Revenue and EBITDA comparisons are available, but they do not establish one undifferentiated or causal ranking of overall segment contribution. |
| `ril_q55` | No exact daily Retail sales amount for a specified FY2024–25 festival date was located. |
| `ril_q56` | No store-by-store quarterly revenue schedule for every Retail outlet was located; aggregate segment figures cannot support it. |
| `ril_q57` | The exact gross Value of Sales & Services figure supports the rounding calculation, but an independent corpus source phrased as “10.7 lakh crore” was not found. Net-of-GST Revenue from Operations is a different measure. |
| `infosys_q39`, `infosys_q40`, `ril_q59`, `ril_q60` | Current market/news information is not established by the indexed annual reports. Do not infer a live price, announcement, commodity price, or current market driver. |

No value was invented for these gaps. An absence in this indexed corpus is not evidence that the company never disclosed the information elsewhere.

## Candidate-action differences for human adjudication

Candidate actions are explicitly unverified and are not gold labels. The proposed actions differ from the current candidates for:

| Case | Candidate action | Proposed action | Evidence-based reason |
|---|---|---|---|
| `infosys_q33` | `partial` | `answer` | Both balances and their different scopes are directly reported, which supports answering the requested same-scope comparison. |
| `infosys_q34` | `partial` | `answer` | Large-deal TCV and revenue-recognition policy support a qualified explanation without asserting guaranteed conversion. |
| `ril_q52` | `partial` | `answer` | The report directly provides comparable FY2024–25 YoY EBITDA growth rates for Retail, Digital Services, and O2C. |

These are proposed adjudications, not a claim that the original candidate annotations have been formally reviewed and rejected.

## Calculation and source checks

- Infosys EPS growth: `(63.39 − 57.63) / 57.63 × 100 = 9.9948…%`, approximately **10.0%**.
- Infosys North America geographic revenue share: `92,411 / 153,670 × 100 = 60.1360…%`, approximately **60.1%**.
- Infosys Financial Services segment share: `42,158 / 153,670 × 100 = 27.4341…%`, approximately **27.4%**.
- Infosys net profit margin: `26,248 / 153,670 × 100 = 17.0808…%`, approximately **17.1%**.
- Reliance Revenue from Operations change: `980,136 − 914,472 = 65,664 crore`; growth is approximately **7.2%**.
- Reliance Profit After Tax change: `80,787 − 78,633 = 2,154 crore`; growth is approximately **2.7%**.
- Reliance Retail net-of-GST share: `291,043 / 980,136 × 100 = 29.6941…%`, approximately **29.7%**.
- Reliance Digital Services/O2C EBITDA shares of consolidated EBITDA are approximately **35.4%** and **30.0%**, respectively; the comparison is measure-specific, not causal.
- Reliance GST reconciliation: `1,071,174 − 91,038 = 980,136 crore`.
- Reliance rounding: `1,071,174 crore = 10.71174 lakh crore`, which rounds to **10.7 lakh crore**; the difference from 10.7 lakh crore is 1,174 crore.

Infosys highlight values on PDF page 15 are on the report's printed pp.28–29 spread. Reliance consolidated-statement evidence on PDF page 100 is on printed pp.196–197; segment evidence on PDF page 130 is printed p.257. Other page references are recorded in the individual evidence rationales.

## Validation and integrity

- Both output JSON files parse and contain exactly 20 unique cases each; together they cover all 40 target IDs exactly once.
- Each draft's question wording, candidate action, and candidate action status match the benchmark.
- All 28 unique cited chunk IDs exist in `all_chunks.jsonl` and match the expected company and Annual Report document type.
- Numeric-reference types, temporal periods, and independently recomputed calculations were checked.
- Benchmark SHA256 remains `1a60f5e235eec5ac24ff8f2bc28c706256a5d6e6ba7de4bce56f520cf79cd0fa`.
- Original 15-question seed SHA256 remains `0256b2da87e340611aaff46b1cf6fd6d2324c879a89e5ffb326062f44c0d1d76`.
- Indexed chunk corpus SHA256 is `b6af8d5df0fee1c59dcca7833e16099652b2a51f5226fab48213c9f46c3927fe`.
- Refined benchmark validator passed; focused refined/seed tests passed (**28 passed**).
- No benchmark, schema, seed, or TCS annotation draft was modified for this task.

**Approval boundary:** These proposed annotations await human review and approval; they are not verified gold standards.
