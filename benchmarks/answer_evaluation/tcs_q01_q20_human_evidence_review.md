# TCS Q01-Q20 Human Evidence Verification Packet

**Purpose:** Human-review support for the proposed TCS annotations. This is not an authoritative gold update and does not mark evidence human-verified.

- Draft: [`tcs_q01_q20_gold_draft.json`](./tcs_q01_q20_gold_draft.json), version 1.0.0; 20 cases.
- Authoritative benchmark: [`financial_rag_60_question_benchmark.json`](./financial_rag_60_question_benchmark.json), inspected only.
- Reviewer identity: not recorded. No evidence reference is changed to `human_verified`; all source references remain proposed/unverified.
- Method: all proposed chunk records were loaded from `data/chunks/all_chunks.jsonl`, cross-checked against company/document/year metadata, and relevant source excerpts transcribed below. Original annual-report PDFs in `data/raw/` were checked with local text extraction; TCS pages 6, 8, 55, 76, 78, and 181, Infosys printed page 28, and TCS FY2014-15 printed page 81 were visually checked. The other cited PDF pages were checked using extracted text and page markers. No external sources were used.
- Document identifiers: chunk metadata includes company, document type, financial year, and page, but no separate source `document_id`. For traceability this packet uses the per-document corpus file stem (for example `tataconsultancyservices_2024`) and original PDF filename.
- Page notation: `chunk page` is index metadata; `printed page` is the page number in the annual report. The Infosys chunk page 15 is a two-page PDF spread containing printed pages 28-29; the 20.7% margin is on printed p.28. TCS FY2014-15 chunk page 83 corresponds to printed p.81.

## Source evidence register

Excerpts below are from the cited chunk text. Table whitespace is normalized for legibility; labels and values are retained. Cases list every reference and point to the source entry that contains its exact passage.

### E01
- Chunk ID: `annualreport_tataconsultancyservices_2024_p181_t165`.
- Corpus document ID (file stem): `tataconsultancyservices_2024`; metadata: TataConsultancyServices, Annual Report, FY `2023-2024`.
- Original report: `data/raw/TataConsultancyServices_AnnualReport_2023-2024.pdf`; chunk page `181`; printed page `181`; PDF physical page/spread `181`.
- Exact source excerpt:
  > Table Heading: Consolidated Statement of Profit and Loss
  > Source PDF unit label (visually checked): (₹ crore)
  > | Revenue from operations | 12 | 2,40,893 | 2,25,458 |
  > | Other income | 13 | 4,422 | 3,449 |
  > | TOTAL INCOME | nan | 2,45,315 | 2,28,907 |
  > | Settlement of legal claim | 20 | 958 | - |
  > | PROFIT FOR THE YEAR | nan | 46,099 | 42,303 |
  > | Shareholders of the Company | nan | 45,908 | 42,147 |
  > | Non-controlling interests | nan | 191 | 156 |
- Evidence assessment: Direct for the labeled consolidated statement rows, periods, units, and values.
- Verification state: proposed/unverified; no `verified_by` assigned.

### E02
- Chunk ID: `annualreport_tataconsultancyservices_2024_txt_p006_0002`.
- Corpus document ID (file stem): `tataconsultancyservices_2024`; metadata: TataConsultancyServices, Annual Report, FY `2023-2024`.
- Original report: `data/raw/TataConsultancyServices_AnnualReport_2023-2024.pdf`; chunk page `6`; printed page `6`; PDF physical page/spread `6`.
- Exact source excerpt:
  > In FY 2024, your company has continued to deliver strong performance. I am pleased to share with you that your company has crossed annual revenues of ₹240,893 crore, a growth of 6.8%, over the previous year. With relentless focus on operational excellence, this growth has come at an industry leading operating margin of 24.6%*, an increase from 24.1% last year. The order book for FY 2024 came at an all-time high of US$ 42.7 billion supported by strong client relationships and engagement. Client metrics continue to exhibit healthy progress with strong client additions. Employee retention continues to be at benchmark levels in the industry. TCS has been selected as a Top Employer of Choice in 32 countries. TCS has retained its ranking as the second most valuable global IT services brand, valued at US$19.2 billion, an increase of US$2 billion from last year.
  > 
  > The global environment around the world continues to go through significant shifts. Post the pandemic, which resulted in supply chain shocks, there was an economic slowdown especially in developed markets. While initial signs of stability began to emerge, the military conflicts have further intensified this year and continue to impact the global supply chains.
  > 
  > *Excludes settlement of legal claim
  > 
  > After two years of recessionary fears, persistently high inflation, and unprecedented monetary tightening, the global macro-outlook looks relatively better now with improving growth, disinflation, and monetary easing in sight. Across industries globally, there are multiple mega trends that are shaping priorities of businesses: AI, New Energy, Supply Chain and Talent.
- Evidence assessment: Direct for the quoted reported metric(s), qualifiers, periods, or table rows identified in the case discussions.
- Verification state: proposed/unverified; no `verified_by` assigned.

### E03
- Chunk ID: `annualreport_tataconsultancyservices_2024_txt_p008_0001`.
- Corpus document ID (file stem): `tataconsultancyservices_2024`; metadata: TataConsultancyServices, Annual Report, FY `2023-2024`.
- Original report: `data/raw/TataConsultancyServices_AnnualReport_2023-2024.pdf`; chunk page `8`; printed page `8`; PDF physical page/spread `8`.
- Exact source excerpt:
  > It is my privilege to be writing to you from this desk as we near 20 years to our listing since August 25, 2004. Over the last two decades, your company's Revenue and Net Profits have grown at a compounded annual growth rate of over 18% each.
  > 
  > It has been almost a full year since my transition as CEO, and I am happy to report that it has been a very stable and seamless experience for all our stakeholders. During this period, we refocused on our industry and technology expertise, emphasized and refreshed the core values that define who we are as a company, and doubled down on customer centricity and employee empathy.
  > 
  > We had a strong finish to the fiscal year FY 2024, with revenue for the year at ₹240,893 crore. This is, a growth of 6.8% over the previous year ( 3.4% in constant currency). Demand for our services showed remarkable resilience as macroeconomic uncertainties and geopolitical volatilities continued in major markets through the year.
  > 
  > This growth came with an industry leading operating margin of 24.6% 2 . More importantly we exited the year with a quarterly operating margin of 26% in Q4, demonstrating our commitment to the margin band of 26% to 28%. Our Net Margin was at 19.3% 2 .
  > 
  > 1 GRI 2-22 2 Excludes settlement of legal claim The rise of GenAI has been catching customers' attention, and it promises a leap in productivity and accelerates the creation of new products and services. Although still in the early stages of adoption, the use of GenAI is expected to transform every industry.
  > 
  > The Earnings Per Share was at ₹127.74 2 , a growth of 10.9% over the prior year.
- Evidence assessment: Direct for the quoted reported metric(s), qualifiers, periods, or table rows identified in the case discussions.
- Verification state: proposed/unverified; no `verified_by` assigned.

### E04
- Chunk ID: `annualreport_tataconsultancyservices_2024_txt_p008_0002`.
- Corpus document ID (file stem): `tataconsultancyservices_2024`; metadata: TataConsultancyServices, Annual Report, FY `2023-2024`.
- Original report: `data/raw/TataConsultancyServices_AnnualReport_2023-2024.pdf`; chunk page `8`; printed page `8`; PDF physical page/spread `8`.
- Exact source excerpt:
  > and accelerates the creation of new products and services. Although still in the early stages of adoption, the use of GenAI is expected to transform every industry. The Earnings Per Share was at ₹127.74 2 , a growth of 10.9% over the prior year. Among the Business Segments, Manufacturing grew 10.6%, Life Sciences and Healthcare grew 8.7%, Banking, Financial Services and Insurance grew 5.6%, Consumer Business grew 4.9%, Communication, Media and Technology grew 4.6%, while Others grew 14.5% (YoY in reported currency).
  > 
  > Among geographies, growth was led by emerging markets: Latin America grew 21.1%, India grew 20.3%, Middle East &amp; Africa grew 14.8% while Asia Pacific grew 4.0%. The UK grew 17.7%, Continental Europe grew 6.5%. North America grew 2.3% (YoY in reported currency).
  > 
  > We are seeing strong deal momentum across markets resulting in double-digit growth in our TCV of US$ 42.7 billion, which reflects our deepening partnership with our clients and gives us optimism for the medium to long term growth outlook.
  > 
  > In keeping with our capital allocation policy of returning substantial free cashflow to shareholders, the Board has recommended a final dividend of ₹28 per share, bringing the total dividend for the year to ₹73 per share. The company also successfully completed its fifth buyback program, distributing ₹17,000 crore to shareholders. For the full year, the company's shareholder payout was ₹47,445 crore, which will be our largest payout to date. Our average shareholder payout has been more than 100% during the last 5 years.
- Evidence assessment: Direct for the quoted reported metric(s), qualifiers, periods, or table rows identified in the case discussions.
- Verification state: proposed/unverified; no `verified_by` assigned.

### E05
- Chunk ID: `annualreport_tataconsultancyservices_2024_txt_p075_0002`.
- Corpus document ID (file stem): `tataconsultancyservices_2024`; metadata: TataConsultancyServices, Annual Report, FY `2023-2024`.
- Original report: `data/raw/TataConsultancyServices_AnnualReport_2023-2024.pdf`; chunk page `75`; printed page `75`; PDF physical page/spread `75`.
- Exact source excerpt:
  > On a reported basis, TCS' revenue grew 6.8% in FY 2024, compared to 17.6% in the prior year. The company's performance showed a remarkable resilience against the backdrop of macro uncertainty and geo-political volatility. Demand for TCS' services, particularly around cloud adoption, cost optimization, vendor consolidation and operating model transformation, resulted in record deal wins. The newly won deals conversion into revenue continued as expected, but the revenue inflows kept getting neutralized by reduction in the existing revenue base as some of the earlier projects got completed, optimized, or in some cases, downsized. This led to overall muted revenue growth in FY 2024.
  > 
  > EBIT margins were 24.6%* in FY 2024, compared to 24.1% in the previous year. Margin improvement was led by improved productivity, better utilization and reduction in subcontractor expenses.
  > 
  > Average currency exchange rates during FY 2024 for the three major currencies are given below:
  > 
  > Movements in currency exchange rates through the year resulted in a positive impact of 3.4% on the reported revenue. The constant currency revenue growth for the year, which is the reported revenue growth stripped of the currency impact, was 3.4%.
- Evidence assessment: Direct for the quoted reported metric(s), qualifiers, periods, or table rows identified in the case discussions.
- Verification state: proposed/unverified; no `verified_by` assigned.

### E06
- Chunk ID: `annualreport_tataconsultancyservices_2024_p055_t027`.
- Corpus document ID (file stem): `tataconsultancyservices_2024`; metadata: TataConsultancyServices, Annual Report, FY `2023-2024`.
- Original report: `data/raw/TataConsultancyServices_AnnualReport_2023-2024.pdf`; chunk page `55`; printed page `55`; PDF physical page/spread `55`.
- Exact source excerpt:
  > | Unnamed: 0 | Standalone.Financial Year 2023-24 (FY 2024) | Standalone.Financial Year 2022-23 (FY 2023) | Consolidated.Financial Year 2023-24 (FY 2024) | Consolidated.Financial Year 2022-23 (FY 2023) |
  > | Revenue from operations | 2,02,359 | 1,90,354 | 2,40,893 | 2,25,458 |
  > | Other income | 7,273 | 5,328 | 4,422 | 3,449 |
  > | Total income | 2,09,632 | 1,95,682 | 2,45,315 | 2,28,907 |
- Evidence assessment: Direct for the quoted reported metric(s), qualifiers, periods, or table rows identified in the case discussions.
- Verification state: proposed/unverified; no `verified_by` assigned.

### E07
- Chunk ID: `annualreport_tataconsultancyservices_2024_p076_t041`.
- Corpus document ID (file stem): `tataconsultancyservices_2024`; metadata: TataConsultancyServices, Annual Report, FY `2023-2024`.
- Original report: `data/raw/TataConsultancyServices_AnnualReport_2023-2024.pdf`; chunk page `76`; printed page `76`; PDF physical page/spread `76`.
- Exact source excerpt:
  > | Banking, Financial Services and Insurance | 90,928 (86,127) | 5.6 |
  > | Communication, Media and Technology | 39,391 | 4.6 |
- Evidence assessment: Direct for the quoted reported metric(s), qualifiers, periods, or table rows identified in the case discussions.
- Verification state: proposed/unverified; no `verified_by` assigned.

### E08
- Chunk ID: `annualreport_tataconsultancyservices_2024_p077_t042`.
- Corpus document ID (file stem): `tataconsultancyservices_2024`; metadata: TataConsultancyServices, Annual Report, FY `2023-2024`.
- Original report: `data/raw/TataConsultancyServices_AnnualReport_2023-2024.pdf`; chunk page `77`; printed page `77`; PDF physical page/spread `77`.
- Exact source excerpt:
  > | Consumer Business | 39,357 | 4.9 |
  > | Life Sciences and Healthcare | 26,745 | 8.7 |
- Evidence assessment: Direct for the quoted reported metric(s), qualifiers, periods, or table rows identified in the case discussions.
- Verification state: proposed/unverified; no `verified_by` assigned.

### E09
- Chunk ID: `annualreport_tataconsultancyservices_2024_p078_t043`.
- Corpus document ID (file stem): `tataconsultancyservices_2024`; metadata: TataConsultancyServices, Annual Report, FY `2023-2024`.
- Original report: `data/raw/TataConsultancyServices_AnnualReport_2023-2024.pdf`; chunk page `78`; printed page `78`; PDF physical page/spread `78`.
- Exact source excerpt:
  > | Manufacturing | 23,491 | 10.6 |
  > | Others | 20,981 | 14.5 |
- Evidence assessment: Direct for the quoted reported metric(s), qualifiers, periods, or table rows identified in the case discussions.
- Verification state: proposed/unverified; no `verified_by` assigned.

### E10
- Chunk ID: `annualreport_tataconsultancyservices_2024_p313_t408`.
- Corpus document ID (file stem): `tataconsultancyservices_2024`; metadata: TataConsultancyServices, Annual Report, FY `2023-2024`.
- Original report: `data/raw/TataConsultancyServices_AnnualReport_2023-2024.pdf`; chunk page `313`; printed page `313`; PDF physical page/spread `313`.
- Exact source excerpt:
  > | Order Book | See Total Contract Value |
- Evidence assessment: Contextual glossary evidence. It maps/defines TCV terminology but does not establish comparability of FY2023 closing order book with FY2024 period TCV.
- Verification state: proposed/unverified; no `verified_by` assigned.

### E11
- Chunk ID: `annualreport_tataconsultancyservices_2024_p314_t409`.
- Corpus document ID (file stem): `tataconsultancyservices_2024`; metadata: TataConsultancyServices, Annual Report, FY `2023-2024`.
- Original report: `data/raw/TataConsultancyServices_AnnualReport_2023-2024.pdf`; chunk page `314`; printed page `314`; PDF physical page/spread `314`.
- Exact source excerpt:
  > | nan | See Total Contract Value |
  > | Total Contract Value | An aggregation of the value of all the contracts signed during a period and a useful indicator of demand, and near-term business visibility. |
- Evidence assessment: Contextual glossary evidence. It maps/defines TCV terminology but does not establish comparability of FY2023 closing order book with FY2024 period TCV.
- Verification state: proposed/unverified; no `verified_by` assigned.

### E12
- Chunk ID: `annualreport_tataconsultancyservices_2023_txt_p011_0002`.
- Corpus document ID (file stem): `tataconsultancyservices_2023`; metadata: TataConsultancyServices, Annual Report, FY `2022-2023`.
- Original report: `data/raw/TataConsultancyServices_AnnualReport_2022-2023.pdf`; chunk page `11`; printed page `11`; PDF physical page/spread `11`.
- Exact source excerpt:
  > These business and technology trends drove a strong deal flow throughout FY 2023, peaking in the fourth quarter. We closed the year with an order book of $34.1 Bn. This along with the strong pipeline replenishment gives us a good visibility for the medium term.
- Evidence assessment: Direct for the quoted reported metric(s), qualifiers, periods, or table rows identified in the case discussions.
- Verification state: proposed/unverified; no `verified_by` assigned.

### E13
- Chunk ID: `annualreport_infosys_2024_txt_p015_0001`.
- Corpus document ID (file stem): `infosys_2024`; metadata: Infosys, Annual Report, FY `2023-2024`.
- Original report: `data/raw/Infosys_AnnualReport_2023-2024.pdf`; chunk page `15`; printed page `28`; PDF physical page/spread `15`.
- Exact source excerpt:
  > 85% of free cash flow for fiscal 2020 to fiscal 2024 returned to shareholders in line with the Capital Allocation Policy.
  > 
  > Operating margin
  > 
  > 20.7%
  > 
  > Robust operating margin
  > 
  > Return on equity (3)
  > 
  > 32.1%
  > 
  > Improved by 0.9% over the last fiscal
  > 
  > Free cash (4)
  > 
  > ₹  23,865 cr
  > 
  > FCF conversion at 90.9% of net profit
- Evidence assessment: Direct for the reported Infosys operating-margin figure; not sufficient alone to establish the same calculation/adjustment basis as TCS.
- Verification state: proposed/unverified; no `verified_by` assigned.

### E14
- Chunk ID: `annualreport_tataconsultancyservices_2015_p083_t055_c01`.
- Corpus document ID (file stem): `tataconsultancyservices_2015`; metadata: TataConsultancyServices, Annual Report, FY `2014-2015`.
- Original report: `data/raw/TataConsultancyServices_AnnualReport_2014-2015.pdf`; chunk page `83`; printed page `81`; PDF physical page/spread `83`.
- Exact source excerpt:
  > Table Heading: PERFORMANCE SUMMARY
  > Context: ( ` crores)
  > | Unnamed: 0 | Fiscal 2015* | Fiscal 2015** | Fiscal 2014 | Fiscal 2013 | Fiscal 2012 | Fiscal 2011 | Fiscal 2010 | Fiscal 2009 | Fiscal 2008 | Fiscal 2007 | Fiscal 2006 |
  > | Total revenue | 94648.4 | 94648.4 | 81809.4 | 62989.5 | 48893.8 | 37324.5 | 30028.9 | 27812.9 | 22619.5 | 18685.2 | 13264 |
- Source-page note: The PDF physical page 83 visibly prints page 81. Its table labels scope as Indian GAAP consolidated; PDF value is 30,028.92, while the indexed chunk rounds to 30,028.9.
- Evidence assessment: Direct for the differently labeled FY2010 Total revenue amount and consolidated table scope; insufficient for the specifically requested Revenue from operations label.
- Verification state: proposed/unverified; no `verified_by` assigned.

## Question-by-question review

### TCS_Q01 - What was TCS consolidated revenue from operations for FY2024, in INR crore?
- **Proposed expected action:** `answer`.
- **Proposed answer/action rationale:** The FY2024 consolidated statement directly reports the requested Revenue from operations line. Do not substitute Total income.
- **Expected assertions (1):**
  - `tcs_q01_a1` - TCS consolidated revenue from operations for FY2024 was INR 240,893 crore.
- **Every proposed evidence reference (all unverified):**
  - Assertion `tcs_q01_a1` -> [E01](#e01), chunk `annualreport_tataconsultancyservices_2024_p181_t165`; relation `supports`; `TataConsultancyServices` / FY `2023-2024`; printed p.181 (chunk page 181). Exact quote: see source register E01.
    Draft rationale: FY2023-24 Annual Report, printed p.181, consolidated Statement of Profit and Loss, INR crore: row 'Revenue from operations' reports 2,40,893. The same statement reports Total income of 2,45,315 as a distinct line.
- **Evidence assessment and calculation check:** Direct statement evidence supports the exact consolidated Revenue from operations line. Retain the INR crore unit and FY2024 scope; do not substitute Total income.
- **Verification recommendation:** Review direct labeled statement row, period, and consolidated scope.

### TCS_Q02 - What was TCS consolidated profit after tax for FY2024, in INR crore?
- **Proposed expected action:** `answer`.
- **Proposed answer/action rationale:** Use the consolidated 'PROFIT FOR THE YEAR' total, which includes both shareholders' and non-controlling interests' shares. Do not use shareholders' share alone as consolidated PAT.
- **Expected assertions (2):**
  - `tcs_q02_a1` - TCS consolidated profit for the year, representing consolidated profit after tax, was INR 46,099 crore in FY2024.
  - `tcs_q02_a2` - Of FY2024 consolidated profit for the year, INR 45,908 crore was attributable to shareholders and INR 191 crore to non-controlling interests.
- **Every proposed evidence reference (all unverified):**
  - Assertion `tcs_q02_a1` -> [E01](#e01), chunk `annualreport_tataconsultancyservices_2024_p181_t165`; relation `supports`; `TataConsultancyServices` / FY `2023-2024`; printed p.181 (chunk page 181). Exact quote: see source register E01.
    Draft rationale: FY2023-24 Annual Report, printed p.181, consolidated Statement of Profit and Loss, INR crore: 'PROFIT FOR THE YEAR' is 46,099.
  - Assertion `tcs_q02_a2` -> [E01](#e01), chunk `annualreport_tataconsultancyservices_2024_p181_t165`; relation `supports`; `TataConsultancyServices` / FY `2023-2024`; printed p.181 (chunk page 181). Exact quote: see source register E01.
    Draft rationale: FY2023-24 Annual Report, printed p.181, consolidated Statement of Profit and Loss, INR crore: attribution rows report 45,908 for shareholders and 191 for non-controlling interests; they sum to 46,099.
- **Evidence assessment and calculation check:** The consolidated statement supports profit for the year of INR 46,099 crore. Attribution rows distinguish INR 45,908 crore to shareholders and INR 191 crore to non-controlling interests; these sum to INR 46,099 crore.
- **Verification recommendation:** Review consolidated total and attribution; do not substitute shareholder-attributable profit alone.

### TCS_Q03 - What operating margin did TCS report for FY2024, and what qualification did it provide regarding the settlement of a legal claim?
- **Proposed expected action:** `answer`.
- **Proposed answer/action rationale:** The report states the FY2024 operating margin and qualifies it as excluding the legal-claim settlement. Preserve that qualification; the settlement is separately disclosed in the consolidated statement.
- **Expected assertions (1):**
  - `tcs_q03_a1` - TCS reported an FY2024 operating margin of 24.6%, excluding settlement of a legal claim.
- **Every proposed evidence reference (all unverified):**
  - Assertion `tcs_q03_a1` -> [E02](#e02), chunk `annualreport_tataconsultancyservices_2024_txt_p006_0002`; relation `supports`; `TataConsultancyServices` / FY `2023-2024`; printed p.6 (chunk page 6). Exact quote: see source register E02.
    Draft rationale: FY2023-24 Annual Report, printed p.6, reports an operating margin of 24.6% and states that it excludes settlement of a legal claim.
  - Assertion `tcs_q03_a1` -> [E05](#e05), chunk `annualreport_tataconsultancyservices_2024_txt_p075_0002`; relation `context_only`; `TataConsultancyServices` / FY `2023-2024`; printed p.75 (chunk page 75). Exact quote: see source register E05.
    Draft rationale: Printed p.75 provides the operating-performance discussion and legal-claim qualification in its annual-analysis context.
  - Assertion `tcs_q03_a1` -> [E01](#e01), chunk `annualreport_tataconsultancyservices_2024_p181_t165`; relation `context_only`; `TataConsultancyServices` / FY `2023-2024`; printed p.181 (chunk page 181). Exact quote: see source register E01.
    Draft rationale: Printed p.181, consolidated Statement of Profit and Loss, separately presents the legal-claim settlement line as INR 958 crore; this amount is not a margin value.
- **Evidence assessment and calculation check:** The annual margin and footnote directly support 24.6% and exclusion of settlement of legal claim. The consolidated statement separately lists the settlement as INR 958 crore; this amount is context, not a margin adjustment calculation.
- **Verification recommendation:** Review exact footnote attachment and preserve the source term operating margin.

### TCS_Q04 - What total contract value (TCV) of new deals did TCS report for FY2024?
- **Proposed expected action:** `answer`.
- **Proposed answer/action rationale:** The FY2024 letter reports TCV of US$42.7 billion. The glossary maps Order Book to Total Contract Value and defines TCV as the aggregate value of contracts signed during a period; preserve the report's terminology.
- **Expected assertions (1):**
  - `tcs_q04_a1` - TCS reported FY2024 total contract value (TCV) of US$42.7 billion.
- **Every proposed evidence reference (all unverified):**
  - Assertion `tcs_q04_a1` -> [E04](#e04), chunk `annualreport_tataconsultancyservices_2024_txt_p008_0002`; relation `supports`; `TataConsultancyServices` / FY `2023-2024`; printed p.8 (chunk page 8). Exact quote: see source register E04.
    Draft rationale: FY2023-24 Annual Report, printed p.8, states 'TCV of US$ 42.7 billion' for the year.
  - Assertion `tcs_q04_a1` -> [E10](#e10), chunk `annualreport_tataconsultancyservices_2024_p313_t408`; relation `context_only`; `TataConsultancyServices` / FY `2023-2024`; printed p.313 (chunk page 313). Exact quote: see source register E10.
    Draft rationale: Printed p.313 glossary: 'Order Book — See Total Contract Value.' This supports the terminology mapping, not a separate unrounded value.
  - Assertion `tcs_q04_a1` -> [E11](#e11), chunk `annualreport_tataconsultancyservices_2024_p314_t409`; relation `context_only`; `TataConsultancyServices` / FY `2023-2024`; printed p.314 (chunk page 314). Exact quote: see source register E11.
    Draft rationale: Printed p.314 glossary defines Total Contract Value as the aggregation of contract values signed during a period.
- **Evidence assessment and calculation check:** The CEO passage directly reports FY2024 TCV of US$42.7 billion. The glossary entries contextualize the term, not an additional amount.
- **Verification recommendation:** Review direct TCV text and period; glossary is contextual.

### TCS_Q05 - What was the year-over-year percentage change in TCS consolidated profit after tax from FY2023 to FY2024?
- **Proposed expected action:** `answer`.
- **Proposed answer/action rationale:** The consolidated statement reports profit for the year of INR 42,303 crore in FY2023 and INR 46,099 crore in FY2024. The increase is INR 3,796 crore; (46,099 - 42,303) / 42,303 x 100 = 8.973358...%, rounded to 9.0% at one decimal.
- **Expected assertions (2):**
  - `tcs_q05_a1` - Consolidated profit for the year was INR 42,303 crore in FY2023 and INR 46,099 crore in FY2024.
  - `tcs_q05_a2` - Consolidated profit after tax increased by INR 3,796 crore, approximately 9.0% year over year from FY2023 to FY2024.
- **Every proposed evidence reference (all unverified):**
  - Assertion `tcs_q05_a1` -> [E01](#e01), chunk `annualreport_tataconsultancyservices_2024_p181_t165`; relation `supports`; `TataConsultancyServices` / FY `2023-2024`; printed p.181 (chunk page 181). Exact quote: see source register E01.
    Draft rationale: FY2023-24 Annual Report, printed p.181, consolidated Statement of Profit and Loss, INR crore, presents FY2024 and FY2023 comparative 'PROFIT FOR THE YEAR' values 46,099 and 42,303.
  - Assertion `tcs_q05_a2` -> [E01](#e01), chunk `annualreport_tataconsultancyservices_2024_p181_t165`; relation `supports`; `TataConsultancyServices` / FY `2023-2024`; printed p.181 (chunk page 181). Exact quote: see source register E01.
    Draft rationale: Printed p.181 supports the two consolidated profit-for-the-year inputs. Calculation: 46,099 - 42,303 = 3,796; 3,796 / 42,303 x 100 = 8.973358...%, rounded to 9.0%.
- **Evidence assessment and calculation check:** Inputs: FY2023 consolidated profit for the year INR 42,303 crore; FY2024 INR 46,099 crore. Increase = 46,099 - 42,303 = INR 3,796 crore. Growth = 3,796 / 42,303 x 100 = 8.973358863...%, approximately 9.0% to one decimal.
- **Verification recommendation:** Arithmetic independently recomputed; review input values and rounding.

### TCS_Q06 - How did TCS annual deal TCV change between FY2023 and FY2024, in value and percentage terms?
- **Proposed expected action:** `partial`.
- **Proposed answer/action rationale:** The corpus reports a FY2023 closing order book of US$34.1 billion and FY2024 TCV of US$42.7 billion, and the FY2024 glossary cross-references Order Book to Total Contract Value. However, the available evidence does not establish that the FY2023 closing order-book balance and FY2024 TCV signed during a period are comparable values for calculating a change. Do not claim an absolute or percentage increase.
- **Expected assertions (1):**
  - `tcs_q06_a1` - TCS reported a FY2023 closing order book of US$34.1 billion and FY2024 TCV of US$42.7 billion.
- **Every proposed evidence reference (all unverified):**
  - Assertion `tcs_q06_a1` -> [E12](#e12), chunk `annualreport_tataconsultancyservices_2023_txt_p011_0002`; relation `supports`; `TataConsultancyServices` / FY `2022-2023`; printed p.11 (chunk page 11). Exact quote: see source register E12.
    Draft rationale: FY2022-23 Annual Report, printed p.11, states that TCS 'closed the year with an order book of $34.1 Bn'.
  - Assertion `tcs_q06_a1` -> [E04](#e04), chunk `annualreport_tataconsultancyservices_2024_txt_p008_0002`; relation `supports`; `TataConsultancyServices` / FY `2023-2024`; printed p.8 (chunk page 8). Exact quote: see source register E04.
    Draft rationale: FY2023-24 Annual Report, printed p.8, reports TCV of US$42.7 billion.
  - Assertion `tcs_q06_a1` -> [E10](#e10), chunk `annualreport_tataconsultancyservices_2024_p313_t408`; relation `context_only`; `TataConsultancyServices` / FY `2023-2024`; printed p.313 (chunk page 313). Exact quote: see source register E10.
    Draft rationale: FY2023-24 glossary, printed p.313, cross-references Order Book to Total Contract Value.
  - Assertion `tcs_q06_a1` -> [E11](#e11), chunk `annualreport_tataconsultancyservices_2024_p314_t409`; relation `context_only`; `TataConsultancyServices` / FY `2023-2024`; printed p.314 (chunk page 314). Exact quote: see source register E11.
    Draft rationale: FY2023-24 glossary, printed p.314, defines TCV as the aggregate value of contracts signed during a period. The definition does not establish that the FY2023 closing order-book balance and FY2024 period TCV are comparable for a change calculation.
- **Evidence assessment and calculation check:** The sources directly report FY2023 closing order book US$34.1bn and FY2024 TCV US$42.7bn. The glossary cross-reference does not establish that the closing order-book balance and TCV signed during FY2024 are comparable bases for a change. The requested absolute and percentage change remain unsupported; no subtraction or growth rate is asserted.
- **Verification recommendation:** Retain partial; do not calculate deal growth until measurement/timing comparability is established.

### TCS_Q07 - What was TCS consolidated revenue from operations in FY2023 and FY2024, and what was its absolute year-over-year change in INR crore?
- **Proposed expected action:** `answer`.
- **Proposed answer/action rationale:** The FY2024 consolidated statement presents both comparative annual values on the same Revenue from operations line. The absolute increase is INR 15,435 crore.
- **Expected assertions (2):**
  - `tcs_q07_a1` - Consolidated revenue from operations was INR 225,458 crore in FY2023 and INR 240,893 crore in FY2024.
  - `tcs_q07_a2` - Consolidated revenue from operations increased by INR 15,435 crore year over year from FY2023 to FY2024.
- **Every proposed evidence reference (all unverified):**
  - Assertion `tcs_q07_a1` -> [E01](#e01), chunk `annualreport_tataconsultancyservices_2024_p181_t165`; relation `supports`; `TataConsultancyServices` / FY `2023-2024`; printed p.181 (chunk page 181). Exact quote: see source register E01.
    Draft rationale: FY2023-24 Annual Report, printed p.181, consolidated Statement of Profit and Loss, INR crore, gives comparative Revenue from operations values 2,25,458 (FY2023) and 2,40,893 (FY2024).
  - Assertion `tcs_q07_a2` -> [E01](#e01), chunk `annualreport_tataconsultancyservices_2024_p181_t165`; relation `supports`; `TataConsultancyServices` / FY `2023-2024`; printed p.181 (chunk page 181). Exact quote: see source register E01.
    Draft rationale: Printed p.181 supplies both comparative values on the same line. Calculation: 240,893 - 225,458 = 15,435 crore.
- **Evidence assessment and calculation check:** The consolidated statement compares the same Revenue from operations line: FY2023 INR 225,458 crore and FY2024 INR 240,893 crore. Difference = 240,893 - 225,458 = INR 15,435 crore.
- **Verification recommendation:** Arithmetic directly follows same-line comparatives; review line label and units.

### TCS_Q08 - What was TCS FY2024 revenue growth compared with FY2023, and was it stated in constant-currency or reported-currency terms?
- **Proposed expected action:** `answer`.
- **Proposed answer/action rationale:** The annual report gives both reported-currency and constant-currency revenue growth. Do not present one basis as the other. The reported-currency result is also consistent with the consolidated revenue values after rounding.
- **Expected assertions (2):**
  - `tcs_q08_a1` - TCS FY2024 revenue growth was 6.8% in reported-currency terms and 3.4% in constant-currency terms.
  - `tcs_q08_a2` - Using consolidated revenue from operations, the calculated FY2024 year-over-year growth is approximately 6.8461%, which rounds to 6.8% at one decimal.
- **Every proposed evidence reference (all unverified):**
  - Assertion `tcs_q08_a1` -> [E03](#e03), chunk `annualreport_tataconsultancyservices_2024_txt_p008_0001`; relation `supports`; `TataConsultancyServices` / FY `2023-2024`; printed p.8 (chunk page 8). Exact quote: see source register E03.
    Draft rationale: FY2023-24 Annual Report, printed p.8, states reported-currency and constant-currency revenue growth of 6.8% and 3.4%, respectively.
  - Assertion `tcs_q08_a1` -> [E05](#e05), chunk `annualreport_tataconsultancyservices_2024_txt_p075_0002`; relation `supports`; `TataConsultancyServices` / FY `2023-2024`; printed p.75 (chunk page 75). Exact quote: see source register E05.
    Draft rationale: Printed p.75 provides the FY2024 revenue-growth analysis and distinguishes constant-currency performance from currency impact.
  - Assertion `tcs_q08_a2` -> [E01](#e01), chunk `annualreport_tataconsultancyservices_2024_p181_t165`; relation `context_only`; `TataConsultancyServices` / FY `2023-2024`; printed p.181 (chunk page 181). Exact quote: see source register E01.
    Draft rationale: Printed p.181 gives consolidated revenue from operations of INR 225,458 crore in FY2023 and INR 240,893 crore in FY2024. Calculation: (240,893 - 225,458) / 225,458 x 100 = 6.846064...%; this arithmetic check is not a replacement for the report's stated currency basis.
- **Evidence assessment and calculation check:** The report states FY2024 growth of 6.8% and constant-currency growth of 3.4%; the analysis describes the currency impact and constant-currency basis. Independent statement-line check: (240,893 - 225,458) / 225,458 x 100 = 6.846064...%, consistent with 6.8% to one decimal. This checks reported revenue from operations, not the constant-currency measure.
- **Verification recommendation:** Review reported versus constant-currency distinction and rounding.

### TCS_Q09 - Which category in TCS's FY2024 Industry Vertical table recorded the highest year-over-year revenue growth, including the Others category, and what was its reported-currency growth rate?
- **Proposed expected action:** `answer`.
- **Proposed answer/action rationale:** The complete annual Industry Vertical comparison, including Others, shows Others at the highest reported growth rate. The report identifies the growth rates as year over year in reported currency.
- **Expected assertions (1):**
  - `tcs_q09_a1` - Others recorded the highest FY2024 year-over-year Industry Vertical revenue growth at 14.5% in reported-currency terms.
- **Every proposed evidence reference (all unverified):**
  - Assertion `tcs_q09_a1` -> [E07](#e07), chunk `annualreport_tataconsultancyservices_2024_p076_t041`; relation `supports`; `TataConsultancyServices` / FY `2023-2024`; printed p.76 (chunk page 76). Exact quote: see source register E07.
    Draft rationale: FY2023-24 Annual Report, printed p.76, Industry Vertical table gives Others growth of 14.5% and the other vertical results, including BFSI at 5.6%.
  - Assertion `tcs_q09_a1` -> [E08](#e08), chunk `annualreport_tataconsultancyservices_2024_p077_t042`; relation `supports`; `TataConsultancyServices` / FY `2023-2024`; printed p.77 (chunk page 77). Exact quote: see source register E08.
    Draft rationale: Printed p.77 continues the Industry Vertical table and supports comparison across all categories.
  - Assertion `tcs_q09_a1` -> [E09](#e09), chunk `annualreport_tataconsultancyservices_2024_p078_t043`; relation `supports`; `TataConsultancyServices` / FY `2023-2024`; printed p.78 (chunk page 78). Exact quote: see source register E09.
    Draft rationale: Printed p.78 completes the Industry Vertical table; the full table is needed to establish the maximum including Others.
  - Assertion `tcs_q09_a1` -> [E04](#e04), chunk `annualreport_tataconsultancyservices_2024_txt_p008_0002`; relation `context_only`; `TataConsultancyServices` / FY `2023-2024`; printed p.8 (chunk page 8). Exact quote: see source register E04.
    Draft rationale: Printed p.8 identifies the reported-currency year-over-year basis for the vertical growth rates.
- **Evidence assessment and calculation check:** The cited pages cover the Industry Vertical table: Banking, Financial Services and Insurance 5.6%; Communication, Media and Technology 4.6%; Consumer Business 4.9%; Life Sciences and Healthcare 8.7%; Manufacturing 10.6%; Others 14.5%. The CEO letter identifies these vertical rates as YoY in reported currency. Others is the maximum among those displayed categories.
- **Verification recommendation:** Review table continuation pp.76-78 and reported-currency qualifier; include Others.

### TCS_Q10 - What year-over-year revenue growth or decline did TCS report for its BFSI vertical in FY2024?
- **Proposed expected action:** `answer`.
- **Proposed answer/action rationale:** The annual Industry Vertical table reports BFSI growth of 5.6% year over year in reported currency. Its displayed revenues also independently reproduce that rounded rate: (90,928 - 86,127) / 86,127 x 100 = 5.5743...%, which rounds to 5.6%.
- **Expected assertions (1):**
  - `tcs_q10_a1` - TCS's BFSI vertical reported FY2024 year-over-year revenue growth of 5.6% in reported-currency terms.
- **Every proposed evidence reference (all unverified):**
  - Assertion `tcs_q10_a1` -> [E07](#e07), chunk `annualreport_tataconsultancyservices_2024_p076_t041`; relation `supports`; `TataConsultancyServices` / FY `2023-2024`; printed p.76 (chunk page 76). Exact quote: see source register E07.
    Draft rationale: FY2023-24 Annual Report, printed p.76, Industry Vertical annual table reports BFSI growth of 5.6%; the row gives revenue of INR 86,127 crore for FY2023 and INR 90,928 crore for FY2024.
  - Assertion `tcs_q10_a1` -> [E04](#e04), chunk `annualreport_tataconsultancyservices_2024_txt_p008_0002`; relation `context_only`; `TataConsultancyServices` / FY `2023-2024`; printed p.8 (chunk page 8). Exact quote: see source register E04.
    Draft rationale: Printed p.8 identifies the vertical growth rates as year over year in reported currency.
- **Evidence assessment and calculation check:** BFSI row reports INR 90,928 crore for FY2024 versus INR 86,127 crore for FY2023 and 5.6% growth. Independent check: difference = INR 4,801 crore; 4,801 / 86,127 x 100 = 5.574326...%, which rounds to 5.6%.
- **Verification recommendation:** Review annual Industry Vertical rate, not a quarterly rate.

### TCS_Q11 - Which geographic market had the highest reported revenue growth for TCS in FY2024, and what was the rate?
- **Proposed expected action:** `answer`.
- **Proposed answer/action rationale:** The annual report's geographic growth comparison identifies Latin America as highest. The reported-currency qualifier and year-over-year basis are explicit; do not confuse growth with revenue share.
- **Expected assertions (1):**
  - `tcs_q11_a1` - Latin America had the highest listed FY2024 geographic revenue growth at 21.1% year over year in reported currency.
- **Every proposed evidence reference (all unverified):**
  - Assertion `tcs_q11_a1` -> [E04](#e04), chunk `annualreport_tataconsultancyservices_2024_txt_p008_0002`; relation `supports`; `TataConsultancyServices` / FY `2023-2024`; printed p.8 (chunk page 8). Exact quote: see source register E04.
    Draft rationale: FY2023-24 Annual Report, printed p.8, states Latin America grew 21.1%, India 20.3%, Middle East & Africa 14.8%, UK 17.7%, Continental Europe 6.5%, Asia Pacific 4.0%, and North America 2.3%, YoY in reported currency.
- **Evidence assessment and calculation check:** The CEO letter lists Latin America 21.1%, India 20.3%, Middle East & Africa 14.8%, Asia Pacific 4.0%, UK 17.7%, Continental Europe 6.5%, and North America 2.3%, explicitly YoY in reported currency. Latin America is the highest listed geographic market.
- **Verification recommendation:** Review complete geography list and preserve reported-currency basis.

### TCS_Q12 - Compare TCS FY2024 operating margin with Infosys FY2024 operating margin. Which was higher, and what comparability caveat should be checked?
- **Proposed expected action:** `answer`.
- **Proposed answer/action rationale:** The reports state TCS at 24.6% and Infosys at 20.7%, a 3.9 percentage-point difference. TCS's reported margin excludes settlement of a legal claim; check the two companies' definitions, scope, and adjustments before treating the comparison as strictly like-for-like.
- **Expected assertions (2):**
  - `tcs_q12_a1` - TCS reported FY2024 operating margin of 24.6%, while Infosys reported FY2024 operating margin of 20.7%; the TCS reported figure is higher by 3.9 percentage points.
  - `tcs_q12_a2` - A like-for-like comparison should verify the companies' margin definitions, scope, and adjustments; TCS's FY2024 disclosure specifically excludes legal-claim settlement.
- **Every proposed evidence reference (all unverified):**
  - Assertion `tcs_q12_a1` -> [E02](#e02), chunk `annualreport_tataconsultancyservices_2024_txt_p006_0002`; relation `supports`; `TataConsultancyServices` / FY `2023-2024`; printed p.6 (chunk page 6). Exact quote: see source register E02.
    Draft rationale: TCS FY2023-24 Annual Report, printed p.6, reports operating margin of 24.6% and notes exclusion of legal-claim settlement.
  - Assertion `tcs_q12_a1` -> [E13](#e13), chunk `annualreport_infosys_2024_txt_p015_0001`; relation `supports`; `Infosys` / FY `2023-2024`; printed p.28 (chunk page 15). Exact quote: see source register E13.
    Draft rationale: The draft says printed p.15; correction: p.15 is chunk metadata and PDF spread number. The 20.7% figure is on printed p.28.
  - Assertion `tcs_q12_a2` -> [E02](#e02), chunk `annualreport_tataconsultancyservices_2024_txt_p006_0002`; relation `context_only`; `TataConsultancyServices` / FY `2023-2024`; printed p.6 (chunk page 6). Exact quote: see source register E02.
    Draft rationale: TCS FY2023-24 Annual Report, printed p.6, qualifies its margin by excluding the legal-claim settlement.
  - Assertion `tcs_q12_a2` -> [E13](#e13), chunk `annualreport_infosys_2024_txt_p015_0001`; relation `context_only`; `Infosys` / FY `2023-2024`; printed p.28 (chunk page 15). Exact quote: see source register E13.
    Draft rationale: The draft says printed p.15; correction: p.15 is chunk metadata and PDF spread number. The 20.7% figure is on printed p.28. Definitions and adjustment bases must be checked rather than assumed identical.
- **Evidence assessment and calculation check:** TCS reports 24.6%; Infosys reports 20.7%. Difference = 24.6 - 20.7 = 3.9 percentage points. TCS marks its margin as excluding settlement of a legal claim. The Infosys excerpt is a performance-overview headline and does not establish an equivalent adjustment basis.
- **Verification recommendation:** Reported values support ranking; verify like-for-like definitions. Infosys page locator is printed p.28, not metadata p.15.

### TCS_Q13 - Did TCS annual deal TCV or consolidated profit after tax grow faster from FY2023 to FY2024? Show both calculations.
- **Proposed expected action:** `partial`.
- **Proposed answer/action rationale:** The consolidated PAT figures support a year-over-year growth calculation of (46,099 - 42,303) / 42,303 x 100 = 8.9733...%, approximately 9.0%. The deal disclosures provide FY2023 closing order book of US$34.1 billion and FY2024 TCV of US$42.7 billion, but the glossary cross-reference does not establish that these differently timed disclosures are comparable values for calculating deal-measure growth. The available evidence therefore does not establish which measure grew faster.
- **Expected assertions (2):**
  - `tcs_q13_a1` - TCS reported a FY2023 closing order book of US$34.1 billion and a FY2024 TCV of US$42.7 billion; the available evidence does not establish that these amounts are comparable values for calculating growth.
  - `tcs_q13_a2` - Consolidated profit for the year increased from INR 42,303 crore in FY2023 to INR 46,099 crore in FY2024, approximately 9.0% year over year.
- **Every proposed evidence reference (all unverified):**
  - Assertion `tcs_q13_a1` -> [E12](#e12), chunk `annualreport_tataconsultancyservices_2023_txt_p011_0002`; relation `supports`; `TataConsultancyServices` / FY `2022-2023`; printed p.11 (chunk page 11). Exact quote: see source register E12.
    Draft rationale: FY2022-23 Annual Report, printed p.11, reports closing order book of US$34.1 billion.
  - Assertion `tcs_q13_a1` -> [E04](#e04), chunk `annualreport_tataconsultancyservices_2024_txt_p008_0002`; relation `supports`; `TataConsultancyServices` / FY `2023-2024`; printed p.8 (chunk page 8). Exact quote: see source register E04.
    Draft rationale: FY2023-24 Annual Report, printed p.8, reports TCV of US$42.7 billion for FY2024. The FY2024 glossary defines TCV as contract value signed during a period but does not establish comparability with the FY2023 closing order-book balance.
  - Assertion `tcs_q13_a1` -> [E10](#e10), chunk `annualreport_tataconsultancyservices_2024_p313_t408`; relation `context_only`; `TataConsultancyServices` / FY `2023-2024`; printed p.313 (chunk page 313). Exact quote: see source register E10.
    Draft rationale: FY2023-24 Annual Report, printed p.313, cross-references Order Book to Total Contract Value.
  - Assertion `tcs_q13_a1` -> [E11](#e11), chunk `annualreport_tataconsultancyservices_2024_p314_t409`; relation `context_only`; `TataConsultancyServices` / FY `2023-2024`; printed p.314 (chunk page 314). Exact quote: see source register E11.
    Draft rationale: Printed p.314 defines TCV as the aggregate contract value signed during a period. Retain the distinct source labels and use only the displayed rounded amounts.
  - Assertion `tcs_q13_a2` -> [E01](#e01), chunk `annualreport_tataconsultancyservices_2024_p181_t165`; relation `supports`; `TataConsultancyServices` / FY `2023-2024`; printed p.181 (chunk page 181). Exact quote: see source register E01.
    Draft rationale: FY2023-24 Annual Report, printed p.181, consolidated Statement of Profit and Loss, INR crore, presents FY2023/FY2024 profit-for-the-year comparatives of 42,303/46,099, including non-controlling interests. Increase = 3,796; 3,796 / 42,303 x 100 = 8.9733...%, approximately 9.0%.
- **Evidence assessment and calculation check:** Deal disclosures: FY2023 closing order book US$34.1bn; FY2024 TCV US$42.7bn. The glossary does not prove that these different-timing disclosures support a comparable change calculation, so no deal growth rate is supported. PAT: FY2023 INR 42,303 crore; FY2024 INR 46,099 crore; increase INR 3,796 crore; 3,796 / 42,303 x 100 = 8.973358863...%, approximately 9.0%. The evidence does not establish which measure grew faster.
- **Verification recommendation:** Retain partial and PAT calculation only; do not reintroduce 25.2% deal growth.

### TCS_Q14 - TCS reported an operating margin of 24.6% for FY2024 and 26% for Q4 FY2024. What do these annual and quarter-specific figures establish, and why do they not by themselves establish employee-cost efficiency?
- **Proposed expected action:** `answer`.
- **Proposed answer/action rationale:** The disclosures establish the two margin figures for different reporting periods. A margin percentage alone does not isolate employee costs or establish a causal efficiency claim; do not infer employee-cost efficiency from these figures.
- **Expected assertions (2):**
  - `tcs_q14_a1` - TCS reported an operating margin of 24.6% for FY2024 and 26% for Q4 FY2024.
  - `tcs_q14_a2` - The annual and Q4 operating-margin figures do not by themselves establish employee-cost efficiency because they do not isolate employee costs or show that employee costs caused the margin result.
- **Every proposed evidence reference (all unverified):**
  - Assertion `tcs_q14_a1` -> [E02](#e02), chunk `annualreport_tataconsultancyservices_2024_txt_p006_0002`; relation `supports`; `TataConsultancyServices` / FY `2023-2024`; printed p.6 (chunk page 6). Exact quote: see source register E02.
    Draft rationale: FY2023-24 Annual Report, printed p.6, reports the FY2024 full-year operating margin of 24.6%.
  - Assertion `tcs_q14_a1` -> [E03](#e03), chunk `annualreport_tataconsultancyservices_2024_txt_p008_0001`; relation `supports`; `TataConsultancyServices` / FY `2023-2024`; printed p.8 (chunk page 8). Exact quote: see source register E03.
    Draft rationale: Printed p.8 reports Q4 FY2024 operating margin of 26%.
  - Assertion `tcs_q14_a2` -> [E02](#e02), chunk `annualreport_tataconsultancyservices_2024_txt_p006_0002`; relation `context_only`; `TataConsultancyServices` / FY `2023-2024`; printed p.6 (chunk page 6). Exact quote: see source register E02.
    Draft rationale: Printed p.6 reports a full-year margin ratio; it does not attribute that ratio to employee costs.
  - Assertion `tcs_q14_a2` -> [E03](#e03), chunk `annualreport_tataconsultancyservices_2024_txt_p008_0001`; relation `context_only`; `TataConsultancyServices` / FY `2023-2024`; printed p.8 (chunk page 8). Exact quote: see source register E03.
    Draft rationale: Printed p.8 reports a quarter-specific margin ratio; it does not isolate employee-cost efficiency or establish causation.
- **Evidence assessment and calculation check:** FY2024 full-year margin is 24.6%; Q4 FY2024 margin is 26%. These refer to distinct periods. The cited margin disclosures do not decompose employee cost or establish a causal employee-cost efficiency claim; that limitation is a reasoning boundary, not a separate report statement.
- **Verification recommendation:** Keep full-year/Q4 distinction; frame efficiency as an evidence limitation, not a report claim.

### TCS_Q15 - Does the indexed TCS corpus establish FY2010 consolidated revenue from operations specifically? If it contains only a differently labelled or scoped historical revenue figure, identify that limitation rather than treating it as the requested metric.
- **Proposed expected action:** `partial`.
- **Proposed answer/action rationale:** The indexed corpus includes an FY2010 historical figure labeled Total revenue in a table whose heading identifies Indian GAAP consolidated scope. The indexed chunk reports INR 30,028.9 crore (the original report prints INR 30,028.92 crore), but it does not label the figure Revenue from operations. The exact requested metric is therefore not established; state the adjacent measure only with that limitation.
- **Expected assertions (1):**
  - `tcs_q15_a1` - The FY2014-15 Annual Report's Indian GAAP consolidated performance-summary table reports FY2010 Total revenue of INR 30,028.9 crore in the indexed chunk; the original report prints INR 30,028.92 crore.
- **Every proposed evidence reference (all unverified):**
  - Assertion `tcs_q15_a1` -> [E14](#e14), chunk `annualreport_tataconsultancyservices_2015_p083_t055_c01`; relation `supports`; `TataConsultancyServices` / FY `2014-2015`; printed p.81 (chunk page 83). Exact quote: see source register E14.
    Draft rationale: The chunk is in the TCS FY2014-15 per-document file and all_chunks.jsonl. It has indexed page 83; the source PDF table is on printed p.81. The table heading specifies Indian GAAP consolidated scope, context gives INR crore, and the Fiscal 2010 row is labeled 'Total revenue' with indexed value 30,028.9; the PDF prints 30,028.92.
- **Evidence assessment and calculation check:** The indexed chunk reports an FY2010 Fiscal Total revenue value of INR 30,028.9 crore. The original FY2014-15 report prints INR 30,028.92 crore and labels the historical table Indian GAAP consolidated. The specific requested label Revenue from operations is absent, so the adjacent figure cannot be presented as that metric.
- **Verification recommendation:** Retain partial; preserve label and precision distinction.

### TCS_Q16 - Does the indexed TCS FY2024 corpus provide revenue amounts attributable to each individual customer? If not, state that the corpus does not establish a customer-by-customer breakdown.
- **Proposed expected action:** `abstain`.
- **Proposed answer/action rationale:** The indexed FY2024 TCS Annual Report chunks were searched for individual-customer revenue, customer-wise and revenue-by-customer breakdowns; no customer-specific revenue amounts were located. This is limited to the indexed corpus and does not establish that TCS never disclosed such information elsewhere.
- **Expected assertions (0):**
  - None; this proposed case abstains without asserting facts.
- **Every proposed evidence reference (all unverified):**
  - None listed in the draft; no supporting chunk ID is proposed.
- **Evidence assessment and calculation check:** The proposed abstention has no evidence references or assertions. A corpus scan counted 1,426 TCS FY2023-24 chunks; searches for `customer revenue`, `revenue by customer`, `revenue attributable to customer`, `individual customer`, `customer-wise`, `customer wise`, `largest customer`, `top customer`, and `customer concentration` matched no chunks. This does not prove that no such disclosure exists elsewhere.
- **Verification recommendation:** Corpus-bounded abstention; targeted keyword search is not proof of global absence.

### TCS_Q17 - How should TCS FY2024 full-year operating margin be reconciled with its Q4 FY2024 operating margin? Are they contradictory or different-period measures?
- **Proposed expected action:** `answer`.
- **Proposed answer/action rationale:** The report gives 24.6% for the full fiscal year and 26% for Q4 alone. These are different-period measures, not contradictory values or interchangeable rates.
- **Expected assertions (1):**
  - `tcs_q17_a1` - TCS FY2024 full-year operating margin was 24.6%, while Q4 FY2024 operating margin was 26%; they are different-period measures and are not contradictory.
- **Every proposed evidence reference (all unverified):**
  - Assertion `tcs_q17_a1` -> [E02](#e02), chunk `annualreport_tataconsultancyservices_2024_txt_p006_0002`; relation `supports`; `TataConsultancyServices` / FY `2023-2024`; printed p.6 (chunk page 6). Exact quote: see source register E02.
    Draft rationale: FY2023-24 Annual Report, printed p.6, reports the 24.6% full-year operating margin.
  - Assertion `tcs_q17_a1` -> [E03](#e03), chunk `annualreport_tataconsultancyservices_2024_txt_p008_0001`; relation `supports`; `TataConsultancyServices` / FY `2023-2024`; printed p.8 (chunk page 8). Exact quote: see source register E03.
    Draft rationale: Printed p.8 reports the 26% Q4 operating margin. The respective passages explicitly distinguish the full year from Q4.
- **Evidence assessment and calculation check:** The annual statement reports 24.6% for FY2024 and the CEO letter reports 26% for Q4. The report names different periods; the values are not contradictory or interchangeable.
- **Verification recommendation:** Review full-year versus quarter period distinction.

### TCS_Q18 - If TCS annual report lists revenue from operations and total income separately, which figure answers a question specifically asking for revenue from operations, and why?
- **Proposed expected action:** `answer`.
- **Proposed answer/action rationale:** Use the consolidated statement's specifically labeled Revenue from operations figure. Total income is a distinct line that includes other income; it is not interchangeable with the requested revenue measure.
- **Expected assertions (2):**
  - `tcs_q18_a1` - For FY2024 consolidated results, revenue from operations was INR 240,893 crore; total income was INR 245,315 crore and other income was INR 4,422 crore.
  - `tcs_q18_a2` - A question specifically asking for revenue from operations should be answered with INR 240,893 crore, not total income of INR 245,315 crore.
- **Every proposed evidence reference (all unverified):**
  - Assertion `tcs_q18_a1` -> [E01](#e01), chunk `annualreport_tataconsultancyservices_2024_p181_t165`; relation `supports`; `TataConsultancyServices` / FY `2023-2024`; printed p.181 (chunk page 181). Exact quote: see source register E01.
    Draft rationale: FY2023-24 Annual Report, printed p.181, consolidated Statement of Profit and Loss, INR crore, labels Revenue from operations as 2,40,893 and Total income as 2,45,315.
  - Assertion `tcs_q18_a1` -> [E06](#e06), chunk `annualreport_tataconsultancyservices_2024_p055_t027`; relation `supports`; `TataConsultancyServices` / FY `2023-2024`; printed p.55 (chunk page 55). Exact quote: see source register E06.
    Draft rationale: Printed p.55 financial-performance table separately reports revenue from operations, other income of INR 4,422 crore, and total income; this makes the line-item distinction explicit.
  - Assertion `tcs_q18_a2` -> [E01](#e01), chunk `annualreport_tataconsultancyservices_2024_p181_t165`; relation `supports`; `TataConsultancyServices` / FY `2023-2024`; printed p.181 (chunk page 181). Exact quote: see source register E01.
    Draft rationale: Printed p.181 presents distinct labeled rows for Revenue from operations and Total income in the consolidated statement.
  - Assertion `tcs_q18_a2` -> [E06](#e06), chunk `annualreport_tataconsultancyservices_2024_p055_t027`; relation `context_only`; `TataConsultancyServices` / FY `2023-2024`; printed p.55 (chunk page 55). Exact quote: see source register E06.
    Draft rationale: Printed p.55 separately presents other income of INR 4,422 crore and total income, showing why total income is not the requested revenue line.
- **Evidence assessment and calculation check:** The consolidated P&L labels Revenue from operations as INR 240,893 crore and Total income as INR 245,315 crore; other income is INR 4,422 crore. Arithmetic check: 240,893 + 4,422 = 245,315. The p.55 summary separates standalone and consolidated columns.
- **Verification recommendation:** Review primary line labels and consolidated scope; do not use Total income as Revenue from operations.

### TCS_Q19 - What is TCS share price on NSE right now, and what is its percentage change today?
- **Proposed expected action:** `current_information_unavailable`.
- **Proposed answer/action rationale:** The indexed TCS corpus consists of annual-report material and does not contain a live NSE quote or current-day price-change feed. Current market information is unavailable from this corpus; do not present historical report data as live.
- **Expected assertions (0):**
  - None; this proposed case abstains without asserting facts.
- **Every proposed evidence reference (all unverified):**
  - None listed in the draft; no supporting chunk ID is proposed.
- **Evidence assessment and calculation check:** No assertions or evidence references are proposed. TCS chunk metadata across the combined corpus identifies Annual Report documents only (14,982 TCS chunks); no live NSE quote source is represented.
- **Verification recommendation:** Current-information-unavailable is consistent with corpus; do not infer a live value.

### TCS_Q20 - Why is TCS share price moving today? Identify the latest verified event or market driver.
- **Proposed expected action:** `current_information_unavailable`.
- **Proposed answer/action rationale:** The indexed TCS Annual Reports do not provide current news or a live market feed to verify today's price movement or its driver. Do not infer a current cause from historical annual-report information.
- **Expected assertions (0):**
  - None; this proposed case abstains without asserting facts.
- **Every proposed evidence reference (all unverified):**
  - None listed in the draft; no supporting chunk ID is proposed.
- **Evidence assessment and calculation check:** No assertions or evidence references are proposed. The indexed TCS source types are Annual Reports, with no current news feed or market-driver source represented.
- **Verification recommendation:** Current-information-unavailable is consistent with corpus; do not invent a current cause.

## Summary table

| Question ID | Expected action | Assertion count | Evidence references | Source-page availability | Verification recommendation |
|---|---|---:|---|---|---|
| tcs_q01 | `answer` | 1 | `annualreport_tataconsultancyservices_2024_p181_t165` | TCS FY2023-24 p.181 | Review direct labeled statement row, period, and consolidated scope. |
| tcs_q02 | `answer` | 2 | `annualreport_tataconsultancyservices_2024_p181_t165` | TCS FY2023-24 p.181 | Review consolidated total and attribution; do not substitute shareholder-attributable profit alone. |
| tcs_q03 | `answer` | 1 | `annualreport_tataconsultancyservices_2024_p181_t165`, `annualreport_tataconsultancyservices_2024_txt_p006_0002`, `annualreport_tataconsultancyservices_2024_txt_p075_0002` | TCS FY2023-24 pp.6, 75, 181 | Review exact footnote attachment and preserve the source term operating margin. |
| tcs_q04 | `answer` | 1 | `annualreport_tataconsultancyservices_2024_p313_t408`, `annualreport_tataconsultancyservices_2024_p314_t409`, `annualreport_tataconsultancyservices_2024_txt_p008_0002` | TCS FY2023-24 pp.8, 313-314 | Review direct TCV text and period; glossary is contextual. |
| tcs_q05 | `answer` | 2 | `annualreport_tataconsultancyservices_2024_p181_t165` | TCS FY2023-24 p.181 | Arithmetic independently recomputed; review input values and rounding. |
| tcs_q06 | `partial` | 1 | `annualreport_tataconsultancyservices_2023_txt_p011_0002`, `annualreport_tataconsultancyservices_2024_p313_t408`, `annualreport_tataconsultancyservices_2024_p314_t409`, `annualreport_tataconsultancyservices_2024_txt_p008_0002` | TCS FY2022-23 p.11; FY2023-24 pp.8, 313-314 | Retain partial; do not calculate deal growth until measurement/timing comparability is established. |
| tcs_q07 | `answer` | 2 | `annualreport_tataconsultancyservices_2024_p181_t165` | TCS FY2023-24 p.181 | Arithmetic directly follows same-line comparatives; review line label and units. |
| tcs_q08 | `answer` | 2 | `annualreport_tataconsultancyservices_2024_p181_t165`, `annualreport_tataconsultancyservices_2024_txt_p008_0001`, `annualreport_tataconsultancyservices_2024_txt_p075_0002` | TCS FY2023-24 pp.8, 75, 181 | Review reported versus constant-currency distinction and rounding. |
| tcs_q09 | `answer` | 1 | `annualreport_tataconsultancyservices_2024_p076_t041`, `annualreport_tataconsultancyservices_2024_p077_t042`, `annualreport_tataconsultancyservices_2024_p078_t043`, `annualreport_tataconsultancyservices_2024_txt_p008_0002` | TCS FY2023-24 pp.8, 76-78 | Review table continuation pp.76-78 and reported-currency qualifier; include Others. |
| tcs_q10 | `answer` | 1 | `annualreport_tataconsultancyservices_2024_p076_t041`, `annualreport_tataconsultancyservices_2024_txt_p008_0002` | TCS FY2023-24 pp.8, 76 | Review annual Industry Vertical rate, not a quarterly rate. |
| tcs_q11 | `answer` | 1 | `annualreport_tataconsultancyservices_2024_txt_p008_0002` | TCS FY2023-24 p.8 | Review complete geography list and preserve reported-currency basis. |
| tcs_q12 | `answer` | 2 | `annualreport_infosys_2024_txt_p015_0001`, `annualreport_tataconsultancyservices_2024_txt_p006_0002` | TCS p.6; Infosys FY2023-24 printed p.28 (chunk page 15) | Reported values support ranking; verify like-for-like definitions. Infosys page locator is printed p.28, not metadata p.15. |
| tcs_q13 | `partial` | 2 | `annualreport_tataconsultancyservices_2023_txt_p011_0002`, `annualreport_tataconsultancyservices_2024_p181_t165`, `annualreport_tataconsultancyservices_2024_p313_t408`, `annualreport_tataconsultancyservices_2024_p314_t409`, `annualreport_tataconsultancyservices_2024_txt_p008_0002` | TCS FY2022-23 p.11; FY2023-24 pp.8, 181, 313-314 | Retain partial and PAT calculation only; do not reintroduce 25.2% deal growth. |
| tcs_q14 | `answer` | 2 | `annualreport_tataconsultancyservices_2024_txt_p006_0002`, `annualreport_tataconsultancyservices_2024_txt_p008_0001` | TCS FY2023-24 pp.6, 8 | Keep full-year/Q4 distinction; frame efficiency as an evidence limitation, not a report claim. |
| tcs_q15 | `partial` | 1 | `annualreport_tataconsultancyservices_2015_p083_t055_c01` | TCS FY2014-15 printed p.81 (chunk page 83) | Retain partial; preserve label and precision distinction. |
| tcs_q16 | `abstain` | 0 | None proposed | No cited evidence page; corpus-only search | Corpus-bounded abstention; targeted keyword search is not proof of global absence. |
| tcs_q17 | `answer` | 1 | `annualreport_tataconsultancyservices_2024_txt_p006_0002`, `annualreport_tataconsultancyservices_2024_txt_p008_0001` | TCS FY2023-24 pp.6, 8 | Review full-year versus quarter period distinction. |
| tcs_q18 | `answer` | 2 | `annualreport_tataconsultancyservices_2024_p055_t027`, `annualreport_tataconsultancyservices_2024_p181_t165` | TCS FY2023-24 pp.55, 181 | Review primary line labels and consolidated scope; do not use Total income as Revenue from operations. |
| tcs_q19 | `current_information_unavailable` | 0 | None proposed | No cited evidence page; no live source | Current-information-unavailable is consistent with corpus; do not infer a live value. |
| tcs_q20 | `current_information_unavailable` | 0 | None proposed | No cited evidence page; no current-news source | Current-information-unavailable is consistent with corpus; do not invent a current cause. |

## Validation and outstanding issues

- SHA256: draft `beab27813877def7d0daa74e3dce171d86879d7f4c652740375a93aab8274b25`; authoritative benchmark `1a60f5e235eec5ac24ff8f2bc28c706256a5d6e6ba7de4bce56f520cf79cd0fa`; original seed dataset `0256b2da87e340611aaff46b1cf6fd6d2324c879a89e5ffb326062f44c0d1d76`; schema `a7261caf9ae5876bb23fc19ae33054a7e08d0efcb8b84bd0fcda9196ede6cde3`.
- The 20 TCS draft cases are present exactly once. All 14 unique cited chunk IDs were found in `all_chunks.jsonl`; every per-case reference was checked, and company/document metadata match the cited company and Annual Report type.
- Printed pages are available for every cited chunk. Infosys printed p.28 (not chunk metadata p.15) and TCS FY2014-15 printed p.81 (not chunk metadata p.83) are specifically reconciled. Q16, Q19, and Q20 have no proposed evidence references, so no source page can be cited for them.
- Q06 and Q13: the glossary establishes a terminology cross-reference and defines TCV but does not establish comparability between the FY2023 closing order-book balance and FY2024 TCV signed during a period. No deal growth calculation is supported; Q13 retains only the PAT growth calculation.
- Q12: direct reported-margin values support the ranking, but a strictly like-for-like definition/scope/adjustment comparison remains unverified. TCS excludes settlement of a legal claim; the cited Infosys overview does not provide equivalent calculation detail.
- Q14: the statement that margins alone do not establish employee-cost efficiency is a limitation of inference, not a report quotation; the cited passages do not isolate employee costs or causation.
- Q15: the original report prints INR 30,028.92 crore for FY2010 Total revenue; the indexed chunk gives INR 30,028.9 crore. It is not labeled Revenue from operations.
- Q16 lexical audit: 1,426 TCS FY2023-24 indexed chunks scanned; searches for `customer revenue`, `revenue by customer`, `revenue attributable to customer`, `individual customer`, `customer-wise`, `customer wise`, `largest customer`, `top customer`, and `customer concentration` matched no chunks. This does not prove that no such disclosure exists outside the retrieved evidence.
- Q19-Q20: TCS corpus metadata contains Annual Report material only; no live quote or current-news evidence is proposed. Do not answer with stale historical data or infer a market driver.
- All evidence references remain `proposed_unverified`; this packet does not set `human_verified`, `verified_by`, or authoritative gold fields.

**Approval boundary:** This is a human evidence review packet. It does not promote or verify the proposed gold annotations.
