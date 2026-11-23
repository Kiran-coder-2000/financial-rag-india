# Financial RAG for Indian Corporate Disclosures

## Project Overview

This project evaluates the reliability of Retrieval-Augmented Generation (RAG)
systems when answering questions about Indian corporate financial disclosures.

The project focuses on hallucination, retrieval reliability, answer correctness,
and citation faithfulness.

## Academic Program

MBA in AI & Data Science

## Current Phase

Phase 0 - Development Environment Setup

## LLM Provider Layer

The answer-generation provider layer supports Google Gemini and Groq. Copy
`.env.example` to `.env`, then set `LLM_PROVIDER=gemini` or
`LLM_PROVIDER=groq`. The query planner separately requires Gemini credentials;
when Groq is selected for answer generation, both Gemini and Groq credentials
are needed for chat. Gemini requires `GEMINI_MODEL` unless `LLM_MODEL`
overrides it; Groq defaults to `openai/gpt-oss-20b`. Set `LLM_MODEL` to
override the answer-generation model. Provider tests use mocked SDK calls:

```powershell
.venv\Scripts\python.exe -m pytest tests\test_llm_settings.py tests\test_llm_providers.py
```

## Streamlit Dashboard

Install the dashboard dependencies and start the application from the repository root:

```powershell
.venv\Scripts\python.exe -m pip install -r requirements-ui.txt
.venv\Scripts\python.exe -m streamlit run app.py
```

The dashboard uses the existing local Chroma index and Python retrieval/answer-generation modules. Configure the Gemini query planner and selected answer-generation provider in `.env` before submitting research questions. Conversation history remains in the current browser session.

## Create a Source ZIP

From the repository root, run:

```powershell
.venv\Scripts\python.exe scripts\create_project_zip.py
```

The archive is written to `financial-rag-india.zip` with files rooted under
`financial-rag-india/`, preserving the `ui/` package. Local corpora and persisted
vector indexes are not included; install dependencies and provision those
assets separately when needed. Virtual environments, Git data, caches, secrets,
credential files, and all `.env*` files are also excluded. When using the ZIP,
create `.env` from the variable names above or obtain `.env.example` from the
source repository. Pass `--output` to choose another destination.

## Human Evidence Annotation Audit

Review the proposed evidence references and assess question-level evidence
completeness using the separate workflow in
[`benchmarks/annotation_audit/README.md`](benchmarks/annotation_audit/README.md).
After manually editing its CSVs, refresh the audit summary with:

```powershell
.venv\Scripts\python.exe scripts\build_annotation_audit.py --summary-only
```

The workflow does not modify the benchmark or original gold annotation files,
and it does not assign verified labels automatically.