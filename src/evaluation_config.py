"""Configuration for the standalone retrieval benchmark evaluation."""

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
BENCHMARK_PATH = PROJECT_ROOT / "benchmarks" / "answer_evaluation" / "financial_rag_60_question_benchmark.json"
ANNOTATION_PATHS = {
    "TataConsultancyServices": (
        PROJECT_ROOT / "benchmarks" / "answer_evaluation" / "tcs_q01_q20_gold_annotations.json"
    ),
    "Infosys": PROJECT_ROOT / "benchmarks" / "answer_evaluation" / "infosys_gold_annotations.json",
    "RelianceIndustries": PROJECT_ROOT / "benchmarks" / "answer_evaluation" / "reliance_gold_annotations.json",
}
CHUNKS_PATH = PROJECT_ROOT / "data" / "chunks" / "all_chunks.jsonl"
CHROMA_PATH = PROJECT_ROOT / "data" / "chroma_db"
CHROMA_COLLECTION = "financial_reports_filtered"
KEYWORD_CHROMA_PILOT_PATH = PROJECT_ROOT / "data" / "keyword_chroma_db_pilot"
KEYWORD_CHROMA_PILOT_COLLECTION = "financial_report_keywords_pilot"
KEYWORD_LEXICAL_PILOT_PATH = (
    PROJECT_ROOT / "data" / "keyword_index" / "financial_keyword_index_pilot.sqlite"
)
RESULTS_PATH = PROJECT_ROOT / "evaluation_results"

K_VALUES = (1, 3, 5, 10)
TOP_K = max(K_VALUES)
STRATEGIES = (
    "baseline",
    "chunk_semantic",
    "keyword_semantic",
    "keyword_lexical",
    "hybrid_chunk_keyword",
    "hybrid_chunk_lexical",
    "hybrid_all",
)

STRATEGY_DESCRIPTIONS = {
    "baseline": "Configured default strategy; direct chunk-semantic retrieval.",
    "chunk_semantic": "Direct chunk-semantic retrieval; same underlying path as baseline.",
    "keyword_semantic": "Keyword-vector retrieval using the available 150-chunk pilot index.",
    "keyword_lexical": "FTS5 retrieval using the available 150-chunk pilot index.",
    "hybrid_chunk_keyword": "RRF over chunk-semantic and keyword-vector pilot results.",
    "hybrid_chunk_lexical": "RRF over chunk-semantic and keyword-lexical pilot results.",
    "hybrid_all": "RRF over chunk-semantic and both keyword pilot paths.",
}
