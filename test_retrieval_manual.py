"""
test_retrieval_manual.py

Manual, interactive inspection tool for the retrieval pipeline --
NOT a pytest test despite the filename, and NOT part of the
production retrieval API (per the task: a developer tool only).

Run directly from the project root:

    python test_retrieval_manual.py

Loads the real persisted chunk corpus and embeddings, takes a
question from the terminal, embeds it via the existing
RetrievalProcessor (which uses embeddings.embed_query() internally
-- the BGE query prefix logic is NOT duplicated here), and prints
the top-5 results from the existing DenseRetriever for manual
eyeballing. No LLM answer generation, no vector DB/ANN index.

Paths used (confirmed from source, not guessed):
    CHUNKS_FILE     = "data/chunks/all_chunks.jsonl"
        -- embeddings.py's own default `chunks_file` in
           run_embedding()'s signature; also build.py's documented
           master-corpus output path.
    EMBEDDINGS_ROOT = "data/embeddings"
        -- embeddings.py's own default `output_root` in
           run_embedding()'s signature.

Requires a completed pipeline run first:
    python -m src.build
    python -m src.embeddings
so both paths above actually exist and correspond to each other --
RetrievalLoader.load() will raise a clear error if they don't.
"""

from src.retrieval import RetrievalLoader, RetrievalProcessor, DenseRetriever

CHUNKS_FILE = "data/chunks/all_chunks.jsonl"
EMBEDDINGS_ROOT = "data/embeddings"

EXIT_COMMANDS = {"q", "quit", "exit"}


def _print_result(result) -> None:
    chunk = result.chunk
    print(f"Rank {result.rank}")
    print(f"Score: {result.score:.4f}")
    print(f"Chunk ID: {result.chunk_id}")
    print(f"Company: {chunk.get('company')}")
    print(f"Document: {chunk.get('document')}")
    print(f"Financial Year: {chunk.get('financial_year')}")
    print(f"Page: {chunk.get('page')}")
    print(f"Chunk Type: {chunk.get('chunk_type')}")
    print()
    print("Text:")
    print(chunk.get("text"))
    print()


def main() -> None:
    print(f"Loading chunks from: {CHUNKS_FILE}")
    print(f"Loading embeddings from: {EMBEDDINGS_ROOT}")

    loader = RetrievalLoader(CHUNKS_FILE, EMBEDDINGS_ROOT)
    chunks, embedding_result = loader.load()

    print(
        f"Loaded {len(chunks)} chunks and {embedding_result.vectors.shape[0]} "
        f"embeddings (dimension {embedding_result.dimension})."
    )

    processor = RetrievalProcessor()
    retriever = DenseRetriever(chunks, embedding_result)

    print()
    print("Enter a question to retrieve the top 5 most similar chunks.")
    print("Type 'q', 'quit', or 'exit' to stop.")
    print()

    while True:
        query = input("Question: ").strip()

        if query.lower() in EXIT_COMMANDS:
            print("Exiting.")
            break

        if not query:
            print("Please enter a question (or 'q' to quit).\n")
            continue

        try:
            query_vector = processor.embed_query(query)
            results = retriever.search(query_vector, top_k=5)
        except Exception as e:
            print(f"Retrieval failed: {e}\n")
            continue

        print()
        print("=" * 60)
        for result in results:
            _print_result(result)
        print("=" * 60)
        print()


if __name__ == "__main__":
    main()