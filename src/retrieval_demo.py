from pathlib import Path

from document_loader import load_documents
from embeddings import EmbeddingModel
from vector_store import VectorStore


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DOCUMENTS_DIR = PROJECT_ROOT / "documents"


def build_vector_store():
    """Load PDFs, create embeddings, and build the FAISS index."""

    print("Loading research documents...")

    documents = load_documents(DOCUMENTS_DIR)

    if not documents:
        raise ValueError(
            "No PDF documents were found in the documents directory."
        )

    print(f"Created {len(documents)} document chunks.")

    embedding_model = EmbeddingModel()

    texts = [document["text"] for document in documents]

    print("Creating document embeddings...")

    embeddings = embedding_model.encode_documents(texts)

    vector_store = VectorStore(
        dimension=embeddings.shape[1]
    )

    vector_store.add(
        embeddings=embeddings,
        documents=documents,
    )

    return embedding_model, vector_store


def main():
    embedding_model, vector_store = build_vector_store()

    print("\nResearch Document Retrieval Demo")
    print("No paid LLM API is required.")
    print("Type 'exit' to stop.\n")

    while True:
        question = input("Question: ").strip()

        if question.lower() == "exit":
            break

        if not question:
            continue

        query_embedding = embedding_model.encode_query(question)

        results = vector_store.search(
            query_embedding,
            top_k=3,
        )

        print("\nTop relevant passages:\n")

        for number, result in enumerate(results, start=1):
            print(f"Result {number}")
            print(f"Source: {result['source']}")
            print(f"Chunk: {result['chunk_id']}")
            print(f"Similarity score: {result['score']:.3f}")
            print()
            print(result["text"][:800])
            print("\n" + "-" * 80 + "\n")


if __name__ == "__main__":
    main()
