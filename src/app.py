from pathlib import Path

from document_loader import load_documents
from embeddings import EmbeddingModel
from vector_store import VectorStore
from rag_pipeline import RAGPipeline


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DOCUMENTS_DIR = PROJECT_ROOT / "documents"


def build_vector_store():
    """Load documents, create embeddings, and build the FAISS index."""

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

    dimension = embeddings.shape[1]

    vector_store = VectorStore(dimension=dimension)

    vector_store.add(
        embeddings=embeddings,
        documents=documents,
    )

    return embedding_model, vector_store


def answer_question(
    question,
    embedding_model,
    vector_store,
    top_k=5,
):
    """Retrieve relevant chunks and generate a grounded answer."""

    query_embedding = embedding_model.encode_query(question)

    retrieved_chunks = vector_store.search(
        query_embedding,
        top_k=top_k,
    )

    rag_pipeline = RAGPipeline()

    answer = rag_pipeline.generate_answer(
        question,
        retrieved_chunks,
    )

    return answer, retrieved_chunks


def main():
    embedding_model, vector_store = build_vector_store()

    print("\nResearch Document RAG")
    print("Type 'exit' to stop.\n")

    while True:
        question = input("Question: ").strip()

        if question.lower() == "exit":
            break

        if not question:
            continue

        answer, retrieved_chunks = answer_question(
            question,
            embedding_model,
            vector_store,
        )

        print("\nAnswer:")
        print(answer)

        print("\nSources:")

        for chunk in retrieved_chunks:
            print(
                f"- {chunk['source']} "
                f"(chunk {chunk['chunk_id']}, "
                f"score={chunk['score']:.3f})"
            )

        print()


if __name__ == "__main__":
    main()
