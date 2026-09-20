from typing import List, Dict

from openai import OpenAI


class RAGPipeline:
    """Generate grounded answers using retrieved document chunks."""

    def __init__(self, model_name: str = "gpt-4.1-mini"):
        self.client = OpenAI()
        self.model_name = model_name

    def build_context(self, retrieved_chunks: List[Dict]) -> str:
        """Combine retrieved chunks into a single context block."""

        context_parts = []

        for chunk in retrieved_chunks:
            source = chunk.get("source", "Unknown source")
            chunk_id = chunk.get("chunk_id", "Unknown chunk")
            text = chunk.get("text", "")

            context_parts.append(
                f"Source: {source} | Chunk: {chunk_id}\n{text}"
            )

        return "\n\n".join(context_parts)

    def generate_answer(
        self,
        question: str,
        retrieved_chunks: List[Dict],
    ) -> str:
        """Generate an answer grounded only in retrieved context."""

        context = self.build_context(retrieved_chunks)

        prompt = f"""
You are answering a question using research documents.

Use only the provided context.
If the context does not contain enough information, say that the available documents do not provide enough information.
Do not invent facts.

Question:
{question}

Context:
{context}
"""

        response = self.client.responses.create(
            model=self.model_name,
            input=prompt,
        )

        return response.output_text
