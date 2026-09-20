from typing import List, Dict

import faiss
import numpy as np


class VectorStore:
    """Store embeddings and retrieve the most similar document chunks."""

    def __init__(self, dimension: int):
        self.index = faiss.IndexFlatIP(dimension)
        self.documents: List[Dict] = []

    def add(
        self,
        embeddings: np.ndarray,
        documents: List[Dict],
    ) -> None:
        """Add document embeddings and their metadata to the vector store."""

        if len(embeddings) != len(documents):
            raise ValueError(
                "Number of embeddings must match number of documents."
            )

        embeddings = embeddings.astype("float32")

        self.index.add(embeddings)
        self.documents.extend(documents)

    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 5,
    ) -> List[Dict]:
        """Return the most similar document chunks."""

        if self.index.ntotal == 0:
            return []

        top_k = min(top_k, self.index.ntotal)

        scores, indices = self.index.search(
            query_embedding.astype("float32"),
            top_k,
        )

        results = []

        for score, index in zip(scores[0], indices[0]):
            document = self.documents[index].copy()
            document["score"] = float(score)
            results.append(document)

        return results
