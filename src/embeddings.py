from typing import List

import numpy as np
from sentence_transformers import SentenceTransformer


DEFAULT_MODEL_NAME = "all-MiniLM-L6-v2"


class EmbeddingModel:
    """Create vector embeddings for document chunks and user queries."""

    def __init__(self, model_name: str = DEFAULT_MODEL_NAME):
        self.model = SentenceTransformer(model_name)

    def encode_documents(self, texts: List[str]) -> np.ndarray:
        """Convert document text chunks into normalized embeddings."""

        embeddings = self.model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )

        return embeddings.astype("float32")

    def encode_query(self, query: str) -> np.ndarray:
        """Convert a user query into a normalized embedding."""

        embedding = self.model.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=True,
        )

        return embedding.astype("float32")
