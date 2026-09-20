from pathlib import Path
import sys

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"

if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

import embeddings


class FakeSentenceTransformer:
    """Fake embedding model used only for unit testing."""

    def __init__(self, model_name):
        self.model_name = model_name

    def encode(
        self,
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True,
    ):
        return np.array(
            [[0.1, 0.2, 0.3] for _ in texts],
            dtype=np.float64,
        )


def test_encode_documents(monkeypatch):
    monkeypatch.setattr(
        embeddings,
        "SentenceTransformer",
        FakeSentenceTransformer,
    )

    model = embeddings.EmbeddingModel()

    result = model.encode_documents(
        ["first document", "second document"]
    )

    assert result.shape == (2, 3)
    assert result.dtype == np.float32


def test_encode_query(monkeypatch):
    monkeypatch.setattr(
        embeddings,
        "SentenceTransformer",
        FakeSentenceTransformer,
    )

    model = embeddings.EmbeddingModel()

    result = model.encode_query(
        "What does the research say?"
    )

    assert result.shape == (1, 3)
    assert result.dtype == np.float32
