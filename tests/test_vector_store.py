from pathlib import Path
import sys

import numpy as np
import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"

if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from vector_store import VectorStore


def test_add_and_search():
    store = VectorStore(dimension=3)

    embeddings = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )

    documents = [
        {"source": "paper1.pdf", "chunk_id": 0, "text": "psychosis research"},
        {"source": "paper2.pdf", "chunk_id": 0, "text": "depression research"},
        {"source": "paper3.pdf", "chunk_id": 0, "text": "substance use research"},
    ]

    store.add(embeddings, documents)

    query = np.array(
        [[1.0, 0.0, 0.0]],
        dtype=np.float32,
    )

    results = store.search(query, top_k=2)

    assert len(results) == 2
    assert results[0]["text"] == "psychosis research"
    assert results[0]["score"] >= results[1]["score"]


def test_empty_store_returns_empty_list():
    store = VectorStore(dimension=3)

    query = np.array(
        [[1.0, 0.0, 0.0]],
        dtype=np.float32,
    )

    results = store.search(query)

    assert results == []


def test_add_requires_matching_lengths():
    store = VectorStore(dimension=3)

    embeddings = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ],
        dtype=np.float32,
    )

    documents = [
        {"source": "paper.pdf", "chunk_id": 0, "text": "sample"}
    ]

    with pytest.raises(ValueError):
        store.add(embeddings, documents)
