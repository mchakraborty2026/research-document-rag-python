from pathlib import Path
import sys

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"

if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from document_loader import chunk_text


def test_chunk_text_creates_multiple_chunks():
    text = "A" * 2500

    chunks = chunk_text(
        text,
        chunk_size=1000,
        overlap=200,
    )

    assert len(chunks) > 1
    assert all(len(chunk) <= 1000 for chunk in chunks)


def test_chunk_text_preserves_overlap():
    text = "0123456789" * 30

    chunks = chunk_text(
        text,
        chunk_size=100,
        overlap=20,
    )

    assert chunks[0][-20:] == chunks[1][:20]


def test_invalid_overlap_raises_error():
    with pytest.raises(ValueError):
        chunk_text(
            "sample text",
            chunk_size=100,
            overlap=100,
        )
