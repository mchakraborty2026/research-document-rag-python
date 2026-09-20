from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"

if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

import rag_pipeline


class FakeResponse:
    output_text = "Psychosis was associated with later suicidal ideation."


class FakeResponses:
    def create(self, model, input):
        return FakeResponse()


class FakeClient:
    def __init__(self):
        self.responses = FakeResponses()


def test_build_context(monkeypatch):
    monkeypatch.setattr(
        rag_pipeline,
        "OpenAI",
        lambda: FakeClient(),
    )

    pipeline = rag_pipeline.RAGPipeline()

    chunks = [
        {
            "source": "paper1.pdf",
            "chunk_id": 0,
            "text": "Psychosis was associated with suicidal ideation.",
        },
        {
            "source": "paper2.pdf",
            "chunk_id": 3,
            "text": "Depression was also examined.",
        },
    ]

    context = pipeline.build_context(chunks)

    assert "paper1.pdf" in context
    assert "paper2.pdf" in context
    assert "Psychosis was associated" in context
    assert "Depression was also examined" in context


def test_generate_answer(monkeypatch):
    monkeypatch.setattr(
        rag_pipeline,
        "OpenAI",
        lambda: FakeClient(),
    )

    pipeline = rag_pipeline.RAGPipeline()

    chunks = [
        {
            "source": "paper1.pdf",
            "chunk_id": 0,
            "text": "Psychosis was associated with later suicidal ideation.",
        }
    ]

    answer = pipeline.generate_answer(
        "What is the relationship between psychosis and suicidal ideation?",
        chunks,
    )

    assert answer == "Psychosis was associated with later suicidal ideation."
