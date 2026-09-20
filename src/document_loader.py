from pathlib import Path
from typing import List, Dict

from pypdf import PdfReader


def extract_text_from_pdf(pdf_path: Path) -> str:
    """Extract text from all pages of a PDF file."""

    reader = PdfReader(pdf_path)

    pages = []

    for page in reader.pages:
        text = page.extract_text()

        if text:
            pages.append(text)

    return "\n".join(pages)


def chunk_text(
    text: str,
    chunk_size: int = 1000,
    overlap: int = 200,
) -> List[str]:
    """Split text into overlapping chunks."""

    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than 0.")

    if overlap < 0:
        raise ValueError("overlap cannot be negative.")

    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size.")

    chunks = []

    start = 0

    while start < len(text):
        end = start + chunk_size

        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        start += chunk_size - overlap

    return chunks


def load_documents(
    documents_dir: Path,
    chunk_size: int = 1000,
    overlap: int = 200,
) -> List[Dict]:
    """Load PDF documents and convert them into text chunks."""

    documents = []

    for pdf_path in sorted(documents_dir.glob("*.pdf")):
        text = extract_text_from_pdf(pdf_path)

        chunks = chunk_text(
            text,
            chunk_size=chunk_size,
            overlap=overlap,
        )

        for chunk_number, chunk in enumerate(chunks):
            documents.append(
                {
                    "source": pdf_path.name,
                    "chunk_id": chunk_number,
                    "text": chunk,
                }
            )

    return documents
