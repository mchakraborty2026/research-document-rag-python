# Research Document RAG in Python

[![Python Tests](https://github.com/mchakraborty2026/research-document-rag-python/actions/workflows/tests.yml/badge.svg)](https://github.com/mchakraborty2026/research-document-rag-python/actions/workflows/tests.yml)

A Python-based Retrieval-Augmented Generation (RAG) project for retrieving relevant information from research documents using semantic search.

The project demonstrates document ingestion, text chunking, sentence-transformer embeddings, FAISS vector search, automated testing, and optional LLM-based answer generation.

## Project Overview

Traditional keyword search can miss relevant information when a user's question uses different wording from the source document.

This project uses semantic retrieval to identify research passages based on meaning rather than exact keyword matches.

The current demonstration uses a research article examining the relationship between psychotic experiences and later suicidal outcomes.

## RAG Architecture

```text
Research PDFs
      ↓
PDF text extraction
      ↓
Overlapping text chunks
      ↓
Sentence Transformer embeddings
      ↓
FAISS vector index
      ↓
User question
      ↓
Question embedding
      ↓
Semantic similarity search
      ↓
Top relevant research passages
      ↓
Optional LLM generation
```

## Features

- PDF text extraction with PyPDF
- Configurable text chunking with overlap
- Sentence-transformer embeddings
- FAISS vector similarity search
- Retrieval of top relevant research passages
- Source and chunk tracking
- Similarity scores for retrieved passages
- Free retrieval-only demonstration
- Optional OpenAI-based grounded answer generation
- Environment-variable handling for API credentials
- Unit testing with pytest
- Automated testing with GitHub Actions

## Project Structure

```text
research-document-rag-python/
│
├── documents/
│   └── psychosis_suicidality_meta_analysis.pdf
│
├── examples/
│   └── retrieval_example.md
│
├── src/
│   ├── app.py
│   ├── document_loader.py
│   ├── embeddings.py
│   ├── rag_pipeline.py
│   ├── retrieval_demo.py
│   └── vector_store.py
│
├── tests/
│   ├── test_document_loader.py
│   ├── test_embeddings.py
│   ├── test_rag_pipeline.py
│   └── test_vector_store.py
│
├── .github/
│   └── workflows/
│       └── tests.yml
│
├── .env.example
├── .gitignore
├── LICENSE
├── README.md
└── requirements.txt
```

## Retrieval Workflow

### 1. Document Loading

PDF documents are read with PyPDF and converted into text.

### 2. Text Chunking

Large documents are divided into smaller overlapping chunks.

The overlap helps preserve information that may occur across chunk boundaries.

### 3. Embeddings

Each document chunk is converted into a numerical vector using the Sentence Transformer model:

```text
all-MiniLM-L6-v2
```

Embeddings represent the semantic meaning of the text.

### 4. Vector Search

The embeddings are stored in a FAISS index.

When a user enters a question, the question is also converted into an embedding. FAISS then finds document chunks whose vectors are most similar to the query vector.

### 5. Retrieval

The application returns the highest-ranked passages along with:

- Source document
- Chunk number
- Similarity score
- Retrieved text

### 6. Optional Generation

The project also includes an LLM generation layer that can send the retrieved context and user question to the OpenAI API.

The retrieval-only application does not require a paid API and can be run without OpenAI API credits.

## Example Retrieval

Example query:

```text
What is the relationship between psychotic experiences and suicidal ideation?
```

The semantic retrieval system successfully identified passages stating that psychotic experiences were associated with increased odds of subsequent suicidal ideation and other suicidal outcomes.

The highest-ranked retrieved passage produced a similarity score of approximately:

```text
0.774
```

The retrieved passage included evidence that psychotic experiences were associated with significantly increased odds of subsequent suicidal ideation, suicide attempts, and suicide death.

This demonstrates that the system can retrieve relevant research evidence based on semantic similarity rather than exact keyword matching.

## Example Output

A saved example of the semantic retrieval results is available here:

[View Example Retrieval Output](examples/retrieval_example.md)

## Run the Free Retrieval Demo

Install the dependencies:

```bash
python -m pip install -r requirements.txt
```

Run the retrieval application:

```bash
python src/retrieval_demo.py
```

Then enter a research question when prompted.

Example:

```text
What is the relationship between psychotic experiences and suicidal ideation?
```

The program returns the top relevant passages together with their source, chunk number, and similarity score.

**No paid LLM API is required for this retrieval demonstration.**

## Optional LLM Generation

The semantic-retrieval demo works without any paid API.

To enable the optional OpenAI generation layer, first copy the example environment file:

```bash
cp .env.example .env
```

Then add your own API key to `.env`:

```text
OPENAI_API_KEY=your_openai_api_key_here
```

Run the full RAG application with:

```bash
python src/app.py
```

The `.env` file is excluded from Git and should never be committed.

OpenAI API generation requires an API account with available credits. The semantic-retrieval pipeline does not require OpenAI API credits.

## Automated Testing

The project contains unit tests for:

- Text chunking
- Chunk overlap
- Input validation
- Document embeddings
- Query embeddings
- FAISS document storage
- Semantic search
- Empty vector-store behavior
- Context construction
- RAG answer-generation logic

Current test suite:

```text
10 tests passing
```

Run the tests locally with:

```bash
pytest tests -v
```

GitHub Actions automatically runs the test suite on pushes and pull requests to the `main` branch.

The CI status is displayed at the top of this README.

## Technologies

- Python
- PyPDF
- Sentence Transformers
- FAISS
- NumPy
- OpenAI API
- python-dotenv
- pytest
- Git
- GitHub Actions

## Skills Demonstrated

This project demonstrates practical experience with:

- Retrieval-Augmented Generation (RAG)
- Semantic search
- Text embeddings
- Vector similarity search
- FAISS
- Research-document processing
- Document chunking
- Prompt grounding
- LLM API integration
- Modular Python development
- Unit testing
- Continuous integration
- Environment-variable management
- Git and GitHub

## Current Status

The document-processing, embedding, FAISS retrieval, and semantic-search pipeline is fully implemented and tested.

The free retrieval-only demonstration successfully retrieves relevant passages from research literature without requiring a paid LLM API.

The optional OpenAI generation layer is also implemented and can generate grounded answers from retrieved context when API credits are available.

The project currently includes:

- A working research-document retrieval pipeline
- Semantic search using Sentence Transformers and FAISS
- Source and similarity-score reporting
- A saved retrieval example
- 10 passing automated tests
- GitHub Actions continuous integration
- Optional LLM-based answer generation
