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
