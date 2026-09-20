# Example Semantic Retrieval

## Research Question

**What is the relationship between psychotic experiences and suicidal ideation?**

## Retrieval Result

The system searched the embedded research-document chunks using semantic similarity and returned the most relevant passages.

### Result 1

- **Source:** `psychosis_suicidality_meta_analysis.pdf`
- **Chunk:** 10
- **Similarity score:** 0.774

The retrieved passage reported that psychotic experiences were associated with increased risk of subsequent suicidal ideation, suicide attempts, and suicide death.

### Result 2

- **Source:** `psychosis_suicidality_meta_analysis.pdf`
- **Chunk:** 3
- **Similarity score:** 0.748

This passage contained additional quantitative evidence showing that the association remained after accounting for co-occurring psychopathology.

### Result 3

- **Source:** `psychosis_suicidality_meta_analysis.pdf`
- **Chunk:** 51
- **Similarity score:** 0.737

This result came from the reference section and demonstrates that semantic retrieval can sometimes return relevant but less useful passages.

## What This Demonstrates

This example shows that the retrieval pipeline can:

- Convert a natural-language research question into an embedding
- Compare it with document embeddings
- Rank research passages by semantic similarity
- Return source information and similarity scores
- Retrieve evidence even when the wording of the question differs from the document text

No paid LLM API was required for this retrieval.
