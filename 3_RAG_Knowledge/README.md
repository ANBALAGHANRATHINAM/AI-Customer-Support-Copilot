# Segment 3 — RAG Knowledge Retrieval

This segment turns the authoritative `docs/ZENDS Communications.pdf` into source-grounded chunks in a persistent ChromaDB collection. It ends at semantic retrieval; it does not contain an LLM or customer-response generator.

- Embedding model: `sentence-transformers/all-MiniLM-L6-v2`, a compact general-purpose semantic model suitable for local retrieval.
- Chunking: 900 characters with 150-character overlap. It prefers complete extracted source sentences, keeping prices and policy statements together while preserving enough neighboring context.
- Retriever: top 4 chunks. Four results generally cover a policy statement plus nearby supporting facts without returning an excessive context bundle.

From the project root:

```powershell
python 3_RAG_Knowledge/scripts/build_vector_db.py
python 3_RAG_Knowledge/scripts/evaluate_retrieval.py
```

The evaluation reports Hit@4: a case is a hit only when all predefined verbatim source phrases expected for that question are present in its top four retrieved chunks.
