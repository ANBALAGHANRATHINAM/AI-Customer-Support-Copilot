"""Reusable semantic retriever that returns source-grounded ZENDS chunks."""

from __future__ import annotations

from pathlib import Path

from embeddings import SentenceTransformerEmbedder
from vector_store import ZendsVectorStore


DEFAULT_TOP_K = 4


class ZendsRetriever:
    """Retrieve relevant source chunks only; this component never generates an answer."""

    def __init__(self, database_path: str | Path, *, top_k: int = DEFAULT_TOP_K, embedder: SentenceTransformerEmbedder | None = None) -> None:
        if top_k <= 0:
            raise ValueError("top_k must be positive.")
        self.store = ZendsVectorStore(database_path)
        self.top_k = top_k
        self.embedder = embedder or SentenceTransformerEmbedder()

    def retrieve(
        self,
        query: str,
        *,
        top_k: int | None = None,
        policy_category: str | None = None,
        support_intent: bool = False,
    ) -> list[dict]:
        if not isinstance(query, str):
            raise TypeError("Retriever query must be a string.")
        cleaned = " ".join(query.split())
        if not cleaned:
            raise ValueError("Retriever query cannot be empty.")
        embedding = self.embedder.encode([cleaned])[0]
        # Support questions use a slightly wider candidate pool only for
        # evidence discovery. Segment 4 still selects at most two focused
        # chunks before answer generation.
        candidate_count = top_k or (8 if support_intent else self.top_k)
        result = self.store.query(embedding, candidate_count)
        chunks = [
            {"text": document, "metadata": metadata, "distance": float(distance)}
            for document, metadata, distance in zip(result["documents"][0], result["metadatas"][0], result["distances"][0])
        ]
        # An explicit policy heading is stronger than broad semantic similarity.
        # Retain the regular retrieval results for traceability, but guarantee
        # that a matching policy section reaches evidence selection when present.
        if policy_category:
            policy_result = self.store.query(embedding, 1, where={"policy_category": policy_category})
            policy_chunks = [
                {"text": document, "metadata": metadata, "distance": float(distance)}
                for document, metadata, distance in zip(
                    policy_result["documents"][0], policy_result["metadatas"][0], policy_result["distances"][0]
                )
            ]
            policy_ids = {str(chunk["metadata"].get("chunk_id")) for chunk in policy_chunks}
            chunks = policy_chunks + [chunk for chunk in chunks if str(chunk["metadata"].get("chunk_id")) not in policy_ids]
        return chunks
