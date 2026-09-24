"""Reusable semantic retriever that returns source-grounded ZENDS chunks."""

from __future__ import annotations

from pathlib import Path
import re

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

    def retrieve_exact_products(self, product_names: list[str]) -> list[dict]:
        """Fetch product records and their next page-local chunk for split price sentences."""
        records = self.store.source_records()
        by_id = {str(record["metadata"].get("chunk_id")): record for record in records}
        found: list[dict] = []
        seen: set[str] = set()
        for name in product_names:
            for record in records:
                chunk_id = str(record["metadata"].get("chunk_id", ""))
                if name.casefold() not in str(record["text"]).casefold():
                    continue
                for candidate in (record, by_id.get(chunk_id[:-2] + f"{int(chunk_id[-2:]) + 1:02d}") if chunk_id[-2:].isdigit() else None):
                    if candidate is None:
                        continue
                    candidate_id = str(candidate["metadata"].get("chunk_id", ""))
                    if candidate_id not in seen:
                        found.append(candidate)
                        seen.add(candidate_id)
                break
        return found

    def source_records(self) -> list[dict]:
        """Expose the persisted PDF records for source-derived query resolution."""
        return self.store.source_records()

    def retrieve_group_capabilities(self, group: str) -> list[dict]:
        """Find the first service list following the PDF's explicit group heading."""
        records = sorted(self.store.source_records(), key=lambda item: (int(item["metadata"]["page"]), str(item["metadata"]["chunk_id"])))
        heading = re.compile(rf"\b[1-5]\.\s+{re.escape(group)}\b", re.I)
        for index, record in enumerate(records):
            match = heading.search(str(record["text"]))
            if not match:
                continue
            for candidate in records[index:]:
                text = str(candidate["text"])
                services = text.find("Services include")
                if services >= 0 and (candidate is not record or services > match.end()):
                    return [candidate]
        return []

    def retrieve_group_products(self, group: str) -> list[dict]:
        """Read records from a PDF group heading up to its service-list boundary."""
        records = sorted(self.store.source_records(), key=lambda item: (int(item["metadata"]["page"]), str(item["metadata"]["chunk_id"])))
        heading = re.compile(rf"\b[1-5]\.\s+{re.escape(group)}\b", re.I)
        for index, record in enumerate(records):
            match = heading.search(str(record["text"]))
            if not match:
                continue
            selected = []
            for candidate in records[index:]:
                text = str(candidate["text"])
                services = text.find("Services include")
                if services >= 0 and (candidate is not record or services > match.end()):
                    break
                selected.append(candidate)
            return selected
        return []
