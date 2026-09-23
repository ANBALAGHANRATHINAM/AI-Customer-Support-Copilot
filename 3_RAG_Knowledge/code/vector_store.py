"""Persistent ChromaDB storage for ZENDS source chunks."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence


COLLECTION_NAME = "zends_communications"


class ZendsVectorStore:
    """Small explicit Chroma wrapper with caller-provided embedding vectors."""

    def __init__(self, database_path: str | Path, collection_name: str = COLLECTION_NAME) -> None:
        import chromadb

        self.database_path = Path(database_path)
        self.database_path.mkdir(parents=True, exist_ok=True)
        self.client = chromadb.PersistentClient(path=str(self.database_path))
        self.collection_name = collection_name
        self.collection = self.client.get_or_create_collection(name=collection_name, metadata={"hnsw:space": "cosine"})

    def upsert(self, *, ids: Sequence[str], documents: Sequence[str], metadatas: Sequence[dict[str, str | int]], embeddings: Sequence[Sequence[float]]) -> None:
        if not (len(ids) == len(documents) == len(metadatas) == len(embeddings)):
            raise ValueError("IDs, documents, metadata, and embeddings must have matching lengths.")
        self.collection.upsert(ids=list(ids), documents=list(documents), metadatas=list(metadatas), embeddings=list(embeddings))

    def count(self) -> int:
        return self.collection.count()

    def reset(self) -> None:
        """Recreate only this named collection for a deterministic source rebuild."""
        self.client.delete_collection(self.collection_name)
        self.collection = self.client.get_or_create_collection(name=self.collection_name, metadata={"hnsw:space": "cosine"})

    def query(self, embedding: Sequence[float], top_k: int, *, where: dict[str, str] | None = None) -> dict:
        if top_k <= 0:
            raise ValueError("top_k must be positive.")
        return self.collection.query(
            query_embeddings=[list(embedding)],
            n_results=min(top_k, self.count()),
            where=where,
            include=["documents", "metadatas", "distances"],
        )
