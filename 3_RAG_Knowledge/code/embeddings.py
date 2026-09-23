"""Pretrained Sentence Transformer embedding adapter."""

from __future__ import annotations

from typing import Sequence


EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


class SentenceTransformerEmbedder:
    """Lazy MiniLM encoder used identically during build and retrieval."""

    def __init__(self, model_name: str = EMBEDDING_MODEL) -> None:
        self.model_name = model_name
        self._model = None

    def _load(self) -> None:
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.model_name)

    def encode(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []
        self._load()
        vectors = self._model.encode(list(texts), normalize_embeddings=True, show_progress_bar=False)
        return vectors.tolist()
