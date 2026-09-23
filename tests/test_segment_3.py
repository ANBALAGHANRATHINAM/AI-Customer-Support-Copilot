"""Behavior-focused tests for Segment 3 ZENDS retrieval."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "3_RAG_Knowledge" / "code"))

from document_loader import load_pdf_pages
from embeddings import SentenceTransformerEmbedder
from metadata import build_metadata
from retriever import ZendsRetriever
from text_processing import chunk_text, clean_document_text
from vector_store import ZendsVectorStore


class DeterministicEmbedder:
    """Tiny test-only embedder to test Chroma persistence without a model download."""

    def encode(self, texts: list[str]) -> list[list[float]]:
        return [[1.0, 0.0] if "refund" in text.lower() else [0.0, 1.0] for text in texts]


def test_document_loader_extracts_all_zends_pages_with_provenance() -> None:
    pages = load_pdf_pages(ROOT / "docs" / "ZENDS Communications.pdf")
    assert [page.page for page in pages] == [1, 2, 3, 4]
    assert all(page.source == "ZENDS Communications.pdf" and page.text for page in pages)
    assert "Prepaid Basic" in pages[0].text
    assert "Full refund within 7 days" in pages[2].text
    assert "minimum 12-month contract" in pages[-1].text.replace("\n", " ")


def test_cleaning_and_chunking_preserve_product_names_prices_and_policy_values() -> None:
    raw = "  ZENDS Prepaid Plus costs $48 in India.\n\nUnlimited plans are capped at 1TB per month. "
    cleaned = clean_document_text(raw)
    chunks = chunk_text(cleaned, chunk_size=60, overlap=10)
    assert "Prepaid Plus" in cleaned and "$48" in cleaned and "1TB" in cleaned
    assert len(chunks) >= 2


def test_metadata_contains_required_provenance_and_only_explicit_tags() -> None:
    metadata = build_metadata(source="ZENDS Communications.pdf", page=4, chunk_id="zends-p04-c01", text="Billing: Monthly billing in advance. Enterprise customers receive consolidated invoices.")
    assert metadata["source"] == "ZENDS Communications.pdf"
    assert metadata["page"] == 4
    assert metadata["chunk_id"] == "zends-p04-c01"
    assert metadata["policy_category"] == "Billing"


def test_embedding_model_creates_normalized_minilm_vectors() -> None:
    vectors = SentenceTransformerEmbedder().encode(["What is the ZENDS refund policy?"])
    assert len(vectors) == 1
    assert len(vectors[0]) == 384
    assert 0.98 < sum(value * value for value in vectors[0]) < 1.02


def test_chromadb_persists_and_reloads_with_metadata(tmp_path: Path) -> None:
    store = ZendsVectorStore(tmp_path)
    store.upsert(ids=["refund-1", "billing-1"], documents=["Full refund within 7 days.", "Monthly billing in advance."], metadatas=[{"source": "ZENDS Communications.pdf", "page": 4, "chunk_id": "refund-1"}, {"source": "ZENDS Communications.pdf", "page": 4, "chunk_id": "billing-1"}], embeddings=[[1.0, 0.0], [0.0, 1.0]])
    reloaded = ZendsVectorStore(tmp_path)
    assert reloaded.count() == 2
    result = reloaded.query([1.0, 0.0], top_k=1)
    assert result["metadatas"][0][0]["chunk_id"] == "refund-1"


def test_retriever_returns_source_grounded_chunks_and_validates_input() -> None:
    retriever = ZendsRetriever(ROOT / "3_RAG_Knowledge" / "vector_db", embedder=SentenceTransformerEmbedder())
    chunks = retriever.retrieve("What is the refund policy for limited usage?", top_k=4)
    assert chunks and all({"text", "metadata", "distance"}.issubset(chunk) for chunk in chunks)
    assert all({"source", "page", "chunk_id"}.issubset(chunk["metadata"]) for chunk in chunks)
    assert any("Full refund within 7 days" in chunk["text"] for chunk in chunks)
    with pytest.raises(ValueError):
        retriever.retrieve("   ")
    with pytest.raises(TypeError):
        retriever.retrieve(None)  # type: ignore[arg-type]


def test_policy_routing_prepends_matching_metadata_chunk() -> None:
    class FakeStore:
        def query(self, _: list[float], __: int, *, where: dict[str, str] | None = None) -> dict:
            if where:
                return {
                    "documents": [["Billing: Monthly billing in advance."]],
                    "metadatas": [[{"source": "ZENDS Communications.pdf", "page": 3, "chunk_id": "billing", "policy_category": "Billing"}]],
                    "distances": [[0.4]],
                }
            return {
                "documents": [["Broad product information.", "Billing: Monthly billing in advance."]],
                "metadatas": [[
                    {"source": "ZENDS Communications.pdf", "page": 1, "chunk_id": "broad"},
                    {"source": "ZENDS Communications.pdf", "page": 3, "chunk_id": "billing", "policy_category": "Billing"},
                ]],
                "distances": [[0.1, 0.4]],
            }

    retriever = ZendsRetriever.__new__(ZendsRetriever)
    retriever.store = FakeStore()
    retriever.top_k = 4
    retriever.embedder = DeterministicEmbedder()
    chunks = retriever.retrieve("How does billing work?", policy_category="Billing")
    assert [chunk["metadata"]["chunk_id"] for chunk in chunks] == ["billing", "broad"]
