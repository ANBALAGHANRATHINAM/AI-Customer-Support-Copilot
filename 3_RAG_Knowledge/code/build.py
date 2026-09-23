"""Build the reproducible ZENDS document chunk collection."""

from __future__ import annotations

from pathlib import Path

from document_loader import load_pdf_pages
from embeddings import EMBEDDING_MODEL, SentenceTransformerEmbedder
from metadata import build_metadata
from text_processing import chunk_text, clean_document_text
from vector_store import COLLECTION_NAME, ZendsVectorStore


def build_knowledge_base(pdf_path: str | Path, database_path: str | Path) -> dict[str, int | str]:
    """Load the authoritative PDF, chunk it, embed it, and upsert deterministic IDs."""
    source_pages = load_pdf_pages(pdf_path)
    ids: list[str] = []
    documents: list[str] = []
    metadatas: list[dict[str, str | int]] = []
    for page in source_pages:
        for index, chunk in enumerate(chunk_text(clean_document_text(page.text)), start=1):
            chunk_id = f"zends-p{page.page:02d}-c{index:02d}"
            ids.append(chunk_id)
            documents.append(chunk)
            metadatas.append(build_metadata(source=page.source, page=page.page, chunk_id=chunk_id, text=chunk))
    if not documents:
        raise ValueError("No chunks were produced from the source document.")
    embedder = SentenceTransformerEmbedder()
    store = ZendsVectorStore(database_path)
    store.reset()
    store.upsert(ids=ids, documents=documents, metadatas=metadatas, embeddings=embedder.encode(documents))
    return {"source_pages": len(source_pages), "chunks": len(documents), "collection": COLLECTION_NAME, "embedding_model": EMBEDDING_MODEL, "stored_chunks": store.count()}
