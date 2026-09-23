"""Faithful, page-aware loading of the authoritative ZENDS PDF."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class SourcePage:
    """Text extracted from one physical source-document page."""

    source: str
    page: int
    text: str


def load_pdf_pages(pdf_path: str | Path) -> list[SourcePage]:
    """Extract non-empty text pages while retaining original page numbers."""
    import pdfplumber

    path = Path(pdf_path)
    if not path.is_file():
        raise FileNotFoundError(f"Source PDF does not exist: {path}")
    pages: list[SourcePage] = []
    with pdfplumber.open(path) as document:
        for page_number, page in enumerate(document.pages, start=1):
            text = page.extract_text() or ""
            if text.strip():
                pages.append(SourcePage(source=path.name, page=page_number, text=text))
    if not pages:
        raise ValueError(f"No readable text was extracted from: {path}")
    return pages
