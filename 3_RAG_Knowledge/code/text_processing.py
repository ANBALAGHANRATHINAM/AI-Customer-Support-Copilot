"""Conservative document cleaning and page-local chunking."""

from __future__ import annotations

import re


POLICY_HEADINGS = ("Billing", "Refund", "Contracts", "SLA", "Data Privacy", "Fair Usage", "Support Tiers", "Discounts")


def clean_document_text(text: str) -> str:
    """Remove extraction noise without deleting facts, values, or named entities."""
    if not isinstance(text, str):
        raise TypeError("Document text must be a string.")
    raw_lines = text.replace("\u00a0", " ").splitlines()
    cleaned_lines = []

    for line in raw_lines:
        line = re.sub(r"\s+", " ", line).strip()
        if not line:
            continue
        if re.fullmatch(r"\d+", line):
            continue
        cleaned_lines.append(line)

    return "\n".join(cleaned_lines)


def chunk_text(text: str, *, chunk_size: int = 900, overlap: int = 150) -> list[str]:
    """Create readable chunks, preferring complete source lines and sentences."""
    if chunk_size <= 0 or overlap < 0 or overlap >= chunk_size:
        raise ValueError("chunk_size must be positive and overlap must be smaller than chunk_size.")
    clean = clean_document_text(text)
    if not clean:
        return []
    # Policy headings are authoritative section boundaries. Keeping each policy
    # separate prevents a broad policy page from diluting a targeted retrieval.
    clean = re.sub(r"(?<!^)(?=(?:" + "|".join(POLICY_HEADINGS) + r"):)", "\n", clean)
    # The source uses "Services include" to introduce a distinct list of
    # product capabilities. Preserve that complete sentence as its own
    # retrievable unit instead of attaching it to a preceding pricing table.
    clean = re.sub(r"(?<!^)(?=Services include )", "\n", clean)
    units = re.split(r"(?<=[.!?])\s+|\n+", clean)
    units = [unit.strip() for unit in units if unit.strip()]
    chunks: list[str] = []
    current = ""
    for unit in units:
        if any(unit.startswith(f"{heading}:") for heading in POLICY_HEADINGS):
            if current:
                chunks.append(current)
            # Start a new source-defined policy section, then retain its
            # following sentences until the next explicit policy heading.
            # This keeps facts such as a rule and its qualifying condition
            # together without inventing a sentence boundary.
            current = unit
            continue
        if unit.startswith("Services include"):
            if current:
                chunks.append(current)
            current = unit
            continue
        candidate = f"{current} {unit}".strip()
        if current and len(candidate) > chunk_size:
            chunks.append(current)
            suffix = current[-overlap:].lstrip() if overlap else ""
            current = f"{suffix} {unit}".strip()
        elif not current and len(unit) > chunk_size:
            # A long extracted line is split only when necessary, with overlap.
            start = 0
            while start < len(unit):
                end = min(start + chunk_size, len(unit))
                chunks.append(unit[start:end].strip())
                if end == len(unit):
                    current = ""
                    break
                start = end - overlap
        else:
            current = candidate
    if current:
        chunks.append(current)
    return chunks
