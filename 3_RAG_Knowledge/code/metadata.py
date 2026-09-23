"""Conservative metadata inferred only from explicit ZENDS source wording."""

from __future__ import annotations


POLICY_CATEGORIES = ("Billing", "Refund", "Contracts", "SLA", "Data Privacy", "Fair Usage", "Support Tiers", "Discounts")


def build_metadata(*, source: str, page: int, chunk_id: str, text: str) -> dict[str, str | int]:
    """Return required provenance plus only explicitly observable semantic tags."""
    metadata: dict[str, str | int] = {"source": source, "page": page, "chunk_id": chunk_id}
    # A policy tag is included only when the chunk begins at the PDF's explicit
    # policy heading; broad term matching would make metadata misleading.
    matched = next((value for value in POLICY_CATEGORIES if text.lstrip().startswith(f"{value}:")), None)
    if matched:
        metadata["policy_category"] = matched
    return metadata
