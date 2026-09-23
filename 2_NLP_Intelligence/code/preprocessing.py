"""Conservative preprocessing that preserves business-critical terms and numbers."""

from __future__ import annotations

import re
import unicodedata


def clean_text(value: str) -> str:
    """Normalize whitespace and control characters without lowercasing or stripping facts."""
    if not isinstance(value, str):
        raise TypeError("Customer query must be a string.")
    normalized = unicodedata.normalize("NFKC", value)
    normalized = normalized.replace("\u200b", "")
    normalized = "".join(character for character in normalized if character.isprintable() or character.isspace())
    normalized = re.sub(r"\s+", " ", normalized).strip()
    if not normalized:
        raise ValueError("Customer query cannot be empty.")
    return normalized

