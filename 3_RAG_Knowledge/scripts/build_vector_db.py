"""Build the persistent Segment 3 ZENDS ChromaDB collection from the PDF."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "3_RAG_Knowledge" / "code"))

from build import build_knowledge_base


if __name__ == "__main__":
    result = build_knowledge_base(ROOT / "docs" / "ZENDS Communications.pdf", ROOT / "3_RAG_Knowledge" / "vector_db")
    print(json.dumps(result, indent=2))
