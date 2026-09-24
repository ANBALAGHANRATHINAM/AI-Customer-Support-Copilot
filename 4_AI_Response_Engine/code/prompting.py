"""Prompt construction for source-grounded customer support assistance."""

from __future__ import annotations

from typing import Any


def format_context(chunks: list[dict[str, Any]]) -> str:
    """Render provenance-bearing retrieval results without hiding their origin."""
    if not chunks:
        return "No ZENDS context was retrieved."
    return "\n\n".join(
        f"[chunk_id={chunk['metadata']['chunk_id']}; page={chunk['metadata']['page']}]\n{chunk['text']}"
        for chunk in chunks
    )


def build_response_prompt(*, customer_query: str, analysis: dict[str, Any], chunks: list[dict[str, Any]]) -> str:
    """Keep the answer request last so the model does not echo instruction lists."""
    return f"""Answer the customer using only the relevant ZENDS evidence. If it does not answer the question, say the information is insufficient. Do not mention internal source details.

Predicted intent: {analysis['intent']}
Predicted sentiment: {analysis['sentiment']}
Predicted priority: {analysis['priority']}

Relevant ZENDS evidence:
{format_context(chunks)}

Customer query: {customer_query}
Answer:"""
