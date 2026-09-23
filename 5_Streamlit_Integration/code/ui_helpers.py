"""Pure presentation helpers used by the Streamlit support workspace."""

from __future__ import annotations

import re
from typing import Any


ABSTENTION_TEXT = "The available ZENDS knowledge does not provide enough information"
OUT_OF_SCOPE_RESPONSE = "I'm here to help with ZENDS Communications services—how can I assist you today?"
TOKEN_PATTERN = re.compile(r"[a-z0-9]+")
GENERIC_QUERY_TERMS = {
    "a", "an", "and", "are", "can", "could", "do", "does", "for", "how", "i", "is", "it", "me",
    "my", "of", "on", "please", "tell", "the", "to", "what", "where", "who", "with", "you", "your",
}
STRONG_ZENDS_SCOPE_TERMS = {
    "bill", "billing", "broadband", "charged", "connectivity", "fiber", "gdpr", "internet", "invoice",
    "iot", "mpls", "payment", "refund", "reimbursement", "roaming", "sla", "troubleshoot", "troubleshooting",
    "wifi", "zendfiber",
}
SERVICE_CONTEXT_TERMS = {"connection", "connectivity", "fiber", "internet", "network", "service", "support", "wifi"}
OPERATIONAL_TERMS = {
    "cannot", "connect", "disconnecting", "down", "dropping", "error", "failing", "issue", "offline", "outage",
    "problem", "problems", "slow", "stopped", "trouble", "unavailable", "working",
}
COMPLAINT_TERMS = {"complain", "complaint", "disappointed", "escalate", "escalation", "unhappy", "unresolved"}
POLICY_TERMS = {"contract", "contracts", "discount", "discounts", "privacy", "refund", "sla"}
PRODUCT_GROUP_TERMS = {"broadband", "cloud", "connectivity", "internet", "iot", "mobile", "network"}
PRODUCT_INQUIRY_TERMS = {
    "available", "cost", "offer", "offering", "offerings", "offers", "option", "options", "plan", "plans",
    "price", "pricing", "product", "products", "provide", "services", "solutions",
}


def format_confidence(value: float) -> str:
    """Format a model probability for the support workspace."""
    return f"{float(value) * 100:.1f}%"


def priority_style(priority: str) -> tuple[str, str]:
    """Return semantic CSS class and accessible display text for triage priority."""
    styles = {
        "High": ("priority-high", "HIGH"),
        "Medium": ("priority-medium", "MEDIUM"),
        "Low": ("priority-low", "LOW"),
    }
    return styles.get(priority, ("priority-low", str(priority).upper()))


def is_abstention(response: str) -> bool:
    """Identify the explicit, existing Segment 4 safe-abstention response."""
    normalized = str(response).lower()
    return ABSTENTION_TEXT.lower() in normalized or OUT_OF_SCOPE_RESPONSE.lower() == normalized


def is_unrelated_to_retrieved_knowledge(query: str, result: dict[str, Any]) -> bool:
    """Determine scope from the query itself, independently of generated output."""
    raw_terms = set(TOKEN_PATTERN.findall(query.lower()))
    query_terms = raw_terms - GENERIC_QUERY_TERMS
    if not query_terms:
        return True

    # An explicit company or product reference is in scope even when the
    # knowledge base cannot answer it (for example, asking who the CEO is).
    if "zends" in raw_terms or any(term.startswith("zend") for term in raw_terms):
        return False
    if raw_terms & STRONG_ZENDS_SCOPE_TERMS or raw_terms & POLICY_TERMS:
        return False

    service_context = bool(raw_terms & SERVICE_CONTEXT_TERMS)
    if service_context and (raw_terms & OPERATIONAL_TERMS or raw_terms & COMPLAINT_TERMS):
        return False
    if raw_terms & PRODUCT_GROUP_TERMS and raw_terms & PRODUCT_INQUIRY_TERMS:
        return False

    # Retrieval can corroborate an already substantive knowledge question, but
    # a generic support word in a weak chunk never establishes scope by itself.
    for chunk in result.get("retrieved_context", []):
        context_terms = set(TOKEN_PATTERN.findall(str(chunk.get("text", "")).lower())) - GENERIC_QUERY_TERMS
        overlap = query_terms & context_terms
        if len(overlap) >= 2 and bool(query_terms & (PRODUCT_GROUP_TERMS | POLICY_TERMS | STRONG_ZENDS_SCOPE_TERMS)):
            return False
    return True


def apply_scope_guard(query: str, result: dict[str, Any]) -> dict[str, Any]:
    """Keep engine output intact unless the query itself is unrelated to ZENDS."""
    if is_unrelated_to_retrieved_knowledge(query, result):
        return {**result, "recommended_response": OUT_OF_SCOPE_RESPONSE, "abstention": True, "reason": "out_of_scope"}
    return result
