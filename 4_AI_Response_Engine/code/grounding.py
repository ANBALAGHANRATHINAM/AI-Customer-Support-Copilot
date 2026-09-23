"""Conservative evidence selection and response composition for Segment 4."""

from __future__ import annotations

import re
from typing import Any


TOKEN_PATTERN = re.compile(r"[a-z0-9]+")
STOPWORDS = {
    "a", "an", "and", "are", "can", "do", "does", "for", "from", "how", "i", "in", "is", "it", "me", "my",
    "of", "on", "please", "the", "to", "what", "when", "where", "with", "you", "your", "zends", "customer",
}
GENERIC_TERMS = {
    "customer", "enterprise", "individual", "india", "price", "prices", "pricing", "service", "services",
    "support", "usa", "user", "users",
}
POLICY_INTENT = {"Billing": "Billing", "Refund": "Refund"}
POLICY_QUERY_TERMS = {
    "contract": "Contracts",
    "contracts": "Contracts",
    "discount": "Discounts",
    "discounts": "Discounts",
    "fair usage": "Fair Usage",
    "privacy": "Data Privacy",
    "sla": "SLA",
    "support tiers": "Support Tiers",
}
SUPPORT_INTENTS = {"Technical", "Complaint"}
TECHNICAL_QUERY_TERMS = {
    "blocking", "cannot", "connect", "connection", "down", "error", "issue", "network", "outage", "problem",
    "problems", "setup", "trouble", "troubleshooting", "wifi", "internet", "dropping", "disconnecting", "fail",
    "failing", "working", "slow", "fiber", "installation",
}
SERVICE_QUERY_TERMS = {
    "broadband", "connection", "connectivity", "fiber", "internet", "network", "service", "wifi",
}
COMPLAINT_QUERY_TERMS = {"complain", "complaint", "disappointed", "escalate", "escalation", "unhappy", "unresolved"}
SUPPORT_EVIDENCE_TERMS = (
    "technical support", "troubleshooting", "network monitoring", "setup guidance", "installation", "fiber connectivity",
)
PRODUCT_GROUPS = {
    "Mobile Connectivity": {
        "aliases": ("mobile connectivity", "mobile services", "mobile plans", "mobile"),
        "evidence": ("mobile connectivity", "prepaid basic", "postpaid silver", "5g mobile data", "sim and esim"),
        "subject": "mobile connectivity services",
    },
    "Home & Office Internet": {
        "aliases": ("home and office internet", "home office internet", "home broadband", "office internet", "zendfiber", "internet"),
        "evidence": ("home office internet", "home broadband", "zendfiber", "fiber connectivity", "router and wifi"),
        "subject": "home and office internet services",
    },
    "Business Connectivity": {
        "aliases": ("business connectivity", "enterprise connectivity", "business internet", "zendbiz", "zendenterprise"),
        "evidence": ("business connectivity", "zendbiz", "zendenterprise", "dedicated bandwidth", "mpls connectivity"),
        "subject": "business connectivity services",
    },
    "Cloud & Data Center Services": {
        "aliases": ("cloud and data center services", "cloud data center", "cloud services", "cloud offerings", "cloud solutions", "cloud"),
        "evidence": ("cloud and data center services", "zendcloud", "virtual machines", "file storage", "cloud networking", "cloud migration"),
        "subject": "cloud and data center services",
    },
    "IoT & Smart Solutions": {
        "aliases": ("iot and smart solutions", "iot smart solutions", "iot services", "smart solutions", "iot"),
        "evidence": ("iot and smart solutions", "zendsmart", "sensor connectivity", "device management", "smart city integrations"),
        "subject": "IoT and smart solutions",
    },
}
FACTUAL_RISK_TERMS = {
    "billing", "contract", "discount", "encryption", "gdpr", "price", "pricing", "refund", "service", "sla", "support",
    "tier", "zends",
}
INTERNAL_TERMS = {"chunk", "context", "document", "embedding", "rag", "retriev"}


def _terms(text: str) -> set[str]:
    return {term for term in TOKEN_PATTERN.findall(text.lower()) if term not in STOPWORDS}


def _meaningful_terms(text: str) -> set[str]:
    """Exclude generic telecom/pricing words from evidence relevance decisions."""
    return _terms(text) - GENERIC_TERMS


def _specific_phrases(query: str) -> set[str]:
    """Extract deterministic two- and three-token product/service phrases from the query."""
    tokens = TOKEN_PATTERN.findall(query.lower())
    phrases: set[str] = set()
    for length in (3, 2):
        for index in range(len(tokens) - length + 1):
            phrase_tokens = tokens[index:index + length]
            if any(token in STOPWORDS or token in GENERIC_TERMS for token in phrase_tokens):
                continue
            if any(token.startswith("zend") for token in phrase_tokens) or all(token.isalpha() or token.isdigit() for token in phrase_tokens):
                phrases.add(" ".join(phrase_tokens))
    return phrases


def _normalized_phrase_text(text: str) -> str:
    """Normalize punctuation variants such as '&' for deterministic alias matching."""
    return " ".join(TOKEN_PATTERN.findall(text.lower()))


def _query_product_group(query: str) -> str | None:
    normalized = _normalized_phrase_text(query)
    for group, configuration in PRODUCT_GROUPS.items():
        aliases = sorted(configuration["aliases"], key=len, reverse=True)
        if any(re.search(rf"(?<!\w){re.escape(alias)}(?!\w)", normalized) for alias in aliases):
            return group
    return None


def _product_group_evidence_strength(document: str, group: str | None) -> int:
    if group is None:
        return 0
    normalized = _normalized_phrase_text(document)
    markers = PRODUCT_GROUPS[group]["evidence"]
    return sum(bool(re.search(rf"(?<!\w){re.escape(marker)}(?!\w)", normalized)) for marker in markers)


def _policy_category(query: str, intent: str) -> str | None:
    """Resolve an explicit source policy heading from the query before intent fallback."""
    normalized = " ".join(query.lower().split())
    for term, category in POLICY_QUERY_TERMS.items():
        if term in normalized:
            return category
    return POLICY_INTENT.get(intent)


def expected_policy_category(query: str, intent: str) -> str | None:
    """Expose policy routing without exposing evidence-selection internals."""
    return _policy_category(query, intent)


def select_grounded_chunks(query: str, intent: str, chunks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Select at most two genuinely relevant, source-retrieved evidence chunks."""
    query_terms = _meaningful_terms(query)
    specific_phrases = _specific_phrases(query)
    named_terms = {term for term in query_terms if term.startswith("zend") and len(term) > 4}
    expected_policy = _policy_category(query, intent)
    raw_query_terms = set(TOKEN_PATTERN.findall(query.lower()))
    service_query_match = bool(raw_query_terms & SERVICE_QUERY_TERMS) or any(term.startswith("zend") for term in raw_query_terms)
    operational_query_match = bool(raw_query_terms & TECHNICAL_QUERY_TERMS)
    complaint_query_match = intent == "Complaint" and bool(raw_query_terms & COMPLAINT_QUERY_TERMS)
    support_query_match = service_query_match and (operational_query_match or complaint_query_match)
    query_product_group = _query_product_group(query)

    # An explicit source policy heading is the strongest evidence for a policy
    # question. Do not dilute it with generic chunks that merely mention terms
    # such as service, enterprise, or support.
    if expected_policy is not None:
        policy_chunks = [chunk for chunk in chunks if chunk["metadata"].get("policy_category") == expected_policy]
        policy_chunks.sort(key=lambda chunk: (chunk["distance"], chunk["metadata"]["chunk_id"]))
        return policy_chunks[:2]

    scored: list[tuple[int, bool, bool, dict[str, Any]]] = []
    for chunk in chunks:
        document_text = str(chunk["text"])
        document_lower = document_text.lower()
        document_terms = _meaningful_terms(document_text)
        overlap = query_terms & document_terms
        named_match = bool(named_terms & document_terms)
        phrase_matches = [phrase for phrase in specific_phrases if phrase in document_lower]
        support_match = (
            intent in SUPPORT_INTENTS
            and support_query_match
            and any(phrase in document_lower for phrase in SUPPORT_EVIDENCE_TERMS)
        )
        product_group_strength = _product_group_evidence_strength(document_text, query_product_group)
        product_group_match = intent == "Product Inquiry" and product_group_strength >= 2
        product_match = bool(phrase_matches) or named_match
        score = (30 * len(phrase_matches)) + (20 if named_match else 0) + (12 if support_match else 0) + (4 if support_match and support_query_match else 0) + (10 * min(product_group_strength, 3)) + len(overlap)
        # Exact product/service phrases, named ZENDS terms, and technical
        # support evidence are meaningful on their own. Other evidence needs
        # two non-generic query terms to qualify.
        if product_match or support_match or product_group_match or len(overlap) >= 2:
            scored.append((score, product_match, support_match, chunk))
    scored.sort(key=lambda item: (-item[0], item[3]["distance"], item[3]["metadata"]["chunk_id"]))
    if not scored:
        return []
    # A second chunk must add a distinct relevant signal. This preserves a
    # product chunk plus a technical-support chunk, while rejecting broad
    # product/pricing context that merely repeats the same product phrase.
    _, first_product, first_support, first_chunk = scored[0]
    selected = [first_chunk]
    for _, product_match, support_match, chunk in scored[1:]:
        if (first_product and support_match and not first_support) or (first_support and product_match and not first_product):
            selected.append(chunk)
            break
    return selected


def safe_acknowledgement(model_text: str, sentiment: str) -> str:
    """Use an LLM tone sentence only when it contains no factual-risk language."""
    candidate = " ".join(str(model_text).replace("\n", " ").split())
    candidate_terms = _terms(candidate)
    factual_risk = any(term == risk or term.startswith(risk) for term in candidate_terms for risk in FACTUAL_RISK_TERMS)
    safe_model_phrases = {
        "i understand your concern.",
        "i understand your question.",
        "thank you for reaching out.",
        "thank you for your question.",
        "i am sorry this has been frustrating.",
    }
    if candidate.lower() in safe_model_phrases and not factual_risk:
        return candidate
    if sentiment == "Angry":
        return "I am sorry that this has been frustrating."
    if sentiment == "Happy":
        return "Thank you for reaching out."
    return "I understand your question."


def _evidence_text(evidence: list[dict[str, Any]]) -> str:
    return " ".join(str(chunk["text"]) for chunk in evidence)


def _source_sentences(text: str) -> list[str]:
    return [sentence.strip() for sentence in re.split(r"(?<=[.!?])\s+", text) if sentence.strip()]


def _policy_answer(evidence: list[dict[str, Any]]) -> str | None:
    """Create short source-derived wording for the explicit policy sections."""
    category = next((chunk["metadata"].get("policy_category") for chunk in evidence if chunk["metadata"].get("policy_category")), None)
    text = _evidence_text(evidence)
    if category == "Billing":
        monthly = re.search(r"Monthly billing in advance", text, re.I)
        invoices = re.search(r"Enterprise customers receive consolidated invoices", text, re.I)
        late = re.search(r"Late payment after (\d+) days may suspend services", text, re.I)
        parts = []
        if monthly:
            parts.append("ZENDS bills customers monthly in advance.")
        if invoices:
            parts.append("Enterprise customers receive consolidated invoices.")
        if late:
            parts.append(f"Payments overdue by more than {late.group(1)} days may lead to service suspension.")
        return " ".join(parts) or None
    if category == "Refund":
        refund = re.search(r"Full refund within (\d+) days if usage is less than (\d+)%", text, re.I)
        cloud = re.search(r"Cloud services are non-refundable after activation", text, re.I)
        parts = []
        if refund:
            parts.append(f"ZENDS offers a full refund within {refund.group(1)} days when usage is below {refund.group(2)}%.")
        if cloud:
            parts.append("Cloud services are not refundable after activation.")
        return " ".join(parts) or None
    if category in {"SLA", "Data Privacy", "Contracts", "Fair Usage", "Support Tiers", "Discounts"}:
        sentences = _source_sentences(re.sub(rf"^{re.escape(str(category))}:\s*", "", text, flags=re.I))
        return " ".join(sentences[:2]) or None
    return None


def _product_answer(query: str, evidence: list[dict[str, Any]]) -> str | None:
    """Summarize source-listed plans or product-group capabilities."""
    text = _evidence_text(evidence)
    normalized = query.lower()
    if "mobile" in normalized and any(word in normalized for word in ("plan", "offer", "option", "available")):
        plans = re.findall(r"\b(?:Prepaid|Postpaid)\s+(?:Basic|Plus|Unlimited|Silver|Gold)\b", text)
        unique_plans = list(dict.fromkeys(plans))
        if unique_plans:
            if len(unique_plans) == 1:
                return f"ZENDS offers the {unique_plans[0]} mobile plan."
            names = ", ".join(unique_plans[:-1]) + f", and {unique_plans[-1]}"
            return f"ZENDS offers {names} mobile plans."
    product_group = _query_product_group(query)
    capability_question = any(
        term in TOKEN_PATTERN.findall(normalized)
        for term in ("available", "capabilities", "offer", "offering", "offerings", "offers", "provide", "provides", "service", "services", "solution", "solutions")
    )
    if product_group and capability_question:
        capabilities = re.search(r"(?:^|(?<=[.!?])\s+)Services include\s+([^.!?]+)[.!?]", text, re.I)
        if capabilities:
            supported_capabilities = capabilities.group(1).strip().rstrip(",")
            subject = PRODUCT_GROUPS[product_group]["subject"]
            return f"ZENDS {subject} include {supported_capabilities}."
    return None


def deterministic_evidence_answer(query: str, intent: str, evidence: list[dict[str, Any]]) -> str | None:
    """Use compact source-derived fallback wording when a draft is unavailable."""
    if not evidence:
        return None
    if intent in SUPPORT_INTENTS:
        text = _evidence_text(evidence).lower()
        support_capabilities = [
            ("24×7 technical support", "24×7 technical support"),
            ("network monitoring", "network monitoring"),
            ("setup guidance", "setup guidance"),
            ("troubleshooting", "troubleshooting assistance"),
            ("installation", "installation support"),
        ]
        supported = [label for phrase, label in support_capabilities if phrase in text]
        if supported:
            capabilities = ", ".join(supported[:-1]) + (f", and {supported[-1]}" if len(supported) > 1 else supported[0])
            product_group = _query_product_group(query)
            if product_group:
                subject = PRODUCT_GROUPS[product_group]["subject"]
                return f"ZENDS provides {capabilities} for its {subject}."
            return f"ZENDS provides {capabilities}."
        # Product/service descriptions alone are not troubleshooting steps.
        return None
    return _policy_answer(evidence) or _product_answer(query, evidence)


def is_grounded_answer(answer: str, evidence: list[dict[str, Any]]) -> bool:
    """Reject internal, overlong, or numerically unsupported generated drafts."""
    candidate = " ".join(str(answer).split())
    if not candidate or len(candidate) > 700:
        return False
    candidate_terms = _terms(candidate)
    if any(term.startswith(internal) for term in candidate_terms for internal in INTERNAL_TERMS):
        return False
    source = _evidence_text(evidence).lower()
    numbers = re.findall(r"\$?\d+(?:\.\d+)?%?", candidate)
    if any(number.lower() not in source for number in numbers):
        return False
    source_terms = _terms(source)
    meaningful = candidate_terms - {"available", "information", "zends"}
    return len(meaningful & source_terms) >= 2


def compose_response(*, sentiment: str, priority: str, answer: str | None) -> str:
    """Return a concise grounded synthesis or a safe evidence abstention."""
    if not answer:
        if sentiment == "Angry":
            return "I’m sorry you’re experiencing this issue. The available ZENDS information does not provide enough detail to safely guide you further."
        return "The available ZENDS knowledge does not provide enough information to answer this question."
    if sentiment == "Angry" and not answer.lower().startswith(("i'm sorry", "i am sorry")):
        answer = f"I’m sorry this has been frustrating. {answer}"
    if priority == "High" and sentiment == "Angry":
        answer = f"{answer} I recognize this needs urgent attention."
    return answer
