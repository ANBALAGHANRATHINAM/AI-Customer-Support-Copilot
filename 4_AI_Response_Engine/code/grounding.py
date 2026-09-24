"""Conservative evidence selection and response composition for Segment 4."""

from __future__ import annotations

import re
from decimal import Decimal
from dataclasses import dataclass
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
POLICY_QUERY_TERMS = {
    "refund": "Refund",
    "refunds": "Refund",
    "refundable": "Refund",
    "reimbursement": "Refund",
    "money back": "Refund",
    "discount": "Discounts",
    "discounts": "Discounts",
    "contract": "Contracts",
    "contracts": "Contracts",
    "service level": "SLA",
    "uptime": "SLA",
    "sla": "SLA",
    "fair usage": "Fair Usage",
    "usage limit": "Fair Usage",
    "support tiers": "Support Tiers",
    "data protection": "Data Privacy",
    "protect customer data": "Data Privacy",
    "protect data": "Data Privacy",
    "privacy": "Data Privacy",
    "gdpr": "Data Privacy",
    "billing": "Billing",
    "bill": "Billing",
    "bills": "Billing",
    "invoice": "Billing",
    "invoices": "Billing",
    "payment": "Billing",
    "payments": "Billing",
    "pays late": "Billing",
    "pay late": "Billing",
    "late payment": "Billing",
    "don't pay": "Billing",
    "overdue": "Billing",
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
OPERATIONAL_QUERY_TERMS = TECHNICAL_QUERY_TERMS - SERVICE_QUERY_TERMS - {"installation"}
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
        "aliases": ("home and office internet", "home and office services", "home office internet", "home office services", "home broadband", "office internet", "zendfiber", "internet"),
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
PRICE_COUNTRY = re.compile(r"\$(\d+(?:\.\d+)?)\s+in\s+(?:the\s+)?([A-Za-z][A-Za-z ]*?)(?=\s*,|\s+and\b|\s+for\b|[.!?]|$)", re.I)
EXPLICIT_PRICE_QUESTION = re.compile(r"\b(?:price|prices|pricing|cost|costs|priced)\b", re.I)
PRICE_QUESTION = re.compile(r"\b(?:price|prices|pricing|cost|costs|priced)\b|how much", re.I)
PRODUCT_NAME_PATTERNS = (
    re.compile(r"\b((?:Prepaid|Postpaid)\s+[A-Za-z]+)\b(?:\s+with\b[^.]*?)?\s*,?\s*(?:is\s+)?priced at", re.I),
    re.compile(r"\b(ZEND[A-Za-z]+(?:\s+[A-Za-z0-9]+){0,3}?)\s+(?:is\s+)?priced at", re.I),
)
UNSUPPORTED_POLICY = re.compile(r"\b(?:cancel|cancellation|terminate|termination)\b", re.I)
ARTIFACT_REQUEST = re.compile(
    r"^\s*(?:(?:please|can you|could you|would you)\s+)?"
    r"(?:write|create|generate|build|draft|produce|make|give me)\b"
    r"[^?.]{0,120}\b(?:code|program|script|poem|essay|email|letter|report)\b",
    re.I,
)
INSTRUCTIONAL_PROGRAMMING_REQUEST = re.compile(
    r"^\s*how\s+(?:do|can|could|would)\s+(?:i|we)\s+"
    r"(?:(?:write|create|generate|build|make)\b[^?.]{0,120}\b(?:code|program|script)\b"
    r"|programmatically\s+\w+)",
    re.I,
)


@dataclass(frozen=True)
class QueryRequest:
    """Requested answer facets, separate from the NLP model's triage label."""

    policy_category: str | None
    pricing_products: tuple[str, ...]
    feature_products: tuple[str, ...]
    product_list_group: str | None
    capability_group: str | None
    support: bool
    unsupported_policy: bool
    asks_price: bool


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
    matches: list[tuple[int, str]] = []
    for group, configuration in PRODUCT_GROUPS.items():
        for alias in configuration["aliases"]:
            if re.search(rf"(?<!\w){re.escape(alias)}(?!\w)", normalized):
                matches.append((len(alias), group))
    return max(matches, default=(0, None))[1]


def requested_capability_group(query: str) -> str | None:
    """Identify a named group only when the user asks about included services."""
    if not re.search(r"\b(?:service|services|capabilities|include|included|offer|offers|available)\b", query, re.I):
        return None
    return _query_product_group(query)


def _product_group_evidence_strength(document: str, group: str | None) -> int:
    if group is None:
        return 0
    normalized = _normalized_phrase_text(document)
    markers = PRODUCT_GROUPS[group]["evidence"]
    return sum(bool(re.search(rf"(?<!\w){re.escape(marker)}(?!\w)", normalized)) for marker in markers)


def _policy_category(query: str, intent: str) -> str | None:
    """Resolve a policy from the question, independent of a noisy model label."""
    normalized = " ".join(query.lower().split())
    for term, category in POLICY_QUERY_TERMS.items():
        if re.search(rf"(?<!\w){re.escape(term)}(?!\w)", normalized):
            return category
    return None


def is_artifact_generation_request(query: str) -> bool:
    """Identify requests to create an artifact rather than answer a support question."""
    return bool(ARTIFACT_REQUEST.search(query) or INSTRUCTIONAL_PROGRAMMING_REQUEST.search(query))


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
    support_query_match = service_query_match and (operational_query_match or complaint_query_match or "technical support" in query.lower())
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


def source_product_names(evidence: list[dict[str, Any]]) -> list[str]:
    """Discover names only from PDF-derived pricing sentences, never the training catalog."""
    names: list[str] = []
    for chunk in evidence:
        text = " ".join(str(chunk["text"]).split())
        for pattern in PRODUCT_NAME_PATTERNS:
            names.extend(match.group(1) for match in pattern.finditer(text))
    return list(dict.fromkeys(names))


def requested_pricing_products(query: str, evidence: list[dict[str, Any]]) -> list[str]:
    names = [name for name in source_product_names(evidence) if re.search(rf"(?<!\w){re.escape(name)}(?!\w)", query, re.I)]
    if _policy_category(query, "") and not EXPLICIT_PRICE_QUESTION.search(query):
        return []
    if not PRICE_QUESTION.search(query) and not (len(names) >= 2 and re.search(r"\b(?:difference|compare|comparison)\b", query, re.I)):
        return []
    return sorted(names, key=lambda name: query.lower().find(name.lower()))


def resolve_query_request(query: str, intent: str, source_records: list[dict[str, Any]]) -> QueryRequest:
    """Route by requested information; triage intent is kept for display only."""
    policy = _policy_category(query, intent)
    unsupported = bool(UNSUPPORTED_POLICY.search(query))
    products = tuple(requested_pricing_products(query, source_records))
    feature_products = tuple(
        name for name in source_product_names(source_records)
        if re.search(rf"(?<!\w){re.escape(name)}(?!\w)", query, re.I)
    ) if not products and policy is None and re.search(r"\b(?:include|includes|feature|features|come with|offer)\b", query, re.I) else ()
    raw_terms = set(TOKEN_PATTERN.findall(query.lower()))
    service_context = bool(raw_terms & SERVICE_QUERY_TERMS) or any(term.startswith("zend") for term in raw_terms)
    support = (
        (service_context and bool(raw_terms & (OPERATIONAL_QUERY_TERMS | COMPLAINT_QUERY_TERMS)))
        or bool(re.search(r"\btechnical support\b", query, re.I))
    ) and policy is None and not products
    group = _query_product_group(query) if policy is None and not products and not feature_products and not support and not unsupported else None
    product_list = group if group and re.search(r"\b(?:plan|plans|products|offerings|options)\b", query, re.I) else None
    capability = requested_capability_group(query) if group and not product_list else None
    asks_price = bool(EXPLICIT_PRICE_QUESTION.search(query) or (products and re.search(r"how much", query, re.I)))
    return QueryRequest(policy, products, feature_products, product_list, capability, support, unsupported, asks_price)


def _joined_source_passages(evidence: list[dict[str, Any]]) -> list[str]:
    """Rejoin adjacent persisted chunks when the PDF sentence crossed a chunk edge."""
    indexed = sorted(evidence, key=lambda item: (str(item["metadata"].get("source", "")), int(item["metadata"].get("page", 0)), str(item["metadata"].get("chunk_id", ""))))
    passages: list[str] = []
    previous_key: tuple[str, int, int] | None = None
    for chunk in indexed:
        metadata = chunk["metadata"]
        chunk_id = str(metadata.get("chunk_id", ""))
        match = re.search(r"-c(\d+)$", chunk_id)
        key = (str(metadata.get("source", "")), int(metadata.get("page", 0)), int(match.group(1)) if match else -1)
        value = " ".join(str(chunk["text"]).split())
        adjacent = previous_key and key[:2] == previous_key[:2] and key[2] == previous_key[2] + 1
        if adjacent:
            prior = passages[-1]
            overlap = next((size for size in range(min(len(prior), len(value)), 19, -1) if prior[-size:] == value[:size]), 0)
            starts_new_product = any(pattern.match(value) for pattern in PRODUCT_NAME_PATTERNS)
            if overlap:
                passages[-1] = prior + value[overlap:]
            elif not prior.endswith((".", "!", "?")) and not starts_new_product:
                passages[-1] = prior + " " + value
            else:
                passages.append(value)
        else:
            passages.append(value)
        previous_key = key
    return passages


def _source_prices(product: str, evidence: list[dict[str, Any]]) -> dict[tuple[str, str], str]:
    """Extract prices only inside this product's own source sentence and user-type clause."""
    prices: dict[tuple[str, str], str] = {}
    for passage in _joined_source_passages(evidence):
        product_starts = sorted(
            (match.start(), match.group(1))
            for pattern in PRODUCT_NAME_PATTERNS for match in pattern.finditer(passage)
        )
        for match in re.finditer(rf"(?<!\w){re.escape(product)}(?!\w)[^.]*\.", passage, re.I):
            next_product = next(
                (start for start, name in product_starts if start > match.start() and name.casefold() != product.casefold()),
                match.end(),
            )
            sentence = passage[match.start():min(match.end(), next_product)]
            if "priced at" not in sentence.lower():
                continue
            individual = re.search(r"\bfor individual users\b", sentence, re.I)
            enterprise = re.search(r"\bfor enterprise customers\b", sentence, re.I)
            if individual and enterprise and individual.start() < enterprise.start():
                sections = (("individual", sentence[:individual.start()]), ("enterprise", sentence[individual.end():enterprise.start()]))
            elif individual:
                sections = (("individual", sentence[:individual.start()]),)
            elif enterprise:
                sections = (("enterprise", sentence[:enterprise.start()]),)
            else:
                continue
            for customer_type, section in sections:
                for amount, country in PRICE_COUNTRY.findall(section):
                    country = country.strip()
                    key = (country.upper() if country.upper() == "USA" else country.title(), customer_type)
                    if key in prices and prices[key] != amount:
                        return {}
                    prices[key] = amount
    return prices


def deterministic_pricing_answer(query: str, evidence: list[dict[str, Any]]) -> str | None:
    """Answer only when every requested product/country/type/price tuple is explicit."""
    products = requested_pricing_products(query, evidence)
    if not products or not evidence:
        return None
    product_prices = {product: _source_prices(product, evidence) for product in products}
    source_countries = list(dict.fromkeys(country for prices in product_prices.values() for country, _ in prices))
    countries = [country for country in source_countries if re.search(rf"(?<!\w){re.escape(country)}(?!\w)", query, re.I)]
    if not countries:
        return None
    requested_types = [customer_type for customer_type in ("individual", "enterprise") if re.search(rf"\b{customer_type}\b", query, re.I)]
    sentences = []
    selected_prices: dict[tuple[str, str, str], str] = {}
    for product in products:
        source_prices = product_prices[product]
        for country in countries:
            types = requested_types or [customer_type for customer_type in ("individual", "enterprise") if (country, customer_type) in source_prices]
            if not types or any((country, customer_type) not in source_prices for customer_type in types):
                return None
            selected_prices.update({(product, country, customer_type): source_prices[(country, customer_type)] for customer_type in types})
            if len(types) == 1:
                customer_type = types[0]
                article = "the " if re.search(rf"\bin the {re.escape(country)}\b", _evidence_text(evidence), re.I) else ""
                sentences.append(f"{product} is priced at ${source_prices[(country, customer_type)]} in {article}{country} for {customer_type} users.")
            else:
                sentences.append(f"{product} in {country}: ${source_prices[(country, 'individual')]} for individual users and ${source_prices[(country, 'enterprise')]} for enterprise customers.")
    if len(countries) == 1 and re.search(r"\bdifference\b", query, re.I):
        country = countries[0]
        pair = None
        if len(products) == 2 and len(requested_types) == 1:
            pair = (selected_prices[(products[0], country, requested_types[0])], selected_prices[(products[1], country, requested_types[0])])
        elif len(products) == 1 and len(types) == 2:
            pair = (selected_prices[(products[0], country, "individual")], selected_prices[(products[0], country, "enterprise")])
        if pair:
            difference = abs(Decimal(pair[0]) - Decimal(pair[1]))
            sentences.append(f"The price difference is ${format(difference, 'f')}.")
    return " ".join(sentences)


def _source_sentences(text: str) -> list[str]:
    return [sentence.strip() for sentence in re.split(r"(?<=[.!?])\s+", text) if sentence.strip()]


def _policy_answer(evidence: list[dict[str, Any]], query: str | None = None) -> str | None:
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
        if query:
            query_terms = _meaningful_terms(query)
            ranked = [(len(query_terms & _meaningful_terms(part)), part) for part in parts]
            best = max((score for score, _ in ranked), default=0)
            if best >= 2 and sum(score == best for score, _ in ranked) == 1:
                return next(part for score, part in ranked if score == best)
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


def deterministic_policy_answer(category: str, evidence: list[dict[str, Any]], *, query: str | None = None) -> str | None:
    """Use only chunks explicitly tagged with the requested PDF policy heading."""
    matching = [chunk for chunk in evidence if chunk["metadata"].get("policy_category") == category]
    return _policy_answer(matching, query=query) if matching else None


def deterministic_product_list_answer(group: str, evidence: list[dict[str, Any]]) -> str | None:
    """List only names appearing in the selected PDF product-group section."""
    names = source_product_names(evidence)
    if not names:
        return None
    names = list(dict.fromkeys(names))
    listed = names[0] if len(names) == 1 else ", ".join(names[:-1]) + f", and {names[-1]}"
    return f"ZENDS {PRODUCT_GROUPS[group]['subject']} include {listed}."


def deterministic_product_feature_answer(query: str, evidence: list[dict[str, Any]]) -> str | None:
    """Use only an explicit product-specific 'with ... priced at' description."""
    products = [name for name in source_product_names(evidence) if re.search(rf"(?<!\w){re.escape(name)}(?!\w)", query, re.I)]
    if len(products) != 1:
        return None
    product = products[0]
    for passage in _joined_source_passages(evidence):
        match = re.search(rf"(?<!\w){re.escape(product)}\s+with\s+([^.]{{1,120}}?)\s*,?\s*(?:is\s+)?priced at", passage, re.I)
        if match:
            return f"{product} includes {match.group(1).strip().rstrip(',')}."
    return None


def policy_scope_note(query: str, category: str, evidence: list[dict[str, Any]]) -> str | None:
    """Avoid implying that a general policy explicitly names a queried product group."""
    group = _query_product_group(query)
    if group is None:
        return None
    source = _normalized_phrase_text(_evidence_text([
        chunk for chunk in evidence if chunk["metadata"].get("policy_category") == category
    ]))
    if not source:
        return None
    aliases = PRODUCT_GROUPS[group]["aliases"]
    if any(re.search(rf"(?<!\w){re.escape(alias)}(?!\w)", source) for alias in aliases):
        return None
    return f"The available policy does not specify a separate rule for {PRODUCT_GROUPS[group]['subject']}."


def _product_answer(query: str, evidence: list[dict[str, Any]]) -> str | None:
    """Summarize source-listed plans or product-group capabilities."""
    text = _evidence_text(evidence)
    normalized = query.lower()
    if "mobile" in normalized and any(word in normalized for word in ("plan", "offer", "option", "available")):
        plans = re.findall(r"\b(?:Prepaid|Postpaid)\s+[A-Za-z]+\b", text)
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


ANSWER_FACETS = {
    "Refund": re.compile(r"\b(?:refund\w*|reimburse\w*|money back)\b", re.I),
    "Discounts": re.compile(r"\bdiscount\w*\b", re.I),
    "Contracts": re.compile(r"\bcontract\w*\b", re.I),
    "SLA": re.compile(r"\b(?:sla|uptime|service level)\b", re.I),
    "Fair Usage": re.compile(r"\b(?:usage|limit|cap)\b", re.I),
    "Support Tiers": re.compile(r"\b(?:support|tier)\b", re.I),
    "Data Privacy": re.compile(r"\b(?:privacy|data|encrypt|gdpr)\w*\b", re.I),
    "Billing": re.compile(r"\b(?:bill|payment|pay|paid|invoice|overdue|suspend)\w*\b", re.I),
}


def _answers_requested_information(query: str, candidate: str) -> bool:
    """Require the draft to address each information type explicitly requested."""
    policy = _policy_category(query, "")
    if policy and not ANSWER_FACETS[policy].search(candidate):
        return False
    if PRICE_QUESTION.search(query) and not re.search(r"\$\d|\b(?:price|cost)\w*\b", candidate, re.I):
        return False
    asks_capability = re.search(r"\b(?:provide|provides|offer|offers|available|capabilities|include|includes|features)\b", query, re.I)
    if asks_capability and not policy and not PRICE_QUESTION.search(query):
        if not re.search(r"\b(?:provid|offer|includ|avail|capabilit|feature|service|product|plan)\w*\b", candidate, re.I):
            return False
    if re.search(r"\b(?:down|outage|troubleshoot|connection issue|technical support)\b", query, re.I):
        if not re.search(r"\b(?:support|troubleshoot|monitor|repair|restore|assist|outage|connection)\w*\b", candidate, re.I):
            return False
    return True


def _relevant_source_sentence(query: str, candidate: str, evidence: list[dict[str, Any]]) -> bool:
    """Tie the answer to source sentences about the question, not other facts in a chunk."""
    def normalized_terms(value: str) -> set[str]:
        return {term[:-1] if term.endswith("s") and len(term) > 3 else term for term in _meaningful_terms(value)}

    query_terms = normalized_terms(query)
    answer_terms = normalized_terms(candidate)
    scored = [
        (len(query_terms & normalized_terms(sentence)), normalized_terms(sentence))
        for chunk in evidence for sentence in _source_sentences(str(chunk["text"]))
    ]
    best = max((score for score, _ in scored), default=0)
    if best == 0:
        return False
    return any(
        score == best
        and len(answer_terms & source_terms) >= 2
        and bool((answer_terms - query_terms) & source_terms)
        for score, source_terms in scored
    )


def is_grounded_answer(answer: str, evidence: list[dict[str, Any]], *, query: str | None = None) -> bool:
    """Accept a draft only when it is sourced and addresses the requested subject."""
    candidate = " ".join(str(answer).split())
    if not candidate or len(candidate) > 700 or not evidence:
        return False
    if re.match(r"^(?:never|do not|don't|use only|answer the|if the supplied)\b", candidate, re.I):
        return False
    candidate_terms = _terms(candidate)
    if any(term.startswith(internal) for term in candidate_terms for internal in INTERNAL_TERMS):
        return False
    source = _evidence_text(evidence).lower()
    if query:
        if not _answers_requested_information(query, candidate):
            return False
        if not _relevant_source_sentence(query, candidate, evidence):
            return False
        query_terms = _meaningful_terms(query)
        if not query_terms & candidate_terms:
            return False
        named_terms = {term for term in query_terms if term.startswith("zend")}
        if named_terms and not named_terms <= candidate_terms:
            return False
        if re.search(r"\b(?:include|includes|feature|features)\b", query, re.I) and named_terms:
            if not any(named_terms <= _terms(sentence) and re.search(r"\b(?:include|includes|feature|features)\b", sentence, re.I) for sentence in _source_sentences(source)):
                return False
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
