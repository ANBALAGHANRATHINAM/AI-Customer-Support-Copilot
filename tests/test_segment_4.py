"""Behavior-focused tests for the isolated Segment 4 response engine."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
import json

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "4_AI_Response_Engine" / "code"))

from grounding import deterministic_pricing_answer, is_grounded_answer, select_grounded_chunks
from response_engine import ZendsResponseEngine


def chunk(text: str, *, chunk_id: str = "zends-p03-c06", page: int = 3, policy: str | None = None, distance: float = 0.1) -> dict:
    metadata = {"source": "ZENDS Communications.pdf", "page": page, "chunk_id": chunk_id}
    if policy:
        metadata["policy_category"] = policy
    return {"text": text, "metadata": metadata, "distance": distance}


class StubNLP:
    def __init__(self, *, intent: str = "Billing", sentiment: str = "Neutral", priority: str = "Low") -> None:
        self.intent = intent
        self.sentiment = sentiment
        self.priority = priority

    def predict(self, query: str) -> dict:
        if not query.strip():
            raise ValueError("Customer query cannot be empty.")
        return {
            "cleaned_text": " ".join(query.split()),
            "intent": self.intent,
            "intent_confidence": 0.91,
            "sentiment": self.sentiment,
            "sentiment_confidence": 0.88,
            "priority": self.priority,
        }


class StubRetriever:
    def __init__(self, chunks: list[dict]) -> None:
        self.chunks = chunks
        self.queries: list[str] = []
        self.policy_categories: list[str | None] = []
        self.support_intents: list[bool] = []

    def retrieve(self, query: str, *, policy_category: str | None = None, support_intent: bool = False) -> list[dict]:
        self.queries.append(query)
        self.policy_categories.append(policy_category)
        self.support_intents.append(support_intent)
        return self.chunks


class CaptureLLM:
    def __init__(self, reply: str = "I understand your question.") -> None:
        self.reply = reply
        self.prompts: list[str] = []

    def generate(self, prompt: str) -> str:
        self.prompts.append(prompt)
        return self.reply


def test_engine_returns_required_schema_and_supported_source_response() -> None:
    source = chunk("Billing: Monthly billing in advance. Late payment after 7 days may suspend services.", policy="Billing")
    engine = ZendsResponseEngine(StubNLP(), StubRetriever([source]), CaptureLLM())
    result = engine.respond("When are bills issued and can service be suspended?")
    assert tuple(result) == (
        "customer_query", "intent", "intent_confidence", "sentiment", "sentiment_confidence", "priority", "retrieved_context", "recommended_response", "abstention", "reason"
    )
    assert result["intent"] == "Billing"
    assert result["retrieved_context"] == [source]
    assert result["recommended_response"] == (
        "ZENDS bills customers monthly in advance. Payments overdue by more than 7 days may lead to service suspension."
    )
    assert result["abstention"] is False
    assert result["reason"] == "grounded_synthesis"
    assert engine.retriever.policy_categories == ["Billing"]


def test_prompt_contains_query_analysis_and_retrieved_context() -> None:
    source = chunk("Refund: Full refund within 7 days if usage is less than 10%.", policy="Refund")
    llm = CaptureLLM()
    engine = ZendsResponseEngine(StubNLP(intent="Refund"), StubRetriever([source]), llm)
    engine.respond("Can I receive a refund within seven days?")
    assert "Can I receive a refund within seven days?" in llm.prompts[0]
    assert "Predicted intent: Refund" in llm.prompts[0]
    assert source["text"] in llm.prompts[0]


def test_llm_prompt_echo_is_not_exposed_to_the_customer() -> None:
    source = chunk("Billing: Monthly billing in advance.", policy="Billing")
    engine = ZendsResponseEngine(StubNLP(), StubRetriever([source]), CaptureLLM("Be concise and professional."))
    result = engine.respond("When does billing happen?")
    assert "Be concise" not in result["recommended_response"]
    assert result["recommended_response"] == "ZENDS bills customers monthly in advance."


@pytest.mark.parametrize("query", [
    "My internet is not working.",
    "My connection keeps dropping.",
    "ZENDFiber setup problem",
])
def test_technical_support_evidence_is_synthesized_without_inventing_steps(query: str) -> None:
    source = chunk(
        "Services include fiber connectivity, installation, 24×7 technical support, network monitoring, setup guidance, and troubleshooting.",
        chunk_id="zends-p02-c03",
    )
    engine = ZendsResponseEngine(StubNLP(intent="Technical", sentiment="Angry", priority="High"), StubRetriever([source]), CaptureLLM("Please accept my apology."))
    result = engine.respond(query)
    assert "sorry" in result["recommended_response"].lower()
    assert "technical support" in result["recommended_response"]
    assert "troubleshooting assistance" in result["recommended_response"]
    assert source["text"] not in result["recommended_response"]
    assert result["abstention"] is False
    assert engine.retriever.support_intents == [True]


@pytest.mark.parametrize("priority", ["High", "Medium", "Low"])
def test_informational_response_does_not_add_unnecessary_urgency(priority: str) -> None:
    source = chunk("Billing: Monthly billing in advance.", policy="Billing")
    engine = ZendsResponseEngine(StubNLP(priority=priority), StubRetriever([source]), CaptureLLM())
    result = engine.respond("When does billing happen?")
    assert "urgent attention" not in result["recommended_response"].lower()


@pytest.mark.parametrize("sentiment", ["Happy", "Neutral"])
def test_non_angry_evidence_gap_is_a_concise_abstention(sentiment: str) -> None:
    engine = ZendsResponseEngine(StubNLP(sentiment=sentiment), StubRetriever([]), CaptureLLM("Refunds are guaranteed."))
    result = engine.respond("Could you help me?")
    assert result["recommended_response"] == "The available ZENDS knowledge does not provide enough information to answer this question."


def test_unsupported_query_abstains_without_using_irrelevant_company_fact() -> None:
    privacy = chunk("Data Privacy: GDPR compliant, ISO 27001 certified, and encrypted data at rest and in transit.", chunk_id="zends-p04-c03", page=4, policy="Data Privacy")
    engine = ZendsResponseEngine(StubNLP(intent="Product Inquiry"), StubRetriever([privacy]), CaptureLLM())
    result = engine.respond("Who is the ZENDS chief executive officer?")
    assert "available ZENDS knowledge does not provide enough information" in result["recommended_response"]
    assert "GDPR compliant" not in result["recommended_response"]
    assert result["abstention"] is True
    assert result["reason"] == "insufficient_grounded_evidence"


def test_known_privacy_retrieval_gap_abstains_instead_of_inventing_privacy_facts() -> None:
    unrelated = chunk("Services include virtual machines, file storage, and cloud networking.", chunk_id="zends-p03-c03")
    engine = ZendsResponseEngine(StubNLP(intent="Product Inquiry"), StubRetriever([unrelated]), CaptureLLM())
    result = engine.respond("How does ZENDS protect customer data?")
    assert "available ZENDS knowledge does not provide enough information" in result["recommended_response"]
    assert "GDPR" not in result["recommended_response"]
    assert "encrypted" not in result["recommended_response"]


def test_technical_retrieval_gap_preserves_nlp_analysis_and_marks_abstention() -> None:
    engine = ZendsResponseEngine(
        StubNLP(intent="Technical", sentiment="Angry", priority="High"),
        StubRetriever([]),
        CaptureLLM(),
    )
    result = engine.respond("My internet is not working.")
    assert result["intent"] == "Technical"
    assert result["sentiment"] == "Angry"
    assert result["priority"] == "High"
    assert result["abstention"] is True
    assert result["reason"] == "insufficient_grounded_evidence"
    assert "available ZENDS information does not provide enough detail" in result["recommended_response"]


def test_complaint_uses_relevant_support_evidence_without_an_escalation_claim() -> None:
    source = chunk("Services include 24×7 technical support, network monitoring, and troubleshooting.", chunk_id="zends-p02-c03")
    engine = ZendsResponseEngine(StubNLP(intent="Complaint", sentiment="Angry", priority="High"), StubRetriever([source]), CaptureLLM())
    result = engine.respond("I want to complain about the service.")
    assert "technical support" in result["recommended_response"]
    assert "escalat" not in result["recommended_response"].lower()
    assert source["text"] not in result["recommended_response"]


def test_mobile_plan_answer_is_compact_and_excludes_unrequested_prices() -> None:
    mobile = chunk(
        "Mobile Connectivity includes Prepaid Basic, Prepaid Plus, Prepaid Unlimited, Postpaid Silver, and Postpaid Gold. "
        "Prepaid Basic is priced at $60 in the USA and $36 in India.",
        chunk_id="zends-p01-c02",
    )
    engine = ZendsResponseEngine(StubNLP(intent="Product Inquiry"), StubRetriever([mobile]), CaptureLLM())
    result = engine.respond("What mobile plans do you offer?")
    assert result["recommended_response"] == (
        "ZENDS offers Prepaid Basic, Prepaid Plus, Prepaid Unlimited, Postpaid Silver, and Postpaid Gold mobile plans."
    )
    assert "$60" not in result["recommended_response"]


def test_cloud_capability_alias_selects_and_answers_from_source_evidence() -> None:
    cloud = chunk(
        "Services include virtual machines, file storage, backup and disaster recovery, cloud networking, "
        "cloud migration support, API integrations, monitoring, and dedicated enterprise support.",
        chunk_id="zends-p03-c03",
    )
    query = "What cloud services does ZENDS offer?"
    assert select_grounded_chunks(query, "Product Inquiry", [cloud]) == [cloud]
    engine = ZendsResponseEngine(StubNLP(intent="Product Inquiry"), StubRetriever([cloud]), CaptureLLM())
    result = engine.respond(query)
    assert result["abstention"] is False
    assert result["recommended_response"] == (
        "ZENDS cloud and data center services include virtual machines, file storage, backup and disaster recovery, "
        "cloud networking, cloud migration support, API integrations, monitoring, and dedicated enterprise support."
    )


@pytest.mark.parametrize(
    ("query", "source_text", "expected_subject"),
    [
        ("What mobile services are available?", "Services include voice calling, SMS, 5G mobile data, SIM and eSIM provisioning.", "mobile connectivity services"),
        ("What home broadband services are available?", "Services include fiber connectivity, router and WiFi equipment, installation, and technical support.", "home and office internet services"),
        ("What business connectivity services are offered?", "Services include dedicated bandwidth, MPLS connectivity, priority troubleshooting, and network monitoring.", "business connectivity services"),
        ("What IoT services are available?", "Services include sensor connectivity, device management, smart city integrations, and remote monitoring.", "IoT and smart solutions"),
    ],
)
def test_product_group_capability_answers_use_only_selected_source(
    query: str, source_text: str, expected_subject: str
) -> None:
    source = chunk(source_text, chunk_id="product-capabilities")
    engine = ZendsResponseEngine(StubNLP(intent="Product Inquiry"), StubRetriever([source]), CaptureLLM())
    result = engine.respond(query)
    assert result["abstention"] is False
    assert result["recommended_response"].startswith(f"ZENDS {expected_subject} include ")
    assert source_text.removeprefix("Services include ") in result["recommended_response"]


def test_weak_support_word_does_not_make_unrelated_technical_query_grounded() -> None:
    weak = chunk(
        "Services include dedicated bandwidth, MPLS connectivity, SLA-backed uptime, priority troubleshooting, "
        "and customized network solutions for enterprises.",
        chunk_id="zends-p02-c05",
        distance=0.97,
    )
    query = "Write me a program."
    assert select_grounded_chunks(query, "Technical", [weak]) == []
    result = ZendsResponseEngine(StubNLP(intent="Technical"), StubRetriever([weak]), CaptureLLM()).respond(query)
    assert result["abstention"] is True
    assert "troubleshooting assistance" not in result["recommended_response"]


@pytest.mark.parametrize(
    ("query", "category", "source_text", "expected"),
    [
        ("What SLA does ZENDS provide?", "SLA", "SLA: Enterprise users have 99.9% uptime.", "99.9% uptime"),
        ("What is the ZENDS data privacy policy?", "Data Privacy", "Data Privacy: GDPR compliant and encrypted data at rest and in transit.", "GDPR compliant"),
        ("What is the fair usage policy?", "Fair Usage", "Fair Usage: Individual broadband plans include 1 TB monthly usage.", "1 TB"),
        ("What support tiers are available?", "Support Tiers", "Support Tiers: Standard, Priority, and Enterprise Dedicated Support.", "Enterprise Dedicated Support"),
        ("What discounts are available?", "Discounts", "Discounts: Annual payment discount is 15%.", "15%"),
    ],
)
def test_existing_policy_answers_remain_grounded(query: str, category: str, source_text: str, expected: str) -> None:
    source = chunk(source_text, chunk_id=f"policy-{category}", page=4, policy=category)
    result = ZendsResponseEngine(StubNLP(intent="Product Inquiry"), StubRetriever([source]), CaptureLLM()).respond(query)
    assert expected in result["recommended_response"]
    assert result["abstention"] is False


def test_existing_product_pricing_generation_remains_grounded() -> None:
    pricing = chunk(
        "ZENDCloud VM Basic is priced at $40 in the USA for individual users.",
        chunk_id="zends-p03-c01",
    )
    draft = "ZENDCloud VM Basic is priced at $40 in the USA for individual users."
    result = ZendsResponseEngine(
        StubNLP(intent="Product Inquiry"), StubRetriever([pricing]), CaptureLLM(draft)
    ).respond("What is the price of ZENDCloud VM Basic in the USA?")
    assert result["recommended_response"] == draft
    assert result["abstention"] is False


def test_exact_fiber_price_uses_source_tuple_when_model_echoes_prompt() -> None:
    pricing = chunk(
        "ZENDFiber Home 100 Mbps priced at $30 in the USA, $18 in India, $33 in Singapore, and $24 in Thailand "
        "for individual users, and $25 in the USA, $15 in India, $27 in Singapore, and $20 in Thailand "
        "for enterprise customers.",
        chunk_id="zends-p02-c01", page=2,
    )
    llm = CaptureLLM("Never invent prices, policies, products, services, contracts, SLAs, refunds, discounts, privacy claims, or escalation procedures.")
    engine = ZendsResponseEngine(StubNLP(intent="Product Inquiry"), StubRetriever([pricing]), llm)
    result = engine.respond("What is the individual price of ZENDFiber Home 100 Mbps in India?")
    assert result["recommended_response"] == "ZENDFiber Home 100 Mbps is priced at $18 in India for individual users."
    assert result["abstention"] is False


def test_pricing_rejects_wrong_country_or_customer_association() -> None:
    pricing = chunk(
        "ZENDFiber Home 100 Mbps priced at $30 in the USA, $18 in India for individual users, "
        "and $25 in the USA, $15 in India for enterprise customers.",
        chunk_id="zends-p02-c01", page=2,
    )
    query = "What is the individual price of ZENDFiber Home 100 Mbps in India?"
    assert deterministic_pricing_answer(query, [pricing]) == "ZENDFiber Home 100 Mbps is priced at $18 in India for individual users."
    assert deterministic_pricing_answer(query, [chunk(pricing["text"].replace("$18 in India", "$19 in India"))]) == "ZENDFiber Home 100 Mbps is priced at $19 in India for individual users."
    assert deterministic_pricing_answer(query, [chunk("ZENDFiber Home 100 Mbps is priced at $15 in India for enterprise customers.")]) is None


def test_adjacent_product_chunks_cannot_transfer_a_price_tuple() -> None:
    first = chunk(
        "ZENDFiber Home 100 Mbps priced at $18 in India for individual users, and",
        chunk_id="zends-p02-c01", page=2,
    )
    second = chunk(
        "ZENDOffice Net 200 is priced at $15 in India for enterprise customers.",
        chunk_id="zends-p02-c02", page=2,
    )
    assert deterministic_pricing_answer(
        "What is the enterprise price of ZENDFiber Home 100 Mbps in India?", [first, second]
    ) is None
    assert deterministic_pricing_answer(
        "What is the enterprise price of ZENDFiber Home 100 Mbps in India?",
        [chunk(f"{first['text']} {second['text']}", chunk_id="zends-p02-c01", page=2)],
    ) is None
    assert deterministic_pricing_answer(
        "What is the enterprise price of ZENDOffice Net 200 in India?", [first, second]
    ) == "ZENDOffice Net 200 is priced at $15 in India for enterprise users."


def test_adjacent_chunks_can_complete_the_same_product_price_sentence() -> None:
    first = chunk(
        "ZENDFiber Home 100 Mbps priced at $18 in India for individual users, and",
        chunk_id="zends-p02-c01", page=2,
    )
    continuation = chunk(
        "$15 in India for enterprise customers.",
        chunk_id="zends-p02-c02", page=2,
    )
    assert deterministic_pricing_answer(
        "What is the enterprise price of ZENDFiber Home 100 Mbps in India?", [first, continuation]
    ) == "ZENDFiber Home 100 Mbps is priced at $15 in India for enterprise users."


def test_generated_answer_must_address_the_requested_information() -> None:
    source = [chunk(
        "ZENDS Communications provides mobile connectivity and cloud infrastructure. "
        "Billing: Monthly billing in advance.",
        chunk_id="zends-p01-c01", page=1,
    )]
    query = "What does ZENDS Communications provide?"
    assert not is_grounded_answer("ZENDS Communications bills customers monthly in advance.", source, query=query)
    assert is_grounded_answer("ZENDS Communications provides mobile connectivity and cloud infrastructure.", source, query=query)


def test_all_pdf_product_country_customer_price_tuples() -> None:
    sys.path.insert(0, str(ROOT / "3_RAG_Knowledge" / "code"))
    from document_loader import load_pdf_pages
    from text_processing import chunk_text

    evidence = [
        chunk(text, chunk_id=f"zends-p{page.page:02d}-c{index:02d}", page=page.page)
        for page in load_pdf_pages(ROOT / "docs" / "ZENDS Communications.pdf")
        for index, text in enumerate(chunk_text(page.text), start=1)
    ]
    facts = json.loads((ROOT / "1_Data_Foundation" / "config" / "zends_facts.json").read_text(encoding="utf-8"))
    for product in facts["products"]:
        for country, prices in product["prices"].items():
            for customer_type, price in prices.items():
                query = f"What is the {customer_type} price of {product['name']} in {country}?"
                answer = deterministic_pricing_answer(query, evidence)
                assert answer is not None, query
                assert f"${price} " in answer, (query, answer)


def test_persisted_chroma_product_records_cover_all_price_tuples() -> None:
    sys.path.insert(0, str(ROOT / "3_RAG_Knowledge" / "code"))
    from retriever import ZendsRetriever

    retriever = ZendsRetriever(ROOT / "3_RAG_Knowledge" / "vector_db")
    facts = json.loads((ROOT / "1_Data_Foundation" / "config" / "zends_facts.json").read_text(encoding="utf-8"))
    for product in facts["products"]:
        evidence = retriever.retrieve_exact_products([product["name"]])
        for country, prices in product["prices"].items():
            for customer_type, price in prices.items():
                query = f"What is the {customer_type} price of {product['name']} in {country}?"
                answer = deterministic_pricing_answer(query, evidence)
                assert answer is not None and f"${price} " in answer, (query, answer)


def test_persisted_group_capabilities_follow_the_correct_pdf_heading() -> None:
    sys.path.insert(0, str(ROOT / "3_RAG_Knowledge" / "code"))
    from retriever import ZendsRetriever

    retriever = ZendsRetriever(ROOT / "3_RAG_Knowledge" / "vector_db")
    expected = {
        "Mobile Connectivity": "voice calling",
        "Home & Office Internet": "fiber connectivity",
        "Business Connectivity": "dedicated bandwidth",
        "Cloud & Data Center Services": "virtual machines",
        "IoT & Smart Solutions": "sensor connectivity",
    }
    for group, term in expected.items():
        evidence = retriever.retrieve_group_capabilities(group)
        assert len(evidence) == 1 and term in evidence[0]["text"]


def test_two_product_difference_uses_each_product_price_tuple() -> None:
    first = chunk("ZENDFiber Home 300 Mbps is priced at $30 in India for individual users, and $27 in India for enterprise customers.", chunk_id="zends-p02-c01", page=2)
    second = chunk("ZENDOffice Net 1G is priced at $90 in India for individual users, and $78 in India for enterprise customers.", chunk_id="zends-p02-c02", page=2)
    query = "What is the difference between ZENDFiber Home 300 Mbps and ZENDOffice Net 1G in India for an enterprise customer?"
    result = ZendsResponseEngine(StubNLP(intent="Product Inquiry"), StubRetriever([first, second]), CaptureLLM("Never invent prices, policies, products, services, contracts, SLAs, refunds, discounts, privacy claims, or escalation procedures.")).respond(query)
    assert "$27" in result["recommended_response"] and "$78" in result["recommended_response"]
    assert "price difference is $51" in result["recommended_response"]
    assert "$30" not in result["recommended_response"] and "$90" not in result["recommended_response"]


def test_late_payment_routes_to_billing_source() -> None:
    source = chunk("Billing: Monthly billing in advance. Late payment after 7 days may suspend services.", policy="Billing")
    engine = ZendsResponseEngine(StubNLP(intent="Product Inquiry"), StubRetriever([source]), CaptureLLM())
    result = engine.respond("What happens if an enterprise customer pays late?")
    assert engine.retriever.policy_categories == ["Billing"]
    assert "7 days" in result["recommended_response"]


def test_grounding_rejects_unsupported_numeric_claim_in_generated_draft() -> None:
    evidence = [chunk("Refund: Full refund within 7 days if usage is less than 10%.", policy="Refund")]
    assert is_grounded_answer("ZENDS offers a full refund within 7 days when usage is below 10%.", evidence)
    assert not is_grounded_answer("ZENDS offers a full refund within 30 days.", evidence)


def test_invalid_query_is_rejected() -> None:
    engine = ZendsResponseEngine(StubNLP(), StubRetriever([]), CaptureLLM())
    with pytest.raises(TypeError):
        engine.respond(None)  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        engine.respond("   ")


def test_response_engine_imports_in_a_fresh_python_process() -> None:
    command = [sys.executable, "-c", "import sys; sys.path.insert(0, r'4_AI_Response_Engine/code'); import response_engine; print('ok')"]
    result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, check=True)
    assert result.stdout.strip() == "ok"
