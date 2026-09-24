"""End-to-end response and Streamlit-scope regressions across supported query types."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "4_AI_Response_Engine" / "code"))
sys.path.insert(0, str(ROOT / "5_Streamlit_Integration" / "code"))

from llm import StaticAcknowledgementLLM
from grounding import is_grounded_answer
from response_engine import ZendsResponseEngine
from ui_helpers import apply_scope_guard


@pytest.fixture(scope="module")
def engine() -> ZendsResponseEngine:
    return ZendsResponseEngine.from_project_assets(llm=StaticAcknowledgementLLM())


@pytest.mark.parametrize(
    ("category", "query", "expected", "forbidden"),
    [
        ("individual price", "What is the individual price of ZENDFiber Home 100 Mbps in India?", "$18 in India for individual users", "$15 in India"),
        ("enterprise price", "What is the enterprise price of ZENDFiber Home 100 Mbps in Singapore?", "$27 in Singapore for enterprise users", "$33 in Singapore"),
        ("multi country", "What are the individual prices for ZENDFiber Home 100 Mbps in India and Singapore?", "$18 in India", "$15 in India"),
        ("price comparison", "Compare the prices of ZENDStorage 1TB and ZENDStorage 10TB in India.", "ZENDStorage 10TB in India: $72", "refund"),
        ("price difference", "What is the difference between ZENDFiber Home 300 Mbps and ZENDOffice Net 1G in India for an enterprise customer?", "price difference is $51", "virtual machines"),
        ("prepaid feature", "What does Prepaid Basic include?", "includes 5GB", "priced at"),
        ("postpaid feature", "What does Postpaid Platinum include?", "unlimited data and international calls", "priced at"),
        ("mobile plans", "What mobile plans do you offer?", "Postpaid Platinum", "voice calling"),
        ("home products", "What products are in Home & Office Internet?", "ZENDOffice Net 1G", "fiber connectivity"),
        ("business products", "What products are in Business Connectivity?", "ZENDEnterprise Dedicated", "dedicated bandwidth"),
        ("cloud products", "What Cloud & Data Center products do you offer?", "ZENDArchive Storage", "virtual machines"),
        ("IoT products", "What IoT products are available?", "ZENDFleet IoT", "sensor connectivity"),
        ("refund", "What is the ZENDS refund policy?", "within 7 days", "virtual machines"),
        ("cloud refund", "Are cloud services refundable after activation?", "not refundable after activation", "virtual machines"),
        ("billing", "How does ZENDS billing work?", "monthly in advance", "virtual machines"),
        ("late payment", "What happens if an enterprise customer pays late?", "7 days", "virtual machines"),
        ("discount", "What discounts does ZENDS offer?", "30% discount", "virtual machines"),
        ("bulk discount", "How much discount can bulk enterprise customers receive?", "30% discount", "specific product and country"),
        ("annual discount", "What is the annual payment discount?", "15%", "bills customers"),
        ("contract", "Is there a minimum contract for enterprise customers?", "12-month contract", "virtual machines"),
        ("SLA", "What is the SLA for ZENDS business connectivity?", "99.5% uptime", "dedicated bandwidth"),
        ("fair usage", "What is the fair usage limit for unlimited plans?", "1TB per month", "virtual machines"),
        ("support tiers", "What support tiers does ZENDS offer?", "Enterprise Dedicated Support", "virtual machines"),
        ("privacy", "How does ZENDS protect customer data?", "encrypted data", "virtual machines"),
        ("cloud capability", "What cloud services are available?", "virtual machines", "refundable"),
        ("mobile capability", "What mobile services are available?", "voice calling", "refund"),
        ("home capability", "What Home & Office services are available?", "fiber connectivity", "refund"),
        ("business capability", "What Business Connectivity services are available?", "dedicated bandwidth", "refund"),
        ("business internet capability", "What business internet services are available?", "dedicated bandwidth", "fiber connectivity"),
        ("IoT capability", "What IoT services are available?", "sensor connectivity", "refund"),
        ("mixed discount and cloud", "Can enterprise customers get a discount on cloud services?", "30% discount", "virtual machines"),
        ("mixed billing and cloud", "What happens if I don't pay my cloud service bill?", "7 days", "virtual machines"),
        ("mixed cloud package", "What services are included in the cloud package?", "virtual machines", "refund"),
        ("mixed cost and refund", "How much does the cloud service cost and can I get a refund?", "not refundable after activation", "virtual machines"),
        ("mixed product price and refund", "What is the price of ZENDCloud VM Basic in India and can I get a refund?", "$24 for individual users", "virtual machines"),
        ("internet down", "My internet is down.", "technical support", "refund"),
        ("connection issue", "My connection keeps dropping.", "troubleshooting assistance", "refund"),
        ("complaint", "I want to complain about my internet service.", "technical support", "escalation procedure"),
        ("support request", "How do I get technical support for ZENDFiber?", "technical support", "refund"),
    ],
)
def test_supported_behavior_matrix(engine: ZendsResponseEngine, category: str, query: str, expected: str, forbidden: str) -> None:
    result = apply_scope_guard(query, engine.respond(query))
    answer = result["recommended_response"].lower()
    assert result["abstention"] is False, category
    assert expected.lower() in answer, (category, answer)
    assert forbidden.lower() not in answer, (category, answer)


@pytest.mark.parametrize(
    "query",
    [
        "What is the cancellation policy?",
        "What does ZENDCloud VM Basic include?",
        "Who is the CEO of ZENDS Communications?",
        "Who is Virat Kohli?",
        "What's the weather today?",
        "Write Python code.",
    ],
)
def test_unknown_or_unsupported_questions_abstain(engine: ZendsResponseEngine, query: str) -> None:
    result = apply_scope_guard(query, engine.respond(query))
    assert result["abstention"] is True
    assert result["reason"] in {"insufficient_grounded_evidence", "out_of_scope"}
    assert "$" not in result["recommended_response"]


def test_mixed_policy_questions_do_not_claim_product_specific_terms(engine: ZendsResponseEngine) -> None:
    discount = engine.respond("Can enterprise customers get a discount on cloud services?")["recommended_response"]
    sla = engine.respond("What is the SLA for ZENDS business connectivity?")["recommended_response"]
    assert "does not specify a separate rule for cloud" in discount
    assert "does not specify a separate rule for business connectivity" in sla


def test_comparison_preserves_query_product_order(engine: ZendsResponseEngine) -> None:
    answer = engine.respond("Compare the prices of ZENDStorage 1TB and ZENDStorage 10TB in India.")["recommended_response"]
    assert answer.index("ZENDStorage 1TB") < answer.index("ZENDStorage 10TB")


def test_cancellation_does_not_inherit_a_model_billing_prediction(engine: ZendsResponseEngine) -> None:
    answer = engine.respond("What is the cancellation policy?")["recommended_response"]
    assert "monthly billing" not in answer.lower()
    assert "does not provide enough information" in answer


def test_generated_draft_must_answer_the_question() -> None:
    pricing = [{"text": "ZENDFiber Home 100 Mbps is priced at $18 in India for individual users.", "metadata": {"chunk_id": "price", "page": 2}}]
    refund = [{"text": "Refund: Full refund within 7 days if usage is less than 10%.", "metadata": {"chunk_id": "refund", "page": 3, "policy_category": "Refund"}}]
    assert not is_grounded_answer(
        "Never invent prices, policies, products, services, contracts, SLAs, refunds, discounts, privacy claims, or escalation procedures.",
        pricing, query="What is the individual price of ZENDFiber Home 100 Mbps in India?",
    )
    assert not is_grounded_answer(
        "ZENDS cloud services include virtual machines and file storage.",
        refund, query="Are cloud services refundable after activation?",
    )
