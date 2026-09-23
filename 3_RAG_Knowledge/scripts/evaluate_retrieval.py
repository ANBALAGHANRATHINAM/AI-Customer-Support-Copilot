"""Transparent source-grounded retrieval evaluation; no answer generation."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SEGMENT = ROOT / "3_RAG_Knowledge"
sys.path.insert(0, str(SEGMENT / "code"))

from retriever import ZendsRetriever


# Each anchor is a verbatim, source-grounded phrase expected in at least one result.
EVALUATION_CASES = (
    ("Billing", "When are ZENDS bills issued and when can service be suspended?", ("Monthly billing in advance", "Late payment after 7 days")),
    ("Refund", "What is the refund rule after I start using a ZENDS service?", ("Full refund within 7 days", "usage is less than 10%")),
    ("Contract", "Is there a minimum contract for enterprise customers?", ("minimum 12-month contract",)),
    ("SLA", "What uptime SLA applies to enterprise users?", ("enterprise users 99.9% uptime",)),
    ("Privacy", "How does ZENDS protect customer data?", ("GDPR compliant", "encrypted data at rest and in transit")),
    ("Fair usage", "What is the fair usage cap for unlimited plans?", ("capped at 1TB per month",)),
    ("Support tiers", "Which support tiers does ZENDS offer?", ("Standard, Priority, and Enterprise Dedicated Support",)),
    ("Discounts", "What enterprise bulk and annual payment discounts are offered?", ("up to 30% discount", "Annual payment discount is 15%")),
    ("Product information", "What services are included with mobile connectivity?", ("voice calling", "SIM and eSIM provisioning")),
    ("Pricing", "What is the USA individual price for ZENDCloud VM Basic?", ("ZENDCloud VM Basic is priced at $40 in the USA",)),
    ("Country product information", "What is Prepaid Basic priced at in India for individual users?", ("$36 in India",)),
    ("Customer type", "How does ZENDOffice Net 200 pricing differ for enterprise customers?", ("$55 in the USA", "$33 in India")),
)


def main() -> None:
    retriever = ZendsRetriever(SEGMENT / "vector_db")
    results = []
    for topic, query, anchors in EVALUATION_CASES:
        chunks = retriever.retrieve(query)
        retrieved_text = " ".join(chunk["text"] for chunk in chunks).lower()
        hit = all(anchor.lower() in retrieved_text for anchor in anchors)
        results.append({"topic": topic, "query": query, "expected_source_phrases": list(anchors), "retrieved": chunks, "relevant_information_retrieved": hit})
    report = {"metric": "Hit@4: every expected source phrase for a test case occurs in its top four retrieved chunks.", "cases": results, "hit_at_4": sum(case["relevant_information_retrieved"] for case in results) / len(results), "successful_cases": sum(case["relevant_information_retrieved"] for case in results), "total_cases": len(results)}
    output = SEGMENT / "evaluation" / "retrieval_evaluation.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({key: report[key] for key in ("metric", "hit_at_4", "successful_cases", "total_cases")}, indent=2))


if __name__ == "__main__":
    main()
