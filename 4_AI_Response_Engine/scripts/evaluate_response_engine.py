"""Evaluate Segment 4 grounding behavior with the existing Segment 2 and 3 assets."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "4_AI_Response_Engine" / "code"))

from llm import StaticAcknowledgementLLM
from response_engine import ZendsResponseEngine


EVALUATION_CASES = (
    ("Billing", "When are ZENDS bills issued and when can service be suspended?"),
    ("Refund", "Can I receive a refund within seven days if I used less than 10%?"),
    ("Technical", "My ZENDFiber connection is down and I need troubleshooting help."),
    ("Complaint", "I am angry because my internet is down and blocking my work; please escalate this."),
    ("Product Inquiry", "What services are included with mobile connectivity?"),
    ("Product pricing", "What is the USA individual price for ZENDCloud VM Basic?"),
    ("Country pricing", "What is Prepaid Basic priced at in India for individual users?"),
    ("Enterprise", "How does ZENDOffice Net 200 pricing differ for enterprise customers?"),
    ("SLA", "What uptime SLA applies to enterprise users?"),
    ("Privacy", "How does ZENDS protect customer data?"),
    ("Fair usage", "What is the fair usage cap for unlimited plans?"),
    ("Unsupported", "Who is the ZENDS chief executive officer?"),
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate source grounding for Segment 4.")
    parser.add_argument("--offline", action="store_true", help="Use deterministic acknowledgement generation for reproducible evaluation.")
    args = parser.parse_args()
    engine = ZendsResponseEngine.from_project_assets(llm=StaticAcknowledgementLLM() if args.offline else None)
    results = []
    for topic, query in EVALUATION_CASES:
        result = engine.respond(query)
        results.append(
            {
                "topic": topic,
                "query": query,
                "intent": result["intent"],
                "sentiment": result["sentiment"],
                "priority": result["priority"],
                "retrieved_chunk_ids": [chunk["metadata"]["chunk_id"] for chunk in result["retrieved_context"]],
                "recommended_response": result["recommended_response"],
                "abstained": "available ZENDS knowledge does not provide enough information" in result["recommended_response"],
            }
        )
    output = ROOT / "4_AI_Response_Engine" / "evaluation" / "response_engine_evaluation.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({"llm_mode": "offline-safe" if args.offline else "pretrained-flan-t5-small", "cases": results}, indent=2), encoding="utf-8")
    print(json.dumps({"cases": len(results), "output": str(output)}, indent=2))


if __name__ == "__main__":
    main()
