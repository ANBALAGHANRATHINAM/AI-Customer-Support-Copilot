"""Tests for the Segment 5 presentation-only Streamlit integration."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "5_Streamlit_Integration" / "code"))

from ui_helpers import ABSTENTION_TEXT, OUT_OF_SCOPE_RESPONSE, apply_scope_guard, display_response, format_confidence, is_abstention, is_unrelated_to_retrieved_knowledge, priority_style


def test_presentation_helpers_format_confidence_priority_and_abstention() -> None:
    assert format_confidence(0.942) == "94.2%"
    assert priority_style("High") == ("priority-high", "HIGH")
    assert priority_style("Medium") == ("priority-medium", "MEDIUM")
    assert priority_style("Low") == ("priority-low", "LOW")
    assert is_abstention("The available ZENDS knowledge does not provide enough information to answer this question.")
    assert not is_abstention("Based on the available ZENDS information: Refund: Full refund within 7 days.")


def test_scope_guard_is_independent_of_response_generation() -> None:
    abstained = {"recommended_response": f"{ABSTENTION_TEXT} to answer this question.", "retrieved_context": [{"text": "ZENDS Communications provides ZENDFiber services."}]}
    incorrectly_answered = {
        "recommended_response": "ZENDS provides troubleshooting assistance.",
        "retrieved_context": [{"text": "Services include priority troubleshooting and support."}],
        "abstention": False,
        "reason": "grounded_synthesis",
    }
    assert is_unrelated_to_retrieved_knowledge("Who is Virat Kohli?", abstained)
    assert is_unrelated_to_retrieved_knowledge("What is the capital of France?", abstained)
    assert is_unrelated_to_retrieved_knowledge("Write me a Python program.", incorrectly_answered)
    assert is_unrelated_to_retrieved_knowledge("Create a sorting algorithm.", incorrectly_answered)
    assert not is_unrelated_to_retrieved_knowledge("Who is the CEO of ZENDS Communications?", abstained)
    assert not is_unrelated_to_retrieved_knowledge("What is the ZENDFiber service?", abstained)
    assert not is_unrelated_to_retrieved_knowledge("My internet is not working.", abstained)
    assert not is_unrelated_to_retrieved_knowledge("My office internet is down.", abstained)
    assert not is_unrelated_to_retrieved_knowledge("I want to complain about the service.", abstained)
    assert not is_unrelated_to_retrieved_knowledge("What cloud services are available?", abstained)
    assert not is_unrelated_to_retrieved_knowledge("What is the price of Prepaid Basic in India?", abstained)
    assert not is_unrelated_to_retrieved_knowledge("How much does Postpaid Gold cost in the USA?", abstained)
    invoice_result = {
        "recommended_response": "Enterprise customers receive consolidated invoices.",
        "retrieved_context": [{
            "text": "Billing: Monthly billing in advance. Enterprise customers receive consolidated invoices.",
            "metadata": {"policy_category": "Billing"},
        }],
    }
    assert not is_unrelated_to_retrieved_knowledge("Do enterprise customers receive consolidated invoices?", invoice_result)
    assert is_unrelated_to_retrieved_knowledge("How do I write Python code to process invoices?", invoice_result)
    assert is_unrelated_to_retrieved_knowledge("What are invoices?", invoice_result)
    assert OUT_OF_SCOPE_RESPONSE == "I'm here to help with ZENDS Communications services—how can I assist you today?"
    assert apply_scope_guard("Who is Virat Kohli?", abstained)["recommended_response"] == OUT_OF_SCOPE_RESPONSE
    guarded = apply_scope_guard("Write me a Python program.", incorrectly_answered)
    assert guarded["recommended_response"] == OUT_OF_SCOPE_RESPONSE
    assert guarded["abstention"] is True
    assert guarded["reason"] == "out_of_scope"
    assert apply_scope_guard("Who is the CEO of ZENDS Communications?", abstained) == abstained
    assert apply_scope_guard("My internet is not working.", abstained) == abstained


def test_user_facing_abstention_preserves_internal_grounding_state() -> None:
    result = {
        "recommended_response": "The available ZENDS knowledge does not provide enough information to answer this question.",
        "abstention": True,
        "reason": "insufficient_grounded_evidence",
    }
    assert display_response(result) == OUT_OF_SCOPE_RESPONSE
    assert result["recommended_response"].startswith(ABSTENTION_TEXT)
    assert result["reason"] == "insufficient_grounded_evidence"
    assert display_response({"recommended_response": "Enterprise customers receive consolidated invoices.", "abstention": False}) == "Enterprise customers receive consolidated invoices."


def test_streamlit_entrypoint_reuses_segment_four_without_duplicate_ai_logic() -> None:
    source = (ROOT / "5_Streamlit_Integration" / "app.py").read_text(encoding="utf-8")
    assert "ZendsResponseEngine.from_project_assets" in source
    assert "StaticAcknowledgementLLM" in source
    assert "llm=StaticAcknowledgementLLM()" in source
    assert ".respond(query)" in source
    assert "except Exception" not in source
    assert "from pipeline import" not in source
    assert "from retriever import" not in source
    assert "C:\\Users\\" not in source


def test_presentation_helpers_import_in_a_fresh_process() -> None:
    command = [sys.executable, "-c", "import sys; sys.path.insert(0, r'5_Streamlit_Integration/code'); import ui_helpers; print(ui_helpers.format_confidence(0.5))"]
    result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, check=True)
    assert result.stdout.strip() == "50.0%"


def test_streamlit_app_loads_without_running_models() -> None:
    from streamlit.testing.v1 import AppTest

    app = AppTest.from_file(ROOT / "5_Streamlit_Integration" / "app.py").run()
    assert not app.exception
    assert any("AI Customer Support Copilot" in heading.value for heading in app.markdown)
