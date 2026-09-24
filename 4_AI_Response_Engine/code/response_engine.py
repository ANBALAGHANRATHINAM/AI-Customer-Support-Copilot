"""Segment 4 orchestration: Segment 2 analysis + Segment 3 retrieval + LLM tone."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
for dependency_path in (ROOT / "2_NLP_Intelligence" / "code", ROOT / "3_RAG_Knowledge" / "code"):
    if str(dependency_path) not in sys.path:
        sys.path.insert(0, str(dependency_path))

from pipeline import NLPPipeline
from retriever import ZendsRetriever

from grounding import (
    compose_response,
    deterministic_evidence_answer,
    deterministic_policy_answer,
    deterministic_pricing_answer,
    deterministic_product_feature_answer,
    deterministic_product_list_answer,
    expected_policy_category,
    is_grounded_answer,
    policy_scope_note,
    resolve_query_request,
    select_grounded_chunks,
)
from llm import HuggingFaceInstructionLLM, InstructionLLM
from prompting import build_response_prompt


class ZendsResponseEngine:
    """Create traceable, source-grounded recommended responses without changing prior segments."""

    def __init__(self, nlp_pipeline: Any, retriever: Any, llm: InstructionLLM) -> None:
        self.nlp_pipeline = nlp_pipeline
        self.retriever = retriever
        self.llm = llm

    @classmethod
    def from_project_assets(cls, *, llm: InstructionLLM | None = None) -> "ZendsResponseEngine":
        database_path = ROOT / "3_RAG_Knowledge" / "vector_db"

        retriever = ZendsRetriever(database_path)

        if retriever.store.count() == 0:
            from build import build_knowledge_base

            build_knowledge_base(
                ROOT / "docs" / "ZENDS Communications.pdf",
                database_path,
            )
            retriever = ZendsRetriever(database_path)

        return cls(
            NLPPipeline.from_model_directory(
                 ROOT / "2_NLP_Intelligence" / "models" / "intent_distilbert"
            ),
            retriever,
            llm or HuggingFaceInstructionLLM(),
        )
    def respond(self, customer_query: str) -> dict[str, Any]:
        if not isinstance(customer_query, str):
            raise TypeError("Customer query must be a string.")
        analysis = self.nlp_pipeline.predict(customer_query)
        cleaned_query = str(analysis["cleaned_text"])
        intent = str(analysis["intent"])
        policy_category = expected_policy_category(cleaned_query, intent)
        retrieved_context = self.retriever.retrieve(
            cleaned_query,
            policy_category=policy_category,
            support_intent=intent in {"Technical", "Complaint"},
        )
        source_records = self.retriever.source_records() if hasattr(self.retriever, "source_records") else retrieved_context
        request = resolve_query_request(cleaned_query, intent, source_records)
        if request.support and intent not in {"Technical", "Complaint"}:
            retrieved_context = self.retriever.retrieve(cleaned_query, support_intent=True)
        exact_products = list(dict.fromkeys((*request.pricing_products, *request.feature_products)))
        if exact_products and hasattr(self.retriever, "retrieve_exact_products"):
            exact = self.retriever.retrieve_exact_products(exact_products)
            seen = {str(chunk["metadata"].get("chunk_id")) for chunk in retrieved_context}
            retrieved_context += [chunk for chunk in exact if str(chunk["metadata"].get("chunk_id")) not in seen]
        product_list_evidence = self.retriever.retrieve_group_products(request.product_list_group) if request.product_list_group and hasattr(self.retriever, "retrieve_group_products") else []
        if product_list_evidence:
            seen = {str(chunk["metadata"].get("chunk_id")) for chunk in retrieved_context}
            retrieved_context += [chunk for chunk in product_list_evidence if str(chunk["metadata"].get("chunk_id")) not in seen]
        capability_evidence = self.retriever.retrieve_group_capabilities(request.capability_group) if request.capability_group and hasattr(self.retriever, "retrieve_group_capabilities") else []
        if capability_evidence:
            seen = {str(chunk["metadata"].get("chunk_id")) for chunk in retrieved_context}
            retrieved_context += [chunk for chunk in capability_evidence if str(chunk["metadata"].get("chunk_id")) not in seen]
        evidence_intent = (intent if intent in {"Technical", "Complaint"} else "Technical") if request.support else intent
        evidence = select_grounded_chunks(cleaned_query, evidence_intent, retrieved_context)
        prompt = build_response_prompt(customer_query=customer_query, analysis=analysis, chunks=evidence)
        generated = self.llm.generate(prompt)
        if request.policy_category:
            parts = [deterministic_policy_answer(request.policy_category, retrieved_context)]
            if parts[0]:
                parts.append(policy_scope_note(customer_query, request.policy_category, retrieved_context))
            if request.pricing_products:
                parts.append(deterministic_pricing_answer(customer_query, retrieved_context))
            elif request.asks_price:
                parts.append("A specific product and country are needed to quote a price.")
            if request.unsupported_policy:
                parts.append("The available ZENDS information does not specify a cancellation policy.")
            answer = " ".join(part for part in parts if part) or None
        elif request.unsupported_policy:
            answer = None
        elif request.pricing_products:
            answer = deterministic_pricing_answer(customer_query, retrieved_context)
        elif request.feature_products:
            answer = deterministic_product_feature_answer(customer_query, retrieved_context)
        elif product_list_evidence:
            answer = deterministic_product_list_answer(request.product_list_group, product_list_evidence)
        elif capability_evidence:
            answer = deterministic_evidence_answer(customer_query, "Product Inquiry", capability_evidence)
        elif request.support:
            answer = deterministic_evidence_answer(customer_query, "Technical", evidence)
        else:
            answer = deterministic_evidence_answer(customer_query, intent, evidence)
            if answer is None and is_grounded_answer(generated, evidence, query=customer_query):
                answer = generated
        abstained = answer is None
        response = compose_response(
            sentiment=str(analysis["sentiment"]),
            priority=str(analysis["priority"]),
            answer=answer,
        )
        return {
            "customer_query": customer_query,
            "intent": analysis["intent"],
            "intent_confidence": analysis["intent_confidence"],
            "sentiment": analysis["sentiment"],
            "sentiment_confidence": analysis["sentiment_confidence"],
            "priority": analysis["priority"],
            "retrieved_context": retrieved_context,
            "recommended_response": response,
            "abstention": abstained,
            "reason": "insufficient_grounded_evidence" if abstained else "grounded_synthesis",
        }
