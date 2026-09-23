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

from grounding import compose_response, deterministic_evidence_answer, expected_policy_category, is_grounded_answer, select_grounded_chunks
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
        retrieved_context = self.retriever.retrieve(
            cleaned_query,
            policy_category=expected_policy_category(cleaned_query, intent),
            support_intent=intent in {"Technical", "Complaint"},
        )
        evidence = select_grounded_chunks(cleaned_query, intent, retrieved_context)
        prompt = build_response_prompt(customer_query=customer_query, analysis=analysis, chunks=evidence)
        generated = self.llm.generate(prompt)
        answer = generated if is_grounded_answer(generated, evidence) else deterministic_evidence_answer(
            customer_query, intent, evidence
        )
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
