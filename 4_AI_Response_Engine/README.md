# Segment 4 - AI Response Engine

Segment 4 orchestrates the locked Segment 2 NLP pipeline and locked Segment 3 Chroma retriever. It does not retrain or replace either component.

## Flow

`customer query -> intent/sentiment/priority -> top-4 ZENDS chunks -> prompt -> pretrained LLM tone -> grounded recommendation`

- Intent, sentiment, confidence, and priority are returned by Segment 2 unchanged.
- The existing Segment 3 retriever returns source text, `chunk_id`, page, optional policy category, and Chroma distance unchanged.
- The default language-model adapter lazily loads `google/flan-t5-small`, a pretrained instruction-tuned Hugging Face model. It is not trained by this project.
- LangChain is intentionally not used: the existing Chroma wrapper and a small explicit prompt interface already provide the required orchestration without another abstraction layer.

## Grounding and safety

The prompt prohibits unsupported ZENDS facts. The LLM is limited to a concise, fact-free acknowledgement, and its output is accepted only when it exactly matches a small allowlist of safe acknowledgements. The response engine appends one verbatim source chunk selected through meaningful query terms, an exact named product, matching Billing/Refund source-policy category, or retrieved troubleshooting context for a Technical/Complaint prediction. When that evidence test fails, the engine explicitly abstains: `The available ZENDS knowledge does not provide enough information to answer this question.`

This prevents the LLM from manufacturing company facts. It also means that the known broad privacy query may abstain: Segment 3 currently ranks its exact `Data Privacy` chunk outside the top four, and Segment 4 must not fill that gap with an invented GDPR or encryption claim.

Happy, Neutral, and Angry sentiment alter the acknowledgement tone. High priority adds only a generic urgency acknowledgement; it does not invent a ZENDS escalation procedure.

## Run

From the project root:

```powershell
# Pretrained FLAN-T5 response generation (downloads the model on first use)
python 4_AI_Response_Engine/scripts/run_response_engine.py "What is the refund policy?"

# Deterministic, no-download demonstration mode
python 4_AI_Response_Engine/scripts/run_response_engine.py --offline "What is the refund policy?"

# Twelve grounding scenarios, without making LLM output affect repeatability
python 4_AI_Response_Engine/scripts/evaluate_response_engine.py --offline
```

## Limitations

- FLAN-T5 Small is a compact local model; the evidence layer, not the LLM, is the factual safety boundary.
- Only facts retrieved by Segment 3 can be used. Retrieval misses therefore lead to abstention, not a fabricated answer.
- Segment 4 creates recommendations for agent review; it does not implement an end-user UI, deployment, or escalation workflow.
