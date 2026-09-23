"""Fine-tuned DistilBERT intent inference and label configuration."""

from __future__ import annotations

import os
import logging
import re
from pathlib import Path
from typing import Any

import numpy as np


# Segment 2 uses PyTorch models exclusively.  These are set while this module
# is imported, before Transformers can resolve optional backend availability.
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_TORCH", "1")


INTENT_LABELS = ("Billing", "Refund", "Technical", "Complaint", "Product Inquiry")
BASE_INTENT_MODEL = "distilbert-base-uncased"
BOUNDARY_FALLBACK_THRESHOLD = 0.50
BOUNDARY_RULES = (
    ("Refund", ("refund", "money back", "reimbursement")),
    ("Billing", ("billing", "bill", "charged", "charge", "payment", "invoice")),
    ("Complaint", ("complaint", "complain", "unhappy", "disappointed", "unresolved problem")),
    ("Technical", ("not working", "connection down", "keeps dropping", "troubleshoot", "troubleshooting", "internet stopped")),
    ("Product Inquiry", ("plans", "plan", "what do you offer", "available", "options")),
)
TECHNICAL_SERVICE_TERMS = (
    "broadband", "connection", "connectivity", "fiber", "internet", "network", "wifi", "zendfiber",
)
OPERATIONAL_DIFFICULTY_TERMS = (
    "cannot connect", "can't connect", "connection down", "disconnecting", "failing", "keeps dropping",
    "keeps failing", "not working", "offline", "outage", "problem", "problems", "slow", "stopped", "trouble",
)
LOGGER = logging.getLogger(__name__)


def _match_phrase(normalized: str, phrases: tuple[str, ...]) -> str | None:
    return next(
        (phrase for phrase in phrases if re.search(rf"(?<!\w){re.escape(phrase)}(?!\w)", normalized)),
        None,
    )


def _boundary_intent(normalized: str) -> tuple[str | None, str | None]:
    """Resolve explicit boundaries, then narrow service-operation ambiguity."""
    # Explicit complaint language must win over a coincident technical symptom.
    complaint_phrases = next(phrases for label, phrases in BOUNDARY_RULES if label == "Complaint")
    complaint_match = _match_phrase(normalized, complaint_phrases)
    if complaint_match:
        return "Complaint", complaint_match

    for candidate, phrases in BOUNDARY_RULES:
        if candidate in {"Complaint", "Technical"}:
            continue
        matched = _match_phrase(normalized, phrases)
        if matched:
            return candidate, matched

    service_match = _match_phrase(normalized, TECHNICAL_SERVICE_TERMS)
    difficulty_match = _match_phrase(normalized, OPERATIONAL_DIFFICULTY_TERMS)
    named_zends_service = bool(re.search(r"(?<!\w)zend[a-z0-9]+(?!\w)", normalized))
    if difficulty_match and (service_match or named_zends_service):
        return "Technical", difficulty_match

    technical_phrases = next(phrases for label, phrases in BOUNDARY_RULES if label == "Technical")
    technical_match = _match_phrase(normalized, technical_phrases)
    if technical_match:
        return "Technical", technical_match
    return None, None


class IntentClassifier:
    """Lazy inference wrapper around the saved fine-tuned DistilBERT model."""

    def __init__(self, model_path: str | Path, tokenizer: Any | None = None, model: Any | None = None) -> None:
        self.model_path = Path(model_path)
        self._tokenizer = tokenizer
        self._model = model
        self._device: Any | None = None
        self.last_decision: dict[str, Any] | None = None

    def _load(self) -> None:
        if self._tokenizer is not None and self._model is not None:
            return
        if self.model_path.exists():
            model_source = self.model_path
        else:
            model_source = "Anbalaghan04/zends-intent-distilbert"

        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        # Keep inference deterministic and fully materialized.  Explicitly opting
        # out of memory-saving dispatch prevents meta-device parameters from
        # reaching the Streamlit inference path.
        
        model_source = (
            self.model_path
            if Path(self.model_path).exists()
            else "Anbalaghan04/zends-intent-distilbert"
        )

        self._model = AutoModelForSequenceClassification.from_pretrained(
            model_source,
            device_map=None,
            low_cpu_mem_usage=False,
            torch_dtype=torch.float32,
            local_files_only=False,
        )
        meta_parameters = [name for name, parameter in self._model.named_parameters() if parameter.is_meta]
        if meta_parameters:
            raise RuntimeError(
                "The saved intent model contains unmaterialized parameters: "
                + ", ".join(meta_parameters[:3])
            )
        self._device = torch.device("cpu")
        self._model.to(self._device)
        self._model.eval()
        self._tokenizer = AutoTokenizer.from_pretrained(
            model_source,
            local_files_only=False,
        )

    def predict(self, text: str) -> dict[str, float | str]:
        self._load()
        import torch

        encoded = self._tokenizer(text, return_tensors="pt", truncation=True, max_length=64)
        if self._device is not None:
            encoded = {name: value.to(self._device) for name, value in encoded.items()}
        with torch.no_grad():
            probabilities = torch.softmax(self._model(**encoded).logits, dim=-1)[0].cpu().numpy()
        index = int(np.argmax(probabilities))
        transformer_label = str(self._model.config.id2label[index])
        transformer_confidence = float(probabilities[index])
        normalized = " ".join(text.lower().split())
        fallback_label, matched_phrase = _boundary_intent(normalized)
        # A rule may correct only a conflicting, low-confidence prediction.
        # Matching transformer predictions always remain primary.
        use_fallback = (
            fallback_label is not None
            and fallback_label != transformer_label
            and transformer_confidence < BOUNDARY_FALLBACK_THRESHOLD
        )
        label = fallback_label if use_fallback else transformer_label
        label_to_index = {str(value): int(key) for key, value in self._model.config.id2label.items()}
        selected_confidence = float(probabilities[label_to_index[label]])
        self.last_decision = {
            "transformer_intent": transformer_label,
            "transformer_confidence": transformer_confidence,
            "fallback_applied": use_fallback,
            "fallback_intent": fallback_label if use_fallback else None,
            "matched_phrase": matched_phrase if use_fallback else None,
            "decision_source": "boundary_rule" if use_fallback else "transformer",
            "selected_confidence": selected_confidence,
        }
        if use_fallback:
            LOGGER.info("Applied intent boundary fallback: %s", self.last_decision)
        return {"intent": str(label), "intent_confidence": selected_confidence}
