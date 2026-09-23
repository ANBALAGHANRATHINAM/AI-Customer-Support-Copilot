"""Combined Segment 2 inference pipeline."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from preprocessing import clean_text
from priority import detect_priority


class NLPPipeline:
    """Combine preprocessing, fine-tuned intent, pretrained sentiment, and priority rules."""

    def __init__(self, intent_classifier: Any, sentiment_classifier: Any) -> None:
        self.intent_classifier = intent_classifier
        self.sentiment_classifier = sentiment_classifier

    @classmethod
    def from_model_directory(cls, model_directory: str | Path) -> "NLPPipeline":
        from intent import IntentClassifier
        from sentiment import EmotionSentimentClassifier

        return cls(IntentClassifier(Path(model_directory)), EmotionSentimentClassifier())

    def predict(self, customer_query: str) -> dict[str, str | float]:
        cleaned_text = clean_text(customer_query)
        intent_result = self.intent_classifier.predict(cleaned_text)
        sentiment_result = self.sentiment_classifier.predict(cleaned_text)
        priority_result = detect_priority(cleaned_text, str(intent_result["intent"]), str(sentiment_result["sentiment"]))
        return {
            "cleaned_text": cleaned_text,
            "intent": str(intent_result["intent"]),
            "intent_confidence": float(intent_result["intent_confidence"]),
            "sentiment": str(sentiment_result["sentiment"]),
            "sentiment_confidence": float(sentiment_result["sentiment_confidence"]),
            "priority": priority_result.priority,
        }
