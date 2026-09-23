"""Pretrained emotion-model wrapper for ZENDS Happy/Neutral/Angry labels."""

from __future__ import annotations

from typing import Any

import numpy as np


SENTIMENT_MODEL = "j-hartmann/emotion-english-distilroberta-base"
SENTIMENT_LABELS = ("Happy", "Neutral", "Angry")

# The model directly includes anger, joy, and neutral. Remaining negative emotions
# belong with Angry; surprise is treated as Neutral for the three-label product UI.
EMOTION_TO_SENTIMENT = {
    "joy": "Happy",
    "neutral": "Neutral",
    "surprise": "Neutral",
    "anger": "Angry",
    "disgust": "Angry",
    "fear": "Angry",
    "sadness": "Angry",
}


class EmotionSentimentClassifier:
    """Aggregate a pretrained seven-emotion model into the required three labels."""

    def __init__(self, tokenizer: Any | None = None, model: Any | None = None, model_name: str = SENTIMENT_MODEL) -> None:
        self.model_name = model_name
        self._tokenizer = tokenizer
        self._model = model

    def _validate_direct_emotion_labels(self) -> None:
        labels = {str(label).lower() for label in self._model.config.id2label.values()}
        required = {"anger", "joy", "neutral"}
        if not required.issubset(labels):
            raise ValueError(f"Sentiment model lacks required emotion labels: {sorted(required - labels)}")

    def _load(self) -> None:
        if self._tokenizer is not None and self._model is not None:
            self._validate_direct_emotion_labels()
            return
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        # Prefer the local cache so normal inference needs no network metadata
        # request. A first-time user can still download the declared model.
        try:
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_name, local_files_only=True)
            self._model = AutoModelForSequenceClassification.from_pretrained(self.model_name, local_files_only=True)
        except OSError:
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self._model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
        self._validate_direct_emotion_labels()
        self._model.eval()

    def predict(self, text: str) -> dict[str, float | str]:
        self._load()
        import torch

        encoded = self._tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
        with torch.no_grad():
            probabilities = torch.softmax(self._model(**encoded).logits, dim=-1)[0].cpu().numpy()
        scores = {str(self._model.config.id2label[index]).lower(): float(probabilities[index]) for index in range(len(probabilities))}
        aggregated = {label: 0.0 for label in SENTIMENT_LABELS}
        for emotion, score in scores.items():
            if emotion not in EMOTION_TO_SENTIMENT:
                raise ValueError(f"Unexpected unmapped emotion label: {emotion}")
            aggregated[EMOTION_TO_SENTIMENT[emotion]] += score
        label = max(aggregated, key=aggregated.get)
        return {"sentiment": label, "sentiment_confidence": float(aggregated[label])}
