"""Verify pretrained model labels and run representative ZENDS-flavoured examples."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SEGMENT_ROOT = ROOT / "2_NLP_Intelligence"
sys.path.insert(0, str(SEGMENT_ROOT / "code"))

from sentiment import EmotionSentimentClassifier, SENTIMENT_MODEL


def main() -> None:
    classifier = EmotionSentimentClassifier()
    examples = {
        "Happy": "I am delighted with the helpful ZENDS support. Thank you so much!",
        "Neutral": "Could you explain the billing rule for Postpaid Gold in USA?",
        "Angry": "I have already tried to get an answer and this complaint remains unresolved. Please escalate it.",
    }
    results = {expected: {"text": text, **classifier.predict(text)} for expected, text in examples.items()}
    native_labels = [classifier._model.config.id2label[index] for index in sorted(classifier._model.config.id2label)]
    path = SEGMENT_ROOT / "evaluation" / "sentiment_verification.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"model": SENTIMENT_MODEL, "native_labels": native_labels, "required_direct_emotions": ["joy", "neutral", "anger"], "aggregation": {"Happy": ["joy"], "Neutral": ["neutral", "surprise"], "Angry": ["anger", "disgust", "fear", "sadness"]}, "examples": results}, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
