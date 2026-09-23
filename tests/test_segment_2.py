"""Unit tests for the isolated Segment 2 NLP Intelligence layer."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "2_NLP_Intelligence" / "code"))

from intent import INTENT_LABELS, IntentClassifier
from pipeline import NLPPipeline
from preprocessing import clean_text
from priority import VALID_PRIORITIES, detect_priority
from sentiment import EMOTION_TO_SENTIMENT, SENTIMENT_LABELS, EmotionSentimentClassifier


class DummyTokenizer:
    def __call__(self, text: str, **_: object) -> dict[str, torch.Tensor]:
        assert isinstance(text, str)
        return {"input_ids": torch.tensor([[1, 2, 3]])}


class DummyModel:
    def __init__(self, labels: dict[int, str], logits: list[float]) -> None:
        self.config = SimpleNamespace(id2label=labels)
        self._logits = torch.tensor([logits], dtype=torch.float)

    def eval(self) -> "DummyModel":
        return self

    def __call__(self, **_: object) -> SimpleNamespace:
        return SimpleNamespace(logits=self._logits)


class StubIntent:
    def predict(self, _: str) -> dict[str, float | str]:
        return {"intent": "Technical", "intent_confidence": 0.91}


class StubSentiment:
    def predict(self, _: str) -> dict[str, float | str]:
        return {"sentiment": "Angry", "sentiment_confidence": 0.87}


def test_preprocessing_preserves_business_terms_numbers_and_case() -> None:
    assert clean_text("  ZENDS Prepaid Plus costs $80\n in India.  ") == "ZENDS Prepaid Plus costs $80 in India."


@pytest.mark.parametrize("value, error", [("   \t", ValueError), (None, TypeError), (123, TypeError)])
def test_preprocessing_rejects_empty_or_invalid_queries(value: object, error: type[Exception]) -> None:
    with pytest.raises(error):
        clean_text(value)  # type: ignore[arg-type]


def test_intent_inference_returns_one_of_the_five_labels() -> None:
    labels = {index: label for index, label in enumerate(INTENT_LABELS)}
    classifier = IntentClassifier("unused", tokenizer=DummyTokenizer(), model=DummyModel(labels, [0.1, 0.2, 4.0, 0.3, 0.4]))
    result = classifier.predict("My connection keeps dropping.")
    assert result["intent"] == "Technical"
    assert result["intent"] in INTENT_LABELS
    assert 0.0 < float(result["intent_confidence"]) <= 1.0


def test_unambiguous_low_confidence_billing_boundary_is_transparent() -> None:
    labels = {index: label for index, label in enumerate(INTENT_LABELS)}
    classifier = IntentClassifier("unused", tokenizer=DummyTokenizer(), model=DummyModel(labels, [0.1, 0.2, 0.3, 0.4, 0.5]))
    result = classifier.predict("How does ZENDS billing work?")
    assert result["intent"] == "Billing"
    assert 0.0 < float(result["intent_confidence"]) < 1.0
    assert classifier.last_decision["transformer_intent"] == "Product Inquiry"
    assert classifier.last_decision["transformer_confidence"] > float(result["intent_confidence"])
    assert classifier.last_decision["fallback_applied"] is True
    assert classifier.last_decision["fallback_intent"] == "Billing"
    assert classifier.last_decision["matched_phrase"] == "billing"
    assert classifier.last_decision["decision_source"] == "boundary_rule"
    assert classifier.last_decision["selected_confidence"] == pytest.approx(float(result["intent_confidence"]))


@pytest.mark.parametrize(
    ("query", "expected"),
    [
        ("I am having trouble with ZENDFiber.", "Technical"),
        ("I'm having trouble with my internet.", "Technical"),
        ("My ZENDFiber connection is having problems.", "Technical"),
        ("My internet connection keeps failing.", "Technical"),
        ("I want to complain about ZENDFiber.", "Complaint"),
        ("I am unhappy with ZENDFiber service.", "Complaint"),
        ("There is a problem with my billing.", "Billing"),
        ("I want a refund for my plan.", "Refund"),
        ("What mobile plans are available?", "Product Inquiry"),
    ],
)
def test_low_confidence_intent_boundaries_preserve_explicit_meaning(query: str, expected: str) -> None:
    labels = {index: label for index, label in enumerate(INTENT_LABELS)}
    classifier = IntentClassifier("unused", tokenizer=DummyTokenizer(), model=DummyModel(labels, [0.0, 0.0, 0.0, 0.4, 0.0]))
    result = classifier.predict(query)
    assert result["intent"] == expected


def test_rule_derived_technical_confidence_is_the_technical_probability() -> None:
    labels = {index: label for index, label in enumerate(INTENT_LABELS)}
    logits = torch.tensor([0.0, 0.0, 0.2, 0.8, 0.0])
    classifier = IntentClassifier("unused", tokenizer=DummyTokenizer(), model=DummyModel(labels, logits.tolist()))
    result = classifier.predict("I am having trouble with ZENDFiber.")
    expected_technical_probability = float(torch.softmax(logits, dim=0)[2])
    assert result == {"intent": "Technical", "intent_confidence": pytest.approx(expected_technical_probability)}
    assert classifier.last_decision["decision_source"] == "boundary_rule"


def test_saved_intent_model_loads_with_expected_label_mapping() -> None:
    classifier = IntentClassifier(ROOT / "2_NLP_Intelligence" / "models" / "intent_distilbert")
    classifier._load()
    assert tuple(classifier._model.config.id2label[index] for index in range(len(INTENT_LABELS))) == INTENT_LABELS


def test_sentiment_inference_aggregates_direct_emotion_labels() -> None:
    labels = {0: "anger", 1: "disgust", 2: "fear", 3: "joy", 4: "neutral", 5: "sadness", 6: "surprise"}
    classifier = EmotionSentimentClassifier(tokenizer=DummyTokenizer(), model=DummyModel(labels, [0.1, 0.1, 0.1, 4.0, 0.3, 0.1, 0.1]))
    result = classifier.predict("Thank you, that was very helpful.")
    assert result["sentiment"] == "Happy"
    assert result["sentiment"] in SENTIMENT_LABELS
    assert 0.0 < float(result["sentiment_confidence"]) <= 1.0
    assert {"joy", "neutral", "anger"}.issubset(EMOTION_TO_SENTIMENT)


def test_priority_rules_cover_service_impact_escalation_and_baseline() -> None:
    assert detect_priority("My service is down and blocking my work.", "Technical", "Angry").priority == "High"
    assert detect_priority("This is unresolved; escalate this complaint.", "Complaint", "Angry").priority == "High"
    assert detect_priority("Could you check why my bill changed?", "Billing", "Neutral").priority == "Low"
    assert detect_priority("Please reply today.", "Product Inquiry", "Neutral").priority == "Medium"


def test_complete_pipeline_has_required_schema_and_valid_labels() -> None:
    result = NLPPipeline(StubIntent(), StubSentiment()).predict("  Service is down; please help. ")
    assert tuple(result) == (
        "cleaned_text", "intent", "intent_confidence", "sentiment", "sentiment_confidence", "priority"
    )
    assert result["cleaned_text"] == "Service is down; please help."
    assert result["intent"] in INTENT_LABELS
    assert result["sentiment"] in SENTIMENT_LABELS
    assert result["priority"] in VALID_PRIORITIES


def test_pipeline_rejects_empty_input_before_model_inference() -> None:
    with pytest.raises(ValueError):
        NLPPipeline(StubIntent(), StubSentiment()).predict("\n  ")


def test_realistic_generalization_set_is_balanced_and_separate_from_hard_boundary_training() -> None:
    segment = ROOT / "2_NLP_Intelligence"
    training = pd.read_csv(segment / "data" / "intent_hard_boundary_train.csv")
    evaluation = pd.read_csv(segment / "evaluation" / "realistic_generalization.csv")
    assert set(training["intent"]) == set(INTENT_LABELS)
    assert set(evaluation["intent"]) == set(INTENT_LABELS)
    assert not set(training["text"]) & set(evaluation["text"])
    assert evaluation.groupby("intent").size().to_dict() == {label: 20 for label in INTENT_LABELS}
