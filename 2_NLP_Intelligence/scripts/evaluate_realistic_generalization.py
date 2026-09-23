"""Evaluate the saved Segment 2 model on held-out natural-language examples."""

from __future__ import annotations

import json
import sys
import argparse
from pathlib import Path

import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_recall_fscore_support

ROOT = Path(__file__).resolve().parents[2]
SEGMENT_ROOT = ROOT / "2_NLP_Intelligence"
sys.path.insert(0, str(SEGMENT_ROOT / "code"))

from intent import INTENT_LABELS, IntentClassifier
from preprocessing import clean_text


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a saved model on the held-out realistic generalization set.")
    parser.add_argument("--model-dir", type=Path, default=SEGMENT_ROOT / "models" / "intent_distilbert")
    parser.add_argument("--output", type=Path, default=SEGMENT_ROOT / "evaluation" / "realistic_generalization_metrics.json")
    args = parser.parse_args()
    data = pd.read_csv(SEGMENT_ROOT / "evaluation" / "realistic_generalization.csv")
    classifier = IntentClassifier(args.model_dir)
    expected = data["intent"].tolist()
    predicted = [str(classifier.predict(clean_text(text))["intent"]) for text in data["text"]]
    precision, recall, f1, _ = precision_recall_fscore_support(expected, predicted, labels=INTENT_LABELS, average="weighted", zero_division=0)
    report = {
        "accuracy": float(accuracy_score(expected, predicted)),
        "precision_weighted": float(precision),
        "recall_weighted": float(recall),
        "f1_weighted": float(f1),
        "examples": int(len(data)),
        "classification_report": classification_report(expected, predicted, labels=INTENT_LABELS, output_dict=True, zero_division=0),
        "confusion_matrix": confusion_matrix(expected, predicted, labels=INTENT_LABELS).tolist(),
        "labels": list(INTENT_LABELS),
    }
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({key: report[key] for key in ("accuracy", "precision_weighted", "recall_weighted", "f1_weighted", "examples")}, indent=2))


if __name__ == "__main__":
    main()
