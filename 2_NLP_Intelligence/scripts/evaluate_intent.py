"""Evaluate the saved Segment 2 intent model on the held-out test split only."""

from __future__ import annotations

import json
import sys
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_recall_fscore_support

ROOT = Path(__file__).resolve().parents[2]
SEGMENT_ROOT = ROOT / "2_NLP_Intelligence"
sys.path.insert(0, str(SEGMENT_ROOT / "code"))

from intent import INTENT_LABELS
from preprocessing import clean_text


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True), encoding="utf-8")


def main() -> None:
    import torch
    from torch.utils.data import DataLoader
    from transformers import AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding

    parser = argparse.ArgumentParser(description="Evaluate a saved Segment 2 intent model on the untouched test split.")
    parser.add_argument("--model-dir", type=Path, default=SEGMENT_ROOT / "models" / "intent_distilbert")
    parser.add_argument("--output-prefix", default="intent_test")
    args = parser.parse_args()
    model_dir = args.model_dir
    if not model_dir.exists():
        raise FileNotFoundError(f"Fine-tuned intent model is missing: {model_dir}")
    test = pd.read_csv(ROOT / "1_Data_Foundation" / "data" / "test.csv")
    label_to_id = {label: index for index, label in enumerate(INTENT_LABELS)}
    labels = test["intent"].map(label_to_id)
    if labels.isna().any():
        raise ValueError("The held-out test split contains an unknown intent label.")
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    model.eval()
    encoded = tokenizer(test["text"].map(clean_text).tolist(), truncation=True, max_length=64)
    examples = [{key: value[index] for key, value in encoded.items()} for index in range(len(test))]
    loader = DataLoader(examples, batch_size=256, collate_fn=DataCollatorWithPadding(tokenizer=tokenizer))
    predictions: list[int] = []
    with torch.no_grad():
        for batch in loader:
            predictions.extend(torch.argmax(model(**batch).logits, dim=-1).tolist())
    actual = labels.astype(int).to_numpy()
    predicted = np.asarray(predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(actual, predicted, average="weighted", zero_division=0)
    metrics = {
        "accuracy": float(accuracy_score(actual, predicted)),
        "precision_weighted": float(precision),
        "recall_weighted": float(recall),
        "f1_weighted": float(f1),
        "test_examples": int(len(actual)),
    }
    evaluation = SEGMENT_ROOT / "evaluation"
    _write_json(evaluation / f"{args.output_prefix}_metrics.json", metrics)
    _write_json(evaluation / f"{args.output_prefix}_classification_report.json", classification_report(actual, predicted, labels=list(range(len(INTENT_LABELS))), target_names=list(INTENT_LABELS), output_dict=True, zero_division=0))
    pd.DataFrame(confusion_matrix(actual, predicted, labels=list(range(len(INTENT_LABELS)))), index=INTENT_LABELS, columns=INTENT_LABELS).to_csv(evaluation / f"{args.output_prefix}_confusion_matrix.csv", index_label="actual")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
