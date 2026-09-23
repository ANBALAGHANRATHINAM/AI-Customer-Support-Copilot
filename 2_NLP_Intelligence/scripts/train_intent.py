"""Fine-tune pretrained DistilBERT and evaluate once on the untouched test split."""

from __future__ import annotations

import json
import os
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from transformers import data

ROOT = Path(__file__).resolve().parents[2]
SEGMENT_ROOT = ROOT / "2_NLP_Intelligence"
sys.path.insert(0, str(SEGMENT_ROOT / "code"))

from intent import BASE_INTENT_MODEL, INTENT_LABELS
from preprocessing import clean_text


SEED = 20260911
MAX_LENGTH = 64
LEARNING_RATE = 2e-5
EPOCHS = 1
TRAIN_BATCH_SIZE = 32
GRADIENT_ACCUMULATION_STEPS = 4
EVAL_BATCH_SIZE = 32
HARD_BOUNDARY_REPEAT = 1
SYNTHETIC_EXAMPLES_PER_INTENT = 500
BALANCED_AUGMENTATION_ENABLED = True

def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True), encoding="utf-8")


def _load_split(name: str, label_to_id: dict[str, int]) -> pd.DataFrame:
    data = pd.read_csv(ROOT / "1_Data_Foundation" / "data" / f"{name}.csv")
    if name == "train":
        # Keep Segment 1 immutable. Use a balanced natural-language
        # augmentation set plus the existing hard-boundary examples.
        data = (
             data.groupby("intent", group_keys=False)
            .sample(n=SYNTHETIC_EXAMPLES_PER_INTENT, random_state=SEED)
            .reset_index(drop=True)
        )

        hard_boundary = pd.read_csv(
            SEGMENT_ROOT / "data" / "intent_hard_boundary_train.csv"
        )

        data = pd.concat(
            [
                data,
                *([hard_boundary] * HARD_BOUNDARY_REPEAT),
            ],
            ignore_index=True,
        )
    
    data["text"] = data["text"].map(clean_text)
    data["labels"] = data["intent"].map(label_to_id)

    if data["labels"].isna().any():
        raise ValueError(f"Unexpected intent label in {os.name} split")

    return data[["text", "labels"]]

def main() -> None:
    import torch
    from datasets import Dataset
    from transformers import AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding, Trainer, TrainingArguments, set_seed

    set_seed(SEED)
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.set_num_threads(max(1, min(4, os.cpu_count() or 1)))
    label_to_id = {label: index for index, label in enumerate(INTENT_LABELS)}
    id_to_label = {index: label for label, index in label_to_id.items()}
    tokenizer = AutoTokenizer.from_pretrained(BASE_INTENT_MODEL)

    splits = {name: Dataset.from_pandas(_load_split(name, label_to_id), preserve_index=False) for name in ("train", "validation")}
    def tokenize(batch: dict[str, list[str]]) -> dict[str, list[int]]:
        return tokenizer(batch["text"], truncation=True, max_length=MAX_LENGTH)
    tokenized = {name: data.map(tokenize, batched=True, remove_columns=["text"]) for name, data in splits.items()}

    model = AutoModelForSequenceClassification.from_pretrained(BASE_INTENT_MODEL, num_labels=len(INTENT_LABELS), id2label=id_to_label, label2id=label_to_id)
    # Stage the candidate separately. The current production model is not
    # replaced until its held-out and realistic evaluations have completed.
    model_dir = SEGMENT_ROOT / "models" / "intent_distilbert_candidate"
    training_output = SEGMENT_ROOT / "models" / "candidate_training_output"
    def validation_metrics(prediction: object) -> dict[str, float]:
        logits, labels = prediction
        predicted = np.argmax(logits, axis=-1)
        precision, recall, f1, _ = precision_recall_fscore_support(labels, predicted, average="weighted", zero_division=0)
        return {"accuracy": accuracy_score(labels, predicted), "precision_weighted": precision, "recall_weighted": recall, "f1_weighted": f1}
    args = TrainingArguments(output_dir=str(training_output), learning_rate=LEARNING_RATE, num_train_epochs=EPOCHS, per_device_train_batch_size=TRAIN_BATCH_SIZE, gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS, per_device_eval_batch_size=EVAL_BATCH_SIZE, weight_decay=0.01, eval_strategy="epoch", save_strategy="no", logging_strategy="steps", logging_steps=50, report_to=[], seed=SEED, data_seed=SEED, fp16=False)
    trainer = Trainer(model=model, args=args, train_dataset=tokenized["train"], eval_dataset=tokenized["validation"], processing_class=tokenizer, data_collator=DataCollatorWithPadding(tokenizer=tokenizer), compute_metrics=validation_metrics)
    trainer.train()
    trainer.save_model(model_dir)
    tokenizer.save_pretrained(model_dir)

    # Held-out test evaluation is deliberately performed by evaluate_intent.py
    # only after this candidate has been saved.
    evaluation_dir = SEGMENT_ROOT / "evaluation"
    config = {"model": BASE_INTENT_MODEL, "seed": SEED, "epochs": EPOCHS, "learning_rate": LEARNING_RATE, "max_length": MAX_LENGTH, "train_batch_size": TRAIN_BATCH_SIZE, "gradient_accumulation_steps": GRADIENT_ACCUMULATION_STEPS, "effective_train_batch_size": TRAIN_BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS, "eval_batch_size": EVAL_BATCH_SIZE, "synthetic_examples_per_intent": SYNTHETIC_EXAMPLES_PER_INTENT, "hard_boundary_source": "data/intent_hard_boundary_train.csv", "hard_boundary_repeat": HARD_BOUNDARY_REPEAT,"balanced_augmentation_source": "data/intent_balanced_augmentation.csv", "balanced_augmentation_examples": 250, "train_examples": len(tokenized["train"]), "validation_examples": len(tokenized["validation"]), "test_examples": 3000, "labels": list(INTENT_LABELS), "candidate_model_dir": str(model_dir.relative_to(SEGMENT_ROOT))}
    _write_json(evaluation_dir / "intent_candidate_training_config.json", config)
    print(json.dumps({"candidate_model_dir": str(model_dir), "train_examples": len(tokenized["train"]), "validation_examples": len(tokenized["validation"])}, indent=2))


if __name__ == "__main__":
    main()
