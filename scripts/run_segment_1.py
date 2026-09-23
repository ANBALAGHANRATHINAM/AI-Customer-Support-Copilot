"""Run the full Segment 1 data foundation workflow."""

from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SEGMENT_ROOT = PROJECT_ROOT / "1_Data_Foundation"
CODE_DIR = SEGMENT_ROOT / "code"
sys.path.insert(0, str(CODE_DIR))

from eda import create_eda, create_splits
from generator import build_manifest, generate_dataset, load_facts
from validation import validate_dataset, validate_splits


SEED = 20260911


def write_json(path: Path, content: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(content, indent=2, sort_keys=True), encoding="utf-8")


def write_eda_summary(path: Path, summary: dict) -> None:
    common_terms = ", ".join(f"{item['term']} ({item['count']})" for item in summary["common_terms"][:10])
    text = f"""# Segment 1 EDA Summary

## Distributions

- Intents: {summary['intent_distribution']}
- Sentiments: {summary['sentiment_distribution']}

## Message length

- Word count: min {summary['message_length_words']['min']}, median {summary['message_length_words']['median']}, mean {summary['message_length_words']['mean']}, max {summary['message_length_words']['max']}.
- IQR outliers: {summary['outliers']['count']} using bounds {summary['outliers']['lower_bound']} to {summary['outliers']['upper_bound']} words.

## Quality checks

- Exact duplicate rows: {summary['duplicate_analysis']['exact_duplicate_rows']}
- Exact duplicate texts: {summary['duplicate_analysis']['exact_duplicate_texts']}
- Frequent non-stopword terms: {common_terms}

See the PNG charts in this folder for the requested univariate and bivariate views.
"""
    path.write_text(text, encoding="utf-8")


def main() -> None:
    facts_path = SEGMENT_ROOT / "config" / "zends_facts.json"
    output_data = SEGMENT_ROOT / "data"
    split_dir = output_data
    evaluation_dir = SEGMENT_ROOT / "evaluation"
    facts = load_facts(facts_path)

    dataset = generate_dataset(facts, records_per_intent=4_000, seed=SEED)
    output_data.mkdir(parents=True, exist_ok=True)
    dataset.to_csv(output_data / "zends_support_queries_20000.csv", index=False)

    validation = validate_dataset(dataset, facts)
    manifest = build_manifest(dataset, facts_path, SEED, PROJECT_ROOT)
    write_json(evaluation_dir / "validation_report.json", validation)
    write_json(evaluation_dir / "generation_manifest.json", manifest)
    if not validation["passed"]:
        raise RuntimeError("Segment 1 validation failed. Inspect 1_Data_Foundation/evaluation/validation_report.json")

    eda_summary = create_eda(dataset, evaluation_dir / "eda")
    write_json(evaluation_dir / "eda" / "eda_summary.json", eda_summary)
    write_eda_summary(evaluation_dir / "eda" / "EDA_SUMMARY.md", eda_summary)

    train, validation_split, test = create_splits(dataset, SEED)
    split_dir.mkdir(parents=True, exist_ok=True)
    train.to_csv(split_dir / "train.csv", index=False)
    validation_split.to_csv(split_dir / "validation.csv", index=False)
    test.to_csv(split_dir / "test.csv", index=False)
    split_summary = {
        "strategy": "70/15/15 stratified jointly by intent and sentiment",
        "seed": SEED,
        **validate_splits(train, validation_split, test),
    }
    write_json(evaluation_dir / "split_report.json", split_summary)
    if not split_summary["passed"]:
        raise RuntimeError("Segment 1 split validation failed. Inspect 1_Data_Foundation/evaluation/split_report.json")
    print(json.dumps({"validation_passed": validation["passed"], "sizes": split_summary["sizes"]}, indent=2))


if __name__ == "__main__":
    main()
