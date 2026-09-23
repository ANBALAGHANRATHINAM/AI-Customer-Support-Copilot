"""EDA and reproducible, stratified split creation for Segment 1."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split


def _save_figure(path: Path) -> None:
    plt.tight_layout()
    plt.savefig(path, dpi=160, bbox_inches="tight")
    plt.close()


def create_splits(frame: pd.DataFrame, seed: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Create 70/15/15 splits stratified jointly by intent and sentiment."""
    strata = frame["intent"] + " | " + frame["sentiment"]
    train, holdout = train_test_split(frame, test_size=0.30, random_state=seed, stratify=strata)
    holdout_strata = holdout["intent"] + " | " + holdout["sentiment"]
    validation, test = train_test_split(holdout, test_size=0.50, random_state=seed, stratify=holdout_strata)
    return tuple(part.reset_index(drop=True) for part in (train, validation, test))


def create_eda(frame: pd.DataFrame, output_dir: Path) -> dict[str, Any]:
    """Write requested EDA charts and return a JSON-serializable summary."""
    output_dir.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid", palette="deep")
    working = frame.copy()
    working["message_length_words"] = working["text"].str.split().str.len()

    plt.figure(figsize=(8, 4.5))
    sns.countplot(data=working, x="intent", order=sorted(working["intent"].unique()))
    plt.xticks(rotation=20, ha="right")
    plt.title("Intent distribution")
    plt.xlabel("")
    _save_figure(output_dir / "intent_distribution.png")

    plt.figure(figsize=(6, 4.5))
    sns.countplot(data=working, x="sentiment", order=["Happy", "Neutral", "Angry"])
    plt.title("Sentiment distribution")
    plt.xlabel("")
    _save_figure(output_dir / "sentiment_distribution.png")

    plt.figure(figsize=(8, 4.5))
    sns.histplot(working["message_length_words"], bins=30)
    plt.title("Message length distribution")
    plt.xlabel("Words per message")
    _save_figure(output_dir / "message_length_distribution.png")

    cross_tab = pd.crosstab(working["intent"], working["sentiment"])
    plt.figure(figsize=(7, 4.5))
    sns.heatmap(cross_tab, annot=True, fmt="d", cmap="Blues")
    plt.title("Intent vs sentiment")
    _save_figure(output_dir / "intent_vs_sentiment.png")

    plt.figure(figsize=(8, 4.5))
    sns.boxplot(data=working, x="intent", y="message_length_words")
    plt.xticks(rotation=20, ha="right")
    plt.title("Message length by intent")
    plt.xlabel("")
    _save_figure(output_dir / "message_length_by_intent.png")

    vectorizer = CountVectorizer(stop_words="english", max_features=20)
    counts = vectorizer.fit_transform(working["text"])
    terms = pd.DataFrame({"term": vectorizer.get_feature_names_out(), "count": counts.sum(axis=0).A1}).sort_values("count", ascending=False)
    plt.figure(figsize=(8, 5))
    sns.barplot(data=terms.head(15), x="count", y="term")
    plt.title("Most common non-stopword terms")
    _save_figure(output_dir / "common_terms.png")

    q1, q3 = working["message_length_words"].quantile([0.25, 0.75])
    iqr = q3 - q1
    lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    outlier_count = int(((working["message_length_words"] < lower) | (working["message_length_words"] > upper)).sum())
    return {
        "intent_distribution": working["intent"].value_counts().sort_index().to_dict(),
        "sentiment_distribution": working["sentiment"].value_counts().reindex(["Happy", "Neutral", "Angry"]).to_dict(),
        "message_length_words": {
            "min": int(working["message_length_words"].min()),
            "max": int(working["message_length_words"].max()),
            "mean": round(float(working["message_length_words"].mean()), 2),
            "median": float(working["message_length_words"].median()),
        },
        "intent_vs_sentiment": cross_tab.to_dict(),
        "duplicate_analysis": {"exact_duplicate_rows": int(working.duplicated().sum()), "exact_duplicate_texts": int(working["text"].str.lower().duplicated().sum())},
        "common_terms": terms.head(15).to_dict(orient="records"),
        "outliers": {"method": "1.5 x IQR on word count", "lower_bound": round(float(lower), 2), "upper_bound": round(float(upper), 2), "count": outlier_count},
        "charts": [
            "intent_distribution.png", "sentiment_distribution.png", "message_length_distribution.png",
            "intent_vs_sentiment.png", "message_length_by_intent.png", "common_terms.png",
        ],
    }
