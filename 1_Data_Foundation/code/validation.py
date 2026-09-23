"""Deterministic quality checks for the Segment 1 synthetic dataset."""

from __future__ import annotations

import re
from typing import Any

import pandas as pd
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.neighbors import NearestNeighbors

from generator import INTENTS, MALFORMED_PATTERNS, SENTIMENTS


REQUIRED_COLUMNS = ["text", "intent", "sentiment"]
INTENT_MARKERS = {
    "Billing": ("billing", "invoice", "payment", "charge", "billed", "bills"),
    "Refund": ("refund",),
    "Technical": ("help with", "technical assistance", "support process", "troubleshooting", "setting up", "trying to use", "get assistance", "support option", "practical help", "clarification on", "clarification about", "how zends supports", "help is available", "help me use"),
    "Complaint": ("contacted support more than once", "formal resolution", "support experience", "raise a complaint", "escalate", "dissatisfied", "repeated", "unresolved", "not resolved", "poor", "did not resolve"),
    "Product Inquiry": ("include", "listed price", "product information", "considering", "available", "choose between", "describe", "costs", "list under", "know about"),
}
COMPLAINT_MARKERS = INTENT_MARKERS["Complaint"]
SENTIMENT_CUES = {
    "Happy": ("glad", "helpful", "promising", "appreciate", "good to know", "pleased"),
    "Angry": ("billing issue clarified", "billing answer today", "billing matter needs", "clear decision about my refund", "refund request promptly", "refund eligibility today", "blocking my work", "technical help without delay", "technical issue handled", "already tried", "happened again", "waiting for a clear answer", "keep asking", "clear product answer", "product details now", "product information clarified"),
}


def _near_duplicate_summary(texts: pd.Series, sample_size: int = 2_500) -> dict[str, Any]:
    sample = texts.sample(n=min(sample_size, len(texts)), random_state=20260911).tolist()
    if len(sample) < 2:
        return {"method": "not enough rows", "sample_size": len(sample), "pairs_at_or_above_0_995": 0}
    matrix = HashingVectorizer(analyzer="char_wb", ngram_range=(4, 5), n_features=2**18, alternate_sign=False, norm="l2").fit_transform(sample)
    distances, _ = NearestNeighbors(n_neighbors=2, metric="cosine", algorithm="brute").fit(matrix).kneighbors(matrix)
    similarities = 1 - distances[:, 1]
    return {"method": "exact nearest-neighbour cosine similarity over character 4-5 grams", "sample_size": len(sample), "threshold": 0.995, "pairs_at_or_above_0_995": int((similarities >= 0.995).sum()), "maximum_nearest_similarity": round(float(similarities.max()), 4)}


def _fact_quality(frame: pd.DataFrame, facts: dict[str, Any]) -> dict[str, Any]:
    products = {product["name"]: product for product in facts["products"]}
    product_names = sorted(products, key=len, reverse=True)
    countries = facts["countries"]
    services_by_group = facts["services"]
    all_services = {service for values in services_by_group.values() for service in values}
    # Do not treat generic words such as "monitoring" or "troubleshooting" as
    # an injected product service when they are also components of longer names.
    services = sorted(
        [service for service in all_services if not any(service != other and service.lower() in other.lower() for other in all_services)],
        key=len,
        reverse=True,
    )
    allowed_prices = {price for product in products.values() for country_prices in product["prices"].values() for price in country_prices.values()}
    invalid_products = invalid_countries = incompatible_services = invalid_price_contexts = 0
    unsupported_prices: set[int] = set()
    for _, row in frame.iterrows():
        text = row["text"].lower()
        mentioned_products = [name for name in product_names if name.lower() in text]
        mentioned_countries = [country for country in countries if re.search(rf"\b{re.escape(country.lower())}\b", text)]
        # Product-comparison questions intentionally mention exactly two products;
        # all other records must mention exactly one.
        if len(mentioned_products) not in (1, 2):
            invalid_products += 1
            continue
        if len(mentioned_countries) > 1:
            invalid_countries += 1
            continue
        country = mentioned_countries[0] if mentioned_countries else None
        mentioned_services = [service for service in services if re.search(rf"\b{re.escape(service.lower())}\b", text)]
        groups = {products[name]["group"] for name in mentioned_products}
        if any(not any(service in services_by_group[group] for group in groups) for service in mentioned_services):
            incompatible_services += 1
        customer_types = [value for value in facts["customer_types"] if re.search(rf"\b{value}\b", text)]
        valid_prices = {products[name]["prices"][country][customer_type] for name in mentioned_products for customer_type in facts["customer_types"]} if country else set()
        for amount in map(int, re.findall(r"\$(\d+)", row["text"])):
            if amount not in allowed_prices:
                unsupported_prices.add(amount)
            elif not country or amount not in valid_prices:
                invalid_price_contexts += 1
    return {
        "catalog_product_count": len(products), "invalid_product_rows": invalid_products, "invalid_country_rows": invalid_countries,
        "incompatible_service_rows": incompatible_services, "invalid_price_context_rows": invalid_price_contexts,
        "unsupported_monetary_values": sorted(unsupported_prices), "all_monetary_values_are_catalog_prices": not unsupported_prices,
        "country_coverage": {country: int(frame["text"].str.contains(country, regex=False).sum()) for country in countries},
    }


def _text_quality(frame: pd.DataFrame) -> dict[str, Any]:
    malformed = {pattern: int(frame["text"].apply(lambda text: bool(re.search(pattern, text, flags=re.IGNORECASE))).sum()) for pattern in MALFORMED_PATTERNS}
    intent_consistency = {}
    for intent, markers in INTENT_MARKERS.items():
        subset = frame.loc[frame["intent"] == intent, "text"].str.lower()
        matched = subset.apply(lambda text: any(marker in text for marker in markers)).sum()
        intent_consistency[intent] = {"checked": int(len(subset)), "marker_consistent": int(matched), "marker_mismatches": int(len(subset) - matched)}
    technical = frame.loc[frame["intent"] == "Technical", "text"].str.lower()
    technical_complaint_overlap = int(technical.apply(lambda text: any(marker in text for marker in COMPLAINT_MARKERS)).sum())
    sentiment_cues = {}
    for sentiment, cues in SENTIMENT_CUES.items():
        subset = frame.loc[frame["sentiment"] == sentiment, "text"].str.lower()
        matched = subset.apply(lambda text: any(cue in text for cue in cues)).sum()
        sentiment_cues[sentiment] = {"checked": int(len(subset)), "cue_present": int(matched), "cue_missing": int(len(subset) - matched)}
    neutral = frame.loc[frame["sentiment"] == "Neutral", "text"].str.lower()
    neutral_conflicts = int(neutral.apply(lambda text: any(cue in text for cues in SENTIMENT_CUES.values() for cue in cues)).sum())
    return {"malformed_pattern_matches": malformed, "intent_text_consistency": intent_consistency, "technical_rows_with_complaint_markers": technical_complaint_overlap, "sentiment_cue_coverage": sentiment_cues, "neutral_rows_with_explicit_happy_or_angry_tone": neutral_conflicts}


def validate_dataset(frame: pd.DataFrame, facts: dict[str, Any], expected_rows: int = 20_000) -> dict[str, Any]:
    """Validate schema, balance, factual grounding, text quality, and duplicates."""
    report: dict[str, Any] = {
        "expected_rows": expected_rows, "actual_rows": int(len(frame)), "required_columns": REQUIRED_COLUMNS,
        "columns_match": list(frame.columns) == REQUIRED_COLUMNS,
        "missing_values": {column: int(frame[column].isna().sum()) for column in REQUIRED_COLUMNS},
        "blank_text_rows": int(frame["text"].fillna("").str.strip().eq("").sum()),
        "intent_counts": {label: int((frame["intent"] == label).sum()) for label in INTENTS},
        "sentiment_counts": {label: int((frame["sentiment"] == label).sum()) for label in SENTIMENTS},
        "unknown_intents": sorted(set(frame["intent"]) - set(INTENTS)), "unknown_sentiments": sorted(set(frame["sentiment"]) - set(SENTIMENTS)),
        "exact_duplicate_rows": int(frame.duplicated().sum()), "exact_duplicate_texts": int(frame["text"].str.lower().duplicated().sum()),
        "normalized_duplicate_texts": int(frame["text"].map(lambda text: re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()).duplicated().sum()),
    }
    report["balanced_intents"] = all(count == expected_rows // len(INTENTS) for count in report["intent_counts"].values())
    spread = max(report["sentiment_counts"].values()) - min(report["sentiment_counts"].values())
    report["sentiment_balance"] = {"max_minus_min": spread, "tolerance": max(3, round(expected_rows * 0.01)), "approximately_balanced": spread <= max(3, round(expected_rows * 0.01))}
    report["factual_consistency"] = _fact_quality(frame, facts)
    report["text_quality"] = _text_quality(frame)
    report["near_duplicate_scan"] = _near_duplicate_summary(frame["text"])
    report["limitations"] = [
        "Rule-based checks catch known malformed patterns and source-fact mismatches but cannot prove every sentence is semantically natural.",
        "The near-duplicate scan is a deterministic 2,500-row character n-gram sample, not an exhaustive semantic-duplicate audit of all 20,000 records.",
    ]
    facts_quality, text_quality = report["factual_consistency"], report["text_quality"]
    report["passed"] = bool(
        report["actual_rows"] == expected_rows and report["columns_match"] and not any(report["missing_values"].values()) and report["blank_text_rows"] == 0
        and report["balanced_intents"] and report["sentiment_balance"]["approximately_balanced"] and not report["unknown_intents"] and not report["unknown_sentiments"]
        and report["exact_duplicate_rows"] == 0 and report["exact_duplicate_texts"] == 0 and report["normalized_duplicate_texts"] == 0
        and not any(facts_quality[key] for key in ("invalid_product_rows", "invalid_country_rows", "incompatible_service_rows", "invalid_price_context_rows", "unsupported_monetary_values"))
        and not any(text_quality["malformed_pattern_matches"].values()) and not any(item["marker_mismatches"] for item in text_quality["intent_text_consistency"].values())
        and text_quality["technical_rows_with_complaint_markers"] == 0 and not any(item["cue_missing"] for item in text_quality["sentiment_cue_coverage"].values()) and text_quality["neutral_rows_with_explicit_happy_or_angry_tone"] == 0
    )
    return report


def validate_splits(train: pd.DataFrame, validation: pd.DataFrame, test: pd.DataFrame) -> dict[str, Any]:
    """Verify target sizes and absence of text leakage between split files."""
    texts = {"train": set(train["text"]), "validation": set(validation["text"]), "test": set(test["text"])}
    overlap = {"train_validation": len(texts["train"] & texts["validation"]), "train_test": len(texts["train"] & texts["test"]), "validation_test": len(texts["validation"] & texts["test"])}
    sizes = {"train": len(train), "validation": len(validation), "test": len(test)}
    return {"sizes": sizes, "overlap": overlap, "passed": sizes == {"train": 14_000, "validation": 3_000, "test": 3_000} and not any(overlap.values())}
