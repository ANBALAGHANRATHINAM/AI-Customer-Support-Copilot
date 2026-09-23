import sys
import re
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SEGMENT_ROOT = ROOT / "1_Data_Foundation"
sys.path.insert(0, str(SEGMENT_ROOT / "code"))

from eda import create_splits
from generator import INTENTS, MALFORMED_PATTERNS, SENTIMENTS, generate_dataset, load_facts
from validation import validate_dataset, validate_splits


def test_zends_catalog_has_all_required_regions_and_products():
    facts = load_facts(SEGMENT_ROOT / "config" / "zends_facts.json")
    assert facts["countries"] == ["USA", "India", "Singapore", "Thailand"]
    assert len(facts["products"]) == 28
    for product in facts["products"]:
        assert set(product["prices"]) == set(facts["countries"])
        assert all(set(prices) == {"individual", "enterprise"} for prices in product["prices"].values())


def test_generator_is_balanced_and_schema_safe():
    facts = load_facts(SEGMENT_ROOT / "config" / "zends_facts.json")
    dataset = generate_dataset(facts, records_per_intent=60, seed=7)
    assert list(dataset.columns) == ["text", "intent", "sentiment"]
    assert len(dataset) == 300
    assert dataset["intent"].value_counts().to_dict() == {intent: 60 for intent in INTENTS}
    assert set(dataset["sentiment"]) == set(SENTIMENTS)
    assert not dataset["text"].str.lower().duplicated().any()


def test_generated_queries_keep_complaints_distinct_from_technical_requests():
    facts = load_facts(SEGMENT_ROOT / "config" / "zends_facts.json")
    dataset = generate_dataset(facts, records_per_intent=120, seed=11)
    complaints = dataset.loc[dataset["intent"] == "Complaint", "text"].str.lower()
    technical = dataset.loc[dataset["intent"] == "Technical", "text"].str.lower()
    complaint_cues = ("unresolved", "escalat", "complaint", "dissatisfied", "did not resolve", "recurring", "poor", "formally addressed", "clear resolution")
    assert complaints.apply(lambda text: any(cue in text for cue in complaint_cues)).all()
    assert not technical.str.contains("raise a complaint|formal resolution|support experience", regex=True).any()


def test_generated_queries_have_no_known_malformed_patterns_and_are_fact_valid():
    facts = load_facts(SEGMENT_ROOT / "config" / "zends_facts.json")
    dataset = generate_dataset(facts, records_per_intent=120, seed=12)
    assert all(not dataset["text"].apply(lambda text: bool(re.search(pattern, text, flags=re.IGNORECASE))).any() for pattern in MALFORMED_PATTERNS)
    report = validate_dataset(dataset, facts, expected_rows=600)
    assert report["factual_consistency"]["invalid_product_rows"] == 0
    assert report["factual_consistency"]["incompatible_service_rows"] == 0


def test_service_queries_do_not_claim_group_services_belong_to_one_product():
    facts = load_facts(SEGMENT_ROOT / "config" / "zends_facts.json")
    dataset = generate_dataset(facts, records_per_intent=120, seed=13)
    service_rows = dataset[dataset["intent"].isin(["Technical", "Product Inquiry"])]["text"].str.lower()
    service_names = [service.lower() for services in facts["services"].values() for service in services]
    rows_with_named_services = service_rows[service_rows.apply(lambda text: any(service in text for service in service_names))]
    service_pattern = "|".join(re.escape(service) for service in service_names)
    assert not rows_with_named_services.str.contains(rf"\bdoes\s+[^?]+?\s+offer\s+(?:{service_pattern})\b|\bincluded with\b", regex=True).any()
    assert service_rows.str.contains("listed under").any()


def test_validator_passes_a_full_size_dataset():
    facts = load_facts(SEGMENT_ROOT / "config" / "zends_facts.json")
    dataset = generate_dataset(facts, records_per_intent=4_000, seed=20260911)
    report = validate_dataset(dataset, facts)
    assert report["passed"]


def test_stratified_splits_are_disjoint_and_preserve_joint_labels():
    rows = []
    for intent in INTENTS:
        for sentiment in SENTIMENTS:
            for number in range(20):
                rows.append({"text": f"{intent} {sentiment} request {number}", "intent": intent, "sentiment": sentiment})
    frame = pd.DataFrame(rows)
    train, validation, test = create_splits(frame, seed=1)
    assert (len(train), len(validation), len(test)) == (210, 45, 45)
    assert not (set(train["text"]) & set(validation["text"]))
    assert not (set(train["text"]) & set(test["text"]))
    assert not (set(validation["text"]) & set(test["text"]))
    assert validate_splits(train, validation, test)["overlap"] == {"train_validation": 0, "train_test": 0, "validation_test": 0}
