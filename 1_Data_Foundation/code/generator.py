"""Generate balanced, source-grounded ZENDS customer-support queries."""

from __future__ import annotations

import hashlib
import json
import random
import re
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd


INTENTS = ("Billing", "Refund", "Technical", "Complaint", "Product Inquiry")
SENTIMENTS = ("Happy", "Neutral", "Angry")

# Intent-aware but globally balanced. Complaint texts are naturally more often angry;
# product inquiries are more often happy or neutral.
SENTIMENT_TARGETS = {
    "Billing": {"Happy": 1450, "Neutral": 1400, "Angry": 1150},
    "Refund": {"Happy": 1200, "Neutral": 1400, "Angry": 1400},
    "Technical": {"Happy": 1200, "Neutral": 1500, "Angry": 1300},
    "Complaint": {"Happy": 600, "Neutral": 1200, "Angry": 2200},
    "Product Inquiry": {"Happy": 2200, "Neutral": 1200, "Angry": 600},
}

MALFORMED_PATTERNS = (
    r"\btechnical guidance for setup guidance\b",
    r"\b(setup guidance|troubleshooting|technical support) (?:for|with) \1\b",
    r"\b(help|assistance) (?:with|for) \1\b",
    r"\b(\w+)\s+\1\b",
    r"\s{2,}",
)


def load_facts(path: Path) -> dict[str, Any]:
    """Load the hand-checked transcription of ZENDS Communications.pdf."""
    with path.open(encoding="utf-8") as file:
        return json.load(file)


def _normalise(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()


def _is_well_formed(text: str) -> bool:
    return not any(re.search(pattern, text, flags=re.IGNORECASE) for pattern in MALFORMED_PATTERNS)


def _tone(sentiment: str, intent: str, rng: random.Random) -> tuple[str, str]:
    """Return varied natural framing instead of a mechanical sentiment prefix."""
    choices = {
        "Happy": (
            "I am glad I checked this in advance. ", "This has been helpful so far. ",
            "That sounds promising. ", "I appreciate the clarification so far. ",
            "Good to know. ", "I am pleased with the guidance so far. ",
        ),
        "Neutral": ("", "Could you clarify one point? ", "I would like to understand this better. "),
        "Angry": (),
    }
    endings = {
        "Happy": ("", " Could you confirm that for me?", " I would appreciate a quick confirmation."),
        "Neutral": ("", " Please let me know.", " Kindly confirm."),
        "Angry": ("", " Please respond clearly.", " I need this resolved promptly."),
    }
    angry_by_intent = {
        "Billing": ("I need this billing issue clarified promptly. ", "I need a clear billing answer today. ", "This billing matter needs to be resolved. "),
        "Refund": ("I need a clear decision about my refund request. ", "Please address this refund request promptly. ", "I need an answer about refund eligibility today. "),
        "Technical": ("This is blocking my work. ", "I need technical help without delay. ", "I need this technical issue handled promptly. "),
        "Complaint": ("I have already tried to get an answer. ", "This has happened again. ", "I have been waiting for a clear answer. ", "I should not have to keep asking about this. "),
        "Product Inquiry": ("I need a clear product answer before I decide. ", "Please provide these product details now. ", "I need this product information clarified. "),
    }
    complaint_happy = ("I appreciate that this is being reviewed, but ", "I am glad someone is looking into this, but ")
    opening = rng.choice(complaint_happy) if sentiment == "Happy" and intent == "Complaint" else rng.choice(angry_by_intent[intent]) if sentiment == "Angry" else rng.choice(choices[sentiment])
    return opening, rng.choice(endings[sentiment])


def _product_context(facts: dict[str, Any], rng: random.Random) -> dict[str, Any]:
    product = rng.choice(facts["products"])
    country = rng.choice(facts["countries"])
    customer_type = rng.choice(facts["customer_types"])
    peers = [item for item in facts["products"] if item["group"] == product["group"] and item["name"] != product["name"]]
    comparison = rng.choice(peers) if peers else product
    return {
        "product": product["name"], "group": product["group"], "details": product["details"],
        "country": country, "customer_type": customer_type, "price": product["prices"][country][customer_type],
        "service": rng.choice(facts["services"][product["group"]]),
        "comparison_product": comparison["name"], "comparison_details": comparison["details"],
        "comparison_price": comparison["prices"][country][customer_type],
    }


def _billing_query(c: dict[str, Any], rng: random.Random) -> str:
    templates = (
        "How does monthly billing in advance apply to my {customer_type} {product} service in {country}?",
        "The listed price for {product} is ${price} for an {customer_type} customer in {country}. When is that charge billed?",
        "Could you explain why ZENDS bills {product} in advance each month for customers in {country}?",
        "What should I expect if payment for my {product} service in {country} is more than 7 days late?",
        "Does an enterprise {product} account in {country} receive a consolidated invoice, and how is it billed?",
        "I am reviewing the ${price} charge for {product} in {country}. Which billing rule applies to an {customer_type} customer?",
        "Before I pay for {product}, can you clarify the monthly billing timing for an {customer_type} account in {country}?",
        "My {product} invoice in {country} is due soon. What does the late-payment policy say after 7 days?",
        "Where can I find the billing details for an {customer_type} customer paying ${price} for {product} in {country}?",
        "Is the ${price} amount for {product} in {country} part of the monthly advance billing cycle?",
        "I need to understand the invoice process for my {customer_type} {product} service in {country}.",
        "Can you confirm whether the billing policy for {product} in {country} changes for enterprise customers?",
    )
    return rng.choice(templates).format(**c)


def _refund_query(c: dict[str, Any], rng: random.Random) -> str:
    templates = (
        "I am within 7 days of starting {product} in {country} and my usage is below 10%. Can I request a full refund?",
        "Does the refund policy apply to an {customer_type} {product} account when less than 10% has been used within 7 days?",
        "Please explain whether my {product} service in {country} is eligible for a refund before the 7-day period ends.",
        "I need to check refund eligibility for {product}: I am still within 7 days and usage is under 10%.",
        "What information is relevant when requesting a refund for an {customer_type} {product} service in {country}?",
        "Could you confirm the 7-day and under-10%-usage conditions for a full refund on {product}?",
        "I am deciding whether to keep {product} in {country}. What does ZENDS require for a full refund?",
        "Would an {customer_type} customer using less than 10% of {product} qualify for a refund within 7 days?",
        "Please clarify the refund rule before I make a decision about my {product} service in {country}.",
        "I want to understand the refund window for {product}; is the usage threshold below 10%?",
    )
    if c["group"] == "Cloud & Data Center Services":
        templates += (
            "My {product} cloud service is already activated. Is it non-refundable after activation even if I am within 7 days?",
            "Before activating {product}, can you explain the cloud-service refund restriction after activation?",
            "Is there any refund option for an activated {product} cloud service in {country}?",
        )
    return rng.choice(templates).format(**c)


def _technical_query(c: dict[str, Any], rng: random.Random) -> str:
    templates = (
        "I use {product} in {country}. What should I check first when I need help with {service} listed under {group}?",
        "Where can an {customer_type} customer using {product} in {country} get technical assistance with {service} under {group}?",
        "Can you point me to the ZENDS support process for {service}, which is listed under {group}? I use {product}.",
        "I am trying to use {service} listed under {group} while using {product} in {country}. Which steps should I follow?",
        "What help is available for {service} under {group} for an {customer_type} customer using {product}?",
        "Could you explain how ZENDS supports {service} under {group}? I use {product} in {country}.",
        "I need practical help with {service} listed under {group}; I use {product}. Is there a recommended troubleshooting path?",
        "For my {product} account in {country}, how do I get assistance with {service} listed under {group}?",
        "I am setting up {product} and need help with {service} under {group}. Where should I start?",
        "Which ZENDS support option covers {service} listed under {group} for an {customer_type} customer using {product}?",
        "Can you help me use {service} listed under {group}? My product is {product} in {country}.",
        "I need clarification about {service}, which ZENDS lists under {group}. I use {product}.",
    )
    return rng.choice(templates).format(**c)


def _complaint_query(c: dict[str, Any], rng: random.Random) -> str:
    """Complaint content always signals dissatisfaction, failed resolution, or escalation."""
    templates = (
        "I have contacted support more than once about {service} for {product} in {country}, but it remains unresolved. How can I escalate this?",
        "The problem with {service} on my {customer_type} {product} account keeps recurring. I need a formal resolution.",
        "My support experience for {service} on {product} in {country} has been poor. Please review this complaint.",
        "The issue with {service} was not resolved after prior support for my {product} service. What is the escalation route?",
        "I need to raise a complaint because {service} for {product} in {country} has not been resolved.",
        "After repeated requests for help with {service}, my {product} concern is still unresolved. Please escalate it.",
        "I am dissatisfied with how the {service} issue on my {product} account has been handled. I need a proper resolution.",
        "The support provided for {service} on {product} did not resolve my problem. Who can formally review this?",
        "I have had to report the {service} issue on my {product} account repeatedly. Please provide an escalation option.",
        "My {customer_type} {product} service in {country} still has an unresolved {service} concern after support contact.",
        "I want this {service} problem with {product} formally addressed because earlier support did not resolve it.",
        "The repeated {service} issue on {product} is affecting my experience. I need a clear resolution from ZENDS.",
    )
    return rng.choice(templates).format(**c)


def _product_query(c: dict[str, Any], rng: random.Random) -> str:
    templates = (
        "What does {product} include for an {customer_type} customer in {country}, and is its listed price ${price}?",
        "I am considering {product} in {country}. What does this {details} offer to an {customer_type} customer?",
        "Can you confirm that {product} costs ${price} for an {customer_type} customer in {country}?",
        "Which services does ZENDS list under {group}, including {service}, for someone considering {product} in {country}?",
        "For an {customer_type} customer in {country}, how do the listed prices of {product} (${price}) and {comparison_product} (${comparison_price}) differ?",
        "What is the product information for {product}, including its {details} and listed price in {country}?",
        "Does ZENDS list {service} under {group}? I am considering {product} as an {customer_type} customer in {country}.",
        "I want to choose between {product} and {comparison_product}. What do their listed prices in {country} show for an {customer_type} customer?",
        "Before choosing {product}, could you explain what is included and the ${price} price for an {customer_type} customer in {country}?",
        "Is {product} available to an {customer_type} customer in {country}, and what does the offering include?",
        "Could you describe {product} and confirm whether ${price} is its listed price in {country}?",
        "What should an {customer_type} customer in {country} know about {product}, a {details}?",
    )
    return rng.choice(templates).format(**c)


QUERY_BUILDERS = {"Billing": _billing_query, "Refund": _refund_query, "Technical": _technical_query, "Complaint": _complaint_query, "Product Inquiry": _product_query}


def _sentiment_schedule(intent: str, records_per_intent: int, rng: random.Random) -> list[str]:
    targets = SENTIMENT_TARGETS[intent]
    raw = {sentiment: records_per_intent * targets[sentiment] / 4_000 for sentiment in SENTIMENTS}
    counts = {sentiment: int(raw[sentiment]) for sentiment in SENTIMENTS}
    for sentiment in sorted(SENTIMENTS, key=lambda item: raw[item] - counts[item], reverse=True)[: records_per_intent - sum(counts.values())]:
        counts[sentiment] += 1
    schedule = [sentiment for sentiment in SENTIMENTS for _ in range(counts[sentiment])]
    rng.shuffle(schedule)
    return schedule


def generate_dataset(facts: dict[str, Any], *, records_per_intent: int = 4_000, seed: int = 20260911) -> pd.DataFrame:
    """Return a balanced, intent-distinct, de-duplicated three-column dataset."""
    if records_per_intent <= 0:
        raise ValueError("records_per_intent must be positive")
    rng = random.Random(seed)
    rows: list[dict[str, str]] = []
    seen_texts: set[str] = set()
    for intent in INTENTS:
        schedule = _sentiment_schedule(intent, records_per_intent, rng)
        accepted = attempts = 0
        while accepted < records_per_intent:
            attempts += 1
            if attempts > records_per_intent * 250:
                raise RuntimeError(f"Could not generate enough unique {intent} records")
            context = _product_context(facts, rng)
            opening, closing = _tone(schedule[accepted], intent, rng)
            body = QUERY_BUILDERS[intent](context, rng)
            if opening.endswith("but ") and not body.startswith("I "):
                body = body[:1].lower() + body[1:]
            text = f"{opening}{body}{closing}".strip()
            normalized = _normalise(text)
            if normalized in seen_texts or not _is_well_formed(text):
                continue
            seen_texts.add(normalized)
            rows.append({"text": text, "intent": intent, "sentiment": schedule[accepted]})
            accepted += 1
    return pd.DataFrame(rows, columns=["text", "intent", "sentiment"]).sample(frac=1, random_state=seed).reset_index(drop=True)


def facts_checksum(facts_path: Path) -> str:
    return hashlib.sha256(facts_path.read_bytes()).hexdigest()


def build_manifest(frame: pd.DataFrame, facts_path: Path, seed: int, project_root: Path) -> dict[str, Any]:
    return {"source": "ZENDS Communications.pdf only for company facts", "facts_file": str(facts_path.relative_to(project_root).as_posix()), "facts_sha256": facts_checksum(facts_path), "seed": seed, "rows": int(len(frame)), "intent_counts": Counter(frame["intent"]).most_common(), "sentiment_counts": Counter(frame["sentiment"]).most_common(), "columns": list(frame.columns)}
