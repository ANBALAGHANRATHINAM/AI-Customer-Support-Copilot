"""Explainable non-ML priority rules for customer-support triage."""

from __future__ import annotations

from dataclasses import asdict, dataclass


VALID_PRIORITIES = ("High", "Medium", "Low")
ESCALATION_TERMS = ("escalate", "unresolved", "not resolved", "formal resolution", "complaint", "repeated", "poor support")
SERVICE_IMPACT_TERMS = ("blocking my work", "service down", "outage", "suspended", "suspend services", "cannot connect", "can't connect", "no connectivity", "business impact")
URGENCY_TERMS = ("urgent", "immediately", "without delay", "today", "promptly", "as soon as possible")


@dataclass(frozen=True)
class PriorityResult:
    priority: str
    reasons: list[str]

    def to_dict(self) -> dict[str, str | list[str]]:
        return asdict(self)


def detect_priority(text: str, intent: str, sentiment: str) -> PriorityResult:
    """Return High/Medium/Low plus deterministic, inspectable reasons."""
    normalized = text.lower()
    reasons: list[str] = []
    escalation = [term for term in ESCALATION_TERMS if term in normalized]
    impact = [term for term in SERVICE_IMPACT_TERMS if term in normalized]
    urgency = [term for term in URGENCY_TERMS if term in normalized]
    if escalation:
        reasons.append(f"escalation language: {', '.join(escalation)}")
    if impact:
        reasons.append(f"service-impact language: {', '.join(impact)}")
    if urgency:
        reasons.append(f"urgency language: {', '.join(urgency)}")

    if escalation or impact or (intent == "Complaint" and sentiment == "Angry") or (sentiment == "Angry" and intent == "Technical"):
        priority = "High"
    elif intent in {"Complaint", "Technical"} or (sentiment == "Angry" and intent in {"Billing", "Refund"}) or urgency:
        priority = "Medium"
    else:
        priority = "Low"
    if not reasons:
        reasons.append(f"baseline triage from intent={intent}, sentiment={sentiment}")
    return PriorityResult(priority=priority, reasons=reasons)

