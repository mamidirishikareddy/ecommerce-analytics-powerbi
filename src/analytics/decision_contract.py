"""
Governed decision-intelligence contract.

Defines the canonical structure for business/product decisions
generated from analytical signals and supporting evidence.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


DecisionPriority = Literal[
    "CRITICAL",
    "HIGH",
    "MEDIUM",
    "LOW",
]

DecisionType = Literal[
    "CUSTOMER_RETENTION",
    "CUSTOMER_VALUE",
    "PRODUCT_PERFORMANCE",
    "MARKET_PERFORMANCE",
    "FORECAST",
    "DATA_QUALITY",
]

DecisionReadiness = Literal[
    "ACTION_CANDIDATE",
    "REVIEW",
    "OBSERVE",
    "NOT_READY",
]


@dataclass(frozen=True)
class DecisionRecord:
    """
    Canonical decision-intelligence record.

    A decision must identify the business signal, its direction,
    supporting evidence, priority, and readiness before it can be
    surfaced to downstream AI or dashboard layers.
    """

    decision_id: str
    decision_type: DecisionType
    entity_id: str
    signal: str
    direction: str
    priority: DecisionPriority
    evidence_level: str
    decision_readiness: DecisionReadiness
    evidence_basis: str
    recommended_action: str


DECISION_CONTRACT_FIELDS = [
    "decision_id",
    "decision_type",
    "entity_id",
    "signal",
    "direction",
    "priority",
    "evidence_level",
    "decision_readiness",
    "evidence_basis",
    "recommended_action",
]


def validate_decision_record(
    record: DecisionRecord,
) -> None:
    """
    Validate a canonical decision record.

    Raises
    ------
    ValueError
        If any required field is empty or invalid.
    """
    for field_name in DECISION_CONTRACT_FIELDS:
        value = getattr(record, field_name)

        if value is None:
            raise ValueError(
                f"Decision field '{field_name}' cannot be None."
            )

        if isinstance(value, str) and not value.strip():
            raise ValueError(
                f"Decision field '{field_name}' cannot be empty."
            )

    if record.priority not in {
        "CRITICAL",
        "HIGH",
        "MEDIUM",
        "LOW",
    }:
        raise ValueError(
            f"Invalid decision priority: {record.priority}"
        )

    if record.decision_readiness not in {
        "ACTION_CANDIDATE",
        "REVIEW",
        "OBSERVE",
        "NOT_READY",
    }:
        raise ValueError(
            f"Invalid decision readiness: "
            f"{record.decision_readiness}"
        )