"""
Decision engine for governed customer decision candidates.

Converts customer signals and evidence qualification into
decision-intelligence records without inventing unsupported actions.
"""

from __future__ import annotations

import pandas as pd

from src.analytics.decision_contract import (
    DecisionRecord,
    validate_decision_record,
)


class DecisionEngineError(ValueError):
    """Raised when decision candidates cannot be generated."""


REQUIRED_COLUMNS = {
    "customer_unique_id",
    "customer_lifecycle",
    "value_segment",
    "behavior_direction",
    "evidence_level",
    "signal_confidence",
    "evidence_qualification",
    "decision_readiness",
    "evidence_basis",
}


def _priority(
    value_segment: str,
    decision_readiness: str,
    behavior_direction: str,
) -> str:
    """Assign decision priority from value, readiness, and behavior."""
    if decision_readiness == "ACTION_CANDIDATE":
        if (
            value_segment == "HIGH_VALUE"
            and behavior_direction == "DECREASING"
        ):
            return "CRITICAL"

        return "HIGH"

    if decision_readiness == "REVIEW":
        if value_segment == "HIGH_VALUE":
            return "HIGH"

        return "MEDIUM"

    if decision_readiness == "OBSERVE":
        return "LOW"

    return "LOW"


def _decision_type(
    lifecycle: str,
    behavior_direction: str,
) -> str:
    """Map customer behavior to a governed decision category."""
    if lifecycle == "DORMANT":
        return "CUSTOMER_RETENTION"

    if behavior_direction == "DECREASING":
        return "CUSTOMER_VALUE"

    if behavior_direction == "INCREASING":
        return "CUSTOMER_VALUE"

    if behavior_direction == "STABLE":
        return "CUSTOMER_VALUE"

    return "CUSTOMER_VALUE"


def _signal(
    lifecycle: str,
    value_segment: str,
    behavior_direction: str,
) -> str:
    """Build a human-readable customer signal."""
    return (
        f"{lifecycle} customer in {value_segment} segment "
        f"with {behavior_direction} behavior."
    )


def _direction(
    behavior_direction: str,
) -> str:
    """Return the governed direction."""
    return behavior_direction


def _recommended_action(
    decision_readiness: str,
    lifecycle: str,
    behavior_direction: str,
) -> str:
    """
    Generate a conservative action statement.

    Actions remain intentionally review-oriented unless the evidence
    layer explicitly qualifies the decision as an action candidate.
    """
    if decision_readiness == "NOT_READY":
        return "Do not act; insufficient behavioral evidence."

    if decision_readiness == "OBSERVE":
        return "Monitor the customer signal for additional evidence."

    if decision_readiness == "REVIEW":
        if lifecycle == "DORMANT":
            return "Review retention intervention options."
        if behavior_direction == "DECREASING":
            return "Review customer value-risk intervention options."
        if behavior_direction == "INCREASING":
            return "Review opportunities to reinforce positive behavior."
        return "Review the customer behavior before intervention."

    if lifecycle == "DORMANT":
        return "Prioritize retention intervention review."

    if behavior_direction == "DECREASING":
        return "Prioritize customer value-risk intervention."

    if behavior_direction == "INCREASING":
        return "Consider reinforcing the positive behavior."

    return "Review the signal before taking action."


def build_customer_decisions(
    signals: pd.DataFrame,
) -> pd.DataFrame:
    """
    Convert governed customer signals into decision candidates.

    Parameters
    ----------
    signals:
        Customer signal/evidence dataset.

    Returns
    -------
    pd.DataFrame
        One governed decision candidate per customer.
    """
    missing_columns = REQUIRED_COLUMNS - set(signals.columns)

    if missing_columns:
        raise DecisionEngineError(
            "Customer signals are missing required columns: "
            f"{sorted(missing_columns)}"
        )

    if signals.empty:
        raise DecisionEngineError(
            "Customer signals cannot be empty."
        )

    if signals["customer_unique_id"].duplicated().any():
        raise DecisionEngineError(
            "Customer signals must contain one row per customer."
        )

    records: list[DecisionRecord] = []

    for row in signals.itertuples(index=False):
        priority = _priority(
    row.value_segment,
    row.decision_readiness,
    row.behavior_direction,
)

        decision = DecisionRecord(
            decision_id=f"DEC-CUSTOMER-{row.customer_unique_id}",
            decision_type=_decision_type(
                row.customer_lifecycle,
                row.behavior_direction,
            ),
            entity_id=row.customer_unique_id,
            signal=_signal(
                row.customer_lifecycle,
                row.value_segment,
                row.behavior_direction,
            ),
            direction=_direction(
                row.behavior_direction,
            ),
            priority=priority,
            evidence_level=row.evidence_level,
            decision_readiness=row.decision_readiness,
            evidence_basis=row.evidence_basis,
            recommended_action=_recommended_action(
                row.decision_readiness,
                row.customer_lifecycle,
                row.behavior_direction,
            ),
        )

        validate_decision_record(decision)
        records.append(decision)

    return pd.DataFrame(
        [
            {
                field: getattr(record, field)
                for field in (
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
                )
            }
            for record in records
        ]
    )