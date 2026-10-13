from __future__ import annotations

import pandas as pd

from src.ai.decision_input import (
    AIDecisionInputError,
    build_ai_decision_input,
)
from src.ai.decision_reasoning import (
    AIDecisionReasoningRecord,
    build_ai_reasoning_frame,
)


class AIReasoningEngineError(ValueError):
    """Raised when deterministic AI reasoning cannot be generated."""


REQUIRED_COLUMNS = {
    "decision_id",
    "decision_type",
    "entity_id",
    "priority",
    "queue_rank",
    "decision_readiness",
    "signal",
    "direction",
    "evidence_level",
    "signal_confidence",
    "customer_lifecycle",
    "value_segment",
    "active_months",
    "total_orders",
    "lifetime_revenue",
    "transition_count",
    "direction_consistency",
    "evidence_basis",
    "recommended_action",
}


def _validate_input_columns(
    frame: pd.DataFrame,
) -> None:
    """Validate the governed AI decision input."""

    missing = REQUIRED_COLUMNS - set(frame.columns)

    if missing:
        raise AIReasoningEngineError(
            "AI decision input is missing required columns: "
            f"{sorted(missing)}"
        )


def _build_executive_summary(row: pd.Series) -> str:
    """Build a concise factual summary from governed fields."""

    return (
        f"{row['value_segment'].replace('_', ' ').lower()} customer "
        f"with {row['customer_lifecycle'].replace('_', ' ').lower()} "
        f"lifecycle and {row['direction'].lower()} behavioral direction."
    )


def _build_evidence_used(row: pd.Series) -> str:
    """Describe the evidence already established by the signal layer."""

    return (
        f"{row['evidence_level']} evidence with "
        f"{int(row['transition_count'])} behavioral transition(s), "
        f"direction consistency of "
        f"{float(row['direction_consistency']):.2f}, "
        f"and {row['signal_confidence'].lower()} signal confidence."
    )


def _build_reasoning(row: pd.Series) -> str:
    """Generate deterministic reasoning from governed attributes."""

    direction = row["direction"]
    readiness = row["decision_readiness"]
    evidence = row["evidence_level"]

    if readiness == "NOT_READY":
        return (
            "The available behavioral evidence is insufficient to support "
            "an intervention decision."
        )

    if readiness == "OBSERVE":
        return (
            f"The observed {direction.lower()} behavioral direction is "
            f"based on {evidence.lower()} evidence and should be observed "
            "until additional behavioral history is available."
        )

    if readiness == "REVIEW":
        return (
            f"The customer shows a {direction.lower()} behavioral pattern "
            f"with {evidence.lower()} evidence. The signal is sufficiently "
            "relevant for human review, but the evidence does not by itself "
            "establish causality or guarantee that an intervention will work."
        )

    if readiness == "ACTION_CANDIDATE":
        return (
            f"The customer shows a {direction.lower()} behavioral pattern "
            f"supported by {evidence.lower()} evidence. The combination of "
            "behavioral direction, value, and evidence strength makes this "
            "an action candidate under the governed decision framework."
        )

    raise AIReasoningEngineError(
        f"Unsupported decision readiness: {readiness}"
    )


def _build_next_step(row: pd.Series) -> str:
    """Translate the governed recommendation into an AI next-step field."""

    return row["recommended_action"]


def _build_risk_or_caveat(row: pd.Series) -> str:
    """Attach evidence and observational limitations."""

    evidence = row["evidence_level"]
    transition_count = int(row["transition_count"])

    if evidence == "HIGH":
        return (
            f"Reasoning is based on {transition_count} observed behavioral "
            "transition(s), but the dataset does not establish causality "
            "or intervention effectiveness."
        )

    if evidence == "MEDIUM":
        return (
            f"Reasoning is based on {transition_count} observed behavioral "
            "transition(s). Evidence remains limited and should be reviewed "
            "before intervention."
        )

    if evidence == "LOW":
        return (
            "Only one behavioral transition is observed, so the signal "
            "should not be treated as a reliable longitudinal pattern."
        )

    return (
        "No longitudinal behavioral transitions are observed; "
        "the signal should not be used to justify intervention."
    )


def _human_review_required(row: pd.Series) -> str:
    """Determine review requirement from the governed decision readiness."""

    if row["decision_readiness"] in {
        "ACTION_CANDIDATE",
        "REVIEW",
    }:
        return "YES"

    return "NO"


def _grounding_status(row: pd.Series) -> str:
    """Determine whether the reasoning is grounded in sufficient evidence."""

    if row["evidence_level"] in {
        "MEDIUM",
        "HIGH",
    }:
        return "GROUNDED"

    return "INSUFFICIENT_EVIDENCE"


def build_deterministic_reasoning(
    ai_input: pd.DataFrame,
) -> pd.DataFrame:
    """
    Generate deterministic AI reasoning from governed decision input.

    This is a reference reasoning implementation. It does not call an
    external AI model and does not introduce new decision scores,
    classifications, or recommendations.
    """

    if not isinstance(ai_input, pd.DataFrame):
        raise AIReasoningEngineError(
            "ai_input must be a pandas DataFrame."
        )

    _validate_input_columns(ai_input)

    if ai_input.empty:
        raise AIReasoningEngineError(
            "AI decision input cannot be empty."
        )

    try:
        governed_input = build_ai_decision_input(ai_input)
    except AIDecisionInputError as exc:
        raise AIReasoningEngineError(
            f"Invalid AI decision input: {exc}"
        ) from exc

    records: list[AIDecisionReasoningRecord] = []

    for _, row in governed_input.iterrows():
        records.append(
            AIDecisionReasoningRecord(
                decision_id=str(row["decision_id"]),
                executive_summary=_build_executive_summary(row),
                evidence_used=_build_evidence_used(row),
                reasoning=_build_reasoning(row),
                recommended_next_step=_build_next_step(row),
                risk_or_caveat=_build_risk_or_caveat(row),
                human_review_required=_human_review_required(row),
                grounding_status=_grounding_status(row),
            )
        )

    try:
        return build_ai_reasoning_frame(records)
    except Exception as exc:
        if isinstance(exc, AIReasoningEngineError):
            raise

        raise AIReasoningEngineError(
            f"Failed to construct AI reasoning output: {exc}"
        ) from exc