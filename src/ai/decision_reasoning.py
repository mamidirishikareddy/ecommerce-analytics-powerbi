from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Literal

import pandas as pd


class AIDecisionReasoningError(ValueError):
    """Raised when an AI decision reasoning record is invalid."""


GroundingStatus = Literal[
    "GROUNDED",
    "INSUFFICIENT_EVIDENCE",
]

HumanReviewRequired = Literal[
    "YES",
    "NO",
]


@dataclass(frozen=True)
class AIDecisionReasoningRecord:
    """Structured, auditable output produced by the AI reasoning layer."""

    decision_id: str
    executive_summary: str
    evidence_used: str
    reasoning: str
    recommended_next_step: str
    risk_or_caveat: str
    human_review_required: HumanReviewRequired
    grounding_status: GroundingStatus


AI_REASONING_COLUMNS = [
    "decision_id",
    "executive_summary",
    "evidence_used",
    "reasoning",
    "recommended_next_step",
    "risk_or_caveat",
    "human_review_required",
    "grounding_status",
]


ALLOWED_GROUNDING_STATUS = {
    "GROUNDED",
    "INSUFFICIENT_EVIDENCE",
}

ALLOWED_HUMAN_REVIEW = {
    "YES",
    "NO",
}


def validate_ai_reasoning_record(
    record: AIDecisionReasoningRecord,
) -> None:
    """Validate a single AI reasoning record."""

    if not isinstance(record, AIDecisionReasoningRecord):
        raise AIDecisionReasoningError(
            "record must be an AIDecisionReasoningRecord."
        )

    values = asdict(record)

    for field, value in values.items():
        if value is None:
            raise AIDecisionReasoningError(
                f"AI reasoning field '{field}' cannot be null."
            )

        if isinstance(value, str) and not value.strip():
            raise AIDecisionReasoningError(
                f"AI reasoning field '{field}' cannot be empty."
            )

    if record.grounding_status not in ALLOWED_GROUNDING_STATUS:
        raise AIDecisionReasoningError(
            "Invalid grounding_status: "
            f"{record.grounding_status}"
        )

    if record.human_review_required not in ALLOWED_HUMAN_REVIEW:
        raise AIDecisionReasoningError(
            "Invalid human_review_required: "
            f"{record.human_review_required}"
        )


def build_ai_reasoning_frame(
    records: list[AIDecisionReasoningRecord],
) -> pd.DataFrame:
    """
    Convert validated AI reasoning records into a governed dataframe.

    This function performs no AI reasoning itself.
    """

    if not records:
        raise AIDecisionReasoningError(
            "At least one AI reasoning record is required."
        )

    for record in records:
        validate_ai_reasoning_record(record)

    decision_ids = [record.decision_id for record in records]

    if len(decision_ids) != len(set(decision_ids)):
        raise AIDecisionReasoningError(
            "AI reasoning contains duplicate decision_id values."
        )

    frame = pd.DataFrame(
        [asdict(record) for record in records],
        columns=AI_REASONING_COLUMNS,
    )

    if list(frame.columns) != AI_REASONING_COLUMNS:
        raise AIDecisionReasoningError(
            "AI reasoning columns do not match the expected contract."
        )

    if frame.isnull().any().any():
        null_columns = frame.columns[
            frame.isnull().any()
        ].tolist()

        raise AIDecisionReasoningError(
            "AI reasoning contains null values in columns: "
            f"{null_columns}"
        )

    return frame