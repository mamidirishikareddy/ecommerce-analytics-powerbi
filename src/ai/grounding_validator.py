from __future__ import annotations

import pandas as pd


class GroundingValidationError(ValueError):
    """Raised when LLM reasoning violates the governed decision contract."""


CONTEXT_REQUIRED_COLUMNS = {
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


REASONING_REQUIRED_COLUMNS = {
    "decision_id",
    "executive_summary",
    "evidence_used",
    "reasoning",
    "recommended_next_step",
    "risk_or_caveat",
    "human_review_required",
    "grounding_status",
}


ALLOWED_HUMAN_REVIEW = {"YES", "NO"}

ALLOWED_GROUNDING_STATUS = {
    "GROUNDED",
    "INSUFFICIENT_EVIDENCE",
}


def _validate_frame(
    frame: pd.DataFrame,
    required_columns: set[str],
    frame_name: str,
) -> None:
    """Validate basic dataframe structure."""

    if not isinstance(frame, pd.DataFrame):
        raise GroundingValidationError(
            f"{frame_name} must be a pandas DataFrame."
        )

    if frame.empty:
        raise GroundingValidationError(
            f"{frame_name} cannot be empty."
        )

    missing = required_columns - set(frame.columns)

    if missing:
        raise GroundingValidationError(
            f"{frame_name} is missing required columns: "
            f"{sorted(missing)}"
        )

    if frame.isnull().any().any():
        null_columns = frame.columns[
            frame.isnull().any()
        ].tolist()

        raise GroundingValidationError(
            f"{frame_name} contains null values in columns: "
            f"{null_columns}"
        )


def _validate_unique_ids(
    context: pd.DataFrame,
    reasoning: pd.DataFrame,
) -> None:
    """Ensure both sides contain one row per decision."""

    if context["decision_id"].duplicated().any():
        raise GroundingValidationError(
            "Decision context contains duplicate decision_id values."
        )

    if reasoning["decision_id"].duplicated().any():
        raise GroundingValidationError(
            "LLM reasoning contains duplicate decision_id values."
        )


def _validate_decision_coverage(
    context: pd.DataFrame,
    reasoning: pd.DataFrame,
) -> None:
    """Ensure LLM output covers exactly the governed decisions."""

    context_ids = set(context["decision_id"])
    reasoning_ids = set(reasoning["decision_id"])

    missing = context_ids - reasoning_ids
    unexpected = reasoning_ids - context_ids

    if missing:
        raise GroundingValidationError(
            "LLM reasoning is missing governed decisions: "
            f"{sorted(missing)}"
        )

    if unexpected:
        raise GroundingValidationError(
            "LLM reasoning contains unknown decision_id values: "
            f"{sorted(unexpected)}"
        )


def _validate_output_statuses(
    reasoning: pd.DataFrame,
) -> None:
    """Validate controlled LLM status fields."""

    invalid_review = set(
        reasoning["human_review_required"]
    ) - ALLOWED_HUMAN_REVIEW

    if invalid_review:
        raise GroundingValidationError(
            "LLM reasoning contains invalid human_review_required "
            f"values: {sorted(invalid_review)}"
        )

    invalid_grounding = set(
        reasoning["grounding_status"]
    ) - ALLOWED_GROUNDING_STATUS

    if invalid_grounding:
        raise GroundingValidationError(
            "LLM reasoning contains invalid grounding_status "
            f"values: {sorted(invalid_grounding)}"
        )


def _validate_governed_values(
    context: pd.DataFrame,
    reasoning: pd.DataFrame,
) -> None:
    """Ensure the LLM did not redefine governed decisions."""

    merged = context[
        [
            "decision_id",
            "recommended_action",
            "decision_readiness",
        ]
    ].merge(
        reasoning[
            [
                "decision_id",
                "recommended_next_step",
                "human_review_required",
            ]
        ],
        on="decision_id",
        how="left",
        validate="one_to_one",
    )

    recommendation_mismatch = (
        merged["recommended_next_step"]
        != merged["recommended_action"]
    )

    if recommendation_mismatch.any():
        decision_ids = merged.loc[
            recommendation_mismatch,
            "decision_id",
        ].tolist()

        raise GroundingValidationError(
            "LLM changed the governed recommended action for "
            f"decision(s): {decision_ids}"
        )

    expected_review = merged["decision_readiness"].isin(
        {"REVIEW", "ACTION_CANDIDATE"}
    )

    actual_review = (
        merged["human_review_required"] == "YES"
    )

    review_mismatch = expected_review != actual_review

    if review_mismatch.any():
        decision_ids = merged.loc[
            review_mismatch,
            "decision_id",
        ].tolist()

        raise GroundingValidationError(
            "LLM changed the governed human-review requirement for "
            f"decision(s): {decision_ids}"
        )


def validate_llm_grounding(
    context: pd.DataFrame,
    reasoning: pd.DataFrame,
) -> bool:
    """
    Validate that LLM reasoning remains grounded in governed decisions.

    The validator does not judge whether the prose is eloquent.
    It enforces that the LLM cannot change the governed recommendation,
    decision identity, or human-review requirement.
    """

    _validate_frame(
        context,
        CONTEXT_REQUIRED_COLUMNS,
        "Decision context",
    )

    _validate_frame(
        reasoning,
        REASONING_REQUIRED_COLUMNS,
        "LLM reasoning",
    )

    _validate_unique_ids(context, reasoning)

    _validate_decision_coverage(context, reasoning)

    _validate_output_statuses(reasoning)

    _validate_governed_values(context, reasoning)

    return True