from __future__ import annotations

import pandas as pd


class AIDecisionInputError(ValueError):
    """Raised when an AI decision input cannot be constructed."""


CONTEXT_REQUIRED_COLUMNS = {
    "decision_id",
    "decision_type",
    "entity_id",
    "signal",
    "direction",
    "priority",
    "queue_rank",
    "decision_readiness",
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


AI_INPUT_COLUMNS = [
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
]


def _validate_required_columns(
    context: pd.DataFrame,
) -> None:
    """Validate the decision context contract."""

    missing = CONTEXT_REQUIRED_COLUMNS - set(context.columns)

    if missing:
        raise AIDecisionInputError(
            "Decision context is missing required columns: "
            f"{sorted(missing)}"
        )


def build_ai_decision_input(
    context: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build the governed input contract for downstream AI reasoning.

    This function performs no new scoring, classification, prioritization,
    recommendation, or behavioral inference.

    It only validates and selects fields already established by the
    Decision Intelligence layers.
    """

    if not isinstance(context, pd.DataFrame):
        raise AIDecisionInputError(
            "Decision context must be a pandas DataFrame."
        )

    _validate_required_columns(context)

    if context.empty:
        raise AIDecisionInputError(
            "Decision context cannot be empty."
        )

    if context["decision_id"].duplicated().any():
        raise AIDecisionInputError(
            "Decision context contains duplicate decision_id values."
        )

    if context["entity_id"].duplicated().any():
        raise AIDecisionInputError(
            "Decision context contains duplicate entity_id values."
        )

    if context.isnull().any().any():
        null_columns = context.columns[
            context.isnull().any()
        ].tolist()

        raise AIDecisionInputError(
            "Decision context contains null values in columns: "
            f"{null_columns}"
        )

    ai_input = context[AI_INPUT_COLUMNS].copy()

    if list(ai_input.columns) != AI_INPUT_COLUMNS:
        raise AIDecisionInputError(
            "AI decision input columns do not match the expected contract."
        )

    return ai_input