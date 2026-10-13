from __future__ import annotations

import pandas as pd


class DecisionContextError(ValueError):
    """Raised when decision context cannot be constructed."""


QUEUE_COLUMNS = {
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
    "queue_rank",
}


SIGNAL_COLUMNS = {
    "customer_unique_id",
    "signal_date",
    "customer_lifecycle",
    "value_segment",
    "behavior_direction",
    "evidence_level",
    "active_months",
    "total_orders",
    "lifetime_revenue",
    "transition_count",
    "direction_consistency",
    "signal_confidence",
}


CONTEXT_COLUMNS = [
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
]


def _validate_required_columns(
    frame: pd.DataFrame,
    required_columns: set[str],
    frame_name: str,
) -> None:
    """Validate that a dataframe contains all required columns."""
    missing = required_columns - set(frame.columns)

    if missing:
        raise DecisionContextError(
            f"{frame_name} is missing required columns: "
            f"{sorted(missing)}"
        )


def build_decision_context(
    queue: pd.DataFrame,
    signals: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build a deterministic decision context table.

    The decision queue supplies the operational decision information,
    while the customer signals supply the governed behavioral evidence.

    No new scoring, classification, or recommendation logic is introduced.
    """

    if not isinstance(queue, pd.DataFrame):
        raise DecisionContextError("queue must be a pandas DataFrame.")

    if not isinstance(signals, pd.DataFrame):
        raise DecisionContextError("signals must be a pandas DataFrame.")

    _validate_required_columns(
        queue,
        QUEUE_COLUMNS,
        "Decision queue",
    )

    _validate_required_columns(
        signals,
        SIGNAL_COLUMNS,
        "Customer signals",
    )

    if queue.empty:
        raise DecisionContextError(
            "Decision queue cannot be empty."
        )

    if signals.empty:
        raise DecisionContextError(
            "Customer signals cannot be empty."
        )

    if queue["decision_id"].duplicated().any():
        raise DecisionContextError(
            "Decision queue contains duplicate decision_id values."
        )

    if signals["customer_unique_id"].duplicated().any():
        raise DecisionContextError(
            "Customer signals contain duplicate customer_unique_id values."
        )

    context = queue.merge(
        signals,
        left_on="entity_id",
        right_on="customer_unique_id",
        how="left",
        validate="one_to_one",
        suffixes=("_decision", "_signal"),
    )

    if context["customer_unique_id"].isna().any():
        unmatched = context.loc[
            context["customer_unique_id"].isna(),
            "entity_id",
        ].tolist()

        raise DecisionContextError(
            "Decision queue contains entities without matching customer "
            f"signals: {unmatched}"
        )

    if not (
        context["evidence_level_decision"]
        == context["evidence_level_signal"]
    ).all():
        raise DecisionContextError(
            "Decision evidence level does not match customer signal "
            "evidence level."
        )

    context["evidence_level"] = context["evidence_level_signal"]

    context = context[CONTEXT_COLUMNS].copy()

    if list(context.columns) != CONTEXT_COLUMNS:
        raise DecisionContextError(
            "Decision context columns do not match the expected contract."
        )

    if context.isnull().any().any():
        null_columns = context.columns[
            context.isnull().any()
        ].tolist()

        raise DecisionContextError(
            "Decision context contains null values in columns: "
            f"{null_columns}"
        )

    return context