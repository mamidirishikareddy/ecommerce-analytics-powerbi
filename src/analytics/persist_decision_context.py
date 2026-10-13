from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.analytics.decision_context import (
    DecisionContextError,
    build_decision_context,
)
from src.marts.persistence import save_mart


class DecisionContextPersistenceError(ValueError):
    """Raised when decision context persistence fails."""


QUEUE_REQUIRED_COLUMNS = {
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


SIGNAL_REQUIRED_COLUMNS = {
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


def persist_decision_context(
    queue_path: str | Path,
    signals_path: str | Path,
    output_dir: str | Path,
    mart_name: str = "decision_context",
) -> Path:
    """
    Build and persist the production decision context artifact.

    Parameters
    ----------
    queue_path:
        Path to the persisted decision queue parquet.

    signals_path:
        Path to the persisted customer signals parquet.

    output_dir:
        Directory where the decision context mart will be persisted.

    mart_name:
        Output mart name without the .parquet extension.

    Returns
    -------
    Path
        Path to the persisted decision context parquet.
    """

    queue_path = Path(queue_path)
    signals_path = Path(signals_path)

    if not queue_path.exists():
        raise DecisionContextPersistenceError(
            f"Decision queue file does not exist: {queue_path}"
        )

    if not signals_path.exists():
        raise DecisionContextPersistenceError(
            f"Customer signals file does not exist: {signals_path}"
        )

    try:
        queue = pd.read_parquet(queue_path)
        signals = pd.read_parquet(signals_path)
    except Exception as exc:
        raise DecisionContextPersistenceError(
            f"Failed to read decision context inputs: {exc}"
        ) from exc

    missing_queue = QUEUE_REQUIRED_COLUMNS - set(queue.columns)

    if missing_queue:
        raise DecisionContextPersistenceError(
            "Decision queue is missing required columns: "
            f"{sorted(missing_queue)}"
        )

    missing_signals = SIGNAL_REQUIRED_COLUMNS - set(signals.columns)

    if missing_signals:
        raise DecisionContextPersistenceError(
            "Customer signals are missing required columns: "
            f"{sorted(missing_signals)}"
        )

    try:
        context = build_decision_context(
            queue=queue,
            signals=signals,
        )
    except DecisionContextError as exc:
        raise DecisionContextPersistenceError(
            f"Failed to build decision context: {exc}"
        ) from exc

    if context.empty:
        raise DecisionContextPersistenceError(
            "Decision context is empty and cannot be persisted."
        )

    if context["decision_id"].duplicated().any():
        raise DecisionContextPersistenceError(
            "Decision context contains duplicate decision_id values."
        )

    if context["entity_id"].duplicated().any():
        raise DecisionContextPersistenceError(
            "Decision context contains duplicate entity_id values."
        )

    if context.isnull().any().any():
        null_columns = context.columns[
            context.isnull().any()
        ].tolist()

        raise DecisionContextPersistenceError(
            "Decision context contains null values in columns: "
            f"{null_columns}"
        )

    try:
        return save_mart(
            context,
            mart_name=mart_name,
            output_dir=output_dir,
        )
    except Exception as exc:
        raise DecisionContextPersistenceError(
            f"Failed to persist decision context: {exc}"
        ) from exc