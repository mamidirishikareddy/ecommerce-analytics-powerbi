from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.analytics.decision_queue import (
    DecisionQueueError,
    build_decision_queue,
)
from src.marts.persistence import MartPersistenceError, save_mart


class DecisionQueuePersistenceError(ValueError):
    """Raised when decision queue persistence fails."""


REQUIRED_DECISION_COLUMNS = {
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
}


def persist_decision_queue(
    decisions_path: str | Path,
    output_dir: str | Path,
) -> Path:
    """
    Build and persist the operational decision queue.

    The queue contains only REVIEW and ACTION_CANDIDATE decisions.
    No new evidence or scoring is created during persistence.
    """
    decisions_path = Path(decisions_path)
    output_dir = Path(output_dir)

    if not decisions_path.exists():
        raise DecisionQueuePersistenceError(
            f"Decision file does not exist: {decisions_path}"
        )

    try:
        decisions = pd.read_parquet(decisions_path)
    except Exception as exc:
        raise DecisionQueuePersistenceError(
            f"Failed to read decisions: {exc}"
        ) from exc

    missing_columns = REQUIRED_DECISION_COLUMNS - set(decisions.columns)

    if missing_columns:
        raise DecisionQueuePersistenceError(
            "Decisions are missing required columns: "
            f"{sorted(missing_columns)}"
        )

    try:
        queue = build_decision_queue(decisions)
    except DecisionQueueError as exc:
        raise DecisionQueuePersistenceError(
            f"Failed to build decision queue: {exc}"
        ) from exc

    try:
        return save_mart(
            queue,
            mart_name="decision_queue",
            output_dir=output_dir,
        )
    except MartPersistenceError as exc:
        raise DecisionQueuePersistenceError(
            f"Failed to persist decision queue: {exc}"
        ) from exc