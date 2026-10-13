from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.ai.reasoning_engine import (
    AIReasoningEngineError,
    build_deterministic_reasoning,
)
from src.marts.persistence import save_mart


class AIReasoningPersistenceError(ValueError):
    """Raised when AI reasoning persistence fails."""


REQUIRED_CONTEXT_COLUMNS = {
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


def persist_ai_reasoning(
    context_path: str | Path,
    output_dir: str | Path,
    mart_name: str = "ai_decision_reasoning",
) -> Path:
    """
    Build and persist deterministic AI reasoning from decision context.
    """

    context_path = Path(context_path)

    if not context_path.exists():
        raise AIReasoningPersistenceError(
            f"Decision context file does not exist: {context_path}"
        )

    try:
        context = pd.read_parquet(context_path)
    except Exception as exc:
        raise AIReasoningPersistenceError(
            f"Failed to read decision context: {exc}"
        ) from exc

    missing = REQUIRED_CONTEXT_COLUMNS - set(context.columns)

    if missing:
        raise AIReasoningPersistenceError(
            "Decision context is missing required columns: "
            f"{sorted(missing)}"
        )

    try:
        reasoning = build_deterministic_reasoning(context)
    except AIReasoningEngineError as exc:
        raise AIReasoningPersistenceError(
            f"Failed to build AI reasoning: {exc}"
        ) from exc

    if reasoning.empty:
        raise AIReasoningPersistenceError(
            "AI reasoning output is empty and cannot be persisted."
        )

    if reasoning["decision_id"].duplicated().any():
        raise AIReasoningPersistenceError(
            "AI reasoning contains duplicate decision_id values."
        )

    if reasoning.isnull().any().any():
        null_columns = reasoning.columns[
            reasoning.isnull().any()
        ].tolist()

        raise AIReasoningPersistenceError(
            "AI reasoning contains null values in columns: "
            f"{null_columns}"
        )

    try:
        return save_mart(
            reasoning,
            mart_name=mart_name,
            output_dir=output_dir,
        )
    except Exception as exc:
        raise AIReasoningPersistenceError(
            f"Failed to persist AI reasoning: {exc}"
        ) from exc