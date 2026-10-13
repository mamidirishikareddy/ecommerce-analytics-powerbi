from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.analytics.decision_engine import build_customer_decisions
from src.marts.persistence import MartPersistenceError, save_mart


class DecisionPersistenceError(ValueError):
    """Raised when customer decision persistence fails."""


REQUIRED_SIGNAL_COLUMNS = {
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


def persist_customer_decisions(
    signals_path: str | Path,
    output_dir: str | Path,
) -> Path:
    """
    Build governed customer decisions from the persisted signal/evidence
    dataset and persist the resulting decision artifact.

    Parameters
    ----------
    signals_path:
        Path to the persisted customer signal/evidence parquet.

    output_dir:
        Directory where the decision artifact will be persisted.

    Returns
    -------
    Path
        Path to the persisted customer decisions parquet.
    """
    signals_path = Path(signals_path)
    output_dir = Path(output_dir)

    if not signals_path.exists():
        raise DecisionPersistenceError(
            f"Customer signals file does not exist: {signals_path}"
        )

    try:
        signals = pd.read_parquet(signals_path)
    except Exception as exc:
        raise DecisionPersistenceError(
            f"Failed to read customer signals: {exc}"
        ) from exc

    missing_columns = REQUIRED_SIGNAL_COLUMNS - set(signals.columns)

    if missing_columns:
        raise DecisionPersistenceError(
            "Customer signals are missing required columns: "
            f"{sorted(missing_columns)}"
        )

    try:
        decisions = build_customer_decisions(signals)
    except Exception as exc:
        raise DecisionPersistenceError(
            f"Failed to build customer decisions: {exc}"
        ) from exc

    try:
        return save_mart(
            decisions,
            mart_name="customer_decisions",
            output_dir=output_dir,
        )
    except MartPersistenceError as exc:
        raise DecisionPersistenceError(
            f"Failed to persist customer decisions: {exc}"
        ) from exc