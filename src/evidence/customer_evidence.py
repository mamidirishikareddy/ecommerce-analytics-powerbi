"""
Customer evidence assessment layer.

This module consumes the governed customer signal mart and derives
decision-readiness classifications from the existing evidence level
and signal confidence.

The module does not recalculate customer lifecycle, value segments,
behavior direction, or behavioral transitions. Those remain governed
by the customer signal layer.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


class CustomerEvidenceError(ValueError):
    """Raised when customer evidence assessment cannot be completed."""


INPUT_COLUMNS = [
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
]


OUTPUT_COLUMNS = [
    *INPUT_COLUMNS,
    "evidence_qualification",
    "decision_readiness",
    "evidence_basis",
]


REQUIRED_INPUT_COLUMNS = {
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


ALLOWED_EVIDENCE_LEVELS = {
    "NONE",
    "LOW",
    "MEDIUM",
    "HIGH",
}


ALLOWED_SIGNAL_CONFIDENCE = {
    "NONE",
    "LOW",
    "MEDIUM",
    "HIGH",
}


ALLOWED_EVIDENCE_QUALIFICATIONS = {
    "INSUFFICIENT_EVIDENCE",
    "LIMITED_EVIDENCE",
    "MODERATE_EVIDENCE",
    "STRONG_EVIDENCE",
}


ALLOWED_DECISION_READINESS = {
    "NOT_READY",
    "OBSERVE",
    "REVIEW",
    "ACTION_CANDIDATE",
}


def _load_customer_signals(
    signals_path: str | Path,
) -> pd.DataFrame:
    """
    Load the persisted customer signal mart.
    """
    signals_path = Path(signals_path)

    if not signals_path.exists():
        raise FileNotFoundError(
            f"Customer signal mart not found: {signals_path}"
        )

    if signals_path.suffix.lower() != ".parquet":
        raise ValueError(
            "Customer signal input must be a Parquet file."
        )

    return pd.read_parquet(signals_path)


def validate_input(df: pd.DataFrame) -> None:
    """
    Validate the customer signal input contract.
    """
    missing_columns = sorted(
        REQUIRED_INPUT_COLUMNS - set(df.columns)
    )

    if missing_columns:
        raise CustomerEvidenceError(
            "Customer signal input is missing required columns: "
            f"{missing_columns}"
        )

    if df["customer_unique_id"].isna().any():
        raise CustomerEvidenceError(
            "customer_unique_id cannot contain null values."
        )

    if df["customer_unique_id"].duplicated().any():
        raise CustomerEvidenceError(
            "Customer signal input must contain one row per customer."
        )

    invalid_evidence_levels = (
        set(df["evidence_level"].dropna())
        - ALLOWED_EVIDENCE_LEVELS
    )

    if invalid_evidence_levels:
        raise CustomerEvidenceError(
            "Invalid evidence_level values: "
            f"{sorted(invalid_evidence_levels)}"
        )

    invalid_confidence = (
        set(df["signal_confidence"].dropna())
        - ALLOWED_SIGNAL_CONFIDENCE
    )

    if invalid_confidence:
        raise CustomerEvidenceError(
            "Invalid signal_confidence values: "
            f"{sorted(invalid_confidence)}"
        )

    if df["transition_count"].lt(0).any():
        raise CustomerEvidenceError(
            "transition_count cannot contain negative values."
        )

    if df["active_months"].lt(0).any():
        raise CustomerEvidenceError(
            "active_months cannot contain negative values."
        )

    if df["total_orders"].lt(0).any():
        raise CustomerEvidenceError(
            "total_orders cannot contain negative values."
        )

    if df["lifetime_revenue"].lt(0).any():
        raise CustomerEvidenceError(
            "lifetime_revenue cannot contain negative values."
        )

    if df["direction_consistency"].lt(0).any():
        raise CustomerEvidenceError(
            "direction_consistency cannot be below 0."
        )

    if df["direction_consistency"].gt(1).any():
        raise CustomerEvidenceError(
            "direction_consistency cannot exceed 1."
        )


def _classify_evidence(
    evidence_level: str,
    signal_confidence: str,
) -> tuple[str, str]:
    """
    Classify evidence strength and decision readiness.

    The classification uses only the governed evidence level and
    signal confidence already produced by the customer signal layer.
    """
    if evidence_level == "NONE":
        return (
            "INSUFFICIENT_EVIDENCE",
            "NOT_READY",
        )

    if evidence_level == "LOW":
        return (
            "LIMITED_EVIDENCE",
            "OBSERVE",
        )

    if evidence_level == "MEDIUM":
        if signal_confidence in {"MEDIUM", "HIGH"}:
            return (
                "MODERATE_EVIDENCE",
                "REVIEW",
            )

        return (
            "LIMITED_EVIDENCE",
            "OBSERVE",
        )

    if evidence_level == "HIGH":
        if signal_confidence == "HIGH":
            return (
                "STRONG_EVIDENCE",
                "ACTION_CANDIDATE",
            )

        if signal_confidence == "MEDIUM":
            return (
                "STRONG_EVIDENCE",
                "REVIEW",
            )

        return (
            "MODERATE_EVIDENCE",
            "REVIEW",
        )

    raise CustomerEvidenceError(
        f"Unsupported evidence level: {evidence_level}"
    )


def _build_evidence_basis(
    evidence_level: str,
    transition_count: int,
    direction_consistency: float,
) -> str:
    """
    Build a deterministic explanation of the evidence classification.
    """
    if evidence_level == "NONE":
        return (
            "No longitudinal behavioral transitions observed."
        )

    if evidence_level == "LOW":
        return (
            f"One behavioral transition observed "
            f"(transition_count={transition_count})."
        )

    if evidence_level == "MEDIUM":
        return (
            f"Two behavioral transitions observed "
            f"(transition_count={transition_count}) with "
            f"directional consistency of "
            f"{direction_consistency:.2f}."
        )

    if evidence_level == "HIGH":
        return (
            f"Three or more behavioral transitions observed "
            f"(transition_count={transition_count}) with "
            f"directional consistency of "
            f"{direction_consistency:.2f}."
        )

    raise CustomerEvidenceError(
        f"Unsupported evidence level: {evidence_level}"
    )


def build_customer_evidence(
    signals_path: str | Path,
) -> pd.DataFrame:
    """
    Build the governed customer evidence assessment dataset.

    Parameters
    ----------
    signals_path:
        Path to mart_customer_signals.parquet.

    Returns
    -------
    pandas.DataFrame
        One governed evidence assessment row per customer.
    """
    signals = _load_customer_signals(
        signals_path
    )

    validate_input(signals)

    result = signals[INPUT_COLUMNS].copy()

    classifications = [
        _classify_evidence(
            evidence_level=row.evidence_level,
            signal_confidence=row.signal_confidence,
        )
        for row in result.itertuples(index=False)
    ]

    result[
        [
            "evidence_qualification",
            "decision_readiness",
        ]
    ] = classifications

    result["evidence_basis"] = [
        _build_evidence_basis(
            evidence_level=row.evidence_level,
            transition_count=int(row.transition_count),
            direction_consistency=float(
                row.direction_consistency
            ),
        )
        for row in result.itertuples(index=False)
    ]

    invalid_qualifications = (
        set(result["evidence_qualification"])
        - ALLOWED_EVIDENCE_QUALIFICATIONS
    )

    if invalid_qualifications:
        raise CustomerEvidenceError(
            "Invalid evidence_qualification values: "
            f"{sorted(invalid_qualifications)}"
        )

    invalid_readiness = (
        set(result["decision_readiness"])
        - ALLOWED_DECISION_READINESS
    )

    if invalid_readiness:
        raise CustomerEvidenceError(
            "Invalid decision_readiness values: "
            f"{sorted(invalid_readiness)}"
        )

    if result["customer_unique_id"].duplicated().any():
        raise CustomerEvidenceError(
            "Customer evidence output must contain one row per customer."
        )

    return result[OUTPUT_COLUMNS].copy()