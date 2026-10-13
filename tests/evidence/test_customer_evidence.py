"""
Tests for the governed customer evidence assessment layer.
"""

from pathlib import Path

import pandas as pd
import pytest

from src.evidence.customer_evidence import (
    ALLOWED_DECISION_READINESS,
    ALLOWED_EVIDENCE_QUALIFICATIONS,
    CustomerEvidenceError,
    _build_evidence_basis,
    _classify_evidence,
    build_customer_evidence,
    validate_input,
)


SIGNALS_PATH = Path(
    "data/processed/marts/mart_customer_signals.parquet"
)


def _make_signal_row(
    customer_id="customer_001",
    evidence_level="NONE",
    signal_confidence="NONE",
    transition_count=0,
    direction_consistency=0.0,
):
    return {
        "customer_unique_id": customer_id,
        "signal_date": pd.Timestamp("2018-09-01"),
        "customer_lifecycle": "ONE_TIME",
        "value_segment": "LOW_VALUE",
        "behavior_direction": "INSUFFICIENT_EVIDENCE",
        "evidence_level": evidence_level,
        "active_months": 1,
        "total_orders": 1,
        "lifetime_revenue": 100.0,
        "transition_count": transition_count,
        "direction_consistency": direction_consistency,
        "signal_confidence": signal_confidence,
    }


def test_none_evidence_is_not_ready():
    qualification, readiness = _classify_evidence(
        "NONE",
        "NONE",
    )

    assert qualification == "INSUFFICIENT_EVIDENCE"
    assert readiness == "NOT_READY"


def test_low_evidence_is_observe():
    qualification, readiness = _classify_evidence(
        "LOW",
        "LOW",
    )

    assert qualification == "LIMITED_EVIDENCE"
    assert readiness == "OBSERVE"


def test_medium_low_confidence_is_observe():
    qualification, readiness = _classify_evidence(
        "MEDIUM",
        "LOW",
    )

    assert qualification == "LIMITED_EVIDENCE"
    assert readiness == "OBSERVE"


def test_medium_medium_confidence_requires_review():
    qualification, readiness = _classify_evidence(
        "MEDIUM",
        "MEDIUM",
    )

    assert qualification == "MODERATE_EVIDENCE"
    assert readiness == "REVIEW"


def test_medium_high_confidence_requires_review():
    qualification, readiness = _classify_evidence(
        "MEDIUM",
        "HIGH",
    )

    assert qualification == "MODERATE_EVIDENCE"
    assert readiness == "REVIEW"


def test_high_medium_confidence_requires_review():
    qualification, readiness = _classify_evidence(
        "HIGH",
        "MEDIUM",
    )

    assert qualification == "STRONG_EVIDENCE"
    assert readiness == "REVIEW"


def test_high_high_confidence_is_action_candidate():
    qualification, readiness = _classify_evidence(
        "HIGH",
        "HIGH",
    )

    assert qualification == "STRONG_EVIDENCE"
    assert readiness == "ACTION_CANDIDATE"


def test_invalid_evidence_level_is_rejected():
    with pytest.raises(CustomerEvidenceError):
        _classify_evidence(
            "INVALID",
            "LOW",
        )


def test_invalid_input_evidence_level_is_rejected():
    df = pd.DataFrame(
        [
            _make_signal_row(
                evidence_level="INVALID",
                signal_confidence="LOW",
            )
        ]
    )

    with pytest.raises(CustomerEvidenceError):
        validate_input(df)


def test_invalid_input_confidence_is_rejected():
    df = pd.DataFrame(
        [
            _make_signal_row(
                evidence_level="LOW",
                signal_confidence="INVALID",
            )
        ]
    )

    with pytest.raises(CustomerEvidenceError):
        validate_input(df)


def test_negative_transition_count_is_rejected():
    df = pd.DataFrame(
        [
            _make_signal_row(
                transition_count=-1,
            )
        ]
    )

    with pytest.raises(CustomerEvidenceError):
        validate_input(df)


def test_direction_consistency_above_one_is_rejected():
    df = pd.DataFrame(
        [
            _make_signal_row(
                direction_consistency=1.1,
            )
        ]
    )

    with pytest.raises(CustomerEvidenceError):
        validate_input(df)


def test_duplicate_customer_is_rejected():
    row = _make_signal_row()

    df = pd.DataFrame(
        [
            row,
            row.copy(),
        ]
    )

    with pytest.raises(CustomerEvidenceError):
        validate_input(df)


def test_missing_required_column_is_rejected():
    row = _make_signal_row()

    df = pd.DataFrame([row]).drop(
        columns=["evidence_level"]
    )

    with pytest.raises(CustomerEvidenceError):
        validate_input(df)


def test_evidence_basis_none():
    basis = _build_evidence_basis(
        "NONE",
        0,
        0.0,
    )

    assert basis == (
        "No longitudinal behavioral transitions observed."
    )


def test_evidence_basis_low():
    basis = _build_evidence_basis(
        "LOW",
        1,
        1.0,
    )

    assert "One behavioral transition observed" in basis
    assert "transition_count=1" in basis


def test_evidence_basis_medium():
    basis = _build_evidence_basis(
        "MEDIUM",
        2,
        0.75,
    )

    assert "Two behavioral transitions observed" in basis
    assert "transition_count=2" in basis
    assert "0.75" in basis


def test_evidence_basis_high():
    basis = _build_evidence_basis(
        "HIGH",
        3,
        1.0,
    )

    assert "Three or more behavioral transitions observed" in basis
    assert "transition_count=3" in basis
    assert "1.00" in basis


def test_real_customer_evidence_output_contract():
    df = build_customer_evidence(
        SIGNALS_PATH
    )

    expected_columns = [
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
        "evidence_qualification",
        "decision_readiness",
        "evidence_basis",
    ]

    assert df.columns.tolist() == expected_columns
    assert len(df) == 95420
    assert df["customer_unique_id"].is_unique


def test_real_customer_evidence_uses_only_governed_categories():
    df = build_customer_evidence(
        SIGNALS_PATH
    )

    assert (
        set(df["evidence_qualification"])
        <= ALLOWED_EVIDENCE_QUALIFICATIONS
    )

    assert (
        set(df["decision_readiness"])
        <= ALLOWED_DECISION_READINESS
    )


def test_real_customer_evidence_population_matches_signal_population():
    signals = pd.read_parquet(SIGNALS_PATH)
    evidence = build_customer_evidence(
        SIGNALS_PATH
    )

    assert set(evidence["customer_unique_id"]) == set(
        signals["customer_unique_id"]
    )


def test_real_customer_evidence_readiness_distribution():
    df = build_customer_evidence(
        SIGNALS_PATH
    )

    readiness_counts = (
        df["decision_readiness"]
        .value_counts()
        .to_dict()
    )

    assert readiness_counts["NOT_READY"] == 94954
    assert readiness_counts["OBSERVE"] == 460
    assert readiness_counts["REVIEW"] == 4
    assert readiness_counts["ACTION_CANDIDATE"] == 2