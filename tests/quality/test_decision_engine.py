import pandas as pd
import pytest

from src.analytics.decision_engine import (
    DecisionEngineError,
    build_customer_decisions,
)


REQUIRED_OUTPUT_COLUMNS = [
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
]


def _signals():
    return pd.DataFrame(
        {
            "customer_unique_id": [
                "customer-001",
                "customer-002",
                "customer-003",
                "customer-004",
            ],
            "customer_lifecycle": [
                "DORMANT",
                "REPEAT_ACTIVE",
                "REPEAT_ACTIVE",
                "ONE_TIME",
            ],
            "value_segment": [
                "HIGH_VALUE",
                "HIGH_VALUE",
                "MID_VALUE",
                "LOW_VALUE",
            ],
            "behavior_direction": [
                "DECREASING",
                "DECREASING",
                "INCREASING",
                "INSUFFICIENT_EVIDENCE",
            ],
            "evidence_level": [
                "HIGH",
                "MEDIUM",
                "LOW",
                "NONE",
            ],
            "signal_confidence": [
                "HIGH",
                "MEDIUM",
                "LOW",
                "NONE",
            ],
            "evidence_qualification": [
                "STRONG_EVIDENCE",
                "MODERATE_EVIDENCE",
                "LIMITED_EVIDENCE",
                "INSUFFICIENT_EVIDENCE",
            ],
            "decision_readiness": [
                "ACTION_CANDIDATE",
                "REVIEW",
                "OBSERVE",
                "NOT_READY",
            ],
            "evidence_basis": [
                "Three or more behavioral transitions observed.",
                "Two behavioral transitions observed.",
                "One behavioral transition observed.",
                "No longitudinal behavioral transitions observed.",
            ],
        }
    )


def test_build_customer_decisions_has_expected_shape():
    result = build_customer_decisions(_signals())

    assert result.shape == (4, 10)


def test_build_customer_decisions_has_expected_columns():
    result = build_customer_decisions(_signals())

    assert list(result.columns) == REQUIRED_OUTPUT_COLUMNS


def test_one_decision_is_created_per_customer():
    result = build_customer_decisions(_signals())

    assert len(result) == 4
    assert result["entity_id"].nunique() == 4


def test_decision_ids_are_unique():
    result = build_customer_decisions(_signals())

    assert result["decision_id"].is_unique


def test_dormant_high_value_action_candidate_is_critical():
    result = build_customer_decisions(_signals())

    row = result.loc[
        result["entity_id"] == "customer-001"
    ].iloc[0]

    assert row["decision_type"] == "CUSTOMER_RETENTION"
    assert row["priority"] == "CRITICAL"
    assert row["decision_readiness"] == "ACTION_CANDIDATE"
    assert row["direction"] == "DECREASING"


def test_medium_evidence_review_is_high_for_high_value():
    result = build_customer_decisions(_signals())

    row = result.loc[
        result["entity_id"] == "customer-002"
    ].iloc[0]

    assert row["priority"] == "HIGH"
    assert row["decision_readiness"] == "REVIEW"


def test_low_evidence_is_observe():
    result = build_customer_decisions(_signals())

    row = result.loc[
        result["entity_id"] == "customer-003"
    ].iloc[0]

    assert row["priority"] == "LOW"
    assert row["decision_readiness"] == "OBSERVE"


def test_insufficient_evidence_is_not_ready():
    result = build_customer_decisions(_signals())

    row = result.loc[
        result["entity_id"] == "customer-004"
    ].iloc[0]

    assert row["decision_readiness"] == "NOT_READY"
    assert row["priority"] == "LOW"
    assert (
        row["recommended_action"]
        == "Do not act; insufficient behavioral evidence."
    )


def test_decreasing_behavior_generates_value_risk_action():
    result = build_customer_decisions(_signals())

    row = result.loc[
        result["entity_id"] == "customer-002"
    ].iloc[0]

    assert (
        row["recommended_action"]
        == "Review customer value-risk intervention options."
    )


def test_increasing_behavior_generates_positive_action():
    result = build_customer_decisions(_signals())

    row = result.loc[
        result["entity_id"] == "customer-003"
    ].iloc[0]

    assert (
        row["recommended_action"]
        == "Monitor the customer signal for additional evidence."
    )


def test_missing_required_column_raises():
    invalid = _signals().drop(
        columns=["evidence_basis"]
    )

    with pytest.raises(DecisionEngineError):
        build_customer_decisions(invalid)


def test_empty_signals_raise():
    empty = pd.DataFrame(
        columns=_signals().columns
    )

    with pytest.raises(DecisionEngineError):
        build_customer_decisions(empty)


def test_duplicate_customer_rows_raise():
    invalid = pd.concat(
        [_signals(), _signals().iloc[[0]]],
        ignore_index=True,
    )

    with pytest.raises(DecisionEngineError):
        build_customer_decisions(invalid)


def test_decisions_have_no_nulls():
    result = build_customer_decisions(_signals())

    assert result.isna().sum().sum() == 0