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


def _signals() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "customer_unique_id": "customer-001",
                "customer_lifecycle": "DORMANT",
                "value_segment": "HIGH_VALUE",
                "behavior_direction": "DECREASING",
                "evidence_level": "HIGH",
                "signal_confidence": "HIGH",
                "evidence_qualification": "STRONG_EVIDENCE",
                "decision_readiness": "ACTION_CANDIDATE",
                "evidence_basis": "Three or more behavioral transitions observed.",
            },
            {
                "customer_unique_id": "customer-002",
                "customer_lifecycle": "REPEAT_ACTIVE",
                "value_segment": "HIGH_VALUE",
                "behavior_direction": "DECREASING",
                "evidence_level": "MEDIUM",
                "signal_confidence": "MEDIUM",
                "evidence_qualification": "MODERATE_EVIDENCE",
                "decision_readiness": "REVIEW",
                "evidence_basis": "Two behavioral transitions observed.",
            },
            {
                "customer_unique_id": "customer-003",
                "customer_lifecycle": "REPEAT_ACTIVE",
                "value_segment": "MID_VALUE",
                "behavior_direction": "INCREASING",
                "evidence_level": "LOW",
                "signal_confidence": "LOW",
                "evidence_qualification": "LIMITED_EVIDENCE",
                "decision_readiness": "OBSERVE",
                "evidence_basis": "One behavioral transition observed.",
            },
            {
                "customer_unique_id": "customer-004",
                "customer_lifecycle": "ONE_TIME",
                "value_segment": "LOW_VALUE",
                "behavior_direction": "INSUFFICIENT_EVIDENCE",
                "evidence_level": "NONE",
                "signal_confidence": "NONE",
                "evidence_qualification": "INSUFFICIENT_EVIDENCE",
                "decision_readiness": "NOT_READY",
                "evidence_basis": "No longitudinal behavioral transitions observed.",
            },
        ]
    )


def test_build_customer_decisions_shape():
    result = build_customer_decisions(_signals())
    assert result.shape == (4, 10)


def test_build_customer_decisions_columns():
    result = build_customer_decisions(_signals())
    assert list(result.columns) == REQUIRED_OUTPUT_COLUMNS


def test_one_decision_per_customer():
    result = build_customer_decisions(_signals())
    assert result["entity_id"].nunique() == 4
    assert result["entity_id"].is_unique


def test_dormant_high_value_action_candidate_is_critical_retention():
    result = build_customer_decisions(_signals())
    row = result.loc[result["entity_id"] == "customer-001"].iloc[0]

    assert row["decision_type"] == "CUSTOMER_RETENTION"
    assert row["priority"] == "CRITICAL"
    assert row["decision_readiness"] == "ACTION_CANDIDATE"
    assert row["direction"] == "DECREASING"


def test_high_value_review_is_high_priority():
    result = build_customer_decisions(_signals())
    row = result.loc[result["entity_id"] == "customer-002"].iloc[0]

    assert row["priority"] == "HIGH"
    assert row["decision_readiness"] == "REVIEW"


def test_observe_signal_is_low_priority():
    result = build_customer_decisions(_signals())
    row = result.loc[result["entity_id"] == "customer-003"].iloc[0]

    assert row["priority"] == "LOW"
    assert row["decision_readiness"] == "OBSERVE"


def test_not_ready_signal_has_no_action():
    result = build_customer_decisions(_signals())
    row = result.loc[result["entity_id"] == "customer-004"].iloc[0]

    assert row["decision_readiness"] == "NOT_READY"
    assert (
        row["recommended_action"]
        == "Do not act; insufficient behavioral evidence."
    )


def test_decreasing_review_recommends_value_risk_review():
    result = build_customer_decisions(_signals())
    row = result.loc[result["entity_id"] == "customer-002"].iloc[0]

    assert (
        row["recommended_action"]
        == "Review customer value-risk intervention options."
    )


def test_observe_increasing_signal_recommends_monitoring():
    result = build_customer_decisions(_signals())
    row = result.loc[result["entity_id"] == "customer-003"].iloc[0]

    assert (
        row["recommended_action"]
        == "Monitor the customer signal for additional evidence."
    )


def test_missing_required_column_raises():
    signals = _signals().drop(columns=["evidence_basis"])

    with pytest.raises(DecisionEngineError):
        build_customer_decisions(signals)


def test_empty_input_raises():
    signals = _signals().iloc[0:0]

    with pytest.raises(DecisionEngineError):
        build_customer_decisions(signals)


def test_duplicate_customer_raises():
    signals = pd.concat(
        [_signals(), _signals().iloc[[0]]],
        ignore_index=True,
    )

    with pytest.raises(DecisionEngineError):
        build_customer_decisions(signals)


def test_output_contains_no_nulls():
    result = build_customer_decisions(_signals())

    assert not result.isna().any().any()

def test_increasing_action_candidate_is_high_not_critical():
    signals = _signals()

    result = build_customer_decisions(signals)

    row = result.loc[
        result["entity_id"] == "customer-001"
    ].iloc[0]

    assert row["direction"] == "DECREASING"
    assert row["decision_readiness"] == "ACTION_CANDIDATE"
    assert row["priority"] == "CRITICAL"

    increasing = signals.copy()
    increasing.loc[
        increasing["customer_unique_id"] == "customer-001",
        "behavior_direction",
    ] = "INCREASING"

    result = build_customer_decisions(increasing)

    row = result.loc[
        result["entity_id"] == "customer-001"
    ].iloc[0]

    assert row["direction"] == "INCREASING"
    assert row["decision_readiness"] == "ACTION_CANDIDATE"
    assert row["priority"] == "HIGH"