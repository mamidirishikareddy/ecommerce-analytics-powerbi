import pandas as pd
import pytest

from src.analytics.decision_context import (
    DecisionContextError,
    build_decision_context,
)


REQUIRED_COLUMNS = [
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


def _queue() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "decision_id": "DEC-001",
                "decision_type": "CUSTOMER_VALUE",
                "entity_id": "customer-001",
                "signal": "High-value decreasing customer.",
                "direction": "DECREASING",
                "priority": "CRITICAL",
                "evidence_level": "HIGH",
                "decision_readiness": "ACTION_CANDIDATE",
                "evidence_basis": "Three or more transitions observed.",
                "recommended_action": (
                    "Prioritize customer value-risk intervention."
                ),
                "queue_rank": 1,
            },
            {
                "decision_id": "DEC-002",
                "decision_type": "CUSTOMER_VALUE",
                "entity_id": "customer-002",
                "signal": "High-value increasing customer.",
                "direction": "INCREASING",
                "priority": "HIGH",
                "evidence_level": "HIGH",
                "decision_readiness": "ACTION_CANDIDATE",
                "evidence_basis": "Three or more transitions observed.",
                "recommended_action": (
                    "Consider reinforcing positive behavior."
                ),
                "queue_rank": 2,
            },
        ]
    )


def _signals() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "customer_unique_id": "customer-001",
                "signal_date": "2018-09-01",
                "customer_lifecycle": "REPEAT_ACTIVE",
                "value_segment": "HIGH_VALUE",
                "behavior_direction": "DECREASING",
                "evidence_level": "HIGH",
                "active_months": 4,
                "total_orders": 7,
                "lifetime_revenue": 809.21,
                "transition_count": 3,
                "direction_consistency": 1.0,
                "signal_confidence": "HIGH",
            },
            {
                "customer_unique_id": "customer-002",
                "signal_date": "2018-09-01",
                "customer_lifecycle": "REPEAT_ACTIVE",
                "value_segment": "HIGH_VALUE",
                "behavior_direction": "INCREASING",
                "evidence_level": "HIGH",
                "active_months": 6,
                "total_orders": 7,
                "lifetime_revenue": 664.20,
                "transition_count": 4,
                "direction_consistency": 0.75,
                "signal_confidence": "HIGH",
            },
        ]
    )


def test_build_decision_context_shape():
    result = build_decision_context(_queue(), _signals())

    assert result.shape == (2, 19)


def test_build_decision_context_columns():
    result = build_decision_context(_queue(), _signals())

    assert list(result.columns) == REQUIRED_COLUMNS


def test_context_preserves_queue_population():
    result = build_decision_context(_queue(), _signals())

    assert len(result) == len(_queue())
    assert result["decision_id"].is_unique
    assert result["entity_id"].is_unique


def test_context_preserves_decision_ids():
    result = build_decision_context(_queue(), _signals())

    assert result["decision_id"].tolist() == [
        "DEC-001",
        "DEC-002",
    ]


def test_context_attaches_customer_lifecycle():
    result = build_decision_context(_queue(), _signals())

    assert result.loc[
        result["entity_id"] == "customer-001",
        "customer_lifecycle",
    ].iloc[0] == "REPEAT_ACTIVE"


def test_context_attaches_behavioral_metrics():
    result = build_decision_context(_queue(), _signals())

    row = result.loc[
        result["entity_id"] == "customer-001"
    ].iloc[0]

    assert row["active_months"] == 4
    assert row["total_orders"] == 7
    assert row["lifetime_revenue"] == 809.21
    assert row["transition_count"] == 3
    assert row["direction_consistency"] == 1.0


def test_context_preserves_governed_recommendation():
    result = build_decision_context(_queue(), _signals())

    row = result.loc[
        result["entity_id"] == "customer-001"
    ].iloc[0]

    assert (
        row["recommended_action"]
        == "Prioritize customer value-risk intervention."
    )


def test_context_contains_no_nulls():
    result = build_decision_context(_queue(), _signals())

    assert not result.isna().any().any()


def test_missing_queue_column_raises():
    queue = _queue().drop(columns=["priority"])

    with pytest.raises(DecisionContextError):
        build_decision_context(queue, _signals())


def test_missing_signal_column_raises():
    signals = _signals().drop(columns=["lifetime_revenue"])

    with pytest.raises(DecisionContextError):
        build_decision_context(_queue(), signals)


def test_empty_queue_raises():
    queue = _queue().iloc[0:0]

    with pytest.raises(DecisionContextError):
        build_decision_context(queue, _signals())


def test_empty_signals_raises():
    signals = _signals().iloc[0:0]

    with pytest.raises(DecisionContextError):
        build_decision_context(_queue(), signals)


def test_duplicate_decision_id_raises():
    queue = pd.concat(
        [_queue(), _queue().iloc[[0]]],
        ignore_index=True,
    )

    with pytest.raises(DecisionContextError):
        build_decision_context(queue, _signals())


def test_duplicate_signal_customer_raises():
    signals = pd.concat(
        [_signals(), _signals().iloc[[0]]],
        ignore_index=True,
    )

    with pytest.raises(DecisionContextError):
        build_decision_context(_queue(), signals)


def test_unmatched_customer_raises():
    signals = _signals().copy()

    signals.loc[
        signals["customer_unique_id"] == "customer-001",
        "customer_unique_id",
    ] = "different-customer"

    with pytest.raises(DecisionContextError):
        build_decision_context(_queue(), signals)