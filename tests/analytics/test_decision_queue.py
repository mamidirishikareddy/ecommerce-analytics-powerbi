import pandas as pd
import pytest

from src.analytics.decision_queue import (
    DecisionQueueError,
    build_decision_queue,
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
    "queue_rank",
]


def _decisions() -> pd.DataFrame:
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
                "recommended_action": "Prioritize customer value-risk intervention.",
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
                "recommended_action": "Consider reinforcing positive behavior.",
            },
            {
                "decision_id": "DEC-003",
                "decision_type": "CUSTOMER_VALUE",
                "entity_id": "customer-003",
                "signal": "High-value customer requiring review.",
                "direction": "DECREASING",
                "priority": "HIGH",
                "evidence_level": "MEDIUM",
                "decision_readiness": "REVIEW",
                "evidence_basis": "Two transitions observed.",
                "recommended_action": "Review customer value-risk intervention options.",
            },
            {
                "decision_id": "DEC-004",
                "decision_type": "CUSTOMER_VALUE",
                "entity_id": "customer-004",
                "signal": "Limited evidence customer.",
                "direction": "INCREASING",
                "priority": "LOW",
                "evidence_level": "LOW",
                "decision_readiness": "OBSERVE",
                "evidence_basis": "One transition observed.",
                "recommended_action": "Monitor the customer signal for additional evidence.",
            },
            {
                "decision_id": "DEC-005",
                "decision_type": "CUSTOMER_VALUE",
                "entity_id": "customer-005",
                "signal": "No longitudinal evidence.",
                "direction": "INSUFFICIENT_EVIDENCE",
                "priority": "LOW",
                "evidence_level": "NONE",
                "decision_readiness": "NOT_READY",
                "evidence_basis": "No longitudinal behavioral transitions observed.",
                "recommended_action": "Do not act; insufficient behavioral evidence.",
            },
        ]
    )


def test_queue_contains_only_actionable_or_review_decisions():
    result = build_decision_queue(_decisions())

    assert set(result["decision_readiness"]) == {
        "ACTION_CANDIDATE",
        "REVIEW",
    }


def test_queue_excludes_observe_and_not_ready():
    result = build_decision_queue(_decisions())

    assert "DEC-004" not in set(result["decision_id"])
    assert "DEC-005" not in set(result["decision_id"])


def test_queue_shape():
    result = build_decision_queue(_decisions())

    assert result.shape == (3, 11)


def test_queue_columns():
    result = build_decision_queue(_decisions())

    assert list(result.columns) == REQUIRED_OUTPUT_COLUMNS


def test_critical_decision_is_ranked_first():
    result = build_decision_queue(_decisions())

    assert result.iloc[0]["decision_id"] == "DEC-001"
    assert result.iloc[0]["priority"] == "CRITICAL"
    assert result.iloc[0]["queue_rank"] == 1


def test_high_action_candidate_precedes_high_review():
    result = build_decision_queue(_decisions())

    assert result.iloc[1]["decision_id"] == "DEC-002"
    assert result.iloc[2]["decision_id"] == "DEC-003"


def test_queue_rank_is_sequential():
    result = build_decision_queue(_decisions())

    assert result["queue_rank"].tolist() == [1, 2, 3]


def test_queue_has_unique_decision_ids():
    result = build_decision_queue(_decisions())

    assert result["decision_id"].is_unique


def test_queue_contains_no_nulls():
    result = build_decision_queue(_decisions())

    assert not result.isna().any().any()


def test_missing_required_column_raises():
    decisions = _decisions().drop(columns=["evidence_basis"])

    with pytest.raises(DecisionQueueError):
        build_decision_queue(decisions)


def test_empty_input_raises():
    decisions = _decisions().iloc[0:0]

    with pytest.raises(DecisionQueueError):
        build_decision_queue(decisions)


def test_duplicate_decision_id_raises():
    decisions = pd.concat(
        [_decisions(), _decisions().iloc[[0]]],
        ignore_index=True,
    )

    with pytest.raises(DecisionQueueError):
        build_decision_queue(decisions)


def test_invalid_queue_priority_raises():
    decisions = _decisions().copy()

    decisions.loc[
        decisions["decision_id"] == "DEC-001",
        "priority",
    ] = "URGENT"

    with pytest.raises(DecisionQueueError):
        build_decision_queue(decisions)