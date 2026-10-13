import pandas as pd
import pytest

from src.ai.reasoning_engine import (
    AIReasoningEngineError,
    build_deterministic_reasoning,
)


def _input() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "decision_id": "DEC-001",
                "decision_type": "CUSTOMER_VALUE",
                "entity_id": "customer-001",
                "priority": "CRITICAL",
                "queue_rank": 1,
                "decision_readiness": "ACTION_CANDIDATE",
                "signal": "High-value decreasing customer.",
                "direction": "DECREASING",
                "evidence_level": "HIGH",
                "signal_confidence": "HIGH",
                "customer_lifecycle": "REPEAT_ACTIVE",
                "value_segment": "HIGH_VALUE",
                "active_months": 4,
                "total_orders": 7,
                "lifetime_revenue": 809.21,
                "transition_count": 3,
                "direction_consistency": 1.0,
                "evidence_basis": (
                    "Three or more behavioral transitions observed."
                ),
                "recommended_action": (
                    "Prioritize customer value-risk intervention."
                ),
            },
            {
                "decision_id": "DEC-002",
                "decision_type": "CUSTOMER_VALUE",
                "entity_id": "customer-002",
                "priority": "HIGH",
                "queue_rank": 2,
                "decision_readiness": "REVIEW",
                "signal": "High-value increasing customer.",
                "direction": "INCREASING",
                "evidence_level": "MEDIUM",
                "signal_confidence": "MEDIUM",
                "customer_lifecycle": "REPEAT_ACTIVE",
                "value_segment": "HIGH_VALUE",
                "active_months": 3,
                "total_orders": 4,
                "lifetime_revenue": 500.00,
                "transition_count": 2,
                "direction_consistency": 0.5,
                "evidence_basis": (
                    "Two behavioral transitions observed."
                ),
                "recommended_action": (
                    "Review opportunities to reinforce positive behavior."
                ),
            },
        ]
    )


def test_reasoning_output_shape():
    result = build_deterministic_reasoning(_input())

    assert result.shape == (2, 8)


def test_reasoning_output_columns():
    result = build_deterministic_reasoning(_input())

    assert list(result.columns) == [
        "decision_id",
        "executive_summary",
        "evidence_used",
        "reasoning",
        "recommended_next_step",
        "risk_or_caveat",
        "human_review_required",
        "grounding_status",
    ]


def test_preserves_decision_ids():
    result = build_deterministic_reasoning(_input())

    assert result["decision_id"].tolist() == [
        "DEC-001",
        "DEC-002",
    ]


def test_action_candidate_is_grounded():
    result = build_deterministic_reasoning(_input())

    row = result.iloc[0]

    assert row["grounding_status"] == "GROUNDED"
    assert row["human_review_required"] == "YES"


def test_review_decision_requires_human_review():
    result = build_deterministic_reasoning(_input())

    row = result.iloc[1]

    assert row["human_review_required"] == "YES"
    assert row["grounding_status"] == "GROUNDED"


def test_recommendation_is_preserved():
    result = build_deterministic_reasoning(_input())

    assert result.loc[
        result["decision_id"] == "DEC-001",
        "recommended_next_step",
    ].iloc[0] == "Prioritize customer value-risk intervention."


def test_high_evidence_reasoning_mentions_transitions():
    result = build_deterministic_reasoning(_input())

    row = result.iloc[0]

    assert "3 behavioral transition(s)" in row["evidence_used"]
    assert "causality" in row["risk_or_caveat"]


def test_medium_evidence_has_caveat():
    result = build_deterministic_reasoning(_input())

    row = result.iloc[1]

    assert "Evidence remains limited" in row["risk_or_caveat"]


def test_missing_column_raises():
    frame = _input().drop(columns=["evidence_level"])

    with pytest.raises(AIReasoningEngineError):
        build_deterministic_reasoning(frame)


def test_empty_input_raises():
    frame = _input().iloc[0:0]

    with pytest.raises(AIReasoningEngineError):
        build_deterministic_reasoning(frame)


def test_null_input_raises():
    frame = _input().copy()
    frame.loc[0, "lifetime_revenue"] = None

    with pytest.raises(AIReasoningEngineError):
        build_deterministic_reasoning(frame)


def test_not_ready_produces_insufficient_evidence():
    frame = _input().copy()

    frame.loc[0, "decision_readiness"] = "NOT_READY"
    frame.loc[0, "evidence_level"] = "NONE"
    frame.loc[0, "transition_count"] = 0
    frame.loc[0, "direction_consistency"] = 0.0

    result = build_deterministic_reasoning(frame)

    row = result.iloc[0]

    assert row["grounding_status"] == "INSUFFICIENT_EVIDENCE"
    assert row["human_review_required"] == "NO"
    assert "insufficient" in row["reasoning"].lower()


def test_observe_does_not_require_human_review():
    frame = _input().copy()

    frame.loc[0, "decision_readiness"] = "OBSERVE"
    frame.loc[0, "evidence_level"] = "LOW"
    frame.loc[0, "transition_count"] = 1

    result = build_deterministic_reasoning(frame)

    row = result.iloc[0]

    assert row["human_review_required"] == "NO"
    assert row["grounding_status"] == "INSUFFICIENT_EVIDENCE"


def test_output_contains_no_nulls():
    result = build_deterministic_reasoning(_input())

    assert not result.isna().any().any()