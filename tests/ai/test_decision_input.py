import pandas as pd
import pytest

from src.ai.decision_input import (
    AIDecisionInputError,
    AI_INPUT_COLUMNS,
    build_ai_decision_input,
)


def _context() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "decision_id": "DEC-001",
                "decision_type": "CUSTOMER_VALUE",
                "entity_id": "customer-001",
                "signal": "High-value decreasing customer.",
                "direction": "DECREASING",
                "priority": "CRITICAL",
                "queue_rank": 1,
                "decision_readiness": "ACTION_CANDIDATE",
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
                "signal": "High-value increasing customer.",
                "direction": "INCREASING",
                "priority": "HIGH",
                "queue_rank": 2,
                "decision_readiness": "ACTION_CANDIDATE",
                "evidence_level": "HIGH",
                "signal_confidence": "HIGH",
                "customer_lifecycle": "REPEAT_ACTIVE",
                "value_segment": "HIGH_VALUE",
                "active_months": 6,
                "total_orders": 7,
                "lifetime_revenue": 664.20,
                "transition_count": 4,
                "direction_consistency": 0.75,
                "evidence_basis": (
                    "Three or more behavioral transitions observed."
                ),
                "recommended_action": (
                    "Consider reinforcing positive behavior."
                ),
            },
        ]
    )


def test_build_ai_decision_input_shape():
    result = build_ai_decision_input(_context())

    assert result.shape == (2, 19)


def test_build_ai_decision_input_columns():
    result = build_ai_decision_input(_context())

    assert list(result.columns) == AI_INPUT_COLUMNS


def test_preserves_decision_ids():
    result = build_ai_decision_input(_context())

    assert result["decision_id"].tolist() == [
        "DEC-001",
        "DEC-002",
    ]


def test_preserves_queue_order():
    result = build_ai_decision_input(_context())

    assert result["queue_rank"].tolist() == [1, 2]


def test_preserves_governed_recommendations():
    result = build_ai_decision_input(_context())

    assert result.loc[
        result["decision_id"] == "DEC-001",
        "recommended_action",
    ].iloc[0] == "Prioritize customer value-risk intervention."


def test_preserves_evidence_fields():
    result = build_ai_decision_input(_context())

    row = result.loc[
        result["decision_id"] == "DEC-001"
    ].iloc[0]

    assert row["evidence_level"] == "HIGH"
    assert row["signal_confidence"] == "HIGH"
    assert row["transition_count"] == 3
    assert row["direction_consistency"] == 1.0


def test_contains_no_nulls():
    result = build_ai_decision_input(_context())

    assert not result.isna().any().any()


def test_missing_context_column_raises():
    context = _context().drop(columns=["evidence_basis"])

    with pytest.raises(AIDecisionInputError):
        build_ai_decision_input(context)


def test_empty_context_raises():
    context = _context().iloc[0:0]

    with pytest.raises(AIDecisionInputError):
        build_ai_decision_input(context)


def test_duplicate_decision_id_raises():
    context = pd.concat(
        [_context(), _context().iloc[[0]]],
        ignore_index=True,
    )

    with pytest.raises(AIDecisionInputError):
        build_ai_decision_input(context)


def test_duplicate_entity_id_raises():
    context = _context().copy()
    context.loc[1, "entity_id"] = context.loc[0, "entity_id"]

    with pytest.raises(AIDecisionInputError):
        build_ai_decision_input(context)


def test_null_values_raise():
    context = _context().copy()
    context.loc[0, "lifetime_revenue"] = None

    with pytest.raises(AIDecisionInputError):
        build_ai_decision_input(context)