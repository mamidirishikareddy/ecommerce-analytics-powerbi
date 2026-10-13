from pathlib import Path

import pandas as pd
import pytest

from src.ai.persist_reasoning import (
    AIReasoningPersistenceError,
    persist_ai_reasoning,
)


def _context() -> pd.DataFrame:
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


def _write_context(tmp_path: Path) -> Path:
    path = tmp_path / "decision_context.parquet"
    _context().to_parquet(path, index=False)
    return path


def test_persist_ai_reasoning_creates_file(tmp_path):
    context_path = _write_context(tmp_path)

    result = persist_ai_reasoning(
        context_path=context_path,
        output_dir=tmp_path / "output",
    )

    assert result.exists()
    assert result.name == "ai_decision_reasoning.parquet"


def test_persisted_reasoning_can_be_read(tmp_path):
    context_path = _write_context(tmp_path)

    result = persist_ai_reasoning(
        context_path=context_path,
        output_dir=tmp_path / "output",
    )

    reasoning = pd.read_parquet(result)

    assert reasoning.shape == (2, 8)


def test_persisted_reasoning_has_expected_decisions(tmp_path):
    context_path = _write_context(tmp_path)

    result = persist_ai_reasoning(
        context_path=context_path,
        output_dir=tmp_path / "output",
    )

    reasoning = pd.read_parquet(result)

    assert reasoning["decision_id"].tolist() == [
        "DEC-001",
        "DEC-002",
    ]


def test_persisted_reasoning_has_no_nulls(tmp_path):
    context_path = _write_context(tmp_path)

    result = persist_ai_reasoning(
        context_path=context_path,
        output_dir=tmp_path / "output",
    )

    reasoning = pd.read_parquet(result)

    assert not reasoning.isna().any().any()


def test_missing_context_file_raises(tmp_path):
    with pytest.raises(AIReasoningPersistenceError):
        persist_ai_reasoning(
            context_path=tmp_path / "missing.parquet",
            output_dir=tmp_path / "output",
        )


def test_missing_context_column_raises(tmp_path):
    context = _context().drop(columns=["evidence_level"])
    context_path = tmp_path / "decision_context.parquet"
    context.to_parquet(context_path, index=False)

    with pytest.raises(AIReasoningPersistenceError):
        persist_ai_reasoning(
            context_path=context_path,
            output_dir=tmp_path / "output",
        )