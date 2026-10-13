from pathlib import Path

import pandas as pd
import pytest

from src.analytics.persist_decision_context import (
    DecisionContextPersistenceError,
    persist_decision_context,
)


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


def _write_inputs(tmp_path: Path):
    queue_path = tmp_path / "decision_queue.parquet"
    signals_path = tmp_path / "customer_signals.parquet"

    _queue().to_parquet(queue_path, index=False)
    _signals().to_parquet(signals_path, index=False)

    return queue_path, signals_path


def test_persist_decision_context_creates_file(tmp_path):
    queue_path, signals_path = _write_inputs(tmp_path)

    output_dir = tmp_path / "output"

    result = persist_decision_context(
        queue_path=queue_path,
        signals_path=signals_path,
        output_dir=output_dir,
    )

    assert result.exists()
    assert result.name == "decision_context.parquet"


def test_persisted_context_can_be_read(tmp_path):
    queue_path, signals_path = _write_inputs(tmp_path)

    result = persist_decision_context(
        queue_path=queue_path,
        signals_path=signals_path,
        output_dir=tmp_path / "output",
    )

    context = pd.read_parquet(result)

    assert context.shape == (2, 19)


def test_persisted_context_has_expected_decisions(tmp_path):
    queue_path, signals_path = _write_inputs(tmp_path)

    result = persist_decision_context(
        queue_path=queue_path,
        signals_path=signals_path,
        output_dir=tmp_path / "output",
    )

    context = pd.read_parquet(result)

    assert context["decision_id"].tolist() == [
        "DEC-001",
        "DEC-002",
    ]


def test_persisted_context_has_no_nulls(tmp_path):
    queue_path, signals_path = _write_inputs(tmp_path)

    result = persist_decision_context(
        queue_path=queue_path,
        signals_path=signals_path,
        output_dir=tmp_path / "output",
    )

    context = pd.read_parquet(result)

    assert not context.isna().any().any()


def test_missing_queue_file_raises(tmp_path):
    signals_path = tmp_path / "signals.parquet"
    _signals().to_parquet(signals_path, index=False)

    with pytest.raises(DecisionContextPersistenceError):
        persist_decision_context(
            queue_path=tmp_path / "missing_queue.parquet",
            signals_path=signals_path,
            output_dir=tmp_path / "output",
        )


def test_missing_signals_file_raises(tmp_path):
    queue_path = tmp_path / "queue.parquet"
    _queue().to_parquet(queue_path, index=False)

    with pytest.raises(DecisionContextPersistenceError):
        persist_decision_context(
            queue_path=queue_path,
            signals_path=tmp_path / "missing_signals.parquet",
            output_dir=tmp_path / "output",
        )


def test_missing_queue_column_raises(tmp_path):
    queue = _queue().drop(columns=["priority"])

    queue_path = tmp_path / "queue.parquet"
    signals_path = tmp_path / "signals.parquet"

    queue.to_parquet(queue_path, index=False)
    _signals().to_parquet(signals_path, index=False)

    with pytest.raises(DecisionContextPersistenceError):
        persist_decision_context(
            queue_path=queue_path,
            signals_path=signals_path,
            output_dir=tmp_path / "output",
        )


def test_missing_signal_column_raises(tmp_path):
    signals = _signals().drop(columns=["lifetime_revenue"])

    queue_path = tmp_path / "queue.parquet"
    signals_path = tmp_path / "signals.parquet"

    _queue().to_parquet(queue_path, index=False)
    signals.to_parquet(signals_path, index=False)

    with pytest.raises(DecisionContextPersistenceError):
        persist_decision_context(
            queue_path=queue_path,
            signals_path=signals_path,
            output_dir=tmp_path / "output",
        )