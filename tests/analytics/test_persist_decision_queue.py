from pathlib import Path

import pandas as pd
import pytest

from src.analytics.persist_decision_queue import (
    DecisionQueuePersistenceError,
    persist_decision_queue,
)


DECISIONS = Path(
    "data/processed/analytics/customer_decisions.parquet"
)


REQUIRED_COLUMNS = [
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


def test_persist_decision_queue_creates_file(tmp_path):
    result = persist_decision_queue(
        decisions_path=DECISIONS,
        output_dir=tmp_path,
    )

    assert result.exists()
    assert result.name == "decision_queue.parquet"


def test_persisted_queue_has_expected_columns(tmp_path):
    result = persist_decision_queue(
        decisions_path=DECISIONS,
        output_dir=tmp_path,
    )

    df = pd.read_parquet(result)

    assert list(df.columns) == REQUIRED_COLUMNS


def test_persisted_queue_contains_only_operational_decisions(tmp_path):
    result = persist_decision_queue(
        decisions_path=DECISIONS,
        output_dir=tmp_path,
    )

    df = pd.read_parquet(result)

    assert set(df["decision_readiness"]) == {
        "ACTION_CANDIDATE",
        "REVIEW",
    }


def test_persisted_queue_has_expected_population(tmp_path):
    result = persist_decision_queue(
        decisions_path=DECISIONS,
        output_dir=tmp_path,
    )

    df = pd.read_parquet(result)

    assert len(df) == 6
    assert df["decision_id"].nunique() == 6
    assert df["decision_id"].is_unique


def test_persisted_queue_ranks_critical_decision_first(tmp_path):
    result = persist_decision_queue(
        decisions_path=DECISIONS,
        output_dir=tmp_path,
    )

    df = pd.read_parquet(result)

    assert df.iloc[0]["priority"] == "CRITICAL"
    assert df.iloc[0]["queue_rank"] == 1
    assert df.iloc[0]["direction"] == "DECREASING"


def test_persisted_queue_contains_two_action_candidates(tmp_path):
    result = persist_decision_queue(
        decisions_path=DECISIONS,
        output_dir=tmp_path,
    )

    df = pd.read_parquet(result)

    assert (
        (df["decision_readiness"] == "ACTION_CANDIDATE").sum()
        == 2
    )


def test_persisted_queue_contains_four_reviews(tmp_path):
    result = persist_decision_queue(
        decisions_path=DECISIONS,
        output_dir=tmp_path,
    )

    df = pd.read_parquet(result)

    assert (
        (df["decision_readiness"] == "REVIEW").sum()
        == 4
    )


def test_persisted_queue_rank_is_sequential(tmp_path):
    result = persist_decision_queue(
        decisions_path=DECISIONS,
        output_dir=tmp_path,
    )

    df = pd.read_parquet(result)

    assert df["queue_rank"].tolist() == list(
        range(1, len(df) + 1)
    )


def test_persisted_queue_contains_no_nulls(tmp_path):
    result = persist_decision_queue(
        decisions_path=DECISIONS,
        output_dir=tmp_path,
    )

    df = pd.read_parquet(result)

    assert not df.isna().any().any()


def test_missing_decision_file_raises(tmp_path):
    with pytest.raises(DecisionQueuePersistenceError):
        persist_decision_queue(
            decisions_path=tmp_path / "missing.parquet",
            output_dir=tmp_path,
        )


def test_missing_required_decision_column_raises(tmp_path):
    decisions = pd.read_parquet(DECISIONS)
    decisions = decisions.drop(columns=["evidence_basis"])

    broken_path = tmp_path / "broken_decisions.parquet"
    decisions.to_parquet(broken_path, index=False)

    with pytest.raises(DecisionQueuePersistenceError):
        persist_decision_queue(
            decisions_path=broken_path,
            output_dir=tmp_path,
        )