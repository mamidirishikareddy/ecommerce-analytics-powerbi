from pathlib import Path

import pandas as pd
import pytest

from src.analytics.persist_decisions import (
    DecisionPersistenceError,
    persist_customer_decisions,
)


SIGNALS = Path(
    "data/processed/marts/mart_customer_evidence.parquet"
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
]


def test_persist_customer_decisions_creates_file(tmp_path):
    result = persist_customer_decisions(
        signals_path=SIGNALS,
        output_dir=tmp_path,
    )

    assert result.exists()
    assert result.name == "customer_decisions.parquet"


def test_persisted_decisions_have_expected_columns(tmp_path):
    result = persist_customer_decisions(
        signals_path=SIGNALS,
        output_dir=tmp_path,
    )

    df = pd.read_parquet(result)

    assert list(df.columns) == REQUIRED_COLUMNS


def test_persisted_decisions_preserve_customer_population(tmp_path):
    result = persist_customer_decisions(
        signals_path=SIGNALS,
        output_dir=tmp_path,
    )

    df = pd.read_parquet(result)

    assert len(df) == 95420
    assert df["entity_id"].nunique() == 95420
    assert df["entity_id"].is_unique


def test_persisted_decisions_contain_no_nulls(tmp_path):
    result = persist_customer_decisions(
        signals_path=SIGNALS,
        output_dir=tmp_path,
    )

    df = pd.read_parquet(result)

    assert not df.isna().any().any()


def test_persisted_decisions_preserve_not_ready_population(tmp_path):
    result = persist_customer_decisions(
        signals_path=SIGNALS,
        output_dir=tmp_path,
    )

    df = pd.read_parquet(result)

    assert (
        df["decision_readiness"]
        .value_counts()
        .get("NOT_READY", 0)
        == 94954
    )


def test_missing_signal_file_raises(tmp_path):
    with pytest.raises(DecisionPersistenceError):
        persist_customer_decisions(
            signals_path=tmp_path / "missing.parquet",
            output_dir=tmp_path,
        )


def test_missing_required_signal_column_raises(tmp_path):
    signals = pd.read_parquet(SIGNALS)
    signals = signals.drop(columns=["evidence_basis"])

    broken_path = tmp_path / "broken_signals.parquet"
    signals.to_parquet(broken_path, index=False)

    with pytest.raises(DecisionPersistenceError):
        persist_customer_decisions(
            signals_path=broken_path,
            output_dir=tmp_path,
        )