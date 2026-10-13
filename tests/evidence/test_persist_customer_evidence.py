"""
Tests for customer evidence persistence.
"""

from pathlib import Path

import pandas as pd

from src.evidence.persist_customer_evidence import (
    persist_customer_evidence,
)


SIGNALS_PATH = Path(
    "data/processed/marts/mart_customer_signals.parquet"
)


def test_persist_customer_evidence_creates_parquet(
    tmp_path,
):
    output_path = persist_customer_evidence(
        signals_path=SIGNALS_PATH,
        output_dir=tmp_path,
    )

    assert output_path.exists()
    assert output_path.name == "mart_customer_evidence.parquet"


def test_persisted_customer_evidence_can_be_read(
    tmp_path,
):
    output_path = persist_customer_evidence(
        signals_path=SIGNALS_PATH,
        output_dir=tmp_path,
    )

    df = pd.read_parquet(output_path)

    assert len(df) == 95420
    assert df["customer_unique_id"].is_unique


def test_persisted_customer_evidence_has_expected_columns(
    tmp_path,
):
    output_path = persist_customer_evidence(
        signals_path=SIGNALS_PATH,
        output_dir=tmp_path,
    )

    df = pd.read_parquet(output_path)

    expected_columns = [
        "customer_unique_id",
        "signal_date",
        "customer_lifecycle",
        "value_segment",
        "behavior_direction",
        "evidence_level",
        "active_months",
        "total_orders",
        "lifetime_revenue",
        "transition_count",
        "direction_consistency",
        "signal_confidence",
        "evidence_qualification",
        "decision_readiness",
        "evidence_basis",
    ]

    assert df.columns.tolist() == expected_columns


def test_persisted_customer_evidence_preserves_readiness_counts(
    tmp_path,
):
    output_path = persist_customer_evidence(
        signals_path=SIGNALS_PATH,
        output_dir=tmp_path,
    )

    df = pd.read_parquet(output_path)

    counts = df["decision_readiness"].value_counts().to_dict()

    assert counts["NOT_READY"] == 94954
    assert counts["OBSERVE"] == 460
    assert counts["REVIEW"] == 4
    assert counts["ACTION_CANDIDATE"] == 2