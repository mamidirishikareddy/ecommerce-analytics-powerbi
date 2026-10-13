from pathlib import Path

import pandas as pd
import pytest

from src.signals.customer_signals import (
    CustomerSignalError,
    build_customer_signals,
    validate_input,
)


def write_customer_mart(tmp_path: Path) -> None:
    """Create a small governed customer mart for testing."""

    data = pd.DataFrame(
        {
            "month": pd.to_datetime(
                [
                    "2018-01-01",
                    "2018-02-01",
                    "2018-03-01",
                ]
            ),
            "customer_unique_id": [
                "customer_1",
                "customer_1",
                "customer_2",
            ],
            "order_count": [
                1,
                1,
                1,
            ],
            "item_line_count": [
                1,
                1,
                1,
            ],
            "revenue": [
                100.0,
                150.0,
                50.0,
            ],
            "freight_value": [
                10.0,
                10.0,
                5.0,
            ],
            "avg_order_value": [
                100.0,
                150.0,
                50.0,
            ],
            "first_purchase_month": pd.to_datetime(
                [
                    "2018-01-01",
                    "2018-01-01",
                    "2018-03-01",
                ]
            ),
            "customer_status": [
                "NEW",
                "REPEAT",
                "NEW",
            ],
        }
    )

    output_path = (
        tmp_path
        / "mart_customer_performance_monthly.parquet"
    )

    data.to_parquet(
        output_path,
        index=False,
    )


def test_validate_input_accepts_valid_input(tmp_path):
    write_customer_mart(tmp_path)

    data = pd.read_parquet(
        tmp_path
        / "mart_customer_performance_monthly.parquet"
    )

    validate_input(data)


def test_validate_input_rejects_missing_columns():
    data = pd.DataFrame(
        {
            "customer_unique_id": [
                "customer_1",
            ]
        }
    )

    with pytest.raises(CustomerSignalError):
        validate_input(data)


def test_missing_customer_mart_raises_error(tmp_path):
    with pytest.raises(CustomerSignalError):
        build_customer_signals(tmp_path)


def test_empty_customer_mart_raises_error(tmp_path):
    data = pd.DataFrame(
        columns=[
            "month",
            "customer_unique_id",
            "order_count",
            "item_line_count",
            "revenue",
            "freight_value",
            "avg_order_value",
            "first_purchase_month",
            "customer_status",
        ]
    )

    data.to_parquet(
        tmp_path
        / "mart_customer_performance_monthly.parquet",
        index=False,
    )

    with pytest.raises(CustomerSignalError):
        build_customer_signals(tmp_path)


def test_customer_signal_output_has_expected_columns(tmp_path):
    write_customer_mart(tmp_path)

    result = build_customer_signals(tmp_path)

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
    ]

    assert list(result.columns) == expected_columns


def test_customer_signal_grain_is_one_row_per_customer(tmp_path):
    write_customer_mart(tmp_path)

    result = build_customer_signals(tmp_path)

    assert not result[
        "customer_unique_id"
    ].duplicated().any()


def test_customer_signal_count_matches_customer_population(
    tmp_path,
):
    write_customer_mart(tmp_path)

    result = build_customer_signals(tmp_path)

    assert len(result) == 2


def test_customer_lifecycle_is_derived_correctly(tmp_path):
    write_customer_mart(tmp_path)

    result = build_customer_signals(tmp_path)

    lifecycle = result.set_index(
        "customer_unique_id"
    )["customer_lifecycle"]

    assert lifecycle["customer_1"] == "REPEAT_ACTIVE"
    assert lifecycle["customer_2"] == "ONE_TIME"


def test_customer_evidence_is_derived_correctly(tmp_path):
    write_customer_mart(tmp_path)

    result = build_customer_signals(tmp_path)

    customer_1 = result[
        result["customer_unique_id"] == "customer_1"
    ].iloc[0]

    assert customer_1["active_months"] == 2
    assert customer_1["total_orders"] == 2
    assert customer_1["lifetime_revenue"] == 250.0


def test_value_segment_is_derived(tmp_path):
    write_customer_mart(tmp_path)

    result = build_customer_signals(tmp_path)

    assert result["value_segment"].notna().all()


def test_transition_count_is_derived_correctly(tmp_path):
    write_customer_mart(tmp_path)

    result = build_customer_signals(tmp_path)

    transition_count = result.set_index(
        "customer_unique_id"
    )["transition_count"]

    assert transition_count["customer_1"] == 1
    assert transition_count["customer_2"] == 0


def test_direction_consistency_is_derived_correctly(tmp_path):
    write_customer_mart(tmp_path)

    result = build_customer_signals(tmp_path)

    customer_1 = result[
        result["customer_unique_id"] == "customer_1"
    ].iloc[0]

    assert customer_1["direction_consistency"] == 1.0


def test_signal_contract_behavior_and_evidence_are_classified(
    tmp_path,
):
    write_customer_mart(tmp_path)

    result = build_customer_signals(tmp_path)

    assert set(
        result["behavior_direction"].dropna().unique()
    ).issubset(
        {
            "INCREASING",
            "DECREASING",
            "STABLE",
            "INSUFFICIENT_EVIDENCE",
        }
    )

    assert set(
        result["evidence_level"].dropna().unique()
    ).issubset(
        {
            "NONE",
            "LOW",
            "MEDIUM",
            "HIGH",
        }
    )

    assert result.loc[
        result["transition_count"] == 0,
        "behavior_direction",
    ].eq("INSUFFICIENT_EVIDENCE").all()

    assert result.loc[
        result["transition_count"] == 0,
        "evidence_level",
    ].eq("NONE").all()

    assert result.loc[
        result["transition_count"] == 1,
        "evidence_level",
    ].eq("LOW").all()

    assert result.loc[
        result["transition_count"] == 0,
        "signal_confidence",
    ].eq("NONE").all()

    assert result.loc[
        result["transition_count"] == 1,
        "signal_confidence",
    ].eq("LOW").all()


def test_signal_confidence_is_none_without_transitions(
    tmp_path,
):
    write_customer_mart(tmp_path)

    result = build_customer_signals(tmp_path)

    zero_transition = result[
        result["transition_count"] == 0
    ]

    assert zero_transition[
        "signal_confidence"
    ].eq("NONE").all()


def test_signal_confidence_is_low_for_one_transition(
    tmp_path,
):
    write_customer_mart(tmp_path)

    result = build_customer_signals(tmp_path)

    one_transition = result[
        result["transition_count"] == 1
    ]

    assert one_transition[
        "signal_confidence"
    ].eq("LOW").all()


def test_signal_confidence_uses_evidence_and_consistency(
    tmp_path,
):
    write_customer_mart(tmp_path)

    result = build_customer_signals(tmp_path)

    medium = result[
        result["transition_count"] == 2
    ]

    high = result[
        result["transition_count"] >= 3
    ]

    assert (
        medium.loc[
            medium["direction_consistency"] >= 0.75,
            "signal_confidence",
        ]
        .eq("MEDIUM")
        .all()
    )

    assert (
        medium.loc[
            medium["direction_consistency"] < 0.75,
            "signal_confidence",
        ]
        .eq("LOW")
        .all()
    )

    assert (
        high.loc[
            high["direction_consistency"] >= 0.75,
            "signal_confidence",
        ]
        .eq("HIGH")
        .all()
    )

    assert (
        high.loc[
            high["direction_consistency"] < 0.75,
            "signal_confidence",
        ]
        .eq("MEDIUM")
        .all()
    )


def test_non_consecutive_months_are_not_transitions(
    tmp_path,
):
    data = pd.DataFrame(
        {
            "month": pd.to_datetime(
                [
                    "2018-01-01",
                    "2018-03-01",
                ]
            ),
            "customer_unique_id": [
                "customer_1",
                "customer_1",
            ],
            "order_count": [
                1,
                1,
            ],
            "item_line_count": [
                1,
                1,
            ],
            "revenue": [
                100.0,
                200.0,
            ],
            "freight_value": [
                10.0,
                10.0,
            ],
            "avg_order_value": [
                100.0,
                200.0,
            ],
            "first_purchase_month": pd.to_datetime(
                [
                    "2018-01-01",
                    "2018-01-01",
                ]
            ),
            "customer_status": [
                "NEW",
                "REPEAT",
            ],
        }
    )

    data.to_parquet(
        tmp_path
        / "mart_customer_performance_monthly.parquet",
        index=False,
    )

    result = build_customer_signals(tmp_path)

    customer = result.iloc[0]

    assert customer["transition_count"] == 0
    assert customer["direction_consistency"] is pd.NA or pd.isna(
        customer["direction_consistency"]
    )
    assert (
        customer["behavior_direction"]
        == "INSUFFICIENT_EVIDENCE"
    )
    assert customer["evidence_level"] == "NONE"
    assert customer["signal_confidence"] == "NONE"


def test_transition_direction_increasing(tmp_path):
    data = pd.DataFrame(
        {
            "month": pd.to_datetime(
                [
                    "2018-01-01",
                    "2018-02-01",
                ]
            ),
            "customer_unique_id": [
                "customer_1",
                "customer_1",
            ],
            "order_count": [
                1,
                1,
            ],
            "item_line_count": [
                1,
                1,
            ],
            "revenue": [
                100.0,
                200.0,
            ],
            "freight_value": [
                10.0,
                10.0,
            ],
            "avg_order_value": [
                100.0,
                200.0,
            ],
            "first_purchase_month": pd.to_datetime(
                [
                    "2018-01-01",
                    "2018-01-01",
                ]
            ),
            "customer_status": [
                "NEW",
                "REPEAT",
            ],
        }
    )

    data.to_parquet(
        tmp_path
        / "mart_customer_performance_monthly.parquet",
        index=False,
    )

    result = build_customer_signals(tmp_path)

    customer = result.iloc[0]

    assert customer["transition_count"] == 1
    assert customer["behavior_direction"] == "INCREASING"
    assert customer["evidence_level"] == "LOW"
    assert customer["signal_confidence"] == "LOW"


def test_transition_direction_decreasing(tmp_path):
    data = pd.DataFrame(
        {
            "month": pd.to_datetime(
                [
                    "2018-01-01",
                    "2018-02-01",
                ]
            ),
            "customer_unique_id": [
                "customer_1",
                "customer_1",
            ],
            "order_count": [
                1,
                1,
            ],
            "item_line_count": [
                1,
                1,
            ],
            "revenue": [
                200.0,
                100.0,
            ],
            "freight_value": [
                10.0,
                10.0,
            ],
            "avg_order_value": [
                200.0,
                100.0,
            ],
            "first_purchase_month": pd.to_datetime(
                [
                    "2018-01-01",
                    "2018-01-01",
                ]
            ),
            "customer_status": [
                "NEW",
                "REPEAT",
            ],
        }
    )

    data.to_parquet(
        tmp_path
        / "mart_customer_performance_monthly.parquet",
        index=False,
    )

    result = build_customer_signals(tmp_path)

    customer = result.iloc[0]

    assert customer["transition_count"] == 1
    assert customer["behavior_direction"] == "DECREASING"
    assert customer["evidence_level"] == "LOW"
    assert customer["signal_confidence"] == "LOW"


def test_equal_revenue_transition_is_stable(tmp_path):
    data = pd.DataFrame(
        {
            "month": pd.to_datetime(
                [
                    "2018-01-01",
                    "2018-02-01",
                ]
            ),
            "customer_unique_id": [
                "customer_1",
                "customer_1",
            ],
            "order_count": [
                1,
                1,
            ],
            "item_line_count": [
                1,
                1,
            ],
            "revenue": [
                100.0,
                100.0,
            ],
            "freight_value": [
                10.0,
                10.0,
            ],
            "avg_order_value": [
                100.0,
                100.0,
            ],
            "first_purchase_month": pd.to_datetime(
                [
                    "2018-01-01",
                    "2018-01-01",
                ]
            ),
            "customer_status": [
                "NEW",
                "REPEAT",
            ],
        }
    )

    data.to_parquet(
        tmp_path
        / "mart_customer_performance_monthly.parquet",
        index=False,
    )

    result = build_customer_signals(tmp_path)

    customer = result.iloc[0]

    assert customer["transition_count"] == 1
    assert customer["behavior_direction"] == "STABLE"
    assert customer["evidence_level"] == "LOW"
    assert customer["signal_confidence"] == "LOW"


def test_two_increasing_transitions_have_medium_confidence(
    tmp_path,
):
    data = pd.DataFrame(
        {
            "month": pd.to_datetime(
                [
                    "2018-01-01",
                    "2018-02-01",
                    "2018-03-01",
                ]
            ),
            "customer_unique_id": [
                "customer_1",
                "customer_1",
                "customer_1",
            ],
            "order_count": [
                1,
                1,
                1,
            ],
            "item_line_count": [
                1,
                1,
                1,
            ],
            "revenue": [
                100.0,
                200.0,
                300.0,
            ],
            "freight_value": [
                10.0,
                10.0,
                10.0,
            ],
            "avg_order_value": [
                100.0,
                200.0,
                300.0,
            ],
            "first_purchase_month": pd.to_datetime(
                [
                    "2018-01-01",
                    "2018-01-01",
                    "2018-01-01",
                ]
            ),
            "customer_status": [
                "NEW",
                "REPEAT",
                "REPEAT",
            ],
        }
    )

    data.to_parquet(
        tmp_path
        / "mart_customer_performance_monthly.parquet",
        index=False,
    )

    result = build_customer_signals(tmp_path)

    customer = result.iloc[0]

    assert customer["transition_count"] == 2
    assert customer["direction_consistency"] == 1.0
    assert customer["behavior_direction"] == "INCREASING"
    assert customer["evidence_level"] == "MEDIUM"
    assert customer["signal_confidence"] == "MEDIUM"


def test_mixed_two_transitions_have_low_confidence(
    tmp_path,
):
    data = pd.DataFrame(
        {
            "month": pd.to_datetime(
                [
                    "2018-01-01",
                    "2018-02-01",
                    "2018-03-01",
                ]
            ),
            "customer_unique_id": [
                "customer_1",
                "customer_1",
                "customer_1",
            ],
            "order_count": [
                1,
                1,
                1,
            ],
            "item_line_count": [
                1,
                1,
                1,
            ],
            "revenue": [
                100.0,
                200.0,
                100.0,
            ],
            "freight_value": [
                10.0,
                10.0,
                10.0,
            ],
            "avg_order_value": [
                100.0,
                200.0,
                100.0,
            ],
            "first_purchase_month": pd.to_datetime(
                [
                    "2018-01-01",
                    "2018-01-01",
                    "2018-01-01",
                ]
            ),
            "customer_status": [
                "NEW",
                "REPEAT",
                "REPEAT",
            ],
        }
    )

    data.to_parquet(
        tmp_path
        / "mart_customer_performance_monthly.parquet",
        index=False,
    )

    result = build_customer_signals(tmp_path)

    customer = result.iloc[0]

    assert customer["transition_count"] == 2
    assert customer["direction_consistency"] == 0.5
    assert customer["behavior_direction"] == "STABLE"
    assert customer["evidence_level"] == "MEDIUM"
    assert customer["signal_confidence"] == "LOW"


def test_three_consistent_transitions_have_high_confidence(
    tmp_path,
):
    data = pd.DataFrame(
        {
            "month": pd.to_datetime(
                [
                    "2018-01-01",
                    "2018-02-01",
                    "2018-03-01",
                    "2018-04-01",
                ]
            ),
            "customer_unique_id": [
                "customer_1",
                "customer_1",
                "customer_1",
                "customer_1",
            ],
            "order_count": [
                1,
                1,
                1,
                1,
            ],
            "item_line_count": [
                1,
                1,
                1,
                1,
            ],
            "revenue": [
                100.0,
                200.0,
                300.0,
                400.0,
            ],
            "freight_value": [
                10.0,
                10.0,
                10.0,
                10.0,
            ],
            "avg_order_value": [
                100.0,
                200.0,
                300.0,
                400.0,
            ],
            "first_purchase_month": pd.to_datetime(
                [
                    "2018-01-01",
                    "2018-01-01",
                    "2018-01-01",
                    "2018-01-01",
                ]
            ),
            "customer_status": [
                "NEW",
                "REPEAT",
                "REPEAT",
                "REPEAT",
            ],
        }
    )

    data.to_parquet(
        tmp_path
        / "mart_customer_performance_monthly.parquet",
        index=False,
    )

    result = build_customer_signals(tmp_path)

    customer = result.iloc[0]

    assert customer["transition_count"] == 3
    assert customer["direction_consistency"] == 1.0
    assert customer["behavior_direction"] == "INCREASING"
    assert customer["evidence_level"] == "HIGH"
    assert customer["signal_confidence"] == "HIGH"


def test_three_mixed_transitions_have_medium_confidence(
    tmp_path,
):
    data = pd.DataFrame(
        {
            "month": pd.to_datetime(
                [
                    "2018-01-01",
                    "2018-02-01",
                    "2018-03-01",
                    "2018-04-01",
                ]
            ),
            "customer_unique_id": [
                "customer_1",
                "customer_1",
                "customer_1",
                "customer_1",
            ],
            "order_count": [
                1,
                1,
                1,
                1,
            ],
            "item_line_count": [
                1,
                1,
                1,
                1,
            ],
            "revenue": [
                100.0,
                200.0,
                100.0,
                200.0,
            ],
            "freight_value": [
                10.0,
                10.0,
                10.0,
                10.0,
            ],
            "avg_order_value": [
                100.0,
                200.0,
                100.0,
                200.0,
            ],
            "first_purchase_month": pd.to_datetime(
                [
                    "2018-01-01",
                    "2018-01-01",
                    "2018-01-01",
                    "2018-01-01",
                ]
            ),
            "customer_status": [
                "NEW",
                "REPEAT",
                "REPEAT",
                "REPEAT",
            ],
        }
    )

    data.to_parquet(
        tmp_path
        / "mart_customer_performance_monthly.parquet",
        index=False,
    )

    result = build_customer_signals(tmp_path)

    customer = result.iloc[0]

    assert customer["transition_count"] == 3
    assert customer["direction_consistency"] < 0.75
    assert customer["evidence_level"] == "HIGH"
    assert customer["signal_confidence"] == "MEDIUM"