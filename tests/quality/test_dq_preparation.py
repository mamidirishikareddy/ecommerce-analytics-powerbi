import pandas as pd
import pytest

from src.quality.dq_preparation import (
    build_order_item_temporal_frame,
    build_order_reconciliation_frame,
)


def test_build_order_item_temporal_frame():
    orders = pd.DataFrame(
        {
            "order_id": ["O1", "O2"],
            "order_purchase_timestamp": pd.to_datetime(
                ["2024-01-01", "2024-01-02"]
            ),
        }
    )

    order_items = pd.DataFrame(
        {
            "order_id": ["O1", "O1", "O2"],
            "shipping_limit_date": pd.to_datetime(
                ["2024-01-03", "2024-01-04", "2024-01-05"]
            ),
        }
    )

    result = build_order_item_temporal_frame(
        orders,
        order_items,
    )

    assert len(result) == 3
    assert list(result.columns) == [
        "order_id",
        "shipping_limit_date",
        "order_purchase_timestamp",
    ]
    assert result["order_purchase_timestamp"].notna().all()


def test_build_order_item_temporal_frame_rejects_invalid_parent_grain():
    orders = pd.DataFrame(
        {
            "order_id": ["O1", "O1"],
            "order_purchase_timestamp": pd.to_datetime(
                ["2024-01-01", "2024-01-02"]
            ),
        }
    )

    order_items = pd.DataFrame(
        {
            "order_id": ["O1"],
            "shipping_limit_date": pd.to_datetime(
                ["2024-01-03"]
            ),
        }
    )

    with pytest.raises(Exception):
        build_order_item_temporal_frame(
            orders,
            order_items,
        )


def test_build_order_reconciliation_frame():
    order_items = pd.DataFrame(
        {
            "order_id": ["O1", "O1", "O2"],
            "price": [100.0, 50.0, 80.0],
            "freight_value": [10.0, 5.0, 8.0],
        }
    )

    payments = pd.DataFrame(
        {
            "order_id": ["O1", "O2"],
            "payment_value": [165.0, 88.0],
        }
    )

    result = build_order_reconciliation_frame(
        order_items,
        payments,
    )

    result = result.sort_values("order_id").reset_index(drop=True)

    assert list(result.columns) == [
        "order_id",
        "expected_value",
        "payment_value",
    ]

    assert result.loc[0, "expected_value"] == 165.0
    assert result.loc[0, "payment_value"] == 165.0
    assert result.loc[1, "expected_value"] == 88.0
    assert result.loc[1, "payment_value"] == 88.0


def test_build_order_reconciliation_frame_preserves_unmatched_orders():
    order_items = pd.DataFrame(
        {
            "order_id": ["O1"],
            "price": [100.0],
            "freight_value": [10.0],
        }
    )

    payments = pd.DataFrame(
        {
            "order_id": ["O2"],
            "payment_value": [50.0],
        }
    )

    result = build_order_reconciliation_frame(
        order_items,
        payments,
    )

    assert len(result) == 2
    assert set(result["order_id"]) == {"O1", "O2"}


def test_build_order_reconciliation_frame_rejects_missing_columns():
    order_items = pd.DataFrame(
        {
            "order_id": ["O1"],
            "price": [100.0],
        }
    )

    payments = pd.DataFrame(
        {
            "order_id": ["O1"],
            "payment_value": [100.0],
        }
    )

    with pytest.raises(KeyError):
        build_order_reconciliation_frame(
            order_items,
            payments,
        )