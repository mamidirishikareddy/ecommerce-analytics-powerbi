from __future__ import annotations

import pandas as pd


def build_order_item_temporal_frame(
    orders: pd.DataFrame,
    order_items: pd.DataFrame,
) -> pd.DataFrame:
    """Build the execution frame required for TIME-006."""
    required_orders = {
        "order_id",
        "order_purchase_timestamp",
    }
    required_items = {
        "order_id",
        "shipping_limit_date",
    }

    missing_orders = required_orders - set(orders.columns)
    missing_items = required_items - set(order_items.columns)

    if missing_orders:
        raise KeyError(
            f"Missing order columns: {sorted(missing_orders)}"
        )

    if missing_items:
        raise KeyError(
            f"Missing order-item columns: {sorted(missing_items)}"
        )

    return order_items[
        ["order_id", "shipping_limit_date"]
    ].merge(
        orders[
            ["order_id", "order_purchase_timestamp"]
        ],
        on="order_id",
        how="left",
        validate="many_to_one",
    )


def build_order_reconciliation_frame(
    order_items: pd.DataFrame,
    payments: pd.DataFrame,
) -> pd.DataFrame:
    """Build the order-level execution frame required for REC-001."""
    required_items = {
        "order_id",
        "price",
        "freight_value",
    }
    required_payments = {
        "order_id",
        "payment_value",
    }

    missing_items = required_items - set(order_items.columns)
    missing_payments = required_payments - set(payments.columns)

    if missing_items:
        raise KeyError(
            f"Missing order-item columns: {sorted(missing_items)}"
        )

    if missing_payments:
        raise KeyError(
            f"Missing payment columns: {sorted(missing_payments)}"
        )

    item_values = (
        order_items
        .assign(
            item_plus_freight=lambda df:
            df["price"] + df["freight_value"]
        )
        .groupby("order_id", as_index=False)
        .agg(
            expected_value=("item_plus_freight", "sum")
        )
    )

    payment_values = (
        payments
        .groupby("order_id", as_index=False)
        .agg(
            payment_value=("payment_value", "sum")
        )
    )

    return item_values.merge(
        payment_values,
        on="order_id",
        how="outer",
    )