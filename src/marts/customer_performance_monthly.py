from pathlib import Path

import pandas as pd


class CustomerPerformanceMartError(Exception):
    """Raised when customer performance mart construction fails."""


REQUIRED_TABLES = {
    "fact_order": [
        "order_id",
        "customer_id",
        "order_purchase_timestamp",
    ],
    "fact_order_item": [
        "order_id",
        "price",
        "freight_value",
    ],
    "dim_customer": [
        "customer_id",
        "customer_unique_id",
    ],
    "dim_date": [
        "date",
    ],
}


OUTPUT_COLUMNS = [
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


def _load_warehouse_table(
    warehouse_dir: str | Path,
    table_name: str,
) -> pd.DataFrame:

    path = Path(warehouse_dir) / f"{table_name}.parquet"

    if not path.exists():
        raise CustomerPerformanceMartError(
            f"Warehouse table not found: {path}"
        )

    return pd.read_parquet(path)


def _validate_required_columns(
    df: pd.DataFrame,
    table_name: str,
) -> None:

    required_columns = REQUIRED_TABLES[table_name]

    missing = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing:
        raise CustomerPerformanceMartError(
            f"{table_name} is missing required columns: {missing}"
        )


def build_customer_performance_monthly(
    warehouse_dir: str | Path,
) -> pd.DataFrame:

    # ---------------------------------------------------------------
    # Load warehouse tables
    # ---------------------------------------------------------------

    orders = _load_warehouse_table(
        warehouse_dir,
        "fact_order",
    )

    items = _load_warehouse_table(
        warehouse_dir,
        "fact_order_item",
    )

    customers = _load_warehouse_table(
        warehouse_dir,
        "dim_customer",
    )

    dates = _load_warehouse_table(
        warehouse_dir,
        "dim_date",
    )

    _validate_required_columns(
        orders,
        "fact_order",
    )

    _validate_required_columns(
        items,
        "fact_order_item",
    )

    _validate_required_columns(
        customers,
        "dim_customer",
    )

    _validate_required_columns(
        dates,
        "dim_date",
    )

    # ---------------------------------------------------------------
    # Validate customer dimension grain
    # ---------------------------------------------------------------

    customer_identity = customers[
        [
            "customer_id",
            "customer_unique_id",
        ]
    ].copy()

    if customer_identity["customer_id"].duplicated().any():
        raise CustomerPerformanceMartError(
            "dim_customer.customer_id is not unique."
        )

    if customer_identity["customer_unique_id"].isna().any():
        raise CustomerPerformanceMartError(
            "customer_unique_id contains null values."
        )

    # ---------------------------------------------------------------
    # Attach persistent customer identity to orders
    #
    # Relationship:
    # fact_order MANY
    #       ↓
    # dim_customer ONE
    # ---------------------------------------------------------------

    order_customer = orders.merge(
        customer_identity,
        on="customer_id",
        how="inner",
        validate="many_to_one",
    )

    if len(order_customer) != len(orders):
        raise CustomerPerformanceMartError(
            "Customer identity join changed the order population."
        )

    # ---------------------------------------------------------------
    # Validate purchase timestamp
    # ---------------------------------------------------------------

    order_customer["order_purchase_timestamp"] = pd.to_datetime(
        order_customer["order_purchase_timestamp"],
        errors="raise",
    )

    # ---------------------------------------------------------------
    # Derive calendar month
    # ---------------------------------------------------------------

    order_customer["month"] = (
        order_customer["order_purchase_timestamp"]
        .dt.to_period("M")
        .dt.to_timestamp()
    )

    # ---------------------------------------------------------------
    # Calculate first purchase month across COMPLETE history
    #
    # This must happen before customer-month aggregation.
    # ---------------------------------------------------------------

    first_purchase = (
        order_customer
        .groupby(
            "customer_unique_id",
            as_index=False,
        )["month"]
        .min()
        .rename(
            columns={
                "month": "first_purchase_month"
            }
        )
    )

    # ---------------------------------------------------------------
    # Attach customer/month information to order-item rows
    #
    # Relationship:
    # fact_order_item MANY
    #       ↓
    # fact_order ONE
    # ---------------------------------------------------------------

    order_items = items.merge(
        order_customer[
            [
                "order_id",
                "customer_unique_id",
                "month",
            ]
        ],
        on="order_id",
        how="inner",
        validate="many_to_one",
    )

    # Every order-item order should have a corresponding order.
    if len(order_items) != len(items):
        raise CustomerPerformanceMartError(
            "Order-item population changed when joining to orders."
        )

    # ---------------------------------------------------------------
    # Customer-month aggregation
    # ---------------------------------------------------------------

    mart = (
        order_items
        .groupby(
            [
                "month",
                "customer_unique_id",
            ],
            as_index=False,
        )
        .agg(
            order_count=(
                "order_id",
                "nunique",
            ),
            item_line_count=(
                "order_id",
                "size",
            ),
            revenue=(
                "price",
                "sum",
            ),
            freight_value=(
                "freight_value",
                "sum",
            ),
        )
    )

    # ---------------------------------------------------------------
    # Average order value
    # ---------------------------------------------------------------

    mart["avg_order_value"] = (
        mart["revenue"]
        / mart["order_count"]
    )

    # ---------------------------------------------------------------
    # Attach first purchase month
    # ---------------------------------------------------------------

    mart = mart.merge(
        first_purchase,
        on="customer_unique_id",
        how="left",
        validate="many_to_one",
    )

    if mart["first_purchase_month"].isna().any():
        raise CustomerPerformanceMartError(
            "Some customer-month rows have no first purchase month."
        )

    # ---------------------------------------------------------------
    # Customer status
    # ---------------------------------------------------------------

    mart["customer_status"] = "REPEAT"

    mart.loc[
        mart["month"] == mart["first_purchase_month"],
        "customer_status",
    ] = "NEW"

    # ---------------------------------------------------------------
    # Final grain validation
    # ---------------------------------------------------------------

    duplicate_count = int(
        mart.duplicated(
            [
                "month",
                "customer_unique_id",
            ]
        ).sum()
    )

    if duplicate_count != 0:
        raise CustomerPerformanceMartError(
            "Customer-month grain violated: "
            f"{duplicate_count} duplicate rows."
        )

    # ---------------------------------------------------------------
    # Metric validations
    # ---------------------------------------------------------------

    if mart["order_count"].le(0).any():
        raise CustomerPerformanceMartError(
            "order_count contains non-positive values."
        )

    if mart["item_line_count"].le(0).any():
        raise CustomerPerformanceMartError(
            "item_line_count contains non-positive values."
        )

    if mart["revenue"].lt(0).any():
        raise CustomerPerformanceMartError(
            "revenue contains negative values."
        )

    if mart["freight_value"].lt(0).any():
        raise CustomerPerformanceMartError(
            "freight_value contains negative values."
        )

    if not mart["customer_status"].isin(
        ["NEW", "REPEAT"]
    ).all():
        raise CustomerPerformanceMartError(
            "Invalid customer_status value detected."
        )

    # ---------------------------------------------------------------
    # Final column order
    # ---------------------------------------------------------------

    mart = mart[OUTPUT_COLUMNS].copy()

    return mart