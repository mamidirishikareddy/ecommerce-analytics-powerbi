from pathlib import Path

import pandas as pd


class MarketPerformanceMartError(Exception):
    """Raised when market performance mart construction fails."""


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
        "customer_zip_code_prefix",
    ],
    "dim_date": [
        "date",
    ],
}


OUTPUT_COLUMNS = [
    "month",
    "customer_zip_code_prefix",
    "order_count",
    "item_line_count",
    "customer_count",
    "revenue",
    "freight_value",
    "avg_order_value",
    "revenue_per_customer",
    "avg_freight_per_order",
]


def _load_warehouse_table(
    warehouse_dir: str | Path,
    table_name: str,
) -> pd.DataFrame:
    """Load one persisted warehouse table."""

    path = Path(warehouse_dir) / f"{table_name}.parquet"

    if not path.exists():
        raise MarketPerformanceMartError(
            f"Warehouse table not found: {path}"
        )

    return pd.read_parquet(path)


def _validate_required_columns(
    df: pd.DataFrame,
    table_name: str,
) -> None:
    """Validate that a warehouse table contains required columns."""

    required_columns = REQUIRED_TABLES[table_name]

    missing = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing:
        raise MarketPerformanceMartError(
            f"{table_name} is missing required columns: {missing}"
        )


def build_market_performance_monthly(
    warehouse_dir: str | Path,
) -> pd.DataFrame:
    """
    Build the monthly customer ZIP-prefix market performance mart.

    Grain:
        One row per calendar month and customer ZIP-prefix market.

    Business key:
        month + customer_zip_code_prefix
    """

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

    customer_attributes = customers[
        [
            "customer_id",
            "customer_unique_id",
            "customer_zip_code_prefix",
        ]
    ].copy()

    if customer_attributes["customer_id"].duplicated().any():
        raise MarketPerformanceMartError(
            "dim_customer.customer_id is not unique."
        )

    if customer_attributes["customer_unique_id"].isna().any():
        raise MarketPerformanceMartError(
            "customer_unique_id contains null values."
        )

    if customer_attributes[
        "customer_zip_code_prefix"
    ].isna().any():
        raise MarketPerformanceMartError(
            "customer_zip_code_prefix contains null values."
        )

    # ---------------------------------------------------------------
    # Attach customer identity and market to orders
    #
    # Relationship:
    # fact_order MANY
    #       |
    #       v
    # dim_customer ONE
    # ---------------------------------------------------------------

    order_customer = orders.merge(
        customer_attributes,
        on="customer_id",
        how="inner",
        validate="many_to_one",
    )

    if len(order_customer) != len(orders):
        raise MarketPerformanceMartError(
            "Customer dimension join changed the order population."
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
    # Attach market/month information to order-item rows
    #
    # Relationship:
    # fact_order_item MANY
    #       |
    #       v
    # fact_order ONE
    # ---------------------------------------------------------------

    order_items = items.merge(
        order_customer[
            [
                "order_id",
                "customer_unique_id",
                "customer_zip_code_prefix",
                "month",
            ]
        ],
        on="order_id",
        how="inner",
        validate="many_to_one",
    )

    if len(order_items) != len(items):
        raise MarketPerformanceMartError(
            "Order-item population changed when joining to orders."
        )

    # ---------------------------------------------------------------
    # Market-month aggregation
    # ---------------------------------------------------------------

    mart = (
        order_items
        .groupby(
            [
                "month",
                "customer_zip_code_prefix",
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
            customer_count=(
                "customer_unique_id",
                "nunique",
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
    # Derived commercial metrics
    # ---------------------------------------------------------------

    mart["avg_order_value"] = (
        mart["revenue"]
        / mart["order_count"]
    )

    mart["revenue_per_customer"] = (
        mart["revenue"]
        / mart["customer_count"]
    )

    mart["avg_freight_per_order"] = (
        mart["freight_value"]
        / mart["order_count"]
    )

    # ---------------------------------------------------------------
    # Final grain validation
    # ---------------------------------------------------------------

    duplicate_count = int(
        mart.duplicated(
            [
                "month",
                "customer_zip_code_prefix",
            ]
        ).sum()
    )

    if duplicate_count != 0:
        raise MarketPerformanceMartError(
            "Market-month grain violated: "
            f"{duplicate_count} duplicate rows."
        )

    # ---------------------------------------------------------------
    # Metric validations
    # ---------------------------------------------------------------

    if mart["order_count"].le(0).any():
        raise MarketPerformanceMartError(
            "order_count contains non-positive values."
        )

    if mart["item_line_count"].le(0).any():
        raise MarketPerformanceMartError(
            "item_line_count contains non-positive values."
        )

    if mart["customer_count"].le(0).any():
        raise MarketPerformanceMartError(
            "customer_count contains non-positive values."
        )

    if mart["revenue"].lt(0).any():
        raise MarketPerformanceMartError(
            "revenue contains negative values."
        )

    if mart["freight_value"].lt(0).any():
        raise MarketPerformanceMartError(
            "freight_value contains negative values."
        )

    # ---------------------------------------------------------------
    # Validate derived metrics
    # ---------------------------------------------------------------

    expected_aov = (
        mart["revenue"]
        / mart["order_count"]
    )

    expected_revenue_per_customer = (
        mart["revenue"]
        / mart["customer_count"]
    )

    expected_avg_freight = (
        mart["freight_value"]
        / mart["order_count"]
    )

    if not (
        mart["avg_order_value"]
        .eq(expected_aov)
        .all()
    ):
        raise MarketPerformanceMartError(
            "avg_order_value calculation is inconsistent."
        )

    if not (
        mart["revenue_per_customer"]
        .eq(expected_revenue_per_customer)
        .all()
    ):
        raise MarketPerformanceMartError(
            "revenue_per_customer calculation is inconsistent."
        )

    if not (
        mart["avg_freight_per_order"]
        .eq(expected_avg_freight)
        .all()
    ):
        raise MarketPerformanceMartError(
            "avg_freight_per_order calculation is inconsistent."
        )

    # ---------------------------------------------------------------
    # Validate final output schema
    # ---------------------------------------------------------------

    mart = mart[OUTPUT_COLUMNS].copy()

    return mart