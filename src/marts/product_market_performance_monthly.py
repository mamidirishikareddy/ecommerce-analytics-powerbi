from __future__ import annotations

from pathlib import Path

import pandas as pd


class ProductMarketPerformanceMartError(Exception):
    """Raised when the product-market performance mart cannot be built."""


REQUIRED_TABLES = [
    "fact_order",
    "fact_order_item",
    "dim_customer",
    "dim_product",
    "dim_category",
    "dim_date",
]

OUTPUT_COLUMNS = [
    "month",
    "product_id",
    "customer_zip_code_prefix",
    "product_category_name",
    "product_category_name_english",
    "order_count",
    "item_line_count",
    "customer_count",
    "revenue",
    "freight_value",
    "avg_order_value",
    "revenue_per_customer",
    "market_contribution_to_product_revenue_pct",
]


def _load_warehouse_table(
    warehouse_dir: Path,
    table_name: str,
) -> pd.DataFrame:
    """Load one persisted warehouse table."""

    path = warehouse_dir / f"{table_name}.parquet"

    if not path.exists():
        raise ProductMarketPerformanceMartError(
            f"Required warehouse table does not exist: {path}"
        )

    try:
        return pd.read_parquet(path)
    except Exception as exc:
        raise ProductMarketPerformanceMartError(
            f"Failed to read warehouse table '{table_name}': {exc}"
        ) from exc


def _validate_required_columns(
    df: pd.DataFrame,
    table_name: str,
    required_columns: list[str],
) -> None:
    """Validate that a warehouse table contains required columns."""

    missing = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing:
        raise ProductMarketPerformanceMartError(
            f"{table_name} is missing required columns: {missing}"
        )


def build_product_market_performance_monthly(
    warehouse_dir: str | Path,
) -> pd.DataFrame:
    """
    Build the governed product-market performance monthly mart.

    Grain:
        One row per month, product_id, and customer ZIP-prefix market.

    Revenue and freight are sourced exclusively from fact_order_item.
    Payments and reviews are intentionally excluded to prevent fan-out.
    """

    warehouse_dir = Path(warehouse_dir)

    # ------------------------------------------------------------------
    # 1. Load governed warehouse tables
    # ------------------------------------------------------------------

    fact_order = _load_warehouse_table(
        warehouse_dir,
        "fact_order",
    )

    fact_order_item = _load_warehouse_table(
        warehouse_dir,
        "fact_order_item",
    )

    dim_customer = _load_warehouse_table(
        warehouse_dir,
        "dim_customer",
    )

    dim_product = _load_warehouse_table(
        warehouse_dir,
        "dim_product",
    )

    dim_category = _load_warehouse_table(
        warehouse_dir,
        "dim_category",
    )

    dim_date = _load_warehouse_table(
        warehouse_dir,
        "dim_date",
    )

    # ------------------------------------------------------------------
    # 2. Validate required columns
    # ------------------------------------------------------------------

    _validate_required_columns(
        fact_order,
        "fact_order",
        [
            "order_id",
            "customer_id",
            "order_purchase_timestamp",
        ],
    )

    _validate_required_columns(
        fact_order_item,
        "fact_order_item",
        [
            "order_id",
            "order_item_id",
            "product_id",
            "price",
            "freight_value",
        ],
    )

    _validate_required_columns(
        dim_customer,
        "dim_customer",
        [
            "customer_id",
            "customer_unique_id",
            "customer_zip_code_prefix",
        ],
    )

    _validate_required_columns(
        dim_product,
        "dim_product",
        [
            "product_id",
            "product_category_name",
        ],
    )

    _validate_required_columns(
        dim_category,
        "dim_category",
        [
            "product_category_name",
            "product_category_name_english",
        ],
    )

    _validate_required_columns(
        dim_date,
        "dim_date",
        ["date"],
    )

    # ------------------------------------------------------------------
    # 3. Validate dimension uniqueness
    # ------------------------------------------------------------------

    if dim_customer["customer_id"].duplicated().any():
        raise ProductMarketPerformanceMartError(
            "dim_customer.customer_id is not unique."
        )

    if dim_product["product_id"].duplicated().any():
        raise ProductMarketPerformanceMartError(
            "dim_product.product_id is not unique."
        )

    if dim_category["product_category_name"].duplicated().any():
        raise ProductMarketPerformanceMartError(
            "dim_category.product_category_name is not unique."
        )

    # ------------------------------------------------------------------
    # 4. Validate required source identifiers
    # ------------------------------------------------------------------

    if fact_order["order_id"].isna().any():
        raise ProductMarketPerformanceMartError(
            "fact_order contains null order_id values."
        )

    if fact_order_item["order_id"].isna().any():
        raise ProductMarketPerformanceMartError(
            "fact_order_item contains null order_id values."
        )

    if fact_order_item["product_id"].isna().any():
        raise ProductMarketPerformanceMartError(
            "fact_order_item contains null product_id values."
        )

    if dim_customer["customer_unique_id"].isna().any():
        raise ProductMarketPerformanceMartError(
            "dim_customer contains null customer_unique_id values."
        )

    if dim_customer["customer_zip_code_prefix"].isna().any():
        raise ProductMarketPerformanceMartError(
            "dim_customer contains null customer_zip_code_prefix values."
        )

    # ------------------------------------------------------------------
    # 5. Attach customer identity and market
    #
    # fact_order.customer_id -> dim_customer.customer_id
    # Many orders can belong to the same customer_id.
    # Therefore the relationship is many-to-one.
    # ------------------------------------------------------------------

    order_customer = fact_order.merge(
        dim_customer[
            [
                "customer_id",
                "customer_unique_id",
                "customer_zip_code_prefix",
            ]
        ],
        on="customer_id",
        how="inner",
        validate="many_to_one",
    )

    if len(order_customer) != len(fact_order):
        raise ProductMarketPerformanceMartError(
            "Customer join changed fact_order population. "
            "Expected a one-to-one row-preserving enrichment."
        )

    # ------------------------------------------------------------------
    # 6. Derive calendar month
    # ------------------------------------------------------------------

    try:
        order_customer["order_purchase_timestamp"] = pd.to_datetime(
            order_customer["order_purchase_timestamp"],
            errors="raise",
        )
    except Exception as exc:
        raise ProductMarketPerformanceMartError(
            "Failed to parse fact_order.order_purchase_timestamp."
        ) from exc

    order_customer["month"] = (
        order_customer["order_purchase_timestamp"]
        .dt.to_period("M")
        .dt.to_timestamp()
    )

    # ------------------------------------------------------------------
    # 7. Join order items
    #
    # fact_order.order_id -> fact_order_item.order_id
    # One order can have many item rows.
    # ------------------------------------------------------------------

    order_item_population_before = len(fact_order_item)

    order_items = order_customer[
        [
            "order_id",
            "customer_unique_id",
            "customer_zip_code_prefix",
            "month",
        ]
    ].merge(
        fact_order_item[
            [
                "order_id",
                "order_item_id",
                "product_id",
                "price",
                "freight_value",
            ]
        ],
        on="order_id",
        how="inner",
        validate="one_to_many",
    )

    if len(order_items) != order_item_population_before:
        raise ProductMarketPerformanceMartError(
            "Order-item join changed fact_order_item population. "
            "Expected one output row per source order-item row."
        )

    # ------------------------------------------------------------------
    # 8. Attach product attributes
    #
    # fact_order_item.product_id -> dim_product.product_id
    # ------------------------------------------------------------------

    order_items = order_items.merge(
        dim_product[
            [
                "product_id",
                "product_category_name",
            ]
        ],
        on="product_id",
        how="inner",
        validate="many_to_one",
    )

    # ------------------------------------------------------------------
    # 9. Attach translated category
    #
    # dim_product.product_category_name ->
    # dim_category.product_category_name
    #
    # LEFT JOIN is intentional because translation coverage is
    # incomplete and missing translations must remain visible.
    # ------------------------------------------------------------------

    order_items = order_items.merge(
        dim_category[
            [
                "product_category_name",
                "product_category_name_english",
            ]
        ],
        on="product_category_name",
        how="left",
        validate="many_to_one",
    )

    # ------------------------------------------------------------------
    # 10. Validate item-level population after dimension enrichment
    # ------------------------------------------------------------------

    if len(order_items) != order_item_population_before:
        raise ProductMarketPerformanceMartError(
            "Product/category enrichment changed order-item population."
        )

    # ------------------------------------------------------------------
    # 11. Validate numeric source fields
    # ------------------------------------------------------------------

    if order_items["price"].isna().any():
        raise ProductMarketPerformanceMartError(
            "Order-item price contains null values."
        )

    if order_items["freight_value"].isna().any():
        raise ProductMarketPerformanceMartError(
            "Order-item freight_value contains null values."
        )

    if (order_items["price"] < 0).any():
        raise ProductMarketPerformanceMartError(
            "Order-item price contains negative values."
        )

    if (order_items["freight_value"] < 0).any():
        raise ProductMarketPerformanceMartError(
            "Order-item freight_value contains negative values."
        )

    # ------------------------------------------------------------------
    # 12. Aggregate at frozen mart grain
    #
    # month + product_id + customer_zip_code_prefix
    # ------------------------------------------------------------------

    group_columns = [
        "month",
        "product_id",
        "customer_zip_code_prefix",
        "product_category_name",
        "product_category_name_english",
    ]

    mart = (
        order_items.groupby(
            group_columns,
            dropna=False,
            as_index=False,
        )
        .agg(
            order_count=("order_id", "nunique"),
            item_line_count=("order_item_id", "size"),
            customer_count=("customer_unique_id", "nunique"),
            revenue=("price", "sum"),
            freight_value=("freight_value", "sum"),
        )
    )

    # ------------------------------------------------------------------
    # 13. Derived metrics
    # ------------------------------------------------------------------

    if (mart["order_count"] <= 0).any():
        raise ProductMarketPerformanceMartError(
            "Derived order_count contains zero or negative values."
        )

    if (mart["item_line_count"] <= 0).any():
        raise ProductMarketPerformanceMartError(
            "Derived item_line_count contains zero or negative values."
        )

    if (mart["customer_count"] <= 0).any():
        raise ProductMarketPerformanceMartError(
            "Derived customer_count contains zero or negative values."
        )

    if (mart["revenue"] < 0).any():
        raise ProductMarketPerformanceMartError(
            "Derived revenue contains negative values."
        )

    if (mart["freight_value"] < 0).any():
        raise ProductMarketPerformanceMartError(
            "Derived freight_value contains negative values."
        )

    mart["avg_order_value"] = (
        mart["revenue"] / mart["order_count"]
    )

    mart["revenue_per_customer"] = (
        mart["revenue"] / mart["customer_count"]
    )

    # ------------------------------------------------------------------
    # 14. Product-month denominator for market contribution
    #
    # Denominator:
    # total product revenue across every customer ZIP-prefix market
    # for the same month and product.
    # ------------------------------------------------------------------

    product_month_revenue = (
        mart.groupby(
            [
                "month",
                "product_id",
            ],
            as_index=False,
        )["revenue"]
        .sum()
        .rename(
            columns={
                "revenue": "product_month_total_revenue"
            }
        )
    )

    mart = mart.merge(
        product_month_revenue,
        on=[
            "month",
            "product_id",
        ],
        how="left",
        validate="many_to_one",
    )

    if mart["product_month_total_revenue"].isna().any():
        raise ProductMarketPerformanceMartError(
            "Product-month revenue denominator contains null values."
        )

    if (mart["product_month_total_revenue"] <= 0).any():
        raise ProductMarketPerformanceMartError(
            "Product-month revenue denominator contains zero or negative values."
        )

    mart["market_contribution_to_product_revenue_pct"] = (
        mart["revenue"]
        / mart["product_month_total_revenue"]
        * 100.0
    )

    # Denominator is an implementation field only.
    mart = mart.drop(
        columns=["product_month_total_revenue"]
    )

    # ------------------------------------------------------------------
    # 15. Validate frozen mart grain
    # ------------------------------------------------------------------

    grain_columns = [
        "month",
        "product_id",
        "customer_zip_code_prefix",
    ]

    duplicate_grain = mart.duplicated(
        subset=grain_columns
    ).sum()

    if duplicate_grain != 0:
        raise ProductMarketPerformanceMartError(
            f"Mart grain violation: {duplicate_grain} duplicate "
            f"month/product/market rows detected."
        )

    # ------------------------------------------------------------------
    # 16. Validate contribution percentages
    # ------------------------------------------------------------------

    contribution_sums = (
        mart.groupby(
            [
                "month",
                "product_id",
            ]
        )["market_contribution_to_product_revenue_pct"]
        .sum()
    )

    if not contribution_sums.between(
        99.999999,
        100.000001,
    ).all():
        raise ProductMarketPerformanceMartError(
            "Market contribution percentages do not sum to approximately "
            "100% for every product-month."
        )

    # ------------------------------------------------------------------
    # 17. Validate derived metric consistency
    # ------------------------------------------------------------------

    expected_aov = (
        mart["revenue"] / mart["order_count"]
    )

    if not (
        mart["avg_order_value"]
        .sub(expected_aov)
        .abs()
        .lt(1e-9)
        .all()
    ):
        raise ProductMarketPerformanceMartError(
            "avg_order_value is inconsistent with revenue / order_count."
        )

    expected_revenue_per_customer = (
        mart["revenue"] / mart["customer_count"]
    )

    if not (
        mart["revenue_per_customer"]
        .sub(expected_revenue_per_customer)
        .abs()
        .lt(1e-9)
        .all()
    ):
        raise ProductMarketPerformanceMartError(
            "revenue_per_customer is inconsistent with "
            "revenue / customer_count."
        )

    # ------------------------------------------------------------------
    # 18. Final column order
    # ------------------------------------------------------------------

    mart = mart[OUTPUT_COLUMNS].copy()

    # ------------------------------------------------------------------
    # 19. Final dtype normalization
    # ------------------------------------------------------------------

    mart["month"] = pd.to_datetime(
        mart["month"],
        errors="raise",
    )

    mart["product_id"] = mart["product_id"].astype("string")

    mart["customer_zip_code_prefix"] = (
        mart["customer_zip_code_prefix"].astype("int64")
    )

    mart["order_count"] = mart["order_count"].astype("int64")
    mart["item_line_count"] = mart["item_line_count"].astype("int64")
    mart["customer_count"] = mart["customer_count"].astype("int64")

    mart["revenue"] = mart["revenue"].astype("float64")
    mart["freight_value"] = mart["freight_value"].astype("float64")
    mart["avg_order_value"] = mart["avg_order_value"].astype("float64")
    mart["revenue_per_customer"] = (
        mart["revenue_per_customer"].astype("float64")
    )
    mart["market_contribution_to_product_revenue_pct"] = (
        mart["market_contribution_to_product_revenue_pct"]
        .astype("float64")
    )

    return mart