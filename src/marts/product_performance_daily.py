from pathlib import Path

import pandas as pd


class ProductPerformanceMartError(Exception):
    """Raised when the product performance mart cannot be built."""


REQUIRED_TABLES = {
    "fact_order",
    "fact_order_item",
    "dim_product",
    "dim_category",
    "dim_customer",
}


def _load_warehouse_table(
    warehouse_dir: str | Path,
    table_name: str,
) -> pd.DataFrame:
    path = Path(warehouse_dir) / f"{table_name}.parquet"

    if not path.exists():
        raise ProductPerformanceMartError(
            f"Warehouse table not found: {path}"
        )

    return pd.read_parquet(path)


def build_product_performance_daily(
    warehouse_dir: str | Path,
) -> pd.DataFrame:
    """
    Build the daily product performance analytical mart.

    Grain:
        one row per calendar date and product.

    Revenue basis:
        SUM(fact_order_item.price)

    Important:
        Payments and reviews are intentionally excluded to prevent
        one-to-many fan-out from inflating product revenue.
    """

    warehouse_dir = Path(warehouse_dir)

    orders = _load_warehouse_table(
        warehouse_dir,
        "fact_order",
    )

    order_items = _load_warehouse_table(
        warehouse_dir,
        "fact_order_item",
    )

    products = _load_warehouse_table(
        warehouse_dir,
        "dim_product",
    )

    categories = _load_warehouse_table(
        warehouse_dir,
        "dim_category",
    )

    customers = _load_warehouse_table(
        warehouse_dir,
        "dim_customer",
    )

    # ---------------------------------------------------------------
    # Validate required columns before transformation
    # ---------------------------------------------------------------

    required_columns = {
        "fact_order": {
            "order_id",
            "customer_id",
            "order_purchase_timestamp",
        },
        "fact_order_item": {
            "order_id",
            "product_id",
            "price",
            "freight_value",
        },
        "dim_product": {
            "product_id",
            "product_category_name",
        },
        "dim_category": {
            "product_category_name",
            "product_category_name_english",
        },
        "dim_customer": {
            "customer_id",
            "customer_unique_id",
        },
    }

    loaded_tables = {
        "fact_order": orders,
        "fact_order_item": order_items,
        "dim_product": products,
        "dim_category": categories,
        "dim_customer": customers,
    }

    for table_name, columns in required_columns.items():
        missing = columns - set(
            loaded_tables[table_name].columns
        )

        if missing:
            raise ProductPerformanceMartError(
                f"{table_name}: missing columns {sorted(missing)}"
            )

    # ---------------------------------------------------------------
    # Prepare order-level analytical attributes
    # ---------------------------------------------------------------

    order_attributes = orders[
        [
            "order_id",
            "customer_id",
            "order_purchase_timestamp",
        ]
    ].copy()

    order_attributes["date"] = (
        pd.to_datetime(
            order_attributes["order_purchase_timestamp"]
        )
        .dt.normalize()
    )

    # ---------------------------------------------------------------
    # Attach customer identity
    # ---------------------------------------------------------------

    customer_attributes = customers[
        [
            "customer_id",
            "customer_unique_id",
        ]
    ].copy()

    order_attributes = order_attributes.merge(
        customer_attributes,
        on="customer_id",
        how="left",
        validate="one_to_one",
    )

    # ---------------------------------------------------------------
    # Attach order date to order items
    # ---------------------------------------------------------------

    item_data = order_items[
        [
            "order_id",
            "product_id",
            "price",
            "freight_value",
        ]
    ].merge(
        order_attributes[
            [
                "order_id",
                "date",
                "customer_unique_id",
            ]
        ],
        on="order_id",
        how="left",
        validate="many_to_one",
    )

    # ---------------------------------------------------------------
    # Attach product category
    # ---------------------------------------------------------------

    product_attributes = products[
        [
            "product_id",
            "product_category_name",
        ]
    ].copy()

    item_data = item_data.merge(
        product_attributes,
        on="product_id",
        how="left",
        validate="many_to_one",
    )

    # ---------------------------------------------------------------
    # Attach translated category
    # ---------------------------------------------------------------

    category_attributes = categories[
        [
            "product_category_name",
            "product_category_name_english",
        ]
    ].copy()

    item_data = item_data.merge(
        category_attributes,
        on="product_category_name",
        how="left",
        validate="many_to_one",
    )

    # ---------------------------------------------------------------
    # Validate required analytical attributes
    # ---------------------------------------------------------------

    if item_data["date"].isna().any():
        raise ProductPerformanceMartError(
            "Some order items could not be assigned an order date."
        )

    if item_data["customer_unique_id"].isna().any():
        raise ProductPerformanceMartError(
            "Some order items could not be assigned a customer identity."
        )

    # ---------------------------------------------------------------
    # Aggregate to mart grain:
    #
    # 1 row = date × product
    # ---------------------------------------------------------------

    mart = (
        item_data
        .groupby(
            [
                "date",
                "product_id",
                "product_category_name",
                "product_category_name_english",
            ],
            dropna=False,
            as_index=False,
        )
        .agg(
            item_line_count=(
                "product_id",
                "size",
            ),
            order_count=(
                "order_id",
                "nunique",
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
            avg_item_price=(
                "price",
                "mean",
            ),
        )
    )

    # Governed definition:
    # gross_merchandise_value = revenue
    mart["gross_merchandise_value"] = mart["revenue"]

    # ---------------------------------------------------------------
    # Enforce final column order
    # ---------------------------------------------------------------

    expected_columns = [
        "date",
        "product_id",
        "product_category_name",
        "product_category_name_english",
        "item_line_count",
        "order_count",
        "customer_count",
        "revenue",
        "freight_value",
        "gross_merchandise_value",
        "avg_item_price",
    ]

    mart = mart[expected_columns]

    # ---------------------------------------------------------------
    # Enforce output dtypes
    # ---------------------------------------------------------------

    mart["date"] = pd.to_datetime(
        mart["date"]
    )

    mart["item_line_count"] = mart[
        "item_line_count"
    ].astype("int64")

    mart["order_count"] = mart[
        "order_count"
    ].astype("int64")

    mart["customer_count"] = mart[
        "customer_count"
    ].astype("int64")

    mart["revenue"] = mart[
        "revenue"
    ].astype("float64")

    mart["freight_value"] = mart[
        "freight_value"
    ].astype("float64")

    mart["gross_merchandise_value"] = mart[
        "gross_merchandise_value"
    ].astype("float64")

    mart["avg_item_price"] = mart[
        "avg_item_price"
    ].astype("float64")

    return mart