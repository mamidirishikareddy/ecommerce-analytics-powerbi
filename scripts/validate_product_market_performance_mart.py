from pathlib import Path

import pandas as pd


MART_PATH = Path(
    "data/processed/marts/"
    "mart_product_market_performance_monthly.parquet"
)

EXPECTED_COLUMNS = [
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

GRAIN_COLUMNS = [
    "month",
    "product_id",
    "customer_zip_code_prefix",
]

WAREHOUSE_ITEM_PATH = Path(
    "data/processed/warehouse/fact_order_item.parquet"
)


def main() -> None:
    if not MART_PATH.exists():
        raise FileNotFoundError(
            f"Mart file does not exist: {MART_PATH}"
        )

    if not WAREHOUSE_ITEM_PATH.exists():
        raise FileNotFoundError(
            f"Warehouse fact does not exist: "
            f"{WAREHOUSE_ITEM_PATH}"
        )

    mart = pd.read_parquet(MART_PATH)
    fact_order_item = pd.read_parquet(
        WAREHOUSE_ITEM_PATH
    )

    # --------------------------------------------------------------
    # Schema
    # --------------------------------------------------------------

    if list(mart.columns) != EXPECTED_COLUMNS:
        raise AssertionError(
            "Mart columns do not match the governed contract."
        )

    # --------------------------------------------------------------
    # Grain
    # --------------------------------------------------------------

    duplicate_grain_rows = mart.duplicated(
        subset=GRAIN_COLUMNS
    ).sum()

    if duplicate_grain_rows != 0:
        raise AssertionError(
            f"Duplicate mart grain rows: "
            f"{duplicate_grain_rows}"
        )

    # --------------------------------------------------------------
    # Required non-null fields
    # --------------------------------------------------------------

    required_non_null = [
        "month",
        "product_id",
        "customer_zip_code_prefix",
        "order_count",
        "item_line_count",
        "customer_count",
        "revenue",
        "freight_value",
        "avg_order_value",
        "revenue_per_customer",
        "market_contribution_to_product_revenue_pct",
    ]

    null_counts = mart[required_non_null].isna().sum()

    if null_counts.sum() != 0:
        raise AssertionError(
            "Required mart fields contain null values:\n"
            f"{null_counts[null_counts > 0]}"
        )

    # --------------------------------------------------------------
    # Positive counts
    # --------------------------------------------------------------

    if (mart["order_count"] <= 0).any():
        raise AssertionError(
            "order_count contains zero or negative values."
        )

    if (mart["item_line_count"] <= 0).any():
        raise AssertionError(
            "item_line_count contains zero or negative values."
        )

    if (mart["customer_count"] <= 0).any():
        raise AssertionError(
            "customer_count contains zero or negative values."
        )

    # --------------------------------------------------------------
    # Non-negative financial metrics
    # --------------------------------------------------------------

    if (mart["revenue"] < 0).any():
        raise AssertionError(
            "revenue contains negative values."
        )

    if (mart["freight_value"] < 0).any():
        raise AssertionError(
            "freight_value contains negative values."
        )

    # --------------------------------------------------------------
    # Derived metric validation
    # --------------------------------------------------------------

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
        raise AssertionError(
            "avg_order_value does not equal "
            "revenue / order_count."
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
        raise AssertionError(
            "revenue_per_customer does not equal "
            "revenue / customer_count."
        )

    # --------------------------------------------------------------
    # Product-month contribution validation
    # --------------------------------------------------------------

    contribution_sums = (
        mart.groupby(
            [
                "month",
                "product_id",
            ]
        )[
            "market_contribution_to_product_revenue_pct"
        ].sum()
    )

    if not contribution_sums.between(
        99.999999,
        100.000001,
    ).all():
        invalid = contribution_sums[
            ~contribution_sums.between(
                99.999999,
                100.000001,
            )
        ]

        raise AssertionError(
            "Market contribution does not sum to "
            "approximately 100% for some product-months.\n"
            f"{invalid.head(10)}"
        )

    # --------------------------------------------------------------
    # Revenue reconciliation
    # --------------------------------------------------------------

    warehouse_revenue = fact_order_item[
        "price"
    ].sum()

    mart_revenue = mart["revenue"].sum()

    revenue_difference = (
        mart_revenue - warehouse_revenue
    )

    if abs(revenue_difference) > 1e-6:
        raise AssertionError(
            "Revenue reconciliation failed: "
            f"difference={revenue_difference}"
        )

    # --------------------------------------------------------------
    # Freight reconciliation
    # --------------------------------------------------------------

    warehouse_freight = fact_order_item[
        "freight_value"
    ].sum()

    mart_freight = mart[
        "freight_value"
    ].sum()

    freight_difference = (
        mart_freight - warehouse_freight
    )

    if abs(freight_difference) > 1e-6:
        raise AssertionError(
            "Freight reconciliation failed: "
            f"difference={freight_difference}"
        )

    # --------------------------------------------------------------
    # Report
    # --------------------------------------------------------------

    print("PRODUCT-MARKET MART VALIDATION: PASS")
    print(f"Rows: {len(mart):,}")
    print(f"Columns: {len(mart.columns)}")
    print(
        f"Duplicate grain rows: "
        f"{duplicate_grain_rows:,}"
    )
    print(
        f"Revenue: "
        f"{mart_revenue:,.2f}"
    )
    print(
        f"Freight: "
        f"{mart_freight:,.2f}"
    )
    print(
        f"Product-month contribution groups: "
        f"{len(contribution_sums):,}"
    )


if __name__ == "__main__":
    main()