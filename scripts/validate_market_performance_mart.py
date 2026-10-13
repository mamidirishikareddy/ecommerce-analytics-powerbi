from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

MART_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "marts"
    / "mart_market_performance_monthly.parquet"
)

EXPECTED_COLUMNS = [
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


def main() -> None:
    if not MART_PATH.exists():
        raise FileNotFoundError(
            f"Market mart not found: {MART_PATH}"
        )

    mart = pd.read_parquet(MART_PATH)

    # ---------------------------------------------------------------
    # Schema
    # ---------------------------------------------------------------

    if list(mart.columns) != EXPECTED_COLUMNS:
        raise ValueError(
            "Unexpected mart schema.\n"
            f"Expected: {EXPECTED_COLUMNS}\n"
            f"Actual: {list(mart.columns)}"
        )

    if mart.empty:
        raise ValueError(
            "Market performance mart is empty."
        )

    # ---------------------------------------------------------------
    # Grain
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
        raise ValueError(
            "Market-month grain violated: "
            f"{duplicate_count} duplicate rows."
        )

    # ---------------------------------------------------------------
    # Required fields
    # ---------------------------------------------------------------

    required_non_null = [
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

    null_counts = mart[required_non_null].isna().sum()

    if null_counts.any():
        raise ValueError(
            "Required columns contain null values:\n"
            f"{null_counts[null_counts > 0]}"
        )

    # ---------------------------------------------------------------
    # Metric validity
    # ---------------------------------------------------------------

    if mart["order_count"].le(0).any():
        raise ValueError(
            "order_count contains non-positive values."
        )

    if mart["item_line_count"].le(0).any():
        raise ValueError(
            "item_line_count contains non-positive values."
        )

    if mart["customer_count"].le(0).any():
        raise ValueError(
            "customer_count contains non-positive values."
        )

    if mart["revenue"].lt(0).any():
        raise ValueError(
            "revenue contains negative values."
        )

    if mart["freight_value"].lt(0).any():
        raise ValueError(
            "freight_value contains negative values."
        )

    # ---------------------------------------------------------------
    # Derived metric consistency
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

    if not mart["avg_order_value"].eq(
        expected_aov
    ).all():
        raise ValueError(
            "avg_order_value is inconsistent."
        )

    if not mart["revenue_per_customer"].eq(
        expected_revenue_per_customer
    ).all():
        raise ValueError(
            "revenue_per_customer is inconsistent."
        )

    if not mart["avg_freight_per_order"].eq(
        expected_avg_freight
    ).all():
        raise ValueError(
            "avg_freight_per_order is inconsistent."
        )

    # ---------------------------------------------------------------
    # Report
    # ---------------------------------------------------------------

    print("MARKET MART VALIDATION: PASS")
    print(f"Mart: {MART_PATH}")
    print(f"Rows: {len(mart):,}")
    print(f"Columns: {len(mart.columns)}")
    print(f"Duplicate grain rows: {duplicate_count:,}")
    print(f"Revenue: {mart['revenue'].sum():,.2f}")
    print(f"Freight: {mart['freight_value'].sum():,.2f}")
    print(f"Orders: {mart['order_count'].sum():,}")
    print(f"Customers: {mart['customer_count'].sum():,}")


if __name__ == "__main__":
    main()