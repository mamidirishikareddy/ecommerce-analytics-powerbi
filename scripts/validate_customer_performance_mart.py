from pathlib import Path

import pandas as pd


MART_PATH = Path(
    "data/processed/marts/mart_customer_performance_monthly.parquet"
)

EXPECTED_COLUMNS = [
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

NUMERIC_COLUMNS = [
    "order_count",
    "item_line_count",
    "revenue",
    "freight_value",
    "avg_order_value",
]


def validate_mart(path: Path) -> None:

    if not path.exists():
        raise FileNotFoundError(
            f"Mart file does not exist: {path}"
        )

    mart = pd.read_parquet(path)

    errors = []

    # ---------------------------------------------------------------
    # Schema
    # ---------------------------------------------------------------

    if list(mart.columns) != EXPECTED_COLUMNS:
        errors.append(
            "Column schema does not match expected mart contract."
        )

    # ---------------------------------------------------------------
    # Row count
    # ---------------------------------------------------------------

    if len(mart) == 0:
        errors.append("Mart contains zero rows.")

    # ---------------------------------------------------------------
    # Grain
    # ---------------------------------------------------------------

    duplicate_count = int(
        mart.duplicated(
            ["month", "customer_unique_id"]
        ).sum()
    )

    if duplicate_count != 0:
        errors.append(
            f"Duplicate customer-month rows: {duplicate_count}"
        )

    # ---------------------------------------------------------------
    # Required grain fields
    # ---------------------------------------------------------------

    if mart["month"].isna().any():
        errors.append("Null values found in month.")

    if mart["customer_unique_id"].isna().any():
        errors.append(
            "Null values found in customer_unique_id."
        )

    # ---------------------------------------------------------------
    # Numeric fields
    # ---------------------------------------------------------------

    for column in NUMERIC_COLUMNS:

        null_count = int(
            mart[column].isna().sum()
        )

        if null_count != 0:
            errors.append(
                f"Numeric column '{column}' contains "
                f"{null_count} null values."
            )

    # ---------------------------------------------------------------
    # Non-negative metrics
    # ---------------------------------------------------------------

    for column in [
        "order_count",
        "item_line_count",
        "revenue",
        "freight_value",
        "avg_order_value",
    ]:

        negative_count = int(
            (mart[column] < 0).sum()
        )

        if negative_count != 0:
            errors.append(
                f"Column '{column}' contains "
                f"{negative_count} negative values."
            )

    # ---------------------------------------------------------------
    # Positive count metrics
    # ---------------------------------------------------------------

    if (mart["order_count"] <= 0).any():
        errors.append(
            "order_count contains non-positive values."
        )

    if (mart["item_line_count"] <= 0).any():
        errors.append(
            "item_line_count contains non-positive values."
        )

    # ---------------------------------------------------------------
    # AOV consistency
    # ---------------------------------------------------------------

    expected_aov = (
        mart["revenue"]
        / mart["order_count"]
    )

    aov_difference = (
        mart["avg_order_value"]
        - expected_aov
    ).abs()

    if not (aov_difference <= 1e-9).all():
        errors.append(
            "avg_order_value does not equal "
            "revenue / order_count."
        )

    # ---------------------------------------------------------------
    # First purchase month
    # ---------------------------------------------------------------

    if mart["first_purchase_month"].isna().any():
        errors.append(
            "Null first_purchase_month values found."
        )

    if (
        mart["first_purchase_month"]
        > mart["month"]
    ).any():
        errors.append(
            "first_purchase_month occurs after customer activity month."
        )

    # ---------------------------------------------------------------
    # Customer status
    # ---------------------------------------------------------------

    allowed_statuses = {
        "NEW",
        "REPEAT",
    }

    invalid_statuses = set(
        mart.loc[
            ~mart["customer_status"].isin(
                allowed_statuses
            ),
            "customer_status",
        ]
        .dropna()
        .unique()
    )

    if invalid_statuses:
        errors.append(
            "Invalid customer_status values: "
            f"{sorted(invalid_statuses)}"
        )

    # ---------------------------------------------------------------
    # Status consistency
    # ---------------------------------------------------------------

    expected_status = (
        mart["first_purchase_month"]
        == mart["month"]
    )

    actual_status = (
        mart["customer_status"] == "NEW"
    )

    status_mismatch = int(
        (expected_status != actual_status).sum()
    )

    if status_mismatch != 0:
        errors.append(
            f"Customer status mismatches: {status_mismatch}"
        )

    # ---------------------------------------------------------------
    # Report
    # ---------------------------------------------------------------

    print("=" * 70)

    if errors:
        print("CUSTOMER MART VALIDATION: FAILED")
        print("=" * 70)

        for error in errors:
            print(f"- {error}")

        raise ValueError(
            f"Customer mart validation failed "
            f"with {len(errors)} error(s)."
        )

    print("CUSTOMER MART VALIDATION: PASS")
    print("=" * 70)

    print(f"Mart: {path}")
    print(f"Rows: {len(mart):,}")
    print(f"Columns: {len(mart.columns)}")
    print(
        f"Duplicate grain rows: {duplicate_count}"
    )
    print(
        f"Revenue: {mart['revenue'].sum():,.2f}"
    )
    print(
        f"Freight: {mart['freight_value'].sum():,.2f}"
    )
    print(
        f"New customer-months: "
        f"{(mart['customer_status'] == 'NEW').sum():,}"
    )
    print(
        f"Repeat customer-months: "
        f"{(mart['customer_status'] == 'REPEAT').sum():,}"
    )


if __name__ == "__main__":
    validate_mart(MART_PATH)