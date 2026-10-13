from pathlib import Path

import pandas as pd


MART_PATH = Path(
    "data/processed/marts/mart_product_performance_daily.parquet"
)

EXPECTED_COLUMNS = [
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

NUMERIC_COLUMNS = [
    "item_line_count",
    "order_count",
    "customer_count",
    "revenue",
    "freight_value",
    "gross_merchandise_value",
    "avg_item_price",
]


def validate_mart(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(
            f"Mart file does not exist: {path}"
        )

    mart = pd.read_parquet(path)

    errors = []

    # 1. Schema
    if list(mart.columns) != EXPECTED_COLUMNS:
        errors.append(
            "Column schema does not match expected mart contract."
        )

    # 2. Row count
    if len(mart) == 0:
        errors.append("Mart contains zero rows.")

    # 3. Grain
    duplicate_count = int(
        mart.duplicated(
            ["date", "product_id"]
        ).sum()
    )

    if duplicate_count != 0:
        errors.append(
            f"Duplicate date/product grain rows: {duplicate_count}"
        )

    # 4. Required grain fields
    if mart["date"].isna().any():
        errors.append("Null values found in date.")

    if mart["product_id"].isna().any():
        errors.append("Null values found in product_id.")

    # 5. Numeric columns must not be null
    for column in NUMERIC_COLUMNS:
        null_count = int(mart[column].isna().sum())

        if null_count != 0:
            errors.append(
                f"Numeric column '{column}' contains "
                f"{null_count} null values."
            )

    # 6. Non-negative monetary/count measures
    for column in [
        "item_line_count",
        "order_count",
        "customer_count",
        "revenue",
        "freight_value",
        "gross_merchandise_value",
    ]:
        negative_count = int(
            (mart[column] < 0).sum()
        )

        if negative_count != 0:
            errors.append(
                f"Column '{column}' contains "
                f"{negative_count} negative values."
            )

    # 7. Revenue/G​​MV consistency
    gmv_difference = (
        mart["revenue"] - mart["gross_merchandise_value"]
    ).abs()

    if not (gmv_difference <= 1e-9).all():
        errors.append(
            "gross_merchandise_value does not equal revenue."
        )

    # 8. Category null behavior is allowed.
    # Category attributes are descriptive source/reference fields.
    # They must not be artificially imputed.

    if errors:
        print("=" * 70)
        print("MART VALIDATION: FAILED")
        print("=" * 70)

        for error in errors:
            print(f"- {error}")

        raise ValueError(
            f"Mart validation failed with {len(errors)} error(s)."
        )

    print("=" * 70)
    print("MART VALIDATION: PASS")
    print("=" * 70)
    print(f"Mart: {path}")
    print(f"Rows: {len(mart):,}")
    print(f"Columns: {len(mart.columns)}")
    print(f"Duplicate grain rows: {duplicate_count}")
    print(
        f"Revenue: {mart['revenue'].sum():,.2f}"
    )
    print(
        f"Freight: {mart['freight_value'].sum():,.2f}"
    )
    print(
        f"Category name nulls: "
        f"{mart['product_category_name'].isna().sum():,}"
    )
    print(
        f"English category nulls: "
        f"{mart['product_category_name_english'].isna().sum():,}"
    )


if __name__ == "__main__":
    validate_mart(MART_PATH)