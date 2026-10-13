from pathlib import Path

from src.marts.persistence import save_mart
from src.marts.product_market_performance_monthly import (
    build_product_market_performance_monthly,
)


WAREHOUSE_DIR = Path("data/processed/warehouse")
OUTPUT_DIR = Path("data/processed/marts")
MART_NAME = "mart_product_market_performance_monthly"


def main() -> None:
    mart = build_product_market_performance_monthly(
        WAREHOUSE_DIR
    )

    grain_columns = [
        "month",
        "product_id",
        "customer_zip_code_prefix",
    ]

    print(f"MART ROWS: {len(mart):,}")
    print(f"MART COLUMNS: {len(mart.columns)}")
    print(
        f"DATE RANGE: "
        f"{mart['month'].min()} to {mart['month'].max()}"
    )
    print(
        f"UNIQUE GRAIN: "
        f"{mart[grain_columns].drop_duplicates().shape[0]:,}"
    )
    print(
        f"DUPLICATE GRAIN ROWS: "
        f"{mart.duplicated(grain_columns).sum():,}"
    )
    print(
        f"TOTAL REVENUE: "
        f"{mart['revenue'].sum():,.2f}"
    )
    print(
        f"TOTAL FREIGHT: "
        f"{mart['freight_value'].sum():,.2f}"
    )
    print(
        f"TOTAL ORDERS: "
        f"{mart['order_count'].sum():,}"
    )
    print(
        f"TOTAL CUSTOMERS: "
        f"{mart['customer_count'].sum():,}"
    )

    save_mart(
        mart,
        MART_NAME,
        OUTPUT_DIR,
    )

    print(
        f"SAVED MART: "
        f"{OUTPUT_DIR / (MART_NAME + '.parquet')}"
    )
    print("PRODUCT-MARKET MART BUILD: SUCCESS")


if __name__ == "__main__":
    main()