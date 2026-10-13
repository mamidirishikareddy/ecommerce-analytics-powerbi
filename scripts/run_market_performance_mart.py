from pathlib import Path

from src.marts.market_performance_monthly import (
    build_market_performance_monthly,
)
from src.marts.persistence import save_mart


PROJECT_ROOT = Path(__file__).resolve().parents[1]

WAREHOUSE_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "warehouse"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "marts"
)

MART_NAME = "mart_market_performance_monthly"


def main() -> None:
    if not WAREHOUSE_DIR.exists():
        raise FileNotFoundError(
            f"Warehouse directory not found: {WAREHOUSE_DIR}"
        )

    mart = build_market_performance_monthly(
        WAREHOUSE_DIR
    )

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
            "Market-month grain validation failed: "
            f"{duplicate_count} duplicate rows."
        )

    print(
        f"MART ROWS: {len(mart):,}"
    )

    print(
        f"MART COLUMNS: {len(mart.columns)}"
    )

    print(
        "DATE RANGE: "
        f"{mart['month'].min()} "
        f"to "
        f"{mart['month'].max()}"
    )

    print(
        f"UNIQUE GRAIN: "
        f"{mart[['month', 'customer_zip_code_prefix']].drop_duplicates().shape[0]:,}"
    )

    print(
        f"DUPLICATE GRAIN ROWS: "
        f"{duplicate_count:,}"
    )

    print(
        f"TOTAL REVENUE: "
        f"{mart['revenue'].sum():.2f}"
    )

    print(
        f"TOTAL FREIGHT: "
        f"{mart['freight_value'].sum():.2f}"
    )

    print(
        f"TOTAL ORDERS: "
        f"{mart['order_count'].sum():,}"
    )

    print(
        f"TOTAL CUSTOMERS: "
        f"{mart['customer_count'].sum():,}"
    )

    output_path = save_mart(
        mart,
        MART_NAME,
        OUTPUT_DIR,
    )

    print(
        f"SAVED MART: {output_path}"
    )

    print("MARKET MART BUILD: SUCCESS")


if __name__ == "__main__":
    main()