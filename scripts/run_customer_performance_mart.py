from pathlib import Path

from src.marts.customer_performance_monthly import (
    build_customer_performance_monthly,
)
from src.marts.persistence import save_mart


MART_NAME = "mart_customer_performance_monthly"
WAREHOUSE_DIR = Path("data/processed/warehouse")
MART_OUTPUT_DIR = Path("data/processed/marts")


def main() -> None:
    print("=" * 70)
    print(f"BUILDING MART: {MART_NAME}")
    print("=" * 70)

    print(f"Warehouse directory: {WAREHOUSE_DIR}")
    print(f"Mart output directory: {MART_OUTPUT_DIR}")
    print()

    if not WAREHOUSE_DIR.exists():
        raise FileNotFoundError(
            f"Warehouse directory does not exist: {WAREHOUSE_DIR}"
        )

    mart = build_customer_performance_monthly(
        str(WAREHOUSE_DIR)
    )

    print(f"Mart rows: {len(mart):,}")
    print(f"Mart columns: {len(mart.columns)}")
    print(
        f"Date range: "
        f"{mart['month'].min()} to {mart['month'].max()}"
    )

    duplicate_rows = int(
        mart.duplicated(
            ["month", "customer_unique_id"]
        ).sum()
    )

    if duplicate_rows != 0:
        raise ValueError(
            f"Mart grain validation failed: "
            f"{duplicate_rows:,} duplicate rows found."
        )

    print(f"Duplicate grain rows: {duplicate_rows:,}")

    output_path = save_mart(
        df=mart,
        mart_name=MART_NAME,
        output_dir=MART_OUTPUT_DIR,
    )

    print()
    print(f"Saved mart: {output_path}")

    print()
    print("=" * 70)
    print("MART BUILD: SUCCESS")
    print("=" * 70)


if __name__ == "__main__":
    main()