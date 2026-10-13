from pathlib import Path

from src.analytics.revenue_decomposition import (
    build_revenue_decomposition,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MARTS_DIR = PROJECT_ROOT / "data" / "processed" / "marts"
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed" / "analytics"


def main() -> None:
    if not MARTS_DIR.exists():
        raise FileNotFoundError(
            f"Required marts directory does not exist: {MARTS_DIR}"
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    result = build_revenue_decomposition(MARTS_DIR)

    duplicate_months = result["month"].duplicated().sum()

    if duplicate_months != 0:
        raise ValueError(
            f"Revenue decomposition contains {duplicate_months} "
            "duplicate month rows."
        )

    output_path = (
        OUTPUT_DIR / "revenue_decomposition_monthly.parquet"
    )

    result.to_parquet(
        output_path,
        index=False,
        engine="pyarrow",
    )

    print(f"ROWS: {len(result):,}")
    print(f"COLUMNS: {len(result.columns):,}")
    print(
        f"DATE RANGE: {result['month'].min()} "
        f"to {result['month'].max()}"
    )
    print(f"TOTAL REVENUE: {result['revenue'].sum():,.2f}")
    print(f"TOTAL FREIGHT: {result['freight_value'].sum():,.2f}")
    print(
        "MAX BRIDGE DIFFERENCE: "
        f"{(result['revenue_change'] - result['order_volume_effect'] - result['avg_order_value_effect']).abs().max():.10f}"
    )
    print(f"SAVED ANALYTICS: {output_path}")
    print("REVENUE DECOMPOSITION: SUCCESS")


if __name__ == "__main__":
    main()