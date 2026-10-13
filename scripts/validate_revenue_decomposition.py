from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "analytics"
    / "revenue_decomposition_monthly.parquet"
)


EXPECTED_COLUMNS = [
    "month",
    "is_partial_month",
    "revenue",
    "freight_value",
    "order_count",
    "customer_count",
    "avg_order_value",
    "revenue_per_customer",
    "revenue_change",
    "order_count_change",
    "avg_order_value_change",
    "order_volume_effect",
    "avg_order_value_effect",
]


def main() -> None:
    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"Revenue decomposition output does not exist: {INPUT_PATH}"
        )

    df = pd.read_parquet(INPUT_PATH)

    if list(df.columns) != EXPECTED_COLUMNS:
        raise ValueError(
            "Unexpected revenue decomposition columns.\n"
            f"Expected: {EXPECTED_COLUMNS}\n"
            f"Actual: {list(df.columns)}"
        )

    if df.empty:
        raise ValueError("Revenue decomposition output is empty.")

    if df["month"].isna().any():
        raise ValueError("Null month values detected.")

    if not pd.api.types.is_bool_dtype(df["is_partial_month"]):
        raise TypeError("is_partial_month must have boolean dtype.")

    if df["is_partial_month"].isna().any():
        raise ValueError("Null is_partial_month values detected.")

    if df["month"].duplicated().any():
        raise ValueError("Duplicate month rows detected.")

    if not df["month"].is_monotonic_increasing:
        raise ValueError("Months are not sorted in ascending order.")

    expected_partial_months = {pd.Timestamp("2018-09-01")}
    actual_partial_months = set(
        df.loc[df["is_partial_month"], "month"]
        .dt.to_period("M")
        .dt.to_timestamp()
    )

    if actual_partial_months != expected_partial_months:
        raise ValueError(
            "Unexpected partial-month flags.\n"
            f"Expected: {sorted(expected_partial_months)}\n"
            f"Actual: {sorted(actual_partial_months)}"
        )

    for column in [
        "revenue",
        "freight_value",
        "order_count",
        "customer_count",
        "avg_order_value",
        "revenue_per_customer",
        "revenue_change",
        "order_count_change",
        "avg_order_value_change",
        "order_volume_effect",
        "avg_order_value_effect",
    ]:
        if df[column].isna().any():
            raise ValueError(f"Null values detected in {column}.")

    if (df["revenue"] < 0).any():
        raise ValueError("Negative revenue detected.")

    if (df["freight_value"] < 0).any():
        raise ValueError("Negative freight detected.")

    if (df["order_count"] <= 0).any():
        raise ValueError("Non-positive order count detected.")

    if (df["customer_count"] <= 0).any():
        raise ValueError("Non-positive customer count detected.")

    aov_difference = (
        df["avg_order_value"]
        - df["revenue"] / df["order_count"]
    ).abs()

    if (aov_difference > 1e-9).any():
        raise ValueError(
            "Average order value does not reconcile with revenue/order_count."
        )

    rpc_difference = (
        df["revenue_per_customer"]
        - df["revenue"] / df["customer_count"]
    ).abs()

    if (rpc_difference > 1e-9).any():
        raise ValueError(
            "Revenue per customer does not reconcile with revenue/customer_count."
        )

    bridge_difference = (
        df["revenue_change"]
        - (
            df["order_volume_effect"]
            + df["avg_order_value_effect"]
        )
    ).abs()

    if (bridge_difference > 1e-6).any():
        raise ValueError(
            "Revenue decomposition bridge does not reconcile."
        )

    print("REVENUE DECOMPOSITION VALIDATION: PASS")
    print(f"Rows: {len(df):,}")
    print(f"Columns: {len(df.columns):,}")
    print(f"Duplicate month rows: {df['month'].duplicated().sum():,}")
    print(
        "Partial months: "
        f"{df.loc[df['is_partial_month'], 'month'].dt.strftime('%Y-%m').tolist()}"
    )
    print(f"Revenue: {df['revenue'].sum():,.2f}")
    print(f"Freight: {df['freight_value'].sum():,.2f}")
    print(
        "Max bridge difference: "
        f"{bridge_difference.max():.10f}"
    )
    print(f"File exists: {INPUT_PATH.exists()}")


if __name__ == "__main__":
    main()
