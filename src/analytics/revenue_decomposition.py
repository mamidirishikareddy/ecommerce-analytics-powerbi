
from __future__ import annotations

from pathlib import Path

import pandas as pd


class RevenueDecompositionError(ValueError):
    """Raised when governed revenue decomposition cannot be built."""


REQUIRED_MARTS = {
    "product": "mart_product_performance_daily.parquet",
    "customer": "mart_customer_performance_monthly.parquet",
    "market": "mart_market_performance_monthly.parquet",
    "product_market": "mart_product_market_performance_monthly.parquet",
}


# Explicitly governed partial periods.
# Keep these periods in the output, but identify them for reporting.
KNOWN_PARTIAL_MONTHS = {
    pd.Timestamp("2018-09-01"),
}


OUTPUT_COLUMNS = [
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


def _load_mart(path: Path, required_columns: list[str]) -> pd.DataFrame:
    if not path.exists():
        raise RevenueDecompositionError(
            f"Required mart does not exist: {path}"
        )

    df = pd.read_parquet(path)

    missing = sorted(set(required_columns) - set(df.columns))
    if missing:
        raise RevenueDecompositionError(
            f"Missing columns in {path.name}: {missing}"
        )

    return df


def _validate_monthly_series(df: pd.DataFrame) -> None:
    if df.empty:
        raise RevenueDecompositionError(
            "Revenue decomposition input is empty."
        )

    if df["month"].duplicated().any():
        raise RevenueDecompositionError(
            "Revenue decomposition contains duplicate month rows."
        )

    if df["month"].isna().any():
        raise RevenueDecompositionError(
            "Revenue decomposition contains null month values."
        )

    if (df["revenue"] < 0).any():
        raise RevenueDecompositionError(
            "Revenue contains negative values."
        )

    if (df["freight_value"] < 0).any():
        raise RevenueDecompositionError(
            "Freight contains negative values."
        )

    if (df["order_count"] <= 0).any():
        raise RevenueDecompositionError(
            "Monthly order_count must be positive."
        )

    if (df["customer_count"] <= 0).any():
        raise RevenueDecompositionError(
            "Monthly customer_count must be positive."
        )


def build_revenue_decomposition(
    marts_dir: str | Path,
) -> pd.DataFrame:
    """
    Build the governed monthly revenue decomposition.

    Grain:
        One row per observed calendar month.

    Revenue and freight are sourced from the product-performance mart.
    Order and customer counts are sourced from their respective
    analytical grains.

    Revenue decomposition uses a two-factor bridge:

        Revenue = Orders × AOV

    For month t versus the previous observed month:

        Order volume effect =
            (Orders_t - Orders_t-1) × AOV_t-1

        AOV effect =
            Orders_t × (AOV_t - AOV_t-1)

    Known partial periods remain in the output and are identified
    through is_partial_month. Comparisons use the previous observed
    month, which may not be a contiguous calendar month.
    """
    marts_dir = Path(marts_dir)

    product = _load_mart(
        marts_dir / REQUIRED_MARTS["product"],
        [
            "date",
            "revenue",
            "freight_value",
        ],
    )

    customer = _load_mart(
        marts_dir / REQUIRED_MARTS["customer"],
        [
            "month",
            "customer_unique_id",
        ],
    )

    market = _load_mart(
        marts_dir / REQUIRED_MARTS["market"],
        [
            "month",
            "order_count",
        ],
    )

    # Validate the product-market mart as a governed diagnostic source.
    # It is not used to calculate overall monthly orders or customers.
    product_market = _load_mart(
        marts_dir / REQUIRED_MARTS["product_market"],
        [
            "month",
            "revenue",
            "product_id",
            "customer_zip_code_prefix",
        ],
    )

    del product_market

    product["date"] = pd.to_datetime(
        product["date"],
        errors="raise",
    )

    customer["month"] = pd.to_datetime(
        customer["month"],
        errors="raise",
    )

    market["month"] = pd.to_datetime(
        market["month"],
        errors="raise",
    )

    # Product mart grain: date × product.
    product["month"] = (
        product["date"]
        .dt.to_period("M")
        .dt.to_timestamp()
    )

    monthly_product = (
        product.groupby("month", as_index=False)
        .agg(
            revenue=("revenue", "sum"),
            freight_value=("freight_value", "sum"),
        )
    )

    # Customer mart grain: month × customer_unique_id.
    monthly_customer = (
        customer.groupby("month", as_index=False)
        .agg(
            customer_count=("customer_unique_id", "nunique"),
        )
    )

    # Market mart grain: month × ZIP-prefix market.
    # Orders can be summed across markets under the governed
    # one-market-per-order mapping.
    monthly_market = (
        market.groupby("month", as_index=False)
        .agg(
            order_count=("order_count", "sum"),
        )
    )

    # Preserve the existing inner-join contract: only months represented
    # in all three calculation sources are included.
    result = monthly_product.merge(
        monthly_customer,
        on="month",
        how="inner",
        validate="one_to_one",
    )

    result = result.merge(
        monthly_market,
        on="month",
        how="inner",
        validate="one_to_one",
    )

    if result.empty:
        raise RevenueDecompositionError(
            "No overlapping months exist across required marts."
        )

    result = (
        result.sort_values("month")
        .reset_index(drop=True)
    )

    # Keep partial months in the result and explicitly identify them.
    result["is_partial_month"] = (
        result["month"]
        .isin(KNOWN_PARTIAL_MONTHS)
        .astype(bool)
    )

    result["avg_order_value"] = (
        result["revenue"] / result["order_count"]
    )

    result["revenue_per_customer"] = (
        result["revenue"] / result["customer_count"]
    )

    result["revenue_change"] = result["revenue"].diff()

    result["order_count_change"] = result["order_count"].diff()

    result["avg_order_value_change"] = (
        result["avg_order_value"].diff()
    )

    previous_orders = result["order_count"].shift(1)
    previous_aov = result["avg_order_value"].shift(1)

    result["order_volume_effect"] = (
        (result["order_count"] - previous_orders)
        * previous_aov
    )

    result["avg_order_value_effect"] = (
        result["order_count"]
        * (result["avg_order_value"] - previous_aov)
    )

    # The first observed month has no prior comparison.
    # Preserve the existing metric-registry convention.
    first_month = result.index[0]

    result.loc[
        first_month,
        [
            "revenue_change",
            "order_count_change",
            "avg_order_value_change",
            "order_volume_effect",
            "avg_order_value_effect",
        ],
    ] = 0.0

    _validate_monthly_series(result)

    # Revenue bridge reconciliation.
    bridge_difference = (
        result["revenue_change"]
        - (
            result["order_volume_effect"]
            + result["avg_order_value_effect"]
        )
    ).abs()

    if not (bridge_difference <= 1e-6).all():
        bad_rows = result.loc[
            bridge_difference > 1e-6,
            [
                "month",
                "revenue_change",
                "order_volume_effect",
                "avg_order_value_effect",
            ],
        ]

        raise RevenueDecompositionError(
            "Revenue decomposition does not reconcile. "
            f"Problem rows:\n{bad_rows.to_string(index=False)}"
        )

    result = result[OUTPUT_COLUMNS].copy()

    result["month"] = pd.to_datetime(
        result["month"],
        errors="raise",
    )

    result["is_partial_month"] = (
        result["is_partial_month"].astype(bool)
    )

    numeric_columns = [
        column
        for column in OUTPUT_COLUMNS
        if column not in {"month", "is_partial_month"}
    ]

    for column in numeric_columns:
        result[column] = pd.to_numeric(
            result[column],
            errors="raise",
        )

    return result
