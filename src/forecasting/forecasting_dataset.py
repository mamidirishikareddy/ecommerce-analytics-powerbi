"""
Forecasting dataset preparation.

This module converts the governed monthly market-performance mart
into a forecasting-ready marketplace revenue series.

The forecasting window is deliberately restricted to the consecutive,
commercially active period from January 2017 through August 2018.
Sparse startup observations and the partial terminal observation are
excluded from model training without modifying the underlying mart.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


class ForecastingDatasetError(ValueError):
    """Raised when the forecasting dataset cannot be constructed."""


INPUT_COLUMNS = [
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


OUTPUT_COLUMNS = [
    "month",
    "revenue",
    "order_count",
    "customer_count",
    "freight_value",
    "item_line_count",
    "avg_order_value",
    "revenue_per_customer",
    "avg_freight_per_order",
]


FORECAST_START = pd.Timestamp("2017-01-01")
FORECAST_END = pd.Timestamp("2018-08-01")


def _load_market_mart(
    mart_path: str | Path,
) -> pd.DataFrame:
    """
    Load the persisted monthly market-performance mart.
    """
    mart_path = Path(mart_path)

    if not mart_path.exists():
        raise FileNotFoundError(
            f"Market performance mart not found: {mart_path}"
        )

    if mart_path.suffix.lower() != ".parquet":
        raise ForecastingDatasetError(
            "Forecasting input must be a Parquet file."
        )

    df = pd.read_parquet(mart_path)

    missing_columns = sorted(
        set(INPUT_COLUMNS) - set(df.columns)
    )

    if missing_columns:
        raise ForecastingDatasetError(
            "Market performance mart is missing required columns: "
            f"{missing_columns}"
        )

    return df[INPUT_COLUMNS].copy()


def _validate_market_mart(
    df: pd.DataFrame,
) -> None:
    """
    Validate the source mart before aggregation.
    """
    if df.empty:
        raise ForecastingDatasetError(
            "Market performance mart is empty."
        )

    if df["month"].isna().any():
        raise ForecastingDatasetError(
            "month cannot contain null values."
        )

    if df[
        ["customer_zip_code_prefix", "order_count"]
    ].isna().any().any():
        raise ForecastingDatasetError(
            "Market grain or order_count contains null values."
        )

    if df["order_count"].le(0).any():
        raise ForecastingDatasetError(
            "order_count must be positive."
        )

    if df["item_line_count"].le(0).any():
        raise ForecastingDatasetError(
            "item_line_count must be positive."
        )

    if df["customer_count"].le(0).any():
        raise ForecastingDatasetError(
            "customer_count must be positive."
        )

    if df["revenue"].lt(0).any():
        raise ForecastingDatasetError(
            "revenue cannot be negative."
        )

    if df["freight_value"].lt(0).any():
        raise ForecastingDatasetError(
            "freight_value cannot be negative."
        )


def _validate_forecasting_window(
    df: pd.DataFrame,
) -> None:
    """
    Validate the resulting monthly forecasting window.
    """
    expected_months = pd.date_range(
        start=FORECAST_START,
        end=FORECAST_END,
        freq="MS",
    )

    actual_months = pd.DatetimeIndex(
        df["month"]
    )

    if len(df) != len(expected_months):
        raise ForecastingDatasetError(
            "Unexpected forecasting-window length: "
            f"expected {len(expected_months)}, "
            f"got {len(df)}."
        )

    if not actual_months.equals(expected_months):
        raise ForecastingDatasetError(
            "Forecasting window is not the expected consecutive "
            "monthly sequence from 2017-01 through 2018-08."
        )


def build_forecasting_dataset(
    mart_path: str | Path,
) -> pd.DataFrame:
    """
    Build the governed monthly marketplace forecasting dataset.

    Parameters
    ----------
    mart_path:
        Path to mart_market_performance_monthly.parquet.

    Returns
    -------
    pandas.DataFrame
        One row per month for the qualified forecasting window.
    """
    mart = _load_market_mart(mart_path)

    _validate_market_mart(mart)

    mart["month"] = pd.to_datetime(mart["month"])

    monthly = (
        mart.groupby("month", as_index=False)
        .agg(
            revenue=("revenue", "sum"),
            order_count=("order_count", "sum"),
            customer_count=("customer_count", "sum"),
            freight_value=("freight_value", "sum"),
            item_line_count=("item_line_count", "sum"),
        )
        .sort_values("month")
        .reset_index(drop=True)
    )

    monthly["avg_order_value"] = (
        monthly["revenue"]
        / monthly["order_count"]
    )

    monthly["revenue_per_customer"] = (
        monthly["revenue"]
        / monthly["customer_count"]
    )

    monthly["avg_freight_per_order"] = (
        monthly["freight_value"]
        / monthly["order_count"]
    )

    forecasting = monthly[
        (monthly["month"] >= FORECAST_START)
        & (monthly["month"] <= FORECAST_END)
    ].copy()

    forecasting = forecasting[
        OUTPUT_COLUMNS
    ].reset_index(drop=True)

    if forecasting["month"].duplicated().any():
        raise ForecastingDatasetError(
            "Forecasting dataset contains duplicate months."
        )

    _validate_forecasting_window(
        forecasting
    )

    if forecasting["revenue"].le(0).any():
        raise ForecastingDatasetError(
            "Forecasting months must have positive revenue."
        )

    if forecasting["order_count"].le(0).any():
        raise ForecastingDatasetError(
            "Forecasting months must have positive order_count."
        )

    if forecasting["customer_count"].le(0).any():
        raise ForecastingDatasetError(
            "Forecasting months must have positive customer_count."
        )

    if forecasting.isna().any().any():
        raise ForecastingDatasetError(
            "Forecasting dataset contains null values."
        )

    return forecasting