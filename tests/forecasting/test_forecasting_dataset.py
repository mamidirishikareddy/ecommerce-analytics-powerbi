"""
Tests for the governed forecasting dataset.
"""

from pathlib import Path

import pandas as pd
import pytest

from src.forecasting.forecasting_dataset import (
    FORECAST_END,
    FORECAST_START,
    OUTPUT_COLUMNS,
    ForecastingDatasetError,
    build_forecasting_dataset,
)


MART_PATH = Path(
    "data/processed/marts/"
    "mart_market_performance_monthly.parquet"
)


def test_forecasting_dataset_has_expected_shape():
    df = build_forecasting_dataset(MART_PATH)

    assert df.shape == (20, 9)


def test_forecasting_dataset_has_expected_columns():
    df = build_forecasting_dataset(MART_PATH)

    assert df.columns.tolist() == OUTPUT_COLUMNS


def test_forecasting_window_is_correct():
    df = build_forecasting_dataset(MART_PATH)

    assert df["month"].min() == FORECAST_START
    assert df["month"].max() == FORECAST_END


def test_forecasting_months_are_consecutive():
    df = build_forecasting_dataset(MART_PATH)

    expected = pd.date_range(
        start=FORECAST_START,
        end=FORECAST_END,
        freq="MS",
    )

    assert pd.DatetimeIndex(
        df["month"]
    ).equals(expected)


def test_forecasting_months_are_unique():
    df = build_forecasting_dataset(MART_PATH)

    assert df["month"].is_unique


def test_forecasting_dataset_has_no_nulls():
    df = build_forecasting_dataset(MART_PATH)

    assert not df.isna().any().any()


def test_forecasting_metrics_are_positive():
    df = build_forecasting_dataset(MART_PATH)

    assert (df["revenue"] > 0).all()
    assert (df["order_count"] > 0).all()
    assert (df["customer_count"] > 0).all()


def test_forecasting_dataset_excludes_sparse_boundary_periods():
    df = build_forecasting_dataset(MART_PATH)

    assert pd.Timestamp("2016-09-01") not in set(df["month"])
    assert pd.Timestamp("2016-12-01") not in set(df["month"])
    assert pd.Timestamp("2018-09-01") not in set(df["month"])


def test_forecasting_dataset_is_monthly_aggregate():
    df = build_forecasting_dataset(MART_PATH)

    source = pd.read_parquet(MART_PATH)

    expected_revenue = (
        source.loc[
            source["month"].between(
                FORECAST_START,
                FORECAST_END,
            ),
            "revenue",
        ]
        .sum()
    )

    assert df["revenue"].sum() == pytest.approx(
        expected_revenue
    )


def test_missing_mart_raises():
    with pytest.raises(FileNotFoundError):
        build_forecasting_dataset(
            "data/processed/marts/"
            "does_not_exist.parquet"
        )


def test_non_parquet_input_raises(tmp_path):
    path = tmp_path / "market.csv"
    path.write_text("test")

    with pytest.raises(ForecastingDatasetError):
        build_forecasting_dataset(path)