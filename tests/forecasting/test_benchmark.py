from pathlib import Path

import pandas as pd
import pytest

from src.forecasting.benchmark import (
    BENCHMARK_COLUMNS,
    ForecastBenchmarkError,
    build_forecast_benchmark,
)


MARKET_MART = Path(
    "data/processed/marts/mart_market_performance_monthly.parquet"
)


def test_benchmark_returns_expected_models():
    result = build_forecast_benchmark(MARKET_MART)

    assert result["model"].tolist() == ["NAIVE", "MA_3"]


def test_benchmark_has_expected_columns():
    result = build_forecast_benchmark(MARKET_MART)

    assert result.columns.tolist() == BENCHMARK_COLUMNS


def test_benchmark_uses_expected_evaluation_window():
    result = build_forecast_benchmark(MARKET_MART)

    assert result["evaluation_start"].eq(
        pd.Timestamp("2018-05-01")
    ).all()

    assert result["evaluation_end"].eq(
        pd.Timestamp("2018-08-01")
    ).all()


def test_benchmark_uses_expected_training_and_evaluation_lengths():
    result = build_forecast_benchmark(MARKET_MART)

    assert result["training_months"].eq(16).all()
    assert result["evaluation_months"].eq(4).all()


def test_moving_average_beats_naive_on_wape():
    result = build_forecast_benchmark(MARKET_MART)

    naive_wape = result.loc[
        result["model"] == "NAIVE",
        "wape_pct",
    ].iloc[0]

    ma3_wape = result.loc[
        result["model"] == "MA_3",
        "wape_pct",
    ].iloc[0]

    assert ma3_wape < naive_wape


def test_moving_average_beats_naive_on_mae():
    result = build_forecast_benchmark(MARKET_MART)

    naive_mae = result.loc[
        result["model"] == "NAIVE",
        "mae",
    ].iloc[0]

    ma3_mae = result.loc[
        result["model"] == "MA_3",
        "mae",
    ].iloc[0]

    assert ma3_mae < naive_mae


def test_moving_average_beats_naive_on_rmse():
    result = build_forecast_benchmark(MARKET_MART)

    naive_rmse = result.loc[
        result["model"] == "NAIVE",
        "rmse",
    ].iloc[0]

    ma3_rmse = result.loc[
        result["model"] == "MA_3",
        "rmse",
    ].iloc[0]

    assert ma3_rmse < naive_rmse


def test_moving_average_beats_naive_on_mape():
    result = build_forecast_benchmark(MARKET_MART)

    naive_mape = result.loc[
        result["model"] == "NAIVE",
        "mape_pct",
    ].iloc[0]

    ma3_mape = result.loc[
        result["model"] == "MA_3",
        "mape_pct",
    ].iloc[0]

    assert ma3_mape < naive_mape


def test_benchmark_rejects_non_positive_training_months():
    with pytest.raises(ForecastBenchmarkError):
        build_forecast_benchmark(
            MARKET_MART,
            training_months=0,
        )


def test_benchmark_rejects_training_window_without_evaluation():
    with pytest.raises(ForecastBenchmarkError):
        build_forecast_benchmark(
            MARKET_MART,
            training_months=20,
        )


def test_benchmark_rejects_moving_average_window_larger_than_training():
    with pytest.raises(ForecastBenchmarkError):
        build_forecast_benchmark(
            MARKET_MART,
            training_months=2,
            moving_average_window=3,
        )