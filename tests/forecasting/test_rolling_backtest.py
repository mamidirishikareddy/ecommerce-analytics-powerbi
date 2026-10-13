from pathlib import Path

import pandas as pd
import pytest

from src.forecasting.rolling_backtest import (
    ROLLING_BACKTEST_COLUMNS,
    RollingBacktestError,
    build_rolling_backtest,
)


MARKET_MART = Path(
    "data/processed/marts/mart_market_performance_monthly.parquet"
)


def test_rolling_backtest_returns_expected_models():
    result = build_rolling_backtest(MARKET_MART)

    assert set(result["model"]) == {
        "NAIVE",
        "MA_3",
        "SES",
    }


def test_rolling_backtest_has_expected_columns():
    result = build_rolling_backtest(MARKET_MART)

    assert result.columns.tolist() == ROLLING_BACKTEST_COLUMNS


def test_rolling_backtest_uses_four_month_horizon():
    result = build_rolling_backtest(
        MARKET_MART,
        horizon=4,
    )

    assert result["evaluation_months"].eq(4).all()


def test_rolling_backtest_starts_after_minimum_training_window():
    result = build_rolling_backtest(
        MARKET_MART,
        minimum_training_months=12,
        horizon=4,
    )

    assert result["training_months"].min() == 12


def test_rolling_backtest_has_expected_number_of_origins():
    result = build_rolling_backtest(
        MARKET_MART,
        minimum_training_months=12,
        horizon=4,
    )

    expected_origins = 20 - 4 - 12 + 1

    assert result["origin"].nunique() == expected_origins


def test_each_origin_contains_all_models():
    result = build_rolling_backtest(MARKET_MART)

    models_per_origin = (
        result.groupby("origin")["model"]
        .nunique()
    )

    assert models_per_origin.eq(3).all()


def test_rolling_backtest_dates_are_chronological():
    result = build_rolling_backtest(MARKET_MART)

    assert (
        result["origin"]
        < result["evaluation_start"]
    ).all()

    assert (
        result["evaluation_start"]
        <= result["evaluation_end"]
    ).all()


def test_rolling_backtest_metrics_are_finite():
    result = build_rolling_backtest(MARKET_MART)

    metric_columns = [
        "mae",
        "rmse",
        "mape_pct",
        "wape_pct",
    ]

    assert result[metric_columns].notna().all().all()
    assert result[metric_columns].apply(
        lambda column: column.map(pd.notna).all()
    ).all()


def test_rolling_backtest_metrics_are_nonnegative():
    result = build_rolling_backtest(MARKET_MART)

    metric_columns = [
        "mae",
        "rmse",
        "mape_pct",
        "wape_pct",
    ]

    assert (result[metric_columns] >= 0).all().all()


def test_rolling_backtest_rejects_non_positive_training_window():
    with pytest.raises(RollingBacktestError):
        build_rolling_backtest(
            MARKET_MART,
            minimum_training_months=0,
        )


def test_rolling_backtest_rejects_non_positive_horizon():
    with pytest.raises(RollingBacktestError):
        build_rolling_backtest(
            MARKET_MART,
            horizon=0,
        )


def test_rolling_backtest_rejects_non_positive_moving_average_window():
    with pytest.raises(RollingBacktestError):
        build_rolling_backtest(
            MARKET_MART,
            moving_average_window=0,
        )


def test_rolling_backtest_rejects_window_larger_than_training():
    with pytest.raises(RollingBacktestError):
        build_rolling_backtest(
            MARKET_MART,
            minimum_training_months=2,
            moving_average_window=3,
        )


def test_rolling_backtest_rejects_insufficient_history():
    with pytest.raises(RollingBacktestError):
        build_rolling_backtest(
            MARKET_MART,
            minimum_training_months=18,
            horizon=4,
        )