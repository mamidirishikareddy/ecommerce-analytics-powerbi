from pathlib import Path

import pandas as pd
import pytest

from src.forecasting.production_forecast import (
    ProductionForecastError,
    build_production_forecast,
)


MARKET_MART = Path(
    "data/processed/marts/mart_market_performance_monthly.parquet"
)

MODEL_SELECTION = Path(
    "data/processed/forecasting/forecast_model_selection.parquet"
)


def test_production_forecast_has_expected_shape():
    result = build_production_forecast(
        market_mart_path=MARKET_MART,
        model_selection_path=MODEL_SELECTION,
        horizon=4,
    )

    assert result.shape == (4, 6)


def test_production_forecast_has_expected_columns():
    result = build_production_forecast(
        market_mart_path=MARKET_MART,
        model_selection_path=MODEL_SELECTION,
        horizon=4,
    )

    assert list(result.columns) == [
        "forecast_month",
        "model",
        "forecast_revenue",
        "training_end_month",
        "training_months",
        "moving_average_window",
    ]


def test_production_forecast_uses_selected_model():
    result = build_production_forecast(
        market_mart_path=MARKET_MART,
        model_selection_path=MODEL_SELECTION,
        horizon=4,
    )

    assert result["model"].nunique() == 1
    assert result["model"].iloc[0] == "MA_3"


def test_production_forecast_has_four_future_months():
    result = build_production_forecast(
        market_mart_path=MARKET_MART,
        model_selection_path=MODEL_SELECTION,
        horizon=4,
    )

    assert result["forecast_month"].nunique() == 4
    assert result["forecast_month"].is_monotonic_increasing


def test_production_forecast_starts_after_training_period():
    result = build_production_forecast(
        market_mart_path=MARKET_MART,
        model_selection_path=MODEL_SELECTION,
        horizon=4,
    )

    assert (
        result["forecast_month"].iloc[0]
        > result["training_end_month"].iloc[0]
    )


def test_production_forecast_has_no_nulls():
    result = build_production_forecast(
        market_mart_path=MARKET_MART,
        model_selection_path=MODEL_SELECTION,
        horizon=4,
    )

    assert result.isna().sum().sum() == 0


def test_production_forecast_is_nonnegative():
    result = build_production_forecast(
        market_mart_path=MARKET_MART,
        model_selection_path=MODEL_SELECTION,
        horizon=4,
    )

    assert (result["forecast_revenue"] >= 0).all()


def test_production_forecast_uses_three_month_window():
    result = build_production_forecast(
        market_mart_path=MARKET_MART,
        model_selection_path=MODEL_SELECTION,
        horizon=4,
    )

    assert result["moving_average_window"].unique().tolist() == [3]


def test_production_forecast_training_months_are_consistent():
    result = build_production_forecast(
        market_mart_path=MARKET_MART,
        model_selection_path=MODEL_SELECTION,
        horizon=4,
    )

    assert result["training_months"].unique().tolist() == [20]


def test_invalid_horizon_raises():
    with pytest.raises(ProductionForecastError):
        build_production_forecast(
            market_mart_path=MARKET_MART,
            model_selection_path=MODEL_SELECTION,
            horizon=0,
        )


def test_invalid_window_raises():
    with pytest.raises(ProductionForecastError):
        build_production_forecast(
            market_mart_path=MARKET_MART,
            model_selection_path=MODEL_SELECTION,
            horizon=4,
            moving_average_window=0,
        )