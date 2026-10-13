"""
Tests for forecast evaluation utilities.
"""

import numpy as np
import pandas as pd
import pytest

from src.forecasting.evaluation import (
    ForecastEvaluationError,
    build_evaluation_frame,
    mean_absolute_error,
    mean_absolute_percentage_error,
    root_mean_squared_error,
    summarize_forecast_accuracy,
    validate_forecast_frame,
    weighted_absolute_percentage_error,
)


def test_validate_forecast_frame_accepts_valid_series():
    actual = pd.Series([100.0, 200.0, 300.0])
    forecast = pd.Series([110.0, 190.0, 310.0])

    validate_forecast_frame(actual, forecast)


def test_validate_rejects_empty_series():
    with pytest.raises(ForecastEvaluationError):
        validate_forecast_frame(
            pd.Series(dtype=float),
            pd.Series(dtype=float),
        )


def test_validate_rejects_length_mismatch():
    with pytest.raises(ForecastEvaluationError):
        validate_forecast_frame(
            pd.Series([100.0, 200.0]),
            pd.Series([100.0]),
        )


def test_validate_rejects_null_actual():
    with pytest.raises(ForecastEvaluationError):
        validate_forecast_frame(
            pd.Series([100.0, np.nan]),
            pd.Series([100.0, 200.0]),
        )


def test_validate_rejects_null_forecast():
    with pytest.raises(ForecastEvaluationError):
        validate_forecast_frame(
            pd.Series([100.0, 200.0]),
            pd.Series([100.0, np.nan]),
        )


def test_validate_rejects_non_finite_values():
    with pytest.raises(ForecastEvaluationError):
        validate_forecast_frame(
            pd.Series([100.0, np.inf]),
            pd.Series([100.0, 200.0]),
        )


def test_mae():
    actual = pd.Series([100.0, 200.0, 300.0])
    forecast = pd.Series([110.0, 190.0, 310.0])

    assert mean_absolute_error(
        actual,
        forecast,
    ) == pytest.approx(10.0)


def test_rmse():
    actual = pd.Series([100.0, 200.0, 300.0])
    forecast = pd.Series([110.0, 190.0, 310.0])

    assert root_mean_squared_error(
        actual,
        forecast,
    ) == pytest.approx(10.0)


def test_mape():
    actual = pd.Series([100.0, 200.0])
    forecast = pd.Series([110.0, 180.0])

    assert mean_absolute_percentage_error(
        actual,
        forecast,
    ) == pytest.approx(10.0)


def test_mape_rejects_zero_actual():
    with pytest.raises(ForecastEvaluationError):
        mean_absolute_percentage_error(
            pd.Series([100.0, 0.0]),
            pd.Series([100.0, 10.0]),
        )


def test_wape():
    actual = pd.Series([100.0, 200.0])
    forecast = pd.Series([110.0, 180.0])

    assert weighted_absolute_percentage_error(
    actual,
    forecast,
) == pytest.approx(
    30.0 / 300.0 * 100
)


def test_wape_rejects_zero_total_actual():
    with pytest.raises(ForecastEvaluationError):
        weighted_absolute_percentage_error(
            pd.Series([0.0, 0.0]),
            pd.Series([0.0, 0.0]),
        )


def test_build_evaluation_frame():
    months = pd.Series(
        pd.date_range(
            "2018-01-01",
            periods=2,
            freq="MS",
        )
    )
    actual = pd.Series([100.0, 200.0])
    forecast = pd.Series([110.0, 190.0])

    result = build_evaluation_frame(
        months,
        actual,
        forecast,
    )

    assert result.columns.tolist() == [
        "month",
        "actual",
        "forecast",
    ]

    assert len(result) == 2


def test_build_evaluation_frame_rejects_duplicate_months():
    months = pd.Series(
        [
            pd.Timestamp("2018-01-01"),
            pd.Timestamp("2018-01-01"),
        ]
    )

    with pytest.raises(ForecastEvaluationError):
        build_evaluation_frame(
            months,
            pd.Series([100.0, 200.0]),
            pd.Series([110.0, 190.0]),
        )


def test_summary_contains_all_metrics():
    actual = pd.Series([100.0, 200.0])
    forecast = pd.Series([110.0, 180.0])

    result = summarize_forecast_accuracy(
        actual,
        forecast,
    )

    assert set(result) == {
        "mae",
        "rmse",
        "mape_pct",
        "wape_pct",
    }