import numpy as np
import pandas as pd
import pytest

from src.forecasting.baselines import (
    ForecastBaselineError,
    moving_average_forecast,
    naive_forecast,
)


def test_naive_forecast_uses_last_observation():
    train = pd.Series([100.0, 120.0, 140.0])

    result = naive_forecast(train, horizon=3)

    np.testing.assert_array_equal(
        result,
        np.array([140.0, 140.0, 140.0]),
    )


def test_naive_forecast_single_step():
    train = pd.Series([100.0, 120.0, 140.0])

    result = naive_forecast(train, horizon=1)

    np.testing.assert_array_equal(
        result,
        np.array([140.0]),
    )


def test_naive_forecast_rejects_empty_series():
    with pytest.raises(ForecastBaselineError):
        naive_forecast(pd.Series(dtype=float), horizon=1)


def test_naive_forecast_rejects_nulls():
    train = pd.Series([100.0, np.nan, 140.0])

    with pytest.raises(ForecastBaselineError):
        naive_forecast(train, horizon=1)


def test_naive_forecast_rejects_non_positive_horizon():
    train = pd.Series([100.0, 120.0, 140.0])

    with pytest.raises(ForecastBaselineError):
        naive_forecast(train, horizon=0)


def test_moving_average_forecast_uses_recent_window():
    train = pd.Series([100.0, 120.0, 140.0])

    result = moving_average_forecast(
        train,
        horizon=2,
        window=2,
    )

    np.testing.assert_array_equal(
        result,
        np.array([130.0, 130.0]),
    )


def test_moving_average_forecast_default_window():
    train = pd.Series([100.0, 120.0, 140.0])

    result = moving_average_forecast(
        train,
        horizon=2,
    )

    np.testing.assert_array_equal(
        result,
        np.array([120.0, 120.0]),
    )


def test_moving_average_forecast_window_equal_to_series_length():
    train = pd.Series([100.0, 120.0, 140.0])

    result = moving_average_forecast(
        train,
        horizon=1,
        window=3,
    )

    np.testing.assert_array_equal(
        result,
        np.array([120.0]),
    )


def test_moving_average_forecast_rejects_window_larger_than_series():
    train = pd.Series([100.0, 120.0])

    with pytest.raises(ForecastBaselineError):
        moving_average_forecast(
            train,
            horizon=1,
            window=3,
        )


def test_moving_average_forecast_rejects_non_positive_window():
    train = pd.Series([100.0, 120.0, 140.0])

    with pytest.raises(ForecastBaselineError):
        moving_average_forecast(
            train,
            horizon=1,
            window=0,
        )


def test_moving_average_forecast_rejects_non_positive_horizon():
    train = pd.Series([100.0, 120.0, 140.0])

    with pytest.raises(ForecastBaselineError):
        moving_average_forecast(
            train,
            horizon=0,
            window=2,
        )


def test_moving_average_forecast_rejects_nulls():
    train = pd.Series([100.0, np.nan, 140.0])

    with pytest.raises(ForecastBaselineError):
        moving_average_forecast(
            train,
            horizon=1,
            window=2,
        )