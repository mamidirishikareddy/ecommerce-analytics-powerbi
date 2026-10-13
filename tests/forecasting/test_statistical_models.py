import numpy as np
import pandas as pd
import pytest

from src.forecasting.statistical_models import (
    StatisticalForecastError,
    damped_holt_forecast,
    holt_forecast,
    simple_exponential_smoothing_forecast,
)


def test_holt_forecast_returns_requested_horizon():
    train = pd.Series(
        [100.0, 120.0, 140.0, 160.0, 180.0]
    )

    result = holt_forecast(
        train,
        horizon=3,
    )

    assert len(result) == 3


def test_holt_forecast_returns_finite_values():
    train = pd.Series(
        [100.0, 120.0, 140.0, 160.0, 180.0]
    )

    result = holt_forecast(
        train,
        horizon=3,
    )

    assert np.isfinite(result).all()


def test_holt_forecast_continues_linear_trend():
    train = pd.Series(
        [100.0, 120.0, 140.0, 160.0, 180.0]
    )

    result = holt_forecast(
        train,
        horizon=2,
    )

    np.testing.assert_allclose(
        result,
        np.array([200.0, 220.0]),
        atol=1e-6,
    )


def test_holt_forecast_rejects_too_short_series():
    train = pd.Series(
        [100.0, 120.0]
    )

    with pytest.raises(StatisticalForecastError):
        holt_forecast(
            train,
            horizon=1,
        )


def test_holt_forecast_rejects_empty_series():
    train = pd.Series(
        dtype=float
    )

    with pytest.raises(StatisticalForecastError):
        holt_forecast(
            train,
            horizon=1,
        )


def test_holt_forecast_rejects_nulls():
    train = pd.Series(
        [100.0, np.nan, 140.0]
    )

    with pytest.raises(StatisticalForecastError):
        holt_forecast(
            train,
            horizon=1,
        )


def test_holt_forecast_rejects_non_finite_values():
    train = pd.Series(
        [100.0, 120.0, np.inf]
    )

    with pytest.raises(StatisticalForecastError):
        holt_forecast(
            train,
            horizon=1,
        )


def test_holt_forecast_rejects_non_positive_horizon():
    train = pd.Series(
        [100.0, 120.0, 140.0]
    )

    with pytest.raises(StatisticalForecastError):
        holt_forecast(
            train,
            horizon=0,
        )


def test_damped_holt_forecast_returns_requested_horizon():
    train = pd.Series(
        [100.0, 120.0, 140.0, 160.0, 180.0]
    )

    result = damped_holt_forecast(
        train,
        horizon=3,
    )

    assert len(result) == 3


def test_damped_holt_forecast_returns_finite_values():
    train = pd.Series(
        [100.0, 120.0, 140.0, 160.0, 180.0]
    )

    result = damped_holt_forecast(
        train,
        horizon=3,
    )

    assert np.isfinite(result).all()


def test_damped_holt_forecast_rejects_invalid_horizon():
    train = pd.Series(
        [100.0, 120.0, 140.0, 160.0, 180.0]
    )

    with pytest.raises(StatisticalForecastError):
        damped_holt_forecast(
            train,
            horizon=0,
        )


def test_damped_holt_forecast_rejects_nulls():
    train = pd.Series(
        [100.0, np.nan, 140.0, 160.0, 180.0]
    )

    with pytest.raises(StatisticalForecastError):
        damped_holt_forecast(
            train,
            horizon=2,
        )


def test_ses_forecast_returns_requested_horizon():
    train = pd.Series(
        [100.0, 120.0, 140.0, 160.0, 180.0]
    )

    result = simple_exponential_smoothing_forecast(
        train,
        horizon=3,
    )

    assert len(result) == 3


def test_ses_forecast_returns_finite_values():
    train = pd.Series(
        [100.0, 120.0, 140.0, 160.0, 180.0]
    )

    result = simple_exponential_smoothing_forecast(
        train,
        horizon=3,
    )

    assert np.isfinite(result).all()


def test_ses_forecast_tracks_current_level():
    train = pd.Series(
        [100.0, 120.0, 140.0, 160.0, 180.0]
    )

    result = simple_exponential_smoothing_forecast(
        train,
        horizon=3,
    )

    np.testing.assert_allclose(
        result,
        np.array([180.0, 180.0, 180.0]),
        atol=1e-4,
    )


def test_ses_forecast_rejects_too_short_series():
    train = pd.Series(
        [100.0, 120.0]
    )

    with pytest.raises(StatisticalForecastError):
        simple_exponential_smoothing_forecast(
            train,
            horizon=1,
        )


def test_ses_forecast_rejects_nulls():
    train = pd.Series(
        [100.0, np.nan, 140.0]
    )

    with pytest.raises(StatisticalForecastError):
        simple_exponential_smoothing_forecast(
            train,
            horizon=1,
        )


def test_ses_forecast_rejects_non_finite_values():
    train = pd.Series(
        [100.0, 120.0, np.inf]
    )

    with pytest.raises(StatisticalForecastError):
        simple_exponential_smoothing_forecast(
            train,
            horizon=1,
        )


def test_ses_forecast_rejects_non_positive_horizon():
    train = pd.Series(
        [100.0, 120.0, 140.0]
    )

    with pytest.raises(StatisticalForecastError):
        simple_exponential_smoothing_forecast(
            train,
            horizon=0,
        )