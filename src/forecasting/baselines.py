"""
Baseline forecasting models.

Provides simple, transparent forecasting benchmarks that future
statistical or machine-learning models must outperform.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


class ForecastBaselineError(ValueError):
    """Raised when a baseline forecast cannot be generated."""


def _validate_series(series: pd.Series) -> None:
    if not isinstance(series, pd.Series):
        raise ForecastBaselineError("Input must be a pandas Series.")

    if len(series) == 0:
        raise ForecastBaselineError("Input series cannot be empty.")

    if series.isna().any():
        raise ForecastBaselineError("Input series cannot contain nulls.")

    if not np.isfinite(series.to_numpy()).all():
        raise ForecastBaselineError("Input series must contain only finite values.")


def naive_forecast(
    train: pd.Series,
    horizon: int,
) -> np.ndarray:
    """
    Forecast every future period using the last observed value.
    """
    _validate_series(train)

    if horizon <= 0:
        raise ForecastBaselineError("Horizon must be greater than zero.")

    last_value = float(train.iloc[-1])

    return np.repeat(last_value, horizon)


def moving_average_forecast(
    train: pd.Series,
    horizon: int,
    window: int = 3,
) -> np.ndarray:
    """
    Forecast future periods using the mean of the most recent observations.
    """
    _validate_series(train)

    if horizon <= 0:
        raise ForecastBaselineError("Horizon must be greater than zero.")

    if window <= 0:
        raise ForecastBaselineError("Window must be greater than zero.")

    if window > len(train):
        raise ForecastBaselineError(
            "Window cannot be larger than the training series."
        )

    recent_values = train.iloc[-window:]
    forecast_value = float(recent_values.mean())

    return np.repeat(forecast_value, horizon)