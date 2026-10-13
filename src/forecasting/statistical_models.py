"""
Statistical forecasting models.

Provides transparent time-series models suitable for the project's
short monthly revenue history.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from statsmodels.tsa.holtwinters import Holt, SimpleExpSmoothing


class StatisticalForecastError(ValueError):
    """Raised when a statistical forecast cannot be generated."""


def _validate_series(series: pd.Series) -> None:
    if not isinstance(series, pd.Series):
        raise StatisticalForecastError("Input must be a pandas Series.")

    if len(series) < 3:
        raise StatisticalForecastError(
            "At least 3 observations are required for forecasting."
        )

    if series.isna().any():
        raise StatisticalForecastError(
            "Input series cannot contain nulls."
        )

    if not np.isfinite(series.to_numpy()).all():
        raise StatisticalForecastError(
            "Input series must contain only finite values."
        )


def holt_forecast(
    train: pd.Series,
    horizon: int,
) -> np.ndarray:
    """
    Generate forecasts using Holt's linear trend method.

    Parameters
    ----------
    train:
        Chronologically ordered historical observations.

    horizon:
        Number of future periods to forecast.

    Returns
    -------
    np.ndarray
        Forecast values of length ``horizon``.
    """
    _validate_series(train)

    if horizon <= 0:
        raise StatisticalForecastError(
            "Horizon must be greater than zero."
        )

    model = Holt(
        train.astype(float),
        initialization_method="estimated",
    )

    fitted_model = model.fit(
        optimized=True,
    )

    forecast = fitted_model.forecast(horizon)

    result = np.asarray(forecast, dtype=float)

    if len(result) != horizon:
        raise StatisticalForecastError(
            "Statistical model returned an unexpected forecast length."
        )

    if not np.isfinite(result).all():
        raise StatisticalForecastError(
            "Statistical model produced non-finite forecasts."
        )

    return result


def damped_holt_forecast(
    train: pd.Series,
    horizon: int,
) -> np.ndarray:
    """
    Generate forecasts using Holt's damped linear trend method.

    The trend is allowed to continue, but its influence gradually
    decreases as the forecast horizon increases.
    """
    _validate_series(train)

    if horizon <= 0:
        raise StatisticalForecastError(
            "Horizon must be greater than zero."
        )

    model = Holt(
        train.astype(float),
        damped_trend=True,
        initialization_method="estimated",
    )

    fitted_model = model.fit(
        optimized=True,
    )

    forecast = fitted_model.forecast(horizon)

    result = np.asarray(forecast, dtype=float)

    if len(result) != horizon:
        raise StatisticalForecastError(
            "Statistical model returned an unexpected forecast length."
        )

    if not np.isfinite(result).all():
        raise StatisticalForecastError(
            "Statistical model produced non-finite forecasts."
        )

    return result


def simple_exponential_smoothing_forecast(
    train: pd.Series,
    horizon: int,
) -> np.ndarray:
    """
    Generate forecasts using simple exponential smoothing.

    SES models the evolving level of the series without explicitly
    extrapolating a trend.
    """
    _validate_series(train)

    if horizon <= 0:
        raise StatisticalForecastError(
            "Horizon must be greater than zero."
        )

    model = SimpleExpSmoothing(
        train.astype(float),
        initialization_method="estimated",
    )

    fitted_model = model.fit(
        optimized=True,
    )

    forecast = fitted_model.forecast(horizon)

    result = np.asarray(forecast, dtype=float)

    if len(result) != horizon:
        raise StatisticalForecastError(
            "Statistical model returned an unexpected forecast length."
        )

    if not np.isfinite(result).all():
        raise StatisticalForecastError(
            "Statistical model produced non-finite forecasts."
        )

    return result