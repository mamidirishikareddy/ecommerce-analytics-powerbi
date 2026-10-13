"""
Forecast evaluation utilities.

Provides chronological train/evaluation splitting and business-oriented
forecast accuracy metrics for the Product-Pulse-AI forecasting layer.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


class ForecastEvaluationError(ValueError):
    """Raised when forecast evaluation cannot be performed."""


EVALUATION_COLUMNS = [
    "month",
    "actual",
    "forecast",
]


def validate_forecast_frame(
    actual: pd.Series,
    forecast: pd.Series,
) -> None:
    """Validate aligned actual and forecast series."""
    if len(actual) == 0:
        raise ForecastEvaluationError(
            "Actual and forecast series cannot be empty."
        )

    if len(actual) != len(forecast):
        raise ForecastEvaluationError(
            "Actual and forecast series must have equal length."
        )

    if actual.isna().any():
        raise ForecastEvaluationError(
            "Actual values cannot contain nulls."
        )

    if forecast.isna().any():
        raise ForecastEvaluationError(
            "Forecast values cannot contain nulls."
        )

    if not np.isfinite(actual.to_numpy()).all():
        raise ForecastEvaluationError(
            "Actual values must be finite."
        )

    if not np.isfinite(forecast.to_numpy()).all():
        raise ForecastEvaluationError(
            "Forecast values must be finite."
        )


def mean_absolute_error(
    actual: pd.Series,
    forecast: pd.Series,
) -> float:
    """Calculate mean absolute error."""
    validate_forecast_frame(actual, forecast)

    return float(
        np.mean(
            np.abs(
                actual.to_numpy()
                - forecast.to_numpy()
            )
        )
    )


def root_mean_squared_error(
    actual: pd.Series,
    forecast: pd.Series,
) -> float:
    """Calculate root mean squared error."""
    validate_forecast_frame(actual, forecast)

    return float(
        np.sqrt(
            np.mean(
                (
                    actual.to_numpy()
                    - forecast.to_numpy()
                )
                ** 2
            )
        )
    )


def mean_absolute_percentage_error(
    actual: pd.Series,
    forecast: pd.Series,
) -> float:
    """
    Calculate MAPE as a percentage.

    Zero actual values are rejected because MAPE is undefined for them.
    """
    validate_forecast_frame(actual, forecast)

    if (actual == 0).any():
        raise ForecastEvaluationError(
            "MAPE cannot be calculated when actual values contain zero."
        )

    return float(
        np.mean(
            np.abs(
                (
                    actual.to_numpy()
                    - forecast.to_numpy()
                )
                / actual.to_numpy()
            )
        )
        * 100
    )


def weighted_absolute_percentage_error(
    actual: pd.Series,
    forecast: pd.Series,
) -> float:
    """
    Calculate WAPE as a percentage.
    """
    validate_forecast_frame(actual, forecast)

    denominator = float(
        np.abs(actual.to_numpy()).sum()
    )

    if denominator == 0:
        raise ForecastEvaluationError(
            "WAPE cannot be calculated when total actual "
            "magnitude is zero."
        )

    return float(
        np.abs(
            actual.to_numpy()
            - forecast.to_numpy()
        ).sum()
        / denominator
        * 100
    )


def build_evaluation_frame(
    months: pd.Series,
    actual: pd.Series,
    forecast: pd.Series,
) -> pd.DataFrame:
    """
    Build an aligned forecast evaluation frame.
    """
    if len(months) != len(actual):
        raise ForecastEvaluationError(
            "months and actual must have equal length."
        )

    if len(actual) != len(forecast):
        raise ForecastEvaluationError(
            "actual and forecast must have equal length."
        )

    validate_forecast_frame(actual, forecast)

    result = pd.DataFrame(
        {
            "month": pd.to_datetime(months),
            "actual": actual.to_numpy(),
            "forecast": forecast.to_numpy(),
        }
    )

    if result["month"].duplicated().any():
        raise ForecastEvaluationError(
            "Evaluation months must be unique."
        )

    return result[EVALUATION_COLUMNS]


def summarize_forecast_accuracy(
    actual: pd.Series,
    forecast: pd.Series,
) -> dict[str, float]:
    """
    Calculate the complete business-oriented accuracy summary.
    """
    return {
        "mae": mean_absolute_error(
            actual,
            forecast,
        ),
        "rmse": root_mean_squared_error(
            actual,
            forecast,
        ),
        "mape_pct": mean_absolute_percentage_error(
            actual,
            forecast,
        ),
        "wape_pct": weighted_absolute_percentage_error(
            actual,
            forecast,
        ),
    }