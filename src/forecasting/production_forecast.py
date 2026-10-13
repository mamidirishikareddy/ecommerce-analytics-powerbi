"""
Production forecasting using the governed champion model.

The production forecast uses the model selected by the model-selection
artifact and generates a forward forecast from the latest qualified
market observation.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.forecasting.baselines import moving_average_forecast
from src.forecasting.forecasting_dataset import build_forecasting_dataset


class ProductionForecastError(ValueError):
    """Raised when a production forecast cannot be generated."""


PRODUCTION_FORECAST_COLUMNS = [
    "forecast_month",
    "model",
    "forecast_revenue",
    "training_end_month",
    "training_months",
    "moving_average_window",
]


def build_production_forecast(
    market_mart_path: str | Path,
    model_selection_path: str | Path,
    horizon: int = 4,
    moving_average_window: int = 3,
) -> pd.DataFrame:
    """
    Generate a forward production forecast using the selected model.

    Parameters
    ----------
    market_mart_path:
        Path to the canonical market-performance monthly mart.

    model_selection_path:
        Path to the persisted model-selection artifact.

    horizon:
        Number of future months to forecast.

    moving_average_window:
        Number of recent observations used by the MA model.

    Returns
    -------
    pd.DataFrame
        Forward monthly revenue forecast with model provenance.
    """
    if horizon <= 0:
        raise ProductionForecastError(
            "horizon must be greater than zero."
        )

    if moving_average_window <= 0:
        raise ProductionForecastError(
            "moving_average_window must be greater than zero."
        )

    forecasting_data = build_forecasting_dataset(
        market_mart_path
    )

    model_selection = pd.read_parquet(
        model_selection_path
    )

    required_selection_columns = {
        "model",
        "selected",
    }

    missing_columns = (
        required_selection_columns
        - set(model_selection.columns)
    )

    if missing_columns:
        raise ProductionForecastError(
            "Model-selection artifact is missing required "
            f"columns: {sorted(missing_columns)}"
        )

    selected_models = model_selection.loc[
        model_selection["selected"],
        "model",
    ].tolist()

    if len(selected_models) != 1:
        raise ProductionForecastError(
            "Model-selection artifact must contain exactly "
            "one selected model."
        )

    selected_model = selected_models[0]

    if selected_model != "MA_3":
        raise ProductionForecastError(
            "Production forecast currently supports only the "
            "MA_3 champion model."
        )

    train = forecasting_data.copy()

    if moving_average_window > len(train):
        raise ProductionForecastError(
            "moving_average_window cannot exceed the number "
            "of qualified training observations."
        )

    forecast_values = moving_average_forecast(
        train["revenue"],
        horizon=horizon,
        window=moving_average_window,
    )

    training_end_month = pd.Timestamp(
        train["month"].iloc[-1]
    )

    forecast_months = pd.date_range(
        start=training_end_month + pd.offsets.MonthBegin(1),
        periods=horizon,
        freq="MS",
    )

    result = pd.DataFrame(
        {
            "forecast_month": forecast_months,
            "model": selected_model,
            "forecast_revenue": forecast_values,
            "training_end_month": training_end_month,
            "training_months": len(train),
            "moving_average_window": moving_average_window,
        }
    )

    if result["forecast_revenue"].isna().any():
        raise ProductionForecastError(
            "Production forecast contains null values."
        )

    if (result["forecast_revenue"] < 0).any():
        raise ProductionForecastError(
            "Production forecast contains negative revenue."
        )

    return result[PRODUCTION_FORECAST_COLUMNS]