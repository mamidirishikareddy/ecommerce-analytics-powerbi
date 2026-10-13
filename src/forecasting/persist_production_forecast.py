"""
Persistence for the production forecast.

Builds the forward forecast using the governed champion model
and persists the resulting forecast artifact.
"""

from __future__ import annotations

from pathlib import Path

from src.forecasting.production_forecast import (
    build_production_forecast,
)
from src.marts.persistence import save_mart


def persist_production_forecast(
    market_mart_path: str | Path,
    model_selection_path: str | Path,
    output_dir: str | Path,
    horizon: int = 4,
    moving_average_window: int = 3,
) -> Path:
    """
    Build and persist the production forecast.

    Parameters
    ----------
    market_mart_path:
        Path to the canonical market-performance monthly mart.

    model_selection_path:
        Path to the persisted model-selection artifact.

    output_dir:
        Directory where the production forecast is persisted.

    horizon:
        Number of future months to forecast.

    moving_average_window:
        Number of recent observations used by MA_3.

    Returns
    -------
    Path
        Path to the persisted production forecast artifact.
    """
    forecast = build_production_forecast(
        market_mart_path=market_mart_path,
        model_selection_path=model_selection_path,
        horizon=horizon,
        moving_average_window=moving_average_window,
    )

    return save_mart(
        forecast,
        mart_name="forecast_production",
        output_dir=output_dir,
    )