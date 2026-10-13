"""
Persistence for forecast evidence.

Builds the governed forecast-evidence artifact from the production
forecast and model-selection artifacts.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.forecasting.forecast_evidence import (
    build_forecast_evidence,
)
from src.marts.persistence import save_mart


def persist_forecast_evidence(
    production_forecast_path: str | Path,
    model_selection_path: str | Path,
    output_dir: str | Path,
) -> Path:
    """
    Build and persist the forecast evidence artifact.

    Parameters
    ----------
    production_forecast_path:
        Path to the persisted production forecast.

    model_selection_path:
        Path to the persisted model-selection artifact.

    output_dir:
        Directory where the forecast evidence is persisted.

    Returns
    -------
    Path
        Path to the persisted forecast evidence artifact.
    """
    production_forecast = pd.read_parquet(
        production_forecast_path
    )

    model_selection = pd.read_parquet(
        model_selection_path
    )

    evidence = build_forecast_evidence(
        production_forecast=production_forecast,
        model_selection=model_selection,
    )

    return save_mart(
        evidence,
        mart_name="forecast_evidence",
        output_dir=output_dir,
    )