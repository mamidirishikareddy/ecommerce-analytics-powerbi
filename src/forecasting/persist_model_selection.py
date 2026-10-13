"""
Persistence for forecast model selection.

Builds the governed model-selection artifact from the persisted
rolling-origin backtest and fixed-holdout benchmark artifacts.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.forecasting.model_selection import select_forecast_model
from src.marts.persistence import save_mart


def persist_model_selection(
    rolling_backtest_path: str | Path,
    holdout_benchmark_path: str | Path,
    output_dir: str | Path,
) -> Path:
    """
    Build and persist the forecast model-selection artifact.

    Parameters
    ----------
    rolling_backtest_path:
        Path to the persisted rolling-origin backtest.

    holdout_benchmark_path:
        Path to the persisted fixed-holdout benchmark.

    output_dir:
        Directory where the model-selection artifact is persisted.

    Returns
    -------
    Path
        Path to the persisted model-selection artifact.
    """
    rolling_backtest = pd.read_parquet(rolling_backtest_path)
    holdout_benchmark = pd.read_parquet(holdout_benchmark_path)

    selection = select_forecast_model(
        rolling_backtest=rolling_backtest,
        holdout_benchmark=holdout_benchmark,
    )

    return save_mart(
        selection,
        mart_name="forecast_model_selection",
        output_dir=output_dir,
    )