"""
Persistence for the rolling-origin forecasting backtest.

Builds the governed rolling-origin evaluation artifact and persists it
using the project's shared mart persistence utility.
"""

from __future__ import annotations

from pathlib import Path

from src.forecasting.rolling_backtest import build_rolling_backtest
from src.marts.persistence import save_mart


def persist_rolling_backtest(
    market_mart_path: str | Path,
    output_dir: str | Path,
) -> Path:
    """
    Build and persist the rolling-origin forecasting backtest.

    Parameters
    ----------
    market_mart_path:
        Path to the canonical market-performance monthly mart.

    output_dir:
        Directory where the rolling backtest artifact will be persisted.

    Returns
    -------
    Path
        Path to the persisted rolling backtest artifact.
    """
    backtest = build_rolling_backtest(
        market_mart_path,
    )

    return save_mart(
        backtest,
        mart_name="forecast_rolling_backtest",
        output_dir=output_dir,
    )