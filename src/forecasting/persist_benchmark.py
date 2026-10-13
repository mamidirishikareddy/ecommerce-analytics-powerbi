"""
Persistence for the governed forecasting benchmark.

This module builds the baseline forecast benchmark and persists the
result using the project's shared mart persistence utility.
"""

from __future__ import annotations

from pathlib import Path

from src.forecasting.benchmark import build_forecast_benchmark
from src.marts.persistence import save_mart


def persist_forecast_benchmark(
    market_mart_path: str | Path,
    output_dir: str | Path,
) -> Path:
    """
    Build and persist the governed forecasting benchmark.

    Parameters
    ----------
    market_mart_path:
        Path to the canonical market-performance monthly mart.

    output_dir:
        Directory where the benchmark artifact will be persisted.

    Returns
    -------
    Path
        Path to the persisted benchmark artifact.
    """
    benchmark = build_forecast_benchmark(
        market_mart_path,
    )

    return save_mart(
        benchmark,
        mart_name="forecast_benchmark",
        output_dir=output_dir,
    )