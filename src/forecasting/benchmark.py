"""
Forecast benchmark orchestration.

Builds a governed comparison of baseline forecasting models on the
fixed chronological evaluation window.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.forecasting.baselines import (
    moving_average_forecast,
    naive_forecast,
)
from src.forecasting.evaluation import summarize_forecast_accuracy
from src.forecasting.forecasting_dataset import build_forecasting_dataset


class ForecastBenchmarkError(ValueError):
    """Raised when forecast benchmarking cannot be performed."""


BENCHMARK_COLUMNS = [
    "model",
    "evaluation_start",
    "evaluation_end",
    "training_months",
    "evaluation_months",
    "mae",
    "rmse",
    "mape_pct",
    "wape_pct",
]


def build_forecast_benchmark(
    market_mart_path: str | Path,
    training_months: int = 16,
    moving_average_window: int = 3,
) -> pd.DataFrame:
    """
    Evaluate baseline forecasting models on a chronological holdout.

    Parameters
    ----------
    market_mart_path:
        Path to the canonical market-performance monthly mart.

    training_months:
        Number of chronological observations used for training.

    moving_average_window:
        Number of recent observations used by the moving-average baseline.

    Returns
    -------
    pd.DataFrame
        One row per baseline model with evaluation metrics.
    """
    if training_months <= 0:
        raise ForecastBenchmarkError(
            "training_months must be greater than zero."
        )

    if moving_average_window <= 0:
        raise ForecastBenchmarkError(
            "moving_average_window must be greater than zero."
        )

    df = build_forecasting_dataset(market_mart_path)

    if training_months >= len(df):
        raise ForecastBenchmarkError(
            "training_months must leave at least one evaluation observation."
        )

    if moving_average_window > training_months:
        raise ForecastBenchmarkError(
            "moving_average_window cannot exceed training_months."
        )

    train = df.iloc[:training_months].copy()
    evaluation = df.iloc[training_months:].copy()

    actual = evaluation["revenue"].reset_index(drop=True)

    naive = pd.Series(
        naive_forecast(
            train["revenue"],
            horizon=len(evaluation),
        )
    )

    moving_average = pd.Series(
        moving_average_forecast(
            train["revenue"],
            horizon=len(evaluation),
            window=moving_average_window,
        )
    )

    models = {
        "NAIVE": naive,
        f"MA_{moving_average_window}": moving_average,
    }

    rows: list[dict[str, object]] = []

    for model_name, forecast in models.items():
        metrics = summarize_forecast_accuracy(
            actual,
            forecast,
        )

        rows.append(
            {
                "model": model_name,
                "evaluation_start": evaluation["month"].min(),
                "evaluation_end": evaluation["month"].max(),
                "training_months": len(train),
                "evaluation_months": len(evaluation),
                "mae": metrics["mae"],
                "rmse": metrics["rmse"],
                "mape_pct": metrics["mape_pct"],
                "wape_pct": metrics["wape_pct"],
            }
        )

    result = pd.DataFrame(rows)

    if result.empty:
        raise ForecastBenchmarkError(
            "Benchmark evaluation produced no model results."
        )

    result["evaluation_start"] = pd.to_datetime(
        result["evaluation_start"]
    )
    result["evaluation_end"] = pd.to_datetime(
        result["evaluation_end"]
    )

    return result[BENCHMARK_COLUMNS]