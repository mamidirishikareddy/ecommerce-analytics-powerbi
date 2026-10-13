"""
Rolling-origin forecasting backtest.

Evaluates forecasting models across multiple chronological
forecast origins without random train/test splitting.
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
from src.forecasting.statistical_models import (
    simple_exponential_smoothing_forecast,
)


class RollingBacktestError(ValueError):
    """Raised when rolling-origin backtesting cannot be performed."""


ROLLING_BACKTEST_COLUMNS = [
    "origin",
    "evaluation_start",
    "evaluation_end",
    "model",
    "training_months",
    "evaluation_months",
    "mae",
    "rmse",
    "mape_pct",
    "wape_pct",
]


def build_rolling_backtest(
    market_mart_path: str | Path,
    minimum_training_months: int = 12,
    horizon: int = 4,
    moving_average_window: int = 3,
) -> pd.DataFrame:
    """
    Evaluate forecasting models across chronological forecast origins.

    Parameters
    ----------
    market_mart_path:
        Path to the canonical market-performance monthly mart.

    minimum_training_months:
        Minimum number of observations available before the first
        rolling forecast origin.

    horizon:
        Number of future months evaluated at each forecast origin.

    moving_average_window:
        Number of recent observations used by the moving-average
        forecasting model.

    Returns
    -------
    pd.DataFrame
        One row per model per rolling forecast origin containing
        evaluation metrics.
    """
    if minimum_training_months <= 0:
        raise RollingBacktestError(
            "minimum_training_months must be greater than zero."
        )

    if horizon <= 0:
        raise RollingBacktestError(
            "horizon must be greater than zero."
        )

    if moving_average_window <= 0:
        raise RollingBacktestError(
            "moving_average_window must be greater than zero."
        )

    if moving_average_window > minimum_training_months:
        raise RollingBacktestError(
            "moving_average_window cannot exceed "
            "minimum_training_months."
        )

    df = build_forecasting_dataset(market_mart_path)

    if minimum_training_months + horizon > len(df):
        raise RollingBacktestError(
            "Insufficient observations for the requested "
            "rolling backtest."
        )

    rows: list[dict[str, object]] = []

    for origin in range(
        minimum_training_months,
        len(df) - horizon + 1,
    ):
        train = df.iloc[:origin].copy()

        evaluation = df.iloc[
            origin : origin + horizon
        ].copy()

        actual = evaluation["revenue"].reset_index(drop=True)

        models = {
            "NAIVE": naive_forecast(
                train["revenue"],
                horizon=horizon,
            ),
            f"MA_{moving_average_window}": moving_average_forecast(
                train["revenue"],
                horizon=horizon,
                window=moving_average_window,
            ),
            "SES": simple_exponential_smoothing_forecast(
                train["revenue"],
                horizon=horizon,
            ),
        }

        for model_name, forecast_values in models.items():
            forecast = pd.Series(
                forecast_values,
                dtype=float,
            )

            metrics = summarize_forecast_accuracy(
                actual,
                forecast,
            )

            rows.append(
                {
                    "origin": train["month"].iloc[-1],
                    "evaluation_start": evaluation["month"].iloc[0],
                    "evaluation_end": evaluation["month"].iloc[-1],
                    "model": model_name,
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
        raise RollingBacktestError(
            "Rolling backtest produced no evaluation results."
        )

    for column in [
        "origin",
        "evaluation_start",
        "evaluation_end",
    ]:
        result[column] = pd.to_datetime(result[column])

    return result[ROLLING_BACKTEST_COLUMNS]