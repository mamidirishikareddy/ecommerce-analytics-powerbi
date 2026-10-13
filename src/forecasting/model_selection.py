"""
Forecast model selection.

Selects a forecasting champion using rolling-origin validation,
with fixed-holdout performance retained as supporting evidence.
"""

from __future__ import annotations

import pandas as pd


class ModelSelectionError(ValueError):
    """Raised when forecast model selection cannot be performed."""


SELECTION_COLUMNS = [
    "model",
    "rolling_mean_wape_pct",
    "rolling_median_wape_pct",
    "rolling_wape_std_pct",
    "rolling_win_count",
    "holdout_wape_pct",
    "selected",
]


def select_forecast_model(
    rolling_backtest: pd.DataFrame,
    holdout_benchmark: pd.DataFrame,
) -> pd.DataFrame:
    """
    Select the forecasting champion using rolling-origin WAPE.

    The primary selection criterion is the lowest mean WAPE across
    rolling-origin evaluations. Fixed-holdout WAPE is retained as
    supporting evidence.

    Parameters
    ----------
    rolling_backtest:
        Rolling-origin backtest results.

    holdout_benchmark:
        Fixed chronological holdout benchmark results.

    Returns
    -------
    pd.DataFrame
        One row per model with model-selection evidence.
    """
    required_rolling = {
        "model",
        "wape_pct",
    }

    required_holdout = {
        "model",
        "wape_pct",
    }

    missing_rolling = required_rolling - set(rolling_backtest.columns)
    if missing_rolling:
        raise ModelSelectionError(
            f"Rolling backtest is missing required columns: "
            f"{sorted(missing_rolling)}"
        )

    missing_holdout = required_holdout - set(holdout_benchmark.columns)
    if missing_holdout:
        raise ModelSelectionError(
            f"Holdout benchmark is missing required columns: "
            f"{sorted(missing_holdout)}"
        )

    if rolling_backtest.empty:
        raise ModelSelectionError(
            "Rolling backtest cannot be empty."
        )

    if holdout_benchmark.empty:
        raise ModelSelectionError(
            "Holdout benchmark cannot be empty."
        )

    rolling = rolling_backtest.copy()
    holdout = holdout_benchmark.copy()

    if rolling["wape_pct"].isna().any():
        raise ModelSelectionError(
            "Rolling backtest contains null WAPE values."
        )

    if holdout["wape_pct"].isna().any():
        raise ModelSelectionError(
            "Holdout benchmark contains null WAPE values."
        )

    rolling_summary = (
        rolling
        .groupby("model", as_index=False)
        .agg(
            rolling_mean_wape_pct=("wape_pct", "mean"),
            rolling_median_wape_pct=("wape_pct", "median"),
            rolling_wape_std_pct=("wape_pct", "std"),
        )
    )

    rolling_wins = (
        rolling.assign(
            rank=rolling.groupby("origin")["wape_pct"]
            .rank(method="min", ascending=True)
        )
        .loc[lambda df: df["rank"] == 1]
        .groupby("model")
        .size()
        .rename("rolling_win_count")
        .reset_index()
    )

    holdout_summary = (
        holdout[
            [
                "model",
                "wape_pct",
            ]
        ]
        .rename(
            columns={
                "wape_pct": "holdout_wape_pct",
            }
        )
    )

    result = (
        rolling_summary
        .merge(
            rolling_wins,
            on="model",
            how="left",
        )
        .merge(
            holdout_summary,
            on="model",
            how="inner",
        )
    )

    result["rolling_win_count"] = (
        result["rolling_win_count"]
        .fillna(0)
        .astype(int)
    )

    if result.empty:
        raise ModelSelectionError(
            "No common models exist between rolling backtest "
            "and holdout benchmark."
        )

    result = result.sort_values(
        [
            "rolling_mean_wape_pct",
            "holdout_wape_pct",
            "model",
        ],
        ascending=[
            True,
            True,
            True,
        ],
    ).reset_index(drop=True)

    result["selected"] = False
    result.loc[0, "selected"] = True

    return result[SELECTION_COLUMNS]