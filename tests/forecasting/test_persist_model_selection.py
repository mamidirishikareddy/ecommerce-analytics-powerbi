from pathlib import Path

import pandas as pd

from src.forecasting.persist_model_selection import (
    persist_model_selection,
)


ROLLING_BACKTEST = Path(
    "data/processed/forecasting/forecast_rolling_backtest.parquet"
)

HOLDOUT_BENCHMARK = Path(
    "data/processed/forecasting/forecast_benchmark.parquet"
)


def test_persist_model_selection_creates_parquet(tmp_path):
    result = persist_model_selection(
        rolling_backtest_path=ROLLING_BACKTEST,
        holdout_benchmark_path=HOLDOUT_BENCHMARK,
        output_dir=tmp_path,
    )

    assert result.exists()
    assert result.name == "forecast_model_selection.parquet"


def test_persisted_model_selection_has_expected_shape(tmp_path):
    result = persist_model_selection(
        rolling_backtest_path=ROLLING_BACKTEST,
        holdout_benchmark_path=HOLDOUT_BENCHMARK,
        output_dir=tmp_path,
    )

    df = pd.read_parquet(result)

    assert len(df) == 2
    assert df["model"].nunique() == 2
    assert set(df["model"]) == {"MA_3", "NAIVE"}


def test_persisted_model_selection_has_expected_columns(tmp_path):
    result = persist_model_selection(
        rolling_backtest_path=ROLLING_BACKTEST,
        holdout_benchmark_path=HOLDOUT_BENCHMARK,
        output_dir=tmp_path,
    )

    df = pd.read_parquet(result)

    assert list(df.columns) == [
        "model",
        "rolling_mean_wape_pct",
        "rolling_median_wape_pct",
        "rolling_wape_std_pct",
        "rolling_win_count",
        "holdout_wape_pct",
        "selected",
    ]


def test_persisted_model_selection_selects_one_model(tmp_path):
    result = persist_model_selection(
        rolling_backtest_path=ROLLING_BACKTEST,
        holdout_benchmark_path=HOLDOUT_BENCHMARK,
        output_dir=tmp_path,
    )

    df = pd.read_parquet(result)

    assert df["selected"].sum() == 1


def test_persisted_model_selection_selects_ma3(tmp_path):
    result = persist_model_selection(
        rolling_backtest_path=ROLLING_BACKTEST,
        holdout_benchmark_path=HOLDOUT_BENCHMARK,
        output_dir=tmp_path,
    )

    df = pd.read_parquet(result)

    selected_model = df.loc[
        df["selected"],
        "model",
    ].iloc[0]

    assert selected_model == "MA_3"


def test_persisted_model_selection_has_no_null_metrics(tmp_path):
    result = persist_model_selection(
        rolling_backtest_path=ROLLING_BACKTEST,
        holdout_benchmark_path=HOLDOUT_BENCHMARK,
        output_dir=tmp_path,
    )

    df = pd.read_parquet(result)

    assert (
        df[
            [
                "rolling_mean_wape_pct",
                "rolling_median_wape_pct",
                "rolling_wape_std_pct",
                "holdout_wape_pct",
            ]
        ]
        .isna()
        .sum()
        .sum()
        == 0
    )