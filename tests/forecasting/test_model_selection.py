import pandas as pd
import pytest

from src.forecasting.model_selection import (
    ModelSelectionError,
    select_forecast_model,
)


def _rolling_backtest():
    return pd.DataFrame(
        {
            "origin": pd.to_datetime(
                [
                    "2018-01-01",
                    "2018-01-01",
                    "2018-02-01",
                    "2018-02-01",
                ]
            ),
            "model": [
                "NAIVE",
                "MA_3",
                "NAIVE",
                "MA_3",
            ],
            "wape_pct": [
                12.0,
                8.0,
                10.0,
                9.0,
            ],
        }
    )


def _holdout_benchmark():
    return pd.DataFrame(
        {
            "model": [
                "NAIVE",
                "MA_3",
            ],
            "wape_pct": [
                11.0,
                7.0,
            ],
        }
    )


def test_selects_lowest_mean_wape():
    result = select_forecast_model(
        _rolling_backtest(),
        _holdout_benchmark(),
    )

    selected = result.loc[result["selected"], "model"].iloc[0]

    assert selected == "MA_3"


def test_returns_expected_columns():
    result = select_forecast_model(
        _rolling_backtest(),
        _holdout_benchmark(),
    )

    assert list(result.columns) == [
        "model",
        "rolling_mean_wape_pct",
        "rolling_median_wape_pct",
        "rolling_wape_std_pct",
        "rolling_win_count",
        "holdout_wape_pct",
        "selected",
    ]


def test_one_model_is_selected():
    result = select_forecast_model(
        _rolling_backtest(),
        _holdout_benchmark(),
    )

    assert result["selected"].sum() == 1


def test_models_are_ranked_by_rolling_mean_wape():
    result = select_forecast_model(
        _rolling_backtest(),
        _holdout_benchmark(),
    )

    assert result.iloc[0]["model"] == "MA_3"
    assert (
        result.iloc[0]["rolling_mean_wape_pct"]
        < result.iloc[1]["rolling_mean_wape_pct"]
    )


def test_rolling_win_count_is_calculated():
    result = select_forecast_model(
        _rolling_backtest(),
        _holdout_benchmark(),
    )

    ma3_wins = result.loc[
        result["model"] == "MA_3",
        "rolling_win_count",
    ].iloc[0]

    assert ma3_wins == 2


def test_holdout_wape_is_preserved():
    result = select_forecast_model(
        _rolling_backtest(),
        _holdout_benchmark(),
    )

    ma3_holdout = result.loc[
        result["model"] == "MA_3",
        "holdout_wape_pct",
    ].iloc[0]

    assert ma3_holdout == 7.0


def test_empty_rolling_backtest_raises():
    with pytest.raises(ModelSelectionError):
        select_forecast_model(
            pd.DataFrame(
                columns=["model", "wape_pct"]
            ),
            _holdout_benchmark(),
        )


def test_empty_holdout_benchmark_raises():
    with pytest.raises(ModelSelectionError):
        select_forecast_model(
            _rolling_backtest(),
            pd.DataFrame(
                columns=["model", "wape_pct"]
            ),
        )


def test_missing_rolling_column_raises():
    invalid = _rolling_backtest().drop(
        columns=["wape_pct"]
    )

    with pytest.raises(ModelSelectionError):
        select_forecast_model(
            invalid,
            _holdout_benchmark(),
        )


def test_missing_holdout_column_raises():
    invalid = _holdout_benchmark().drop(
        columns=["wape_pct"]
    )

    with pytest.raises(ModelSelectionError):
        select_forecast_model(
            _rolling_backtest(),
            invalid,
        )


def test_null_rolling_wape_raises():
    invalid = _rolling_backtest()
    invalid.loc[0, "wape_pct"] = None

    with pytest.raises(ModelSelectionError):
        select_forecast_model(
            invalid,
            _holdout_benchmark(),
        )


def test_null_holdout_wape_raises():
    invalid = _holdout_benchmark()
    invalid.loc[0, "wape_pct"] = None

    with pytest.raises(ModelSelectionError):
        select_forecast_model(
            _rolling_backtest(),
            invalid,
        )