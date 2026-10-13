from pathlib import Path

import pandas as pd
import pytest

from src.forecasting.persist_rolling_backtest import (
    persist_rolling_backtest,
)


MARKET_MART = Path(
    "data/processed/marts/mart_market_performance_monthly.parquet"
)


def test_persist_rolling_backtest_creates_parquet(tmp_path):
    result = persist_rolling_backtest(
        market_mart_path=MARKET_MART,
        output_dir=tmp_path,
    )

    assert result.exists()
    assert result.name == "forecast_rolling_backtest.parquet"


def test_persisted_rolling_backtest_has_expected_shape(tmp_path):
    result = persist_rolling_backtest(
        market_mart_path=MARKET_MART,
        output_dir=tmp_path,
    )

    df = pd.read_parquet(result)

    assert len(df) == 15
    assert df["model"].nunique() == 3
    assert df["origin"].nunique() == 5


def test_persisted_rolling_backtest_has_expected_models(tmp_path):
    result = persist_rolling_backtest(
        market_mart_path=MARKET_MART,
        output_dir=tmp_path,
    )

    df = pd.read_parquet(result)

    assert set(df["model"]) == {
        "NAIVE",
        "MA_3",
        "SES",
    }


def test_persisted_rolling_backtest_preserves_metrics(tmp_path):
    result = persist_rolling_backtest(
        market_mart_path=MARKET_MART,
        output_dir=tmp_path,
    )

    df = pd.read_parquet(result)

    assert df["wape_pct"].notna().all()
    assert (df["wape_pct"] >= 0).all()
    assert df["mae"].notna().all()
    assert df["rmse"].notna().all()
    assert df["mape_pct"].notna().all()