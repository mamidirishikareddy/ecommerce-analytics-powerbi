from pathlib import Path

import pandas as pd

from src.forecasting.persist_production_forecast import (
    persist_production_forecast,
)


MARKET_MART = Path(
    "data/processed/marts/mart_market_performance_monthly.parquet"
)

MODEL_SELECTION = Path(
    "data/processed/forecasting/forecast_model_selection.parquet"
)


def test_persist_production_forecast_creates_parquet(tmp_path):
    result = persist_production_forecast(
        market_mart_path=MARKET_MART,
        model_selection_path=MODEL_SELECTION,
        output_dir=tmp_path,
    )

    assert result.exists()
    assert result.name == "forecast_production.parquet"


def test_persisted_production_forecast_has_expected_shape(tmp_path):
    result = persist_production_forecast(
        market_mart_path=MARKET_MART,
        model_selection_path=MODEL_SELECTION,
        output_dir=tmp_path,
    )

    df = pd.read_parquet(result)

    assert df.shape == (4, 6)


def test_persisted_production_forecast_uses_ma3(tmp_path):
    result = persist_production_forecast(
        market_mart_path=MARKET_MART,
        model_selection_path=MODEL_SELECTION,
        output_dir=tmp_path,
    )

    df = pd.read_parquet(result)

    assert df["model"].unique().tolist() == ["MA_3"]


def test_persisted_production_forecast_has_four_months(tmp_path):
    result = persist_production_forecast(
        market_mart_path=MARKET_MART,
        model_selection_path=MODEL_SELECTION,
        output_dir=tmp_path,
    )

    df = pd.read_parquet(result)

    assert df["forecast_month"].nunique() == 4
    assert df["forecast_month"].is_monotonic_increasing


def test_persisted_production_forecast_has_no_nulls(tmp_path):
    result = persist_production_forecast(
        market_mart_path=MARKET_MART,
        model_selection_path=MODEL_SELECTION,
        output_dir=tmp_path,
    )

    df = pd.read_parquet(result)

    assert df.isna().sum().sum() == 0


def test_persisted_production_forecast_is_nonnegative(tmp_path):
    result = persist_production_forecast(
        market_mart_path=MARKET_MART,
        model_selection_path=MODEL_SELECTION,
        output_dir=tmp_path,
    )

    df = pd.read_parquet(result)

    assert (df["forecast_revenue"] >= 0).all()