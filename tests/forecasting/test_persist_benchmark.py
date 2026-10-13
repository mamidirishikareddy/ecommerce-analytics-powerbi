from pathlib import Path

import pandas as pd

from src.forecasting.persist_benchmark import persist_forecast_benchmark


MARKET_MART = Path(
    "data/processed/marts/mart_market_performance_monthly.parquet"
)


def test_persist_forecast_benchmark_creates_file(tmp_path):
    output_path = persist_forecast_benchmark(
        market_mart_path=MARKET_MART,
        output_dir=tmp_path,
    )

    assert output_path.exists()
    assert output_path.name == "forecast_benchmark.parquet"


def test_persisted_benchmark_can_be_read(tmp_path):
    output_path = persist_forecast_benchmark(
        market_mart_path=MARKET_MART,
        output_dir=tmp_path,
    )

    result = pd.read_parquet(output_path)

    assert result.shape == (2, 9)
    assert result["model"].tolist() == ["NAIVE", "MA_3"]


def test_persisted_benchmark_has_expected_columns(tmp_path):
    output_path = persist_forecast_benchmark(
        market_mart_path=MARKET_MART,
        output_dir=tmp_path,
    )

    result = pd.read_parquet(output_path)

    expected_columns = [
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

    assert result.columns.tolist() == expected_columns


def test_persisted_benchmark_preserves_metrics(tmp_path):
    output_path = persist_forecast_benchmark(
        market_mart_path=MARKET_MART,
        output_dir=tmp_path,
    )

    result = pd.read_parquet(output_path)

    naive_wape = result.loc[
        result["model"] == "NAIVE",
        "wape_pct",
    ].iloc[0]

    ma3_wape = result.loc[
        result["model"] == "MA_3",
        "wape_pct",
    ].iloc[0]

    assert naive_wape > ma3_wape
    assert ma3_wape < 8.0