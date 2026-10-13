from pathlib import Path

import pandas as pd
import pytest

from src.forecasting.persist_forecast_evidence import (
    persist_forecast_evidence,
)


PRODUCTION_FORECAST = Path(
    "data/processed/forecasting/forecast_production.parquet"
)

MODEL_SELECTION = Path(
    "data/processed/forecasting/forecast_model_selection.parquet"
)


def test_persist_forecast_evidence_creates_parquet(tmp_path):
    result = persist_forecast_evidence(
        production_forecast_path=PRODUCTION_FORECAST,
        model_selection_path=MODEL_SELECTION,
        output_dir=tmp_path,
    )

    assert result.exists()
    assert result.name == "forecast_evidence.parquet"


def test_persisted_forecast_evidence_has_expected_shape(tmp_path):
    result = persist_forecast_evidence(
        production_forecast_path=PRODUCTION_FORECAST,
        model_selection_path=MODEL_SELECTION,
        output_dir=tmp_path,
    )

    df = pd.read_parquet(result)

    assert df.shape == (4, 12)


def test_persisted_forecast_evidence_uses_ma3(tmp_path):
    result = persist_forecast_evidence(
        production_forecast_path=PRODUCTION_FORECAST,
        model_selection_path=MODEL_SELECTION,
        output_dir=tmp_path,
    )

    df = pd.read_parquet(result)

    assert df["model"].unique().tolist() == ["MA_3"]


def test_persisted_forecast_evidence_is_validated(tmp_path):
    result = persist_forecast_evidence(
        production_forecast_path=PRODUCTION_FORECAST,
        model_selection_path=MODEL_SELECTION,
        output_dir=tmp_path,
    )

    df = pd.read_parquet(result)

    assert (
        df["forecast_evidence_level"]
        .unique()
        .tolist()
        == ["VALIDATED"]
    )


def test_persisted_forecast_evidence_has_no_nulls(tmp_path):
    result = persist_forecast_evidence(
        production_forecast_path=PRODUCTION_FORECAST,
        model_selection_path=MODEL_SELECTION,
        output_dir=tmp_path,
    )

    df = pd.read_parquet(result)

    assert df.isna().sum().sum() == 0


def test_persisted_forecast_evidence_preserves_holdout_wape(tmp_path):
    result = persist_forecast_evidence(
        production_forecast_path=PRODUCTION_FORECAST,
        model_selection_path=MODEL_SELECTION,
        output_dir=tmp_path,
    )

    df = pd.read_parquet(result)

    assert df["holdout_wape_pct"].tolist() == pytest.approx(
        [7.306343] * 4
    )