import pandas as pd
import pytest

from src.forecasting.forecast_evidence import (
    ForecastEvidenceError,
    build_forecast_evidence,
)


def _production_forecast():
    return pd.DataFrame(
        {
            "forecast_month": pd.to_datetime(
                [
                    "2018-09-01",
                    "2018-10-01",
                ]
            ),
            "model": [
                "MA_3",
                "MA_3",
            ],
            "forecast_revenue": [
                871772.62,
                871772.62,
            ],
            "training_end_month": pd.to_datetime(
                [
                    "2018-08-01",
                    "2018-08-01",
                ]
            ),
            "training_months": [
                20,
                20,
            ],
            "moving_average_window": [
                3,
                3,
            ],
        }
    )


def _model_selection():
    return pd.DataFrame(
        {
            "model": [
                "MA_3",
                "NAIVE",
            ],
            "rolling_mean_wape_pct": [
                9.717673,
                11.180164,
            ],
            "rolling_median_wape_pct": [
                8.621505,
                10.375762,
            ],
            "rolling_wape_std_pct": [
                3.452098,
                6.162408,
            ],
            "rolling_win_count": [
                3,
                2,
            ],
            "holdout_wape_pct": [
                7.306343,
                10.375762,
            ],
            "selected": [
                True,
                False,
            ],
        }
    )


def test_forecast_evidence_has_expected_shape():
    result = build_forecast_evidence(
        _production_forecast(),
        _model_selection(),
    )

    assert result.shape == (2, 12)


def test_forecast_evidence_has_expected_columns():
    result = build_forecast_evidence(
        _production_forecast(),
        _model_selection(),
    )

    assert list(result.columns) == [
        "forecast_month",
        "model",
        "forecast_revenue",
        "training_end_month",
        "training_months",
        "moving_average_window",
        "rolling_mean_wape_pct",
        "rolling_median_wape_pct",
        "rolling_wape_std_pct",
        "rolling_win_count",
        "holdout_wape_pct",
        "forecast_evidence_level",
    ]


def test_forecast_evidence_uses_selected_model():
    result = build_forecast_evidence(
        _production_forecast(),
        _model_selection(),
    )

    assert result["model"].unique().tolist() == ["MA_3"]


def test_validation_metrics_are_attached():
    result = build_forecast_evidence(
        _production_forecast(),
        _model_selection(),
    )

    assert (
        result["rolling_mean_wape_pct"]
        == 9.717673
    ).all()

    assert (
        result["holdout_wape_pct"]
        == 7.306343
    ).all()


def test_forecast_evidence_is_marked_validated():
    result = build_forecast_evidence(
        _production_forecast(),
        _model_selection(),
    )

    assert (
        result["forecast_evidence_level"]
        .unique()
        .tolist()
        == ["VALIDATED"]
    )


def test_selected_model_must_exist():
    invalid_selection = _model_selection().copy()
    invalid_selection["selected"] = False

    with pytest.raises(ForecastEvidenceError):
        build_forecast_evidence(
            _production_forecast(),
            invalid_selection,
        )


def test_multiple_selected_models_raise():
    invalid_selection = _model_selection().copy()
    invalid_selection.loc[1, "selected"] = True

    with pytest.raises(ForecastEvidenceError):
        build_forecast_evidence(
            _production_forecast(),
            invalid_selection,
        )


def test_model_mismatch_raises():
    invalid_forecast = _production_forecast().copy()
    invalid_forecast["model"] = "SES"

    with pytest.raises(ForecastEvidenceError):
        build_forecast_evidence(
            invalid_forecast,
            _model_selection(),
        )


def test_missing_forecast_column_raises():
    invalid_forecast = _production_forecast().drop(
        columns=["forecast_revenue"]
    )

    with pytest.raises(ForecastEvidenceError):
        build_forecast_evidence(
            invalid_forecast,
            _model_selection(),
        )


def test_missing_selection_column_raises():
    invalid_selection = _model_selection().drop(
        columns=["holdout_wape_pct"]
    )

    with pytest.raises(ForecastEvidenceError):
        build_forecast_evidence(
            _production_forecast(),
            invalid_selection,
        )


def test_empty_forecast_raises():
    with pytest.raises(ForecastEvidenceError):
        build_forecast_evidence(
            pd.DataFrame(
                columns=_production_forecast().columns
            ),
            _model_selection(),
        )


def test_empty_selection_raises():
    with pytest.raises(ForecastEvidenceError):
        build_forecast_evidence(
            _production_forecast(),
            pd.DataFrame(
                columns=_model_selection().columns
            ),
        )


def test_missing_validation_metric_raises():
    invalid_selection = _model_selection().copy()
    invalid_selection.loc[
        0,
        "holdout_wape_pct",
    ] = None

    with pytest.raises(ForecastEvidenceError):
        build_forecast_evidence(
            _production_forecast(),
            invalid_selection,
        )