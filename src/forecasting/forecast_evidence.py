"""
Forecast evidence construction.

Combines the production forecast with model-selection and validation
evidence so downstream decision layers can distinguish the forecast
itself from the strength of evidence supporting it.
"""

from __future__ import annotations

import pandas as pd


class ForecastEvidenceError(ValueError):
    """Raised when forecast evidence cannot be constructed."""


FORECAST_EVIDENCE_COLUMNS = [
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


def build_forecast_evidence(
    production_forecast: pd.DataFrame,
    model_selection: pd.DataFrame,
) -> pd.DataFrame:
    """
    Combine production forecast with model-validation evidence.

    Parameters
    ----------
    production_forecast:
        Persisted production forecast.

    model_selection:
        Persisted model-selection artifact.

    Returns
    -------
    pd.DataFrame
        Forecast rows enriched with model-validation evidence.
    """
    required_forecast = {
        "forecast_month",
        "model",
        "forecast_revenue",
        "training_end_month",
        "training_months",
        "moving_average_window",
    }

    required_selection = {
        "model",
        "rolling_mean_wape_pct",
        "rolling_median_wape_pct",
        "rolling_wape_std_pct",
        "rolling_win_count",
        "holdout_wape_pct",
        "selected",
    }

    missing_forecast = (
        required_forecast - set(production_forecast.columns)
    )

    if missing_forecast:
        raise ForecastEvidenceError(
            "Production forecast is missing required columns: "
            f"{sorted(missing_forecast)}"
        )

    missing_selection = (
        required_selection - set(model_selection.columns)
    )

    if missing_selection:
        raise ForecastEvidenceError(
            "Model selection is missing required columns: "
            f"{sorted(missing_selection)}"
        )

    if production_forecast.empty:
        raise ForecastEvidenceError(
            "Production forecast cannot be empty."
        )

    if model_selection.empty:
        raise ForecastEvidenceError(
            "Model selection cannot be empty."
        )

    selected_models = model_selection.loc[
        model_selection["selected"],
        "model",
    ].tolist()

    if len(selected_models) != 1:
        raise ForecastEvidenceError(
            "Model selection must contain exactly one selected model."
        )

    selected_model = selected_models[0]

    if set(production_forecast["model"]) != {selected_model}:
        raise ForecastEvidenceError(
            "Production forecast model does not match the selected "
            "model."
        )

    evidence = production_forecast.merge(
        model_selection[
            [
                "model",
                "rolling_mean_wape_pct",
                "rolling_median_wape_pct",
                "rolling_wape_std_pct",
                "rolling_win_count",
                "holdout_wape_pct",
                "selected",
            ]
        ],
        on="model",
        how="left",
        validate="many_to_one",
    )

    if evidence[
        [
            "rolling_mean_wape_pct",
            "rolling_median_wape_pct",
            "rolling_wape_std_pct",
            "holdout_wape_pct",
        ]
    ].isna().any().any():
        raise ForecastEvidenceError(
            "Forecast evidence contains missing model-validation metrics."
        )

    evidence["forecast_evidence_level"] = "VALIDATED"

    return evidence[
        FORECAST_EVIDENCE_COLUMNS
    ]