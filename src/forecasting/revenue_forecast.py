from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX


class RevenueForecastError(ValueError):
    """Raised when governed revenue forecasting cannot be completed."""


REQUIRED_COLUMNS = [
    "month",
    "revenue",
]

OUTPUT_COLUMNS = [
    "month",
    "actual_revenue",
    "forecast_revenue",
    "lower_bound",
    "upper_bound",
    "forecast_type",
]

EVALUATION_COLUMNS = [
    "model",
    "mae",
    "rmse",
    "wape",
    "bias",
    "selected",
    "holdout_size",
    "training_observation_count",
    "evaluation_start_month",
    "evaluation_end_month",
]

VERIFIED_STRUCTURAL_ZERO_MONTHS = {
    pd.Timestamp("2016-11-01"),
}

KNOWN_PARTIAL_MONTHS = {
    pd.Timestamp("2018-09-01"),
}


def _validate_monthly_series(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """Validate and normalize governed monthly revenue."""

    if df.empty:
        raise RevenueForecastError(
            "Revenue forecast input is empty."
        )

    missing = sorted(
        set(REQUIRED_COLUMNS) - set(df.columns)
    )

    if missing:
        raise RevenueForecastError(
            f"Missing required columns: {missing}"
        )

    result = df[REQUIRED_COLUMNS].copy()

    result["month"] = pd.to_datetime(
        result["month"],
        errors="raise",
    )

    result["revenue"] = pd.to_numeric(
        result["revenue"],
        errors="raise",
    )

    if result["month"].duplicated().any():
        raise RevenueForecastError(
            "Revenue forecast input contains duplicate months."
        )

    if result["month"].isna().any():
        raise RevenueForecastError(
            "Revenue forecast input contains null months."
        )

    if result["revenue"].isna().any():
        raise RevenueForecastError(
            "Revenue forecast input contains null revenue values."
        )

    if (result["revenue"] < 0).any():
        raise RevenueForecastError(
            "Revenue forecast input contains negative revenue values."
        )

    result = (
        result
        .sort_values("month")
        .reset_index(drop=True)
    )

    expected_months = pd.date_range(
        start=result["month"].min(),
        end=result["month"].max(),
        freq="MS",
    )

    observed_months = pd.DatetimeIndex(
        result["month"]
    )

    missing_months = expected_months.difference(
        observed_months
    )

    unexpected_missing = [
        month
        for month in missing_months
        if month not in VERIFIED_STRUCTURAL_ZERO_MONTHS
    ]

    if unexpected_missing:
        raise RevenueForecastError(
            "Revenue forecast input contains unexpected "
            f"missing calendar months: "
            f"{[m.strftime('%Y-%m') for m in unexpected_missing]}"
        )

    if len(missing_months) > 0:
        zero_rows = pd.DataFrame(
            {
                "month": missing_months,
                "revenue": 0.0,
            }
        )

        result = pd.concat(
            [
                result,
                zero_rows,
            ],
            ignore_index=True,
        )

        result = (
            result
            .sort_values("month")
            .reset_index(drop=True)
        )

    return result


def _remove_partial_periods(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """Remove known partial periods from model training."""

    result = df[
        ~df["month"].isin(KNOWN_PARTIAL_MONTHS)
    ].copy()

    if result.empty:
        raise RevenueForecastError(
            "No complete monthly observations remain "
            "after excluding partial periods."
        )

    if len(result) < 18:
        raise RevenueForecastError(
            "At least 18 complete monthly observations "
            "are required for governed forecasting."
        )

    return result


def _seasonal_naive_forecast(
    train: pd.Series,
    horizon: int,
    season_length: int = 12,
) -> np.ndarray:
    """Forecast using the corresponding month from the prior year."""

    if len(train) < season_length:
        raise RevenueForecastError(
            "Insufficient history for seasonal-naive forecasting."
        )

    values = train.to_numpy(
        dtype=float
    )

    return np.array(
        [
            values[
                -season_length
                + (i % season_length)
            ]
            for i in range(horizon)
        ],
        dtype=float,
    )


def _mae(
    actual: np.ndarray,
    predicted: np.ndarray,
) -> float:
    """Mean Absolute Error."""

    return float(
        np.mean(
            np.abs(
                actual - predicted
            )
        )
    )


def _rmse(
    actual: np.ndarray,
    predicted: np.ndarray,
) -> float:
    """Root Mean Squared Error."""

    return float(
        np.sqrt(
            np.mean(
                (actual - predicted) ** 2
            )
        )
    )


def _wape(
    actual: np.ndarray,
    predicted: np.ndarray,
) -> float:
    """Weighted Absolute Percentage Error."""

    denominator = np.sum(
        np.abs(actual)
    )

    if denominator == 0:
        return 0.0

    return float(
        np.sum(
            np.abs(
                actual - predicted
            )
        )
        / denominator
    )


def _bias(
    actual: np.ndarray,
    predicted: np.ndarray,
) -> float:
    """Mean forecast bias."""

    return float(
        np.mean(
            predicted - actual
        )
    )


def _fit_sarimax(
    train: pd.Series,
    horizon: int,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """Fit SARIMAX and return forecasts with intervals."""

    model = SARIMAX(
        train,
        order=(1, 1, 1),
        seasonal_order=(1, 1, 1, 12),
        trend=None,
        enforce_stationarity=False,
        enforce_invertibility=False,
    )

    fitted = model.fit(
        disp=False,
    )

    forecast = fitted.get_forecast(
        steps=horizon,
    )

    predicted = np.asarray(
        forecast.predicted_mean,
        dtype=float,
    )

    interval = forecast.conf_int()

    lower = np.asarray(
        interval.iloc[:, 0],
        dtype=float,
    )

    upper = np.asarray(
        interval.iloc[:, 1],
        dtype=float,
    )

    return (
        predicted,
        lower,
        upper,
    )


def build_revenue_forecast(
    input_path: str | Path,
    output_path: str | Path,
    forecast_horizon: int = 3,
) -> pd.DataFrame:
    """Build and persist the governed monthly revenue forecast."""

    if forecast_horizon <= 0:
        raise RevenueForecastError(
            "forecast_horizon must be positive."
        )

    input_path = Path(
        input_path
    )

    output_path = Path(
        output_path
    )

    if not input_path.exists():
        raise RevenueForecastError(
            f"Forecast input does not exist: "
            f"{input_path}"
        )

    df = pd.read_parquet(
        input_path
    )

    df = _validate_monthly_series(
        df
    )

    source_observation_count = len(df)

    df = _remove_partial_periods(
        df
    )

    series = (
        df
        .set_index("month")["revenue"]
        .asfreq("MS")
    )

    complete_observation_count = len(series)

    holdout_size = min(
        3,
        max(
            1,
            len(series) // 5,
        ),
    )

    train = series.iloc[
        :-holdout_size
    ]

    test = series.iloc[
        -holdout_size:
    ]

    actual_holdout = test.to_numpy(
        dtype=float
    )

    seasonal_naive_prediction = (
        _seasonal_naive_forecast(
            train=train,
            horizon=holdout_size,
        )
    )

    baseline_mae = _mae(
        actual_holdout,
        seasonal_naive_prediction,
    )

    baseline_rmse = _rmse(
        actual_holdout,
        seasonal_naive_prediction,
    )

    baseline_wape = _wape(
        actual_holdout,
        seasonal_naive_prediction,
    )

    baseline_bias = _bias(
        actual_holdout,
        seasonal_naive_prediction,
    )

    try:
        sarimax_prediction, _, _ = (
            _fit_sarimax(
                train=train,
                horizon=holdout_size,
            )
        )

        sarimax_mae = _mae(
            actual_holdout,
            sarimax_prediction,
        )

        sarimax_rmse = _rmse(
            actual_holdout,
            sarimax_prediction,
        )

        sarimax_wape = _wape(
            actual_holdout,
            sarimax_prediction,
        )

        sarimax_bias = _bias(
            actual_holdout,
            sarimax_prediction,
        )

    except Exception:
        sarimax_prediction = None
        sarimax_mae = np.nan
        sarimax_rmse = np.nan
        sarimax_wape = np.inf
        sarimax_bias = np.nan

    if sarimax_wape < baseline_wape:
        selected_model = "sarimax"
    else:
        selected_model = "seasonal_naive"

    evaluation_path = output_path.with_name(
        "revenue_forecast_evaluation.parquet"
    )

    evaluation = pd.DataFrame(
        [
            {
                "model": "seasonal_naive",
                "mae": baseline_mae,
                "rmse": baseline_rmse,
                "wape": baseline_wape,
                "bias": baseline_bias,
                "selected": selected_model == "seasonal_naive",
                "holdout_size": holdout_size,
                "training_observation_count": len(train),
                "evaluation_start_month": test.index.min(),
                "evaluation_end_month": test.index.max(),
            },
            {
                "model": "sarimax",
                "mae": sarimax_mae,
                "rmse": sarimax_rmse,
                "wape": sarimax_wape,
                "bias": sarimax_bias,
                "selected": selected_model == "sarimax",
                "holdout_size": holdout_size,
                "training_observation_count": len(train),
                "evaluation_start_month": test.index.min(),
                "evaluation_end_month": test.index.max(),
            },
        ],
        columns=EVALUATION_COLUMNS,
    )

    evaluation.to_parquet(
        evaluation_path,
        index=False,
    )

    if selected_model == "sarimax":

        future_forecast, lower, upper = (
            _fit_sarimax(
                train=series,
                horizon=forecast_horizon,
            )
        )

    else:

        future_forecast = (
            _seasonal_naive_forecast(
                train=series,
                horizon=forecast_horizon,
            )
        )

        residuals = (
            series.iloc[12:].to_numpy(
                dtype=float
            )
            - series.iloc[:-12].to_numpy(
                dtype=float
            )
        )

        if len(residuals) > 1:
            residual_std = float(
                np.std(
                    residuals,
                    ddof=1,
                )
            )
        else:
            residual_std = 0.0

        lower = (
            future_forecast
            - 1.96 * residual_std
        )

        upper = (
            future_forecast
            + 1.96 * residual_std
        )

    future_forecast = np.maximum(
        future_forecast,
        0.0,
    )

    lower = np.maximum(
        lower,
        0.0,
    )

    upper = np.maximum(
        upper,
        0.0,
    )

    last_complete_month = series.index.max()

    last_source_month = max(
        pd.Timestamp("2018-09-01"),
        last_complete_month,
    )

    future_start = (
        last_source_month
        + pd.offsets.MonthBegin(1)
    )

    future_months = pd.date_range(
        start=future_start,
        periods=forecast_horizon,
        freq="MS",
    )

    historical = pd.DataFrame(
        {
            "month": series.index,
            "actual_revenue": series.values,
            "forecast_revenue": np.nan,
            "lower_bound": np.nan,
            "upper_bound": np.nan,
            "forecast_type": "actual",
        }
    )

    future = pd.DataFrame(
        {
            "month": future_months,
            "actual_revenue": np.nan,
            "forecast_revenue": future_forecast,
            "lower_bound": lower,
            "upper_bound": upper,
            "forecast_type": selected_model,
        }
    )

    result = pd.concat(
        [
            historical,
            future,
        ],
        ignore_index=True,
    )

    result = result[
        OUTPUT_COLUMNS
    ]

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    result.to_parquet(
        output_path,
        index=False,
    )

    print(
        "Forecast model evaluation:"
    )

    print(
        f"Seasonal Naive | "
        f"MAE={baseline_mae:.2f} | "
        f"RMSE={baseline_rmse:.2f} | "
        f"WAPE={baseline_wape:.6f} | "
        f"Bias={baseline_bias:.2f}"
    )

    if np.isfinite(
        sarimax_wape
    ):
        print(
            f"SARIMAX | "
            f"MAE={sarimax_mae:.2f} | "
            f"RMSE={sarimax_rmse:.2f} | "
            f"WAPE={sarimax_wape:.6f} | "
            f"Bias={sarimax_bias:.2f}"
        )
    else:
        print(
            "SARIMAX | "
            "Model evaluation failed; "
            "Seasonal Naive retained."
        )

    print(
        f"Selected model: {selected_model}"
    )

    print(
        f"Source observations: "
        f"{source_observation_count}"
    )

    print(
        f"Complete observations used: "
        f"{complete_observation_count}"
    )

    print(
        f"Last complete month: "
        f"{last_complete_month.strftime('%Y-%m')}"
    )

    print(
        f"Forecast starts: "
        f"{future_start.strftime('%Y-%m')}"
    )

    print(
        f"Forecast horizon: "
        f"{forecast_horizon} month(s)"
    )

    print(
        f"Output: {output_path}"
    )

    print(
        f"Evaluation output: {evaluation_path}"
    )

    return result