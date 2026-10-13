"""
Customer-level behavioral signal generation.

This module converts the governed customer performance mart into
interpretable customer signals.

Design principles:
- deterministic
- testable
- explainable
- governed input grain
- explicit signal contract
- no external API dependencies
- no hidden business thresholds
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


class CustomerSignalError(ValueError):
    """Raised when governed customer signal generation fails."""


REQUIRED_COLUMNS = {
    "month",
    "customer_unique_id",
    "order_count",
    "item_line_count",
    "revenue",
    "freight_value",
    "avg_order_value",
    "first_purchase_month",
    "customer_status",
}


OUTPUT_COLUMNS = [
    "customer_unique_id",
    "signal_date",
    "customer_lifecycle",
    "value_segment",
    "behavior_direction",
    "evidence_level",
    "active_months",
    "total_orders",
    "lifetime_revenue",
    "transition_count",
    "direction_consistency",
    "signal_confidence",
]


def _load_customer_mart(
    marts_dir: str | Path,
) -> pd.DataFrame:
    """Load the governed customer performance mart."""

    path = (
        Path(marts_dir)
        / "mart_customer_performance_monthly.parquet"
    )

    if not path.exists():
        raise CustomerSignalError(
            f"Customer performance mart not found: {path}"
        )

    df = pd.read_parquet(path)

    missing_columns = sorted(
        REQUIRED_COLUMNS.difference(df.columns)
    )

    if missing_columns:
        raise CustomerSignalError(
            "Customer performance mart is missing required "
            f"columns: {missing_columns}"
        )

    return df


def validate_input(df: pd.DataFrame) -> None:
    """Validate the governed customer performance mart input."""

    missing_columns = sorted(
        REQUIRED_COLUMNS.difference(df.columns)
    )

    if missing_columns:
        raise CustomerSignalError(
            f"Missing required columns: {missing_columns}"
        )


def _validate_customer_mart(df: pd.DataFrame) -> None:
    """Validate the customer-month grain and core metrics."""

    if df.empty:
        raise CustomerSignalError(
            "Customer performance mart is empty."
        )

    if df[
        ["month", "customer_unique_id"]
    ].duplicated().any():
        raise CustomerSignalError(
            "Customer performance mart violates "
            "month × customer grain."
        )

    if df["customer_unique_id"].isna().any():
        raise CustomerSignalError(
            "customer_unique_id contains null values."
        )

    if df["month"].isna().any():
        raise CustomerSignalError(
            "month contains null values."
        )

    if df["order_count"].le(0).any():
        raise CustomerSignalError(
            "order_count contains non-positive values."
        )

    if df["item_line_count"].le(0).any():
        raise CustomerSignalError(
            "item_line_count contains non-positive values."
        )

    if df["revenue"].lt(0).any():
        raise CustomerSignalError(
            "revenue contains negative values."
        )

    if df["freight_value"].lt(0).any():
        raise CustomerSignalError(
            "freight_value contains negative values."
        )


def _derive_customer_lifecycle(
    data: pd.DataFrame,
    dataset_end: pd.Timestamp,
) -> pd.DataFrame:
    """Derive customer lifecycle from observed purchase history."""

    customer_summary = (
        data.groupby(
            "customer_unique_id",
            as_index=False,
        )
        .agg(
            active_months=("month", "nunique"),
            last_purchase_month=("month", "max"),
        )
    )

    customer_summary["months_since_last_purchase"] = (
        (
            dataset_end.year
            - customer_summary[
                "last_purchase_month"
            ].dt.year
        )
        * 12
        + (
            dataset_end.month
            - customer_summary[
                "last_purchase_month"
            ].dt.month
        )
    )

    customer_summary["customer_lifecycle"] = "ONE_TIME"

    repeat_mask = (
        customer_summary["active_months"] > 1
    )

    customer_summary.loc[
        repeat_mask
        & customer_summary[
            "months_since_last_purchase"
        ].lt(12),
        "customer_lifecycle",
    ] = "REPEAT_ACTIVE"

    customer_summary.loc[
        repeat_mask
        & customer_summary[
            "months_since_last_purchase"
        ].ge(12),
        "customer_lifecycle",
    ] = "DORMANT"

    return customer_summary[
        [
            "customer_unique_id",
            "customer_lifecycle",
        ]
    ]


def _derive_customer_evidence(
    data: pd.DataFrame,
) -> pd.DataFrame:
    """Derive customer-level supporting evidence attributes."""

    customer_evidence = (
        data.groupby(
            "customer_unique_id",
            as_index=False,
        )
        .agg(
            active_months=("month", "nunique"),
            total_orders=("order_count", "sum"),
            lifetime_revenue=("revenue", "sum"),
        )
    )

    return customer_evidence


def _derive_value_segment(
    data: pd.DataFrame,
) -> pd.DataFrame:
    """Segment customers by lifetime revenue percentile."""

    customer_revenue = (
        data.groupby(
            "customer_unique_id",
            as_index=False,
        )
        .agg(
            lifetime_revenue=("revenue", "sum"),
        )
    )

    median_revenue = customer_revenue[
        "lifetime_revenue"
    ].quantile(0.50)

    p95_revenue = customer_revenue[
        "lifetime_revenue"
    ].quantile(0.95)

    customer_revenue["value_segment"] = "MID_VALUE"

    customer_revenue.loc[
        customer_revenue[
            "lifetime_revenue"
        ].le(median_revenue),
        "value_segment",
    ] = "LOW_VALUE"

    customer_revenue.loc[
        customer_revenue[
            "lifetime_revenue"
        ].ge(p95_revenue),
        "value_segment",
    ] = "HIGH_VALUE"

    return customer_revenue[
        [
            "customer_unique_id",
            "value_segment",
        ]
    ]


def _derive_transition_evidence(
    data: pd.DataFrame,
) -> pd.DataFrame:
    """
    Derive longitudinal transition evidence per customer.

    A transition exists only when two consecutive observed customer
    records are exactly one calendar month apart.

    This prevents a gap in observation history from being treated
    as a behavioral transition.
    """

    ordered = data.sort_values(
        [
            "customer_unique_id",
            "month",
        ]
    ).copy()

    ordered["previous_month"] = (
        ordered.groupby("customer_unique_id")["month"]
        .shift(1)
    )

    ordered["previous_revenue"] = (
        ordered.groupby("customer_unique_id")["revenue"]
        .shift(1)
    )

    ordered["month_gap"] = (
        (
            ordered["month"].dt.year
            - ordered["previous_month"].dt.year
        )
        * 12
        + (
            ordered["month"].dt.month
            - ordered["previous_month"].dt.month
        )
    )

    transition_mask = (
        ordered["previous_revenue"].notna()
        & ordered["month_gap"].eq(1)
    )

    transitions = ordered.loc[
        transition_mask
    ].copy()

    transitions["direction"] = 0

    transitions.loc[
        transitions["revenue"]
        > transitions["previous_revenue"],
        "direction",
    ] = 1

    transitions.loc[
        transitions["revenue"]
        < transitions["previous_revenue"],
        "direction",
    ] = -1

    transition_summary = (
        transitions.groupby(
            "customer_unique_id",
            as_index=False,
        )
        .agg(
            transition_count=(
                "direction",
                "size",
            ),
            positive_transitions=(
                "direction",
                lambda values: (values > 0).sum(),
            ),
            negative_transitions=(
                "direction",
                lambda values: (values < 0).sum(),
            ),
        )
    )

    transition_summary["direction_consistency"] = (
        transition_summary[
            [
                "positive_transitions",
                "negative_transitions",
            ]
        ].max(axis=1)
        / transition_summary["transition_count"]
    )

    transition_summary["behavior_direction"] = "STABLE"

    transition_summary.loc[
        transition_summary["positive_transitions"]
        > transition_summary["negative_transitions"],
        "behavior_direction",
    ] = "INCREASING"

    transition_summary.loc[
        transition_summary["negative_transitions"]
        > transition_summary["positive_transitions"],
        "behavior_direction",
    ] = "DECREASING"

    transition_summary["evidence_level"] = "LOW"

    transition_summary.loc[
        transition_summary["transition_count"].eq(2),
        "evidence_level",
    ] = "MEDIUM"

    transition_summary.loc[
        transition_summary["transition_count"].ge(3),
        "evidence_level",
    ] = "HIGH"

    return transition_summary[
        [
            "customer_unique_id",
            "transition_count",
            "direction_consistency",
            "behavior_direction",
            "evidence_level",
        ]
    ]


def build_customer_signals(
    marts_dir: str | Path,
) -> pd.DataFrame:
    """
    Build the governed customer signal dataset.

    Parameters
    ----------
    marts_dir:
        Directory containing the governed analytical marts.

    Returns
    -------
    pd.DataFrame
        One row per customer containing the defined signal contract.

    Implemented dimensions
    ----------------------
    - customer lifecycle
    - value segment
    - supporting customer evidence
    - transition count
    - direction consistency
    - behavior direction
    - evidence level
    - signal confidence
    """

    mart = _load_customer_mart(marts_dir)

    validate_input(mart)

    data = mart.copy()

    data["month"] = pd.to_datetime(
        data["month"],
        errors="raise",
    )

    data["first_purchase_month"] = pd.to_datetime(
        data["first_purchase_month"],
        errors="raise",
    )

    _validate_customer_mart(data)

    dataset_end = data["month"].max()

    customer_count = data[
        "customer_unique_id"
    ].nunique()

    result = pd.DataFrame(
        {
            "customer_unique_id": data[
                "customer_unique_id"
            ].drop_duplicates()
        }
    )

    result["signal_date"] = dataset_end

    lifecycle = _derive_customer_lifecycle(
        data,
        dataset_end,
    )

    result = result.merge(
        lifecycle,
        on="customer_unique_id",
        how="left",
        validate="one_to_one",
    )

    evidence = _derive_customer_evidence(data)

    result = result.merge(
        evidence,
        on="customer_unique_id",
        how="left",
        validate="one_to_one",
    )

    value_segment = _derive_value_segment(data)

    result = result.merge(
        value_segment,
        on="customer_unique_id",
        how="left",
        validate="one_to_one",
    )

    transition_evidence = _derive_transition_evidence(
        data
    )

    result = result.merge(
        transition_evidence,
        on="customer_unique_id",
        how="left",
        validate="one_to_one",
    )

    result["transition_count"] = (
        result["transition_count"]
        .fillna(0)
        .astype(int)
    )

    result["behavior_direction"] = (
        result["behavior_direction"]
        .fillna("INSUFFICIENT_EVIDENCE")
    )

    result["evidence_level"] = (
        result["evidence_level"]
        .fillna("NONE")
    )

    result["signal_confidence"] = "NONE"

    result.loc[
        result["evidence_level"].eq("LOW"),
        "signal_confidence",
    ] = "LOW"

    result.loc[
        result["evidence_level"].eq("MEDIUM")
        & result["direction_consistency"].ge(0.75),
        "signal_confidence",
    ] = "MEDIUM"

    result.loc[
        result["evidence_level"].eq("HIGH")
        & result["direction_consistency"].ge(0.75),
        "signal_confidence",
    ] = "HIGH"

    result.loc[
        result["evidence_level"].eq("MEDIUM")
        & result["direction_consistency"].lt(0.75),
        "signal_confidence",
    ] = "LOW"

    result.loc[
        result["evidence_level"].eq("HIGH")
        & result["direction_consistency"].lt(0.75),
        "signal_confidence",
    ] = "MEDIUM"

    if len(result) != customer_count:
        raise CustomerSignalError(
            "Customer signal grain validation failed."
        )

    if result[
        "customer_unique_id"
    ].duplicated().any():
        raise CustomerSignalError(
            "Customer signal output contains "
            "duplicate customers."
        )

    if result[
        "customer_lifecycle"
    ].isna().any():
        raise CustomerSignalError(
            "Customer lifecycle could not be derived "
            "for all customers."
        )

    evidence_columns = [
        "active_months",
        "total_orders",
        "lifetime_revenue",
    ]

    if result[evidence_columns].isna().any().any():
        raise CustomerSignalError(
            "Customer evidence could not be derived "
            "for all customers."
        )

    if result["active_months"].le(0).any():
        raise CustomerSignalError(
            "active_months contains non-positive values."
        )

    if result["total_orders"].le(0).any():
        raise CustomerSignalError(
            "total_orders contains non-positive values."
        )

    if result["lifetime_revenue"].lt(0).any():
        raise CustomerSignalError(
            "lifetime_revenue contains negative values."
        )

    if result["transition_count"].lt(0).any():
        raise CustomerSignalError(
            "transition_count contains negative values."
        )

    valid_lifecycle_values = {
        "ONE_TIME",
        "REPEAT_ACTIVE",
        "DORMANT",
    }

    invalid_lifecycle = set(
        result[
            "customer_lifecycle"
        ].dropna().unique()
    ).difference(valid_lifecycle_values)

    if invalid_lifecycle:
        raise CustomerSignalError(
            "Invalid customer lifecycle values: "
            f"{sorted(invalid_lifecycle)}"
        )

    valid_value_segments = {
        "LOW_VALUE",
        "MID_VALUE",
        "HIGH_VALUE",
    }

    if result["value_segment"].isna().any():
        raise CustomerSignalError(
            "Value segment could not be derived "
            "for all customers."
        )

    invalid_value_segments = set(
        result["value_segment"].unique()
    ).difference(valid_value_segments)

    if invalid_value_segments:
        raise CustomerSignalError(
            "Invalid value segment values: "
            f"{sorted(invalid_value_segments)}"
        )

    valid_behavior_values = {
        "INCREASING",
        "DECREASING",
        "STABLE",
        "INSUFFICIENT_EVIDENCE",
    }

    invalid_behavior = set(
        result["behavior_direction"].unique()
    ).difference(valid_behavior_values)

    if invalid_behavior:
        raise CustomerSignalError(
            "Invalid behavior direction values: "
            f"{sorted(invalid_behavior)}"
        )

    valid_evidence_values = {
        "NONE",
        "LOW",
        "MEDIUM",
        "HIGH",
    }

    invalid_evidence = set(
        result["evidence_level"].unique()
    ).difference(valid_evidence_values)

    if invalid_evidence:
        raise CustomerSignalError(
            "Invalid evidence level values: "
            f"{sorted(invalid_evidence)}"
        )

    valid_confidence_values = {
        "NONE",
        "LOW",
        "MEDIUM",
        "HIGH",
    }

    invalid_confidence = set(
        result["signal_confidence"].unique()
    ).difference(valid_confidence_values)

    if invalid_confidence:
        raise CustomerSignalError(
            "Invalid signal confidence values: "
            f"{sorted(invalid_confidence)}"
        )

    if (
        result["direction_consistency"]
        .dropna()
        .lt(0)
        .any()
    ):
        raise CustomerSignalError(
            "direction_consistency contains values "
            "below 0."
        )

    if (
        result["direction_consistency"]
        .dropna()
        .gt(1)
        .any()
    ):
        raise CustomerSignalError(
            "direction_consistency contains values "
            "above 1."
        )

    if (
        result.loc[
            result["transition_count"] == 0,
            "direction_consistency",
        ]
        .notna()
        .any()
    ):
        raise CustomerSignalError(
            "Customers with zero transitions must have "
            "null direction_consistency."
        )

    if (
        result.loc[
            result["transition_count"] == 0,
            "behavior_direction",
        ]
        .ne("INSUFFICIENT_EVIDENCE")
        .any()
    ):
        raise CustomerSignalError(
            "Customers with zero transitions must have "
            "INSUFFICIENT_EVIDENCE behavior direction."
        )

    if (
        result.loc[
            result["transition_count"] == 0,
            "evidence_level",
        ]
        .ne("NONE")
        .any()
    ):
        raise CustomerSignalError(
            "Customers with zero transitions must have "
            "NONE evidence level."
        )

    if (
        result.loc[
            result["transition_count"] == 0,
            "signal_confidence",
        ]
        .ne("NONE")
        .any()
    ):
        raise CustomerSignalError(
            "Customers with zero transitions must have "
            "NONE signal confidence."
        )

    if (
        result.loc[
            result["transition_count"] == 1,
            "evidence_level",
        ]
        .ne("LOW")
        .any()
    ):
        raise CustomerSignalError(
            "Customers with one transition must have "
            "LOW evidence level."
        )

    if (
        result.loc[
            result["transition_count"] == 1,
            "signal_confidence",
        ]
        .ne("LOW")
        .any()
    ):
        raise CustomerSignalError(
            "Customers with one transition must have "
            "LOW signal confidence."
        )

    if (
        result.loc[
            result["transition_count"] == 2,
            "evidence_level",
        ]
        .ne("MEDIUM")
        .any()
    ):
        raise CustomerSignalError(
            "Customers with two transitions must have "
            "MEDIUM evidence level."
        )

    if (
        result.loc[
            result["transition_count"] >= 3,
            "evidence_level",
        ]
        .ne("HIGH")
        .any()
    ):
        raise CustomerSignalError(
            "Customers with three or more transitions "
            "must have HIGH evidence level."
        )

    return result[OUTPUT_COLUMNS].copy()