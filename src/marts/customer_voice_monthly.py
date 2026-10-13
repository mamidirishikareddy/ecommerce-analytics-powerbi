from __future__ import annotations

from pathlib import Path

import pandas as pd


GRAIN_COLUMNS = ["month"]

OUTPUT_COLUMNS = [
    "month",
    "review_count",
    "text_review_count",
    "text_coverage_pct",
    "average_review_score",
    "positive_review_rate_pct",
    "negative_review_rate_pct",
    "neutral_review_rate_pct",
    "complaint_volume",
    "complaint_rate_pct",
]


def build_customer_voice_monthly(
    customer_voice_path: str | Path,
    complaint_path: str | Path,
) -> pd.DataFrame:
    """
    Build the governed Customer Voice Monthly Analytical Mart.

    Grain:
        One row per calendar month.

    Sources:
        customer_voice.parquet
        customer_voice_complaints.parquet

    The function performs aggregation only.
    It does not:
        - deduplicate reviews
        - alter review scores
        - reclassify sentiment
        - infer complaints for textless reviews
        - perform causal attribution
    """

    customer_voice_path = Path(customer_voice_path)
    complaint_path = Path(complaint_path)

    voice = pd.read_parquet(customer_voice_path)
    complaints = pd.read_parquet(complaint_path)

    required_voice_columns = {
        "review_id",
        "order_id",
        "review_score",
        "review_creation_date",
        "has_review_text",
        "sentiment",
    }

    required_complaint_columns = {
        "review_id",
        "order_id",
        "complaint_flag",
    }

    missing_voice = required_voice_columns - set(voice.columns)
    missing_complaint = required_complaint_columns - set(complaints.columns)

    if missing_voice:
        raise ValueError(
            f"Customer Voice input missing columns: {sorted(missing_voice)}"
        )

    if missing_complaint:
        raise ValueError(
            f"Complaint input missing columns: {sorted(missing_complaint)}"
        )

    # Preserve the governed review grain.
    voice_keys = ["review_id", "order_id"]

    if voice.duplicated(voice_keys).any():
        raise ValueError(
            "Customer Voice source violates review_id + order_id grain."
        )

    if complaints.duplicated(voice_keys).any():
        raise ValueError(
            "Complaint source violates review_id + order_id grain."
        )

    # Complaint enrichment must be one-to-one.
    complaint_columns = voice_keys + ["complaint_flag"]

    enriched = voice.merge(
    complaints[complaint_columns],
    on=voice_keys,
    how="left",
    validate="one_to_one",
)

# Complaint detection was intentionally performed only for
# text-bearing reviews. Textless reviews therefore remain unmatched
# and must retain a null complaint_flag.

    # Calendar month.
    enriched["review_creation_date"] = pd.to_datetime(
    enriched["review_creation_date"],
    errors="raise",
)

    enriched["month"] = (
    enriched["review_creation_date"]
    .dt.to_period("M")
    .dt.to_timestamp()
)

    # Text-bearing denominator.
    text_mask = enriched["has_review_text"].fillna(False)

    enriched["text_review"] = text_mask.astype("int64")

    enriched["positive_review"] = (
        text_mask & enriched["sentiment"].eq("positive")
    ).astype("int64")

    enriched["negative_review"] = (
        text_mask & enriched["sentiment"].eq("negative")
    ).astype("int64")

    enriched["neutral_review"] = (
        text_mask & enriched["sentiment"].eq("neutral")
    ).astype("int64")

    # Complaint detection is only evaluated where the complaint pipeline
    # produced a classification.
    enriched["complaint_detected"] = (
        enriched["complaint_flag"].eq(True)
    ).astype("int64")

    monthly = (
        enriched.groupby("month", as_index=False)
        .agg(
            review_count=("review_id", "size"),
            text_review_count=("text_review", "sum"),
            average_review_score=("review_score", "mean"),
            positive_review_count=("positive_review", "sum"),
            negative_review_count=("negative_review", "sum"),
            neutral_review_count=("neutral_review", "sum"),
            complaint_volume=("complaint_detected", "sum"),
        )
    )

    # Derived percentages.
    monthly["text_coverage_pct"] = (
        monthly["text_review_count"]
        / monthly["review_count"]
        * 100
    )

    monthly["positive_review_rate_pct"] = (
        monthly["positive_review_count"]
        / monthly["text_review_count"]
        * 100
    )

    monthly["negative_review_rate_pct"] = (
        monthly["negative_review_count"]
        / monthly["text_review_count"]
        * 100
    )

    monthly["neutral_review_rate_pct"] = (
        monthly["neutral_review_count"]
        / monthly["text_review_count"]
        * 100
    )

    monthly["complaint_rate_pct"] = (
        monthly["complaint_volume"]
        / monthly["text_review_count"]
        * 100
    )

    # Avoid divide-by-zero artifacts for months without text-bearing reviews.
    rate_columns = [
        "positive_review_rate_pct",
        "negative_review_rate_pct",
        "neutral_review_rate_pct",
        "complaint_rate_pct",
    ]

    monthly.loc[
        monthly["text_review_count"].eq(0),
        rate_columns,
    ] = pd.NA

    # Keep only the governed contract output.
    result = monthly[
        OUTPUT_COLUMNS
    ].copy()

    # Stable chronological ordering.
    result = result.sort_values("month").reset_index(drop=True)

    return result