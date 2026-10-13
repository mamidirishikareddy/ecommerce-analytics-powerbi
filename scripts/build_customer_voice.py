from __future__ import annotations

from pathlib import Path

import pandas as pd


# =============================================================================
# PATHS
# =============================================================================

INPUT_PATH = Path(
    "data/processed/nlp/customer_voice_input.parquet"
)

CHECKPOINT_DIR = Path(
    "data/processed/nlp/sentiment_checkpoints"
)

OUTPUT_DIR = Path(
    "data/processed/nlp"
)

OUTPUT_PATH = OUTPUT_DIR / (
    "customer_voice.parquet"
)


# =============================================================================
# EXPECTED CONTRACT
# =============================================================================

EXPECTED_TOTAL_REVIEWS = 99_224
EXPECTED_TEXT_BEARING_REVIEWS = 42_687
EXPECTED_TEXTLESS_REVIEWS = 56_537

ALLOWED_SENTIMENTS = {
    "positive",
    "negative",
    "neutral",
}


INPUT_REQUIRED_COLUMNS = [
    "review_id",
    "order_id",
    "review_score",
    "review_creation_date",
    "review_answer_timestamp",
    "customer_id",
    "review_comment_title",
    "review_comment_message",
    "review_text",
    "has_review_text",
]


PREDICTION_REQUIRED_COLUMNS = [
    "review_id",
    "order_id",
    "review_key",
    "sentiment",
    "sentiment_confidence",
]


FINAL_COLUMNS = [
    "review_id",
    "order_id",
    "review_score",
    "review_creation_date",
    "review_answer_timestamp",
    "customer_id",
    "review_comment_title",
    "review_comment_message",
    "review_text",
    "has_review_text",
    "sentiment",
    "sentiment_confidence",
]


# =============================================================================
# EXCEPTION
# =============================================================================


class CustomerVoiceBuildError(Exception):
    """Raised when the governed Customer Voice build fails."""


# =============================================================================
# INPUT LOADING
# =============================================================================


def load_customer_voice_input() -> pd.DataFrame:
    """Load and validate the governed Customer Voice NLP input."""

    if not INPUT_PATH.exists():
        raise CustomerVoiceBuildError(
            f"Customer Voice input not found: {INPUT_PATH}"
        )

    input_df = pd.read_parquet(
        INPUT_PATH
    )

    missing_columns = [
        column
        for column in INPUT_REQUIRED_COLUMNS
        if column not in input_df.columns
    ]

    if missing_columns:
        raise CustomerVoiceBuildError(
            "Customer Voice input missing columns: "
            f"{missing_columns}"
        )

    input_df = input_df[
        INPUT_REQUIRED_COLUMNS
    ].copy()

    # -------------------------------------------------------------------------
    # Row-count validation
    # -------------------------------------------------------------------------

    if len(input_df) != EXPECTED_TOTAL_REVIEWS:
        raise CustomerVoiceBuildError(
            "Unexpected Customer Voice input row count: "
            f"{len(input_df):,} "
            f"(expected {EXPECTED_TOTAL_REVIEWS:,})."
        )

    # -------------------------------------------------------------------------
    # Review grain validation
    # -------------------------------------------------------------------------

    duplicate_count = int(
        input_df.duplicated(
            subset=[
                "review_id",
                "order_id",
            ]
        ).sum()
    )

    if duplicate_count:
        raise CustomerVoiceBuildError(
            "Customer Voice input contains "
            f"{duplicate_count:,} duplicate "
            "review_id + order_id keys."
        )

    # -------------------------------------------------------------------------
    # Required identifier validation
    # -------------------------------------------------------------------------

    if input_df["review_id"].isna().any():
        raise CustomerVoiceBuildError(
            "Customer Voice input contains NULL review_id."
        )

    if input_df["order_id"].isna().any():
        raise CustomerVoiceBuildError(
            "Customer Voice input contains NULL order_id."
        )

    if input_df["customer_id"].isna().any():
        raise CustomerVoiceBuildError(
            "Customer Voice input contains NULL customer_id."
        )

    # -------------------------------------------------------------------------
    # Text population validation
    # -------------------------------------------------------------------------

    text_bearing_count = int(
        input_df["has_review_text"].sum()
    )

    textless_count = int(
        (~input_df["has_review_text"]).sum()
    )

    if text_bearing_count != EXPECTED_TEXT_BEARING_REVIEWS:
        raise CustomerVoiceBuildError(
            "Unexpected text-bearing review count: "
            f"{text_bearing_count:,} "
            f"(expected {EXPECTED_TEXT_BEARING_REVIEWS:,})."
        )

    if textless_count != EXPECTED_TEXTLESS_REVIEWS:
        raise CustomerVoiceBuildError(
            "Unexpected textless review count: "
            f"{textless_count:,} "
            f"(expected {EXPECTED_TEXTLESS_REVIEWS:,})."
        )

    # -------------------------------------------------------------------------
    # Text flag consistency
    # -------------------------------------------------------------------------

    empty_text_with_true_flag = int(
        (
            input_df["has_review_text"]
            & input_df["review_text"].fillna("").eq("")
        ).sum()
    )

    nonempty_text_with_false_flag = int(
        (
            (~input_df["has_review_text"])
            & input_df["review_text"].fillna("").ne("")
        ).sum()
    )

    if empty_text_with_true_flag:
        raise CustomerVoiceBuildError(
            "Customer Voice input contains "
            f"{empty_text_with_true_flag:,} empty review_text "
            "values incorrectly marked as text-bearing."
        )

    if nonempty_text_with_false_flag:
        raise CustomerVoiceBuildError(
            "Customer Voice input contains "
            f"{nonempty_text_with_false_flag:,} non-empty review_text "
            "values incorrectly marked as textless."
        )

    # -------------------------------------------------------------------------
    # Review key used by sentiment checkpoints
    # -------------------------------------------------------------------------

    input_df["review_key"] = (
        input_df["review_id"].astype(str)
        + "||"
        + input_df["order_id"].astype(str)
    )

    duplicate_review_keys = int(
        input_df["review_key"].duplicated().sum()
    )

    if duplicate_review_keys:
        raise CustomerVoiceBuildError(
            "Customer Voice input contains "
            f"{duplicate_review_keys:,} duplicate review_key values."
        )

    return input_df


# =============================================================================
# SENTIMENT CHECKPOINT LOADING
# =============================================================================


def load_sentiment_checkpoints() -> pd.DataFrame:
    """
    Load all sentiment checkpoints and validate global prediction integrity.
    """

    if not CHECKPOINT_DIR.exists():
        raise CustomerVoiceBuildError(
            f"Sentiment checkpoint directory not found: "
            f"{CHECKPOINT_DIR}"
        )

    checkpoint_paths = sorted(
        CHECKPOINT_DIR.glob(
            "checkpoint_*.parquet"
        )
    )

    if not checkpoint_paths:
        raise CustomerVoiceBuildError(
            "No sentiment checkpoint files found."
        )

    frames: list[pd.DataFrame] = []

    for checkpoint_path in checkpoint_paths:

        checkpoint = pd.read_parquet(
            checkpoint_path
        )

        missing_columns = [
            column
            for column in PREDICTION_REQUIRED_COLUMNS
            if column not in checkpoint.columns
        ]

        if missing_columns:
            raise CustomerVoiceBuildError(
                f"Checkpoint {checkpoint_path.name} "
                "missing columns: "
                f"{missing_columns}"
            )

        checkpoint = checkpoint[
            PREDICTION_REQUIRED_COLUMNS
        ].copy()

        frames.append(
            checkpoint
        )

    predictions = pd.concat(
        frames,
        ignore_index=True,
    )

    # -------------------------------------------------------------------------
    # Prediction row count
    # -------------------------------------------------------------------------

    if len(predictions) != EXPECTED_TEXT_BEARING_REVIEWS:
        raise CustomerVoiceBuildError(
            "Unexpected sentiment prediction count: "
            f"{len(predictions):,} "
            f"(expected {EXPECTED_TEXT_BEARING_REVIEWS:,})."
        )

    # -------------------------------------------------------------------------
    # Prediction key validation
    # -------------------------------------------------------------------------

    if predictions["review_key"].isna().any():
        raise CustomerVoiceBuildError(
            "Sentiment predictions contain NULL review_key."
        )

    duplicate_prediction_keys = int(
        predictions["review_key"].duplicated().sum()
    )

    if duplicate_prediction_keys:
        raise CustomerVoiceBuildError(
            "Sentiment checkpoints contain "
            f"{duplicate_prediction_keys:,} duplicate review keys."
        )

    # -------------------------------------------------------------------------
    # Required prediction fields
    # -------------------------------------------------------------------------

    null_sentiments = int(
        predictions["sentiment"].isna().sum()
    )

    null_confidences = int(
        predictions["sentiment_confidence"].isna().sum()
    )

    if null_sentiments:
        raise CustomerVoiceBuildError(
            "Sentiment predictions contain "
            f"{null_sentiments:,} NULL sentiment values."
        )

    if null_confidences:
        raise CustomerVoiceBuildError(
            "Sentiment predictions contain "
            f"{null_confidences:,} NULL confidence values."
        )

    # -------------------------------------------------------------------------
    # Sentiment label validation
    # -------------------------------------------------------------------------

    observed_sentiments = set(
        predictions["sentiment"].astype(str).unique()
    )

    unexpected_sentiments = (
        observed_sentiments
        - ALLOWED_SENTIMENTS
    )

    if unexpected_sentiments:
        raise CustomerVoiceBuildError(
            "Unexpected sentiment labels found: "
            f"{sorted(unexpected_sentiments)}"
        )

    # -------------------------------------------------------------------------
    # Confidence validation
    # -------------------------------------------------------------------------

    invalid_confidence = (
        (predictions["sentiment_confidence"] < 0)
        | (predictions["sentiment_confidence"] > 1)
    )

    invalid_confidence_count = int(
        invalid_confidence.sum()
    )

    if invalid_confidence_count:
        raise CustomerVoiceBuildError(
            "Sentiment predictions contain "
            f"{invalid_confidence_count:,} confidence values "
            "outside the [0, 1] range."
        )

    return predictions


# =============================================================================
# BUILD FINAL CUSTOMER VOICE DATASET
# =============================================================================


def build_customer_voice(
    input_df: pd.DataFrame,
    predictions: pd.DataFrame,
) -> pd.DataFrame:
    """
    Merge governed review input with sentiment predictions.

    Important population rule:

        Text-bearing reviews
            -> sentiment prediction REQUIRED

        Textless reviews
            -> sentiment prediction NOT expected
            -> sentiment remains NULL

    The final dataset retains all review records.
    """

    input_keys = set(
        input_df["review_key"]
    )

    prediction_keys = set(
        predictions["review_key"]
    )

    input_only_keys = (
        input_keys
        - prediction_keys
    )

    prediction_only_keys = (
        prediction_keys
        - input_keys
    )

    print()
    print(
        "MERGING CUSTOMER VOICE + SENTIMENT"
    )
    print(
        "----------------------------------"
    )

    print(
        f"Input-only review keys : "
        f"{len(input_only_keys):,}"
    )

    print(
        f"Prediction-only keys   : "
        f"{len(prediction_only_keys):,}"
    )

    # -------------------------------------------------------------------------
    # Input-only keys are expected for TEXTLESS reviews.
    #
    # Only text-bearing reviews are required to have a prediction.
    # -------------------------------------------------------------------------

    text_bearing_keys = set(
        input_df.loc[
            input_df["has_review_text"],
            "review_key",
        ]
    )

    input_only_text_bearing_keys = (
        input_only_keys
        & text_bearing_keys
    )

    if input_only_text_bearing_keys:
        raise CustomerVoiceBuildError(
            "Text-bearing reviews are missing sentiment "
            "predictions: "
            f"{len(input_only_text_bearing_keys):,}"
        )

    # -------------------------------------------------------------------------
    # Every prediction must correspond to a valid input review.
    # -------------------------------------------------------------------------

    if prediction_only_keys:
        raise CustomerVoiceBuildError(
            "Sentiment predictions contain unexpected "
            f"review keys: {len(prediction_only_keys):,}"
        )

    # -------------------------------------------------------------------------
    # Merge
    # -------------------------------------------------------------------------

    final_df = input_df.merge(
        predictions[
            [
                "review_key",
                "sentiment",
                "sentiment_confidence",
            ]
        ],
        on="review_key",
        how="left",
        validate="one_to_one",
        indicator=True,
    )

    # -------------------------------------------------------------------------
    # Merge integrity
    # -------------------------------------------------------------------------

    unexpected_merge_status = int(
        (
            final_df["_merge"]
            != "both"
        ).sum()
    )

    # Textless reviews are allowed to be left-only.
    # Therefore the merge indicator is checked explicitly below.
    if unexpected_merge_status:

        left_only = int(
            (
                final_df["_merge"]
                == "left_only"
            ).sum()
        )

        right_only = int(
            (
                final_df["_merge"]
                == "right_only"
            ).sum()
        )

        # Right-only should already be impossible because of the
        # prediction_only_keys check.
        if right_only:
            raise CustomerVoiceBuildError(
                "Unexpected right-only rows after sentiment merge: "
                f"{right_only:,}"
            )

        # Every left-only row must be textless.
        left_only_text_bearing = int(
            (
                (final_df["_merge"] == "left_only")
                & final_df["has_review_text"]
            ).sum()
        )

        if left_only_text_bearing:
            raise CustomerVoiceBuildError(
                "Text-bearing reviews remain unmatched after merge: "
                f"{left_only_text_bearing:,}"
            )

        # The expected left-only population is exactly the textless
        # review population.
        if left_only != EXPECTED_TEXTLESS_REVIEWS:
            raise CustomerVoiceBuildError(
                "Unexpected number of unmatched textless reviews: "
                f"{left_only:,} "
                f"(expected {EXPECTED_TEXTLESS_REVIEWS:,})."
            )

    final_df = final_df.drop(
        columns=["_merge"]
    )

    # -------------------------------------------------------------------------
    # Text-bearing reviews MUST have sentiment
    # -------------------------------------------------------------------------

    missing_text_sentiment = int(
        (
            final_df["has_review_text"]
            & final_df["sentiment"].isna()
        ).sum()
    )

    if missing_text_sentiment:
        raise CustomerVoiceBuildError(
            "Text-bearing reviews missing sentiment: "
            f"{missing_text_sentiment:,}"
        )

    # -------------------------------------------------------------------------
    # Textless reviews MUST NOT have sentiment
    # -------------------------------------------------------------------------

    textless_with_sentiment = int(
        (
            (~final_df["has_review_text"])
            & final_df["sentiment"].notna()
        ).sum()
    )

    if textless_with_sentiment:
        raise CustomerVoiceBuildError(
            "Textless reviews incorrectly received sentiment: "
            f"{textless_with_sentiment:,}"
        )

    textless_with_confidence = int(
        (
            (~final_df["has_review_text"])
            & final_df["sentiment_confidence"].notna()
        ).sum()
    )

    if textless_with_confidence:
        raise CustomerVoiceBuildError(
            "Textless reviews incorrectly received sentiment "
            "confidence values: "
            f"{textless_with_confidence:,}"
        )

    # -------------------------------------------------------------------------
    # Final row count
    # -------------------------------------------------------------------------

    if len(final_df) != EXPECTED_TOTAL_REVIEWS:
        raise CustomerVoiceBuildError(
            "Final Customer Voice row count changed unexpectedly: "
            f"{len(final_df):,} "
            f"(expected {EXPECTED_TOTAL_REVIEWS:,})."
        )

    # -------------------------------------------------------------------------
    # Final review grain
    # -------------------------------------------------------------------------

    duplicate_final_keys = int(
        final_df.duplicated(
            subset=[
                "review_id",
                "order_id",
            ]
        ).sum()
    )

    if duplicate_final_keys:
        raise CustomerVoiceBuildError(
            "Final Customer Voice dataset contains "
            f"{duplicate_final_keys:,} duplicate "
            "review_id + order_id keys."
        )

    # -------------------------------------------------------------------------
    # Final customer ID validation
    # -------------------------------------------------------------------------

    if final_df["customer_id"].isna().any():
        raise CustomerVoiceBuildError(
            "Final Customer Voice dataset contains NULL customer_id."
        )

    # -------------------------------------------------------------------------
    # Final column selection
    # -------------------------------------------------------------------------

    final_df = final_df[
        FINAL_COLUMNS
    ].copy()

    return final_df


# =============================================================================
# FINAL VALIDATION
# =============================================================================


def validate_final_customer_voice(
    final_df: pd.DataFrame,
) -> None:
    """Run the final governed Customer Voice validation gate."""

    # -------------------------------------------------------------------------
    # Schema
    # -------------------------------------------------------------------------

    missing_columns = [
        column
        for column in FINAL_COLUMNS
        if column not in final_df.columns
    ]

    if missing_columns:
        raise CustomerVoiceBuildError(
            "Final Customer Voice dataset missing columns: "
            f"{missing_columns}"
        )

    if list(final_df.columns) != FINAL_COLUMNS:
        raise CustomerVoiceBuildError(
            "Final Customer Voice column order does not "
            "match the governed contract."
        )

    # -------------------------------------------------------------------------
    # Row count
    # -------------------------------------------------------------------------

    if len(final_df) != EXPECTED_TOTAL_REVIEWS:
        raise CustomerVoiceBuildError(
            "Final Customer Voice row count is incorrect: "
            f"{len(final_df):,}"
        )

    # -------------------------------------------------------------------------
    # Grain
    # -------------------------------------------------------------------------

    duplicate_count = int(
        final_df.duplicated(
            subset=[
                "review_id",
                "order_id",
            ]
        ).sum()
    )

    if duplicate_count:
        raise CustomerVoiceBuildError(
            "Final Customer Voice dataset contains "
            f"{duplicate_count:,} duplicate review keys."
        )

    # -------------------------------------------------------------------------
    # Text population
    # -------------------------------------------------------------------------

    text_bearing_count = int(
        final_df["has_review_text"].sum()
    )

    textless_count = int(
        (~final_df["has_review_text"]).sum()
    )

    if text_bearing_count != EXPECTED_TEXT_BEARING_REVIEWS:
        raise CustomerVoiceBuildError(
            "Final text-bearing count is incorrect: "
            f"{text_bearing_count:,}"
        )

    if textless_count != EXPECTED_TEXTLESS_REVIEWS:
        raise CustomerVoiceBuildError(
            "Final textless count is incorrect: "
            f"{textless_count:,}"
        )

    # -------------------------------------------------------------------------
    # Sentiment population
    # -------------------------------------------------------------------------

    text_bearing_missing_sentiment = int(
        (
            final_df["has_review_text"]
            & final_df["sentiment"].isna()
        ).sum()
    )

    if text_bearing_missing_sentiment:
        raise CustomerVoiceBuildError(
            "Final dataset contains "
            f"{text_bearing_missing_sentiment:,} text-bearing "
            "reviews without sentiment."
        )

    textless_sentiment_count = int(
        (
            (~final_df["has_review_text"])
            & final_df["sentiment"].notna()
        ).sum()
    )

    if textless_sentiment_count:
        raise CustomerVoiceBuildError(
            "Final dataset contains "
            f"{textless_sentiment_count:,} textless reviews "
            "with sentiment."
        )

    # -------------------------------------------------------------------------
    # Confidence population
    # -------------------------------------------------------------------------

    text_bearing_missing_confidence = int(
        (
            final_df["has_review_text"]
            & final_df["sentiment_confidence"].isna()
        ).sum()
    )

    if text_bearing_missing_confidence:
        raise CustomerVoiceBuildError(
            "Final dataset contains "
            f"{text_bearing_missing_confidence:,} text-bearing "
            "reviews without sentiment confidence."
        )

    textless_confidence_count = int(
        (
            (~final_df["has_review_text"])
            & final_df["sentiment_confidence"].notna()
        ).sum()
    )

    if textless_confidence_count:
        raise CustomerVoiceBuildError(
            "Final dataset contains "
            f"{textless_confidence_count:,} textless reviews "
            "with sentiment confidence."
        )

    # -------------------------------------------------------------------------
    # Sentiment labels
    # -------------------------------------------------------------------------

    observed_sentiments = set(
        final_df.loc[
            final_df["sentiment"].notna(),
            "sentiment",
        ].astype(str).unique()
    )

    unexpected_sentiments = (
        observed_sentiments
        - ALLOWED_SENTIMENTS
    )

    if unexpected_sentiments:
        raise CustomerVoiceBuildError(
            "Final dataset contains unexpected sentiment labels: "
            f"{sorted(unexpected_sentiments)}"
        )

    # -------------------------------------------------------------------------
    # Confidence range
    # -------------------------------------------------------------------------

    confidence_values = final_df.loc[
        final_df["sentiment_confidence"].notna(),
        "sentiment_confidence",
    ]

    if not confidence_values.empty:

        invalid_confidence = (
            (confidence_values < 0)
            | (confidence_values > 1)
        )

        invalid_count = int(
            invalid_confidence.sum()
        )

        if invalid_count:
            raise CustomerVoiceBuildError(
                "Final dataset contains "
                f"{invalid_count:,} invalid confidence values."
            )

    # -------------------------------------------------------------------------
    # Customer IDs
    # -------------------------------------------------------------------------

    missing_customer_id = int(
        final_df["customer_id"].isna().sum()
    )

    if missing_customer_id:
        raise CustomerVoiceBuildError(
            "Final dataset contains "
            f"{missing_customer_id:,} NULL customer_id values."
        )


# =============================================================================
# MAIN
# =============================================================================


def main() -> None:

    print(
        "BUILDING CUSTOMER VOICE DATASET"
    )
    print(
        "================================"
    )
    print()

    # -------------------------------------------------------------------------
    # Load input
    # -------------------------------------------------------------------------

    input_df = load_customer_voice_input()

    print(
        f"Input reviews         : "
        f"{len(input_df):,}"
    )

    print(
        f"Input duplicate keys  : "
        f"{input_df.duplicated(subset=['review_id', 'order_id']).sum():,}"
    )

    print(
        f"Text-bearing reviews  : "
        f"{input_df['has_review_text'].sum():,}"
    )

    print(
        f"Textless reviews      : "
        f"{(~input_df['has_review_text']).sum():,}"
    )

    # -------------------------------------------------------------------------
    # Load sentiment checkpoints
    # -------------------------------------------------------------------------

    print()
    print(
        "LOADING SENTIMENT CHECKPOINTS"
    )
    print(
        "-----------------------------"
    )

    predictions = load_sentiment_checkpoints()

    print(
        f"Sentiment predictions : "
        f"{len(predictions):,}"
    )

    print(
        f"Duplicate prediction keys : "
        f"{predictions['review_key'].duplicated().sum():,}"
    )

    print(
        f"NULL sentiments        : "
        f"{predictions['sentiment'].isna().sum():,}"
    )

    print(
        f"NULL confidences       : "
        f"{predictions['sentiment_confidence'].isna().sum():,}"
    )

    # -------------------------------------------------------------------------
    # Build final dataset
    # -------------------------------------------------------------------------

    final_df = build_customer_voice(
        input_df=input_df,
        predictions=predictions,
    )

    # -------------------------------------------------------------------------
    # Final validation
    # -------------------------------------------------------------------------

    validate_final_customer_voice(
        final_df
    )

    # -------------------------------------------------------------------------
    # Print final summary
    # -------------------------------------------------------------------------

    print()
    print(
        "FINAL CUSTOMER VOICE DATASET"
    )
    print(
        "----------------------------"
    )

    print(
        f"Rows                         : "
        f"{len(final_df):,}"
    )

    print(
        f"Columns                      : "
        f"{len(final_df.columns)}"
    )

    print(
        f"Text-bearing reviews         : "
        f"{final_df['has_review_text'].sum():,}"
    )

    print(
        f"Textless reviews             : "
        f"{(~final_df['has_review_text']).sum():,}"
    )

    print(
        f"Reviews with sentiment       : "
        f"{final_df['sentiment'].notna().sum():,}"
    )

    print(
        f"Reviews without sentiment    : "
        f"{final_df['sentiment'].isna().sum():,}"
    )

    print(
        f"Duplicate review keys        : "
        f"{final_df.duplicated(subset=['review_id', 'order_id']).sum():,}"
    )

    print()
    print(
        "SENTIMENT DISTRIBUTION"
    )
    print(
        "----------------------"
    )

    sentiment_counts = (
        final_df["sentiment"]
        .value_counts(dropna=False)
    )

    for label, count in sentiment_counts.items():

        label_text = (
            "NULL / TEXTLESS"
            if pd.isna(label)
            else str(label)
        )

        print(
            f"{label_text:<18}: "
            f"{int(count):,}"
        )

    # -------------------------------------------------------------------------
    # Save atomically
    # -------------------------------------------------------------------------

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    temp_output = OUTPUT_DIR / (
        "customer_voice.parquet.tmp"
    )

    if temp_output.exists():
        temp_output.unlink()

    final_df.to_parquet(
        temp_output,
        index=False,
    )

    # Never silently overwrite an existing final artifact.
    if OUTPUT_PATH.exists():
        raise CustomerVoiceBuildError(
            f"Final Customer Voice output already exists: "
            f"{OUTPUT_PATH}. "
            "Delete or archive it explicitly before rebuilding."
        )

    temp_output.replace(
        OUTPUT_PATH
    )

    # -------------------------------------------------------------------------
    # Read-back validation
    # -------------------------------------------------------------------------

    persisted = pd.read_parquet(
        OUTPUT_PATH
    )

    validate_final_customer_voice(
        persisted
    )

    print()
    print(
        f"OUTPUT: {OUTPUT_PATH}"
    )

    print()
    print(
        "PERSISTED CUSTOMER VOICE VALIDATION: PASS"
    )

    print()
    print(
        "CUSTOMER VOICE BUILD: SUCCESS"
    )


if __name__ == "__main__":
    try:
        main()
    except CustomerVoiceBuildError as exc:
        print()
        print(
            f"CustomerVoiceBuildError: {exc}"
        )
        raise