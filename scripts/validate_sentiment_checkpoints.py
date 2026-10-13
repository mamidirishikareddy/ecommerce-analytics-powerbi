from __future__ import annotations

from pathlib import Path

import pandas as pd


INPUT_PATH = Path(
    "data/processed/nlp/customer_voice_input.parquet"
)

CHECKPOINT_DIR = Path(
    "data/processed/nlp/sentiment_checkpoints"
)

EXPECTED_TEXT_REVIEWS = 42_687

EXPECTED_COLUMNS = [
    "review_id",
    "order_id",
    "review_key",
    "sentiment",
    "sentiment_confidence",
]

ALLOWED_SENTIMENTS = {
    "positive",
    "negative",
    "neutral",
}


class ValidationError(Exception):
    pass


def build_review_key(
    review_id: pd.Series,
    order_id: pd.Series,
) -> pd.Series:

    return (
        review_id.astype(str)
        + "||"
        + order_id.astype(str)
    )


def fail(message: str):
    raise ValidationError(message)


def main():

    print("SENTIMENT CHECKPOINT VALIDATION")
    print("--------------------------------")
    print()

    # ============================================================
    # 1. INPUT FILE
    # ============================================================

    if not INPUT_PATH.exists():
        fail(
            f"Input file not found: {INPUT_PATH}"
        )

    input_df = pd.read_parquet(INPUT_PATH)

    required_input_columns = [
        "review_id",
        "order_id",
        "review_text",
        "has_review_text",
    ]

    missing = [
        column
        for column in required_input_columns
        if column not in input_df.columns
    ]

    if missing:
        fail(
            f"Input missing required columns: {missing}"
        )

    input_df["review_id"] = (
        input_df["review_id"].astype(str)
    )

    input_df["order_id"] = (
        input_df["order_id"].astype(str)
    )

    input_df["review_key"] = build_review_key(
        input_df["review_id"],
        input_df["order_id"],
    )

    # ============================================================
    # 2. INPUT GRAIN
    # ============================================================

    input_duplicate_count = int(
        input_df["review_key"].duplicated().sum()
    )

    print(
        f"Input reviews              : {len(input_df):,}"
    )

    print(
        f"Input duplicate keys       : "
        f"{input_duplicate_count:,}"
    )

    if input_duplicate_count != 0:
        fail(
            "Input contains duplicate "
            "review_id + order_id keys."
        )

    # ============================================================
    # 3. TEXT-BEARING POPULATION
    # ============================================================

    text_reviews = input_df[
        input_df["has_review_text"]
    ].copy()

    text_reviews["review_key"] = build_review_key(
        text_reviews["review_id"],
        text_reviews["order_id"],
    )

    text_review_keys = set(
        text_reviews["review_key"]
    )

    textless_reviews = input_df[
        ~input_df["has_review_text"]
    ].copy()

    print(
        f"Text-bearing reviews       : "
        f"{len(text_reviews):,}"
    )

    print(
        f"Textless reviews            : "
        f"{len(textless_reviews):,}"
    )

    if len(text_reviews) != EXPECTED_TEXT_REVIEWS:
        fail(
            "Unexpected text-bearing review count. "
            f"Expected {EXPECTED_TEXT_REVIEWS:,}, "
            f"found {len(text_reviews):,}."
        )

    # ============================================================
    # 4. CHECKPOINT DIRECTORY
    # ============================================================

    if not CHECKPOINT_DIR.exists():
        fail(
            f"Checkpoint directory not found: "
            f"{CHECKPOINT_DIR}"
        )

    checkpoint_files = sorted(
        CHECKPOINT_DIR.glob(
            "checkpoint_*.parquet"
        )
    )

    print(
        f"Checkpoint files           : "
        f"{len(checkpoint_files)}"
    )

    if len(checkpoint_files) != 43:
        fail(
            "Expected exactly 43 checkpoint files. "
            f"Found {len(checkpoint_files)}."
        )

    print()

    # ============================================================
    # 5. VALIDATE EACH CHECKPOINT
    # ============================================================

    all_keys = set()

    total_rows = 0

    total_positive = 0
    total_negative = 0
    total_neutral = 0

    confidence_min = 1.0
    confidence_max = 0.0

    checkpoint_row_counts = []

    for checkpoint_number, path in enumerate(
        checkpoint_files,
        start=1,
    ):

        df = pd.read_parquet(path)

        # --------------------------------------------------------
        # Schema
        # --------------------------------------------------------

        if df.columns.tolist() != EXPECTED_COLUMNS:
            fail(
                f"{path.name}: unexpected columns.\n"
                f"Expected: {EXPECTED_COLUMNS}\n"
                f"Found:    {df.columns.tolist()}"
            )

        # --------------------------------------------------------
        # Non-empty
        # --------------------------------------------------------

        if df.empty:
            fail(
                f"{path.name}: checkpoint is empty."
            )

        # --------------------------------------------------------
        # Composite key
        # --------------------------------------------------------

        if df["review_key"].isna().any():
            fail(
                f"{path.name}: NULL review_key found."
            )

        duplicate_keys = int(
            df["review_key"].duplicated().sum()
        )

        if duplicate_keys:
            fail(
                f"{path.name}: "
                f"{duplicate_keys:,} duplicate review keys."
            )

        checkpoint_keys = set(
            df["review_key"].astype(str)
        )

        # --------------------------------------------------------
        # Cross-checkpoint duplication
        # --------------------------------------------------------

        overlap = (
            all_keys
            & checkpoint_keys
        )

        if overlap:
            fail(
                f"{path.name}: "
                f"{len(overlap):,} keys already exist "
                "in another checkpoint."
            )

        # --------------------------------------------------------
        # Check keys against text-bearing population
        # --------------------------------------------------------

        unknown_keys = (
            checkpoint_keys
            - text_review_keys
        )

        if unknown_keys:
            fail(
                f"{path.name}: "
                f"{len(unknown_keys):,} keys do not "
                "exist in text-bearing input."
            )

        # --------------------------------------------------------
        # Sentiment
        # --------------------------------------------------------

        null_sentiment = int(
            df["sentiment"].isna().sum()
        )

        if null_sentiment:
            fail(
                f"{path.name}: "
                f"{null_sentiment:,} NULL sentiment values."
            )

        unexpected_sentiments = (
            set(df["sentiment"].astype(str))
            - ALLOWED_SENTIMENTS
        )

        if unexpected_sentiments:
            fail(
                f"{path.name}: unexpected sentiment labels: "
                f"{sorted(unexpected_sentiments)}"
            )

        # --------------------------------------------------------
        # Confidence
        # --------------------------------------------------------

        null_confidence = int(
            df[
                "sentiment_confidence"
            ].isna().sum()
        )

        if null_confidence:
            fail(
                f"{path.name}: "
                f"{null_confidence:,} NULL confidence values."
            )

        invalid_confidence = (
            (df["sentiment_confidence"] < 0)
            | (df["sentiment_confidence"] > 1)
        )

        invalid_count = int(
            invalid_confidence.sum()
        )

        if invalid_count:
            fail(
                f"{path.name}: "
                f"{invalid_count:,} confidence values "
                "outside [0, 1]."
            )

        # --------------------------------------------------------
        # Aggregate
        # --------------------------------------------------------

        positive = int(
            (df["sentiment"] == "positive").sum()
        )

        negative = int(
            (df["sentiment"] == "negative").sum()
        )

        neutral = int(
            (df["sentiment"] == "neutral").sum()
        )

        total_positive += positive
        total_negative += negative
        total_neutral += neutral

        checkpoint_min = float(
            df["sentiment_confidence"].min()
        )

        checkpoint_max = float(
            df["sentiment_confidence"].max()
        )

        confidence_min = min(
            confidence_min,
            checkpoint_min,
        )

        confidence_max = max(
            confidence_max,
            checkpoint_max,
        )

        rows = len(df)

        total_rows += rows

        all_keys.update(
            checkpoint_keys
        )

        checkpoint_row_counts.append(rows)

        print(
            f"Checkpoint {checkpoint_number:02d} "
            f"| rows={rows:,} "
            f"| positive={positive:,} "
            f"| negative={negative:,} "
            f"| neutral={neutral:,} "
            f"| confidence="
            f"{checkpoint_min:.4f}-"
            f"{checkpoint_max:.4f}"
        )

    # ============================================================
    # 6. GLOBAL COVERAGE
    # ============================================================

    print()
    print("GLOBAL COVERAGE")
    print("----------------")

    print(
        f"Checkpoint rows           : "
        f"{total_rows:,}"
    )

    print(
        f"Unique checkpoint keys    : "
        f"{len(all_keys):,}"
    )

    print(
        f"Expected text-bearing     : "
        f"{EXPECTED_TEXT_REVIEWS:,}"
    )

    if total_rows != EXPECTED_TEXT_REVIEWS:
        fail(
            "Total checkpoint rows do not equal "
            f"{EXPECTED_TEXT_REVIEWS:,}."
        )

    if len(all_keys) != EXPECTED_TEXT_REVIEWS:
        fail(
            "Unique checkpoint keys do not equal "
            f"{EXPECTED_TEXT_REVIEWS:,}."
        )

    # ============================================================
    # 7. MISSING TEXT-BEARING REVIEWS
    # ============================================================

    missing_keys = (
        text_review_keys
        - all_keys
    )

    print(
        f"Missing predictions       : "
        f"{len(missing_keys):,}"
    )

    if missing_keys:
        fail(
            f"{len(missing_keys):,} text-bearing reviews "
            "have no sentiment prediction."
        )

    # ============================================================
    # 8. UNEXPECTED PREDICTIONS
    # ============================================================

    unexpected_keys = (
        all_keys
        - text_review_keys
    )

    print(
        f"Unexpected predictions    : "
        f"{len(unexpected_keys):,}"
    )

    if unexpected_keys:
        fail(
            f"{len(unexpected_keys):,} predictions "
            "do not belong to text-bearing reviews."
        )

    # ============================================================
    # 9. TEXTLESS REVIEW SAFETY
    # ============================================================

    textless_keys = set(
        textless_reviews["review_key"]
    )

    textless_predictions = (
        all_keys
        & textless_keys
    )

    print(
        f"Textless predictions      : "
        f"{len(textless_predictions):,}"
    )

    if textless_predictions:
        fail(
            f"{len(textless_predictions):,} textless reviews "
            "received sentiment predictions."
        )

    # ============================================================
    # 10. CHECKPOINT SIZE STRUCTURE
    # ============================================================

    expected_checkpoint_sizes = (
        [1000] * 42
        + [687]
    )

    if checkpoint_row_counts != (
        expected_checkpoint_sizes
    ):
        fail(
            "Checkpoint row distribution is unexpected.\n"
            f"Expected: {expected_checkpoint_sizes}\n"
            f"Found:    {checkpoint_row_counts}"
        )

    # ============================================================
    # 11. FINAL DISTRIBUTION
    # ============================================================

    print()
    print("FINAL SENTIMENT DISTRIBUTION")
    print("----------------------------")

    print(
        f"Positive                  : "
        f"{total_positive:,}"
    )

    print(
        f"Negative                  : "
        f"{total_negative:,}"
    )

    print(
        f"Neutral                   : "
        f"{total_neutral:,}"
    )

    print(
        f"Total                     : "
        f"{total_positive + total_negative + total_neutral:,}"
    )

    print()
    print("CONFIDENCE")
    print("----------")

    print(
        f"Minimum                   : "
        f"{confidence_min:.6f}"
    )

    print(
        f"Maximum                   : "
        f"{confidence_max:.6f}"
    )

    # ============================================================
    # 12. FINAL RESULT
    # ============================================================

    print()
    print(
        "=========================================="
    )
    print(
        "SENTIMENT CHECKPOINT VALIDATION: PASS"
    )
    print(
        "=========================================="
    )


if __name__ == "__main__":
    try:
        main()

    except Exception as exc:
        print()
        print(
            "=========================================="
        )
        print(
            "SENTIMENT CHECKPOINT VALIDATION: FAIL"
        )
        print(
            "=========================================="
        )
        print(
            f"ERROR: {exc}"
        )
        raise