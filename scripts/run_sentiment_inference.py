from __future__ import annotations

import argparse
import time
from pathlib import Path

import pandas as pd

from src.nlp.sentiment_inference import (
    load_sentiment_model,
    predict_sentiment,
)


# ---------------------------------------------------------------------
# PATHS
# ---------------------------------------------------------------------

INPUT_PATH = Path(
    "data/processed/nlp/customer_voice_input.parquet"
)

CHECKPOINT_DIR = Path(
    "data/processed/nlp/sentiment_checkpoints"
)


# ---------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------------

CHECKPOINT_COLUMNS = [
    "review_id",
    "order_id",
    "review_key",
    "sentiment",
    "sentiment_confidence",
]

CHECKPOINT_SIZE = 1000
BATCH_SIZE = 16
MAX_LENGTH = 256


# ---------------------------------------------------------------------
# EXCEPTIONS
# ---------------------------------------------------------------------

class SentimentInferenceRunnerError(Exception):
    """Raised when the sentiment inference runner encounters an error."""


# ---------------------------------------------------------------------
# ARGUMENTS
# ---------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Run resumable customer voice sentiment inference "
            "using the governed review_id + order_id grain."
        )
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help=(
            "Maximum number of text-bearing reviews to process "
            "in this execution. Intended for controlled testing."
        ),
    )

    return parser.parse_args()


# ---------------------------------------------------------------------
# REVIEW GRAIN
# ---------------------------------------------------------------------

def build_review_key(
    review_id: pd.Series,
    order_id: pd.Series,
) -> pd.Series:
    """
    Build the governed Customer Voice business key.

    Grain:
        review_id + order_id
    """

    return (
        review_id.astype(str)
        + "||"
        + order_id.astype(str)
    )


# ---------------------------------------------------------------------
# CHECKPOINT PATH
# ---------------------------------------------------------------------

def checkpoint_path(
    checkpoint_number: int,
) -> Path:

    return CHECKPOINT_DIR / (
        f"checkpoint_{checkpoint_number:05d}.parquet"
    )


# ---------------------------------------------------------------------
# LOAD COMPLETED CHECKPOINTS
# ---------------------------------------------------------------------

def load_completed_review_keys() -> set[str]:
    """
    Read all existing checkpoints and return their completed
    review_id + order_id business keys.

    Existing checkpoints are treated as immutable.
    """

    completed_keys: set[str] = set()

    if not CHECKPOINT_DIR.exists():
        return completed_keys

    checkpoint_files = sorted(
        CHECKPOINT_DIR.glob("checkpoint_*.parquet")
    )

    for path in checkpoint_files:

        df = pd.read_parquet(path)

        missing = [
            column
            for column in CHECKPOINT_COLUMNS
            if column not in df.columns
        ]

        if missing:
            raise SentimentInferenceRunnerError(
                f"Invalid checkpoint {path}. "
                f"Missing columns: {missing}"
            )

        if df.empty:
            raise SentimentInferenceRunnerError(
                f"Invalid checkpoint {path}: "
                "checkpoint contains zero rows."
            )

        if df["review_key"].isna().any():
            raise SentimentInferenceRunnerError(
                f"Invalid checkpoint {path}: "
                "NULL review_key found."
            )

        if df["review_key"].duplicated().any():
            duplicate_count = int(
                df["review_key"].duplicated().sum()
            )

            raise SentimentInferenceRunnerError(
                f"Invalid checkpoint {path}: "
                f"{duplicate_count} duplicate review_key rows found."
            )

        completed_keys.update(
            df["review_key"].astype(str).tolist()
        )

    return completed_keys


# ---------------------------------------------------------------------
# SAVE CHECKPOINT
# ---------------------------------------------------------------------

def save_checkpoint(
    output: pd.DataFrame,
    checkpoint_number: int,
) -> Path:
    """
    Save one immutable checkpoint.

    The temporary file is written first and then renamed to the
    final checkpoint path.
    """

    CHECKPOINT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    output = output[
        CHECKPOINT_COLUMNS
    ].copy()

    output["review_id"] = (
        output["review_id"].astype(str)
    )

    output["order_id"] = (
        output["order_id"].astype(str)
    )

    output["review_key"] = (
        output["review_key"].astype(str)
    )

    final_path = checkpoint_path(
        checkpoint_number
    )

    if final_path.exists():
        raise SentimentInferenceRunnerError(
            "Refusing to overwrite existing checkpoint: "
            f"{final_path}"
        )

    temporary_path = final_path.with_suffix(
        ".tmp.parquet"
    )

    if temporary_path.exists():
        temporary_path.unlink()

    output.to_parquet(
        temporary_path,
        index=False,
    )

    temporary_path.replace(
        final_path
    )

    return final_path


# ---------------------------------------------------------------------
# VALIDATE INPUT
# ---------------------------------------------------------------------

def validate_input(
    input_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Validate the persisted Customer Voice NLP input.

    The governed grain is:
        review_id + order_id

    review_id alone is NOT required to be unique.
    """

    required_columns = [
        "review_id",
        "order_id",
        "review_text",
        "has_review_text",
    ]

    missing = [
        column
        for column in required_columns
        if column not in input_df.columns
    ]

    if missing:
        raise SentimentInferenceRunnerError(
            "Input is missing required columns: "
            f"{missing}"
        )

    input_df = input_df.copy()

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

    duplicate_count = int(
        input_df["review_key"].duplicated().sum()
    )

    if duplicate_count:
        raise SentimentInferenceRunnerError(
            "Input grain violation: duplicate "
            "review_id + order_id combinations found: "
            f"{duplicate_count}"
        )

    if input_df["review_id"].eq("").any():
        raise SentimentInferenceRunnerError(
            "Input contains empty review_id values."
        )

    if input_df["order_id"].eq("").any():
        raise SentimentInferenceRunnerError(
            "Input contains empty order_id values."
        )

    return input_df


# ---------------------------------------------------------------------
# VALIDATE PREDICTIONS
# ---------------------------------------------------------------------

def validate_predictions(
    output: pd.DataFrame,
) -> None:
    """
    Validate one completed inference checkpoint before persistence.
    """

    required_columns = CHECKPOINT_COLUMNS

    missing = [
        column
        for column in required_columns
        if column not in output.columns
    ]

    if missing:
        raise SentimentInferenceRunnerError(
            "Prediction output is missing columns: "
            f"{missing}"
        )

    if output.empty:
        raise SentimentInferenceRunnerError(
            "Prediction output contains zero rows."
        )

    if output["review_key"].duplicated().any():
        raise SentimentInferenceRunnerError(
            "Duplicate review_id + order_id detected "
            "in checkpoint output."
        )

    if output["sentiment"].isna().any():
        raise SentimentInferenceRunnerError(
            "NULL sentiment detected."
        )

    if output[
        "sentiment_confidence"
    ].isna().any():
        raise SentimentInferenceRunnerError(
            "NULL sentiment confidence detected."
        )

    confidence_invalid = (
        (output["sentiment_confidence"] < 0)
        | (output["sentiment_confidence"] > 1)
    )

    if confidence_invalid.any():
        invalid_count = int(
            confidence_invalid.sum()
        )

        raise SentimentInferenceRunnerError(
            "Sentiment confidence outside [0, 1]: "
            f"{invalid_count} rows."
        )

    allowed_sentiments = {
        "negative",
        "neutral",
        "positive",
    }

    unexpected_sentiments = set(
        output["sentiment"].astype(str)
    ) - allowed_sentiments

    if unexpected_sentiments:
        raise SentimentInferenceRunnerError(
            "Unexpected sentiment labels detected: "
            f"{sorted(unexpected_sentiments)}"
        )


# ---------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------

def main():

    args = parse_args()

    # -------------------------------------------------------------
    # INPUT EXISTENCE
    # -------------------------------------------------------------

    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"Customer voice input not found: "
            f"{INPUT_PATH}"
        )

    print(
        "CUSTOMER VOICE SENTIMENT INFERENCE"
    )
    print(
        "----------------------------------"
    )

    # -------------------------------------------------------------
    # LOAD INPUT
    # -------------------------------------------------------------

    input_df = pd.read_parquet(
        INPUT_PATH
    )

    input_df = validate_input(
        input_df
    )

    # -------------------------------------------------------------
    # SELECT TEXT-BEARING REVIEWS
    # -------------------------------------------------------------

    text_reviews = input_df[
        input_df["has_review_text"]
    ].copy()

    total_reviews = len(input_df)

    total_text_reviews = len(
        text_reviews
    )

    # -------------------------------------------------------------
    # LOAD EXISTING CHECKPOINTS
    # -------------------------------------------------------------

    completed_keys = (
        load_completed_review_keys()
    )

    already_completed_mask = (
        text_reviews["review_key"].isin(
            completed_keys
        )
    )

    already_completed = int(
        already_completed_mask.sum()
    )

    remaining = text_reviews[
        ~already_completed_mask
    ].copy()

    # -------------------------------------------------------------
    # CONTROLLED EXECUTION LIMIT
    # -------------------------------------------------------------

    if args.limit is not None:

        if args.limit <= 0:
            raise ValueError(
                "--limit must be greater than 0."
            )

        remaining = remaining.head(
            args.limit
        ).copy()

    # -------------------------------------------------------------
    # RUN INFORMATION
    # -------------------------------------------------------------

    print(
        f"Total reviews       : "
        f"{total_reviews:,}"
    )

    print(
        f"Text-bearing        : "
        f"{total_text_reviews:,}"
    )

    print(
        f"Already completed   : "
        f"{already_completed:,}"
    )

    print(
        f"Remaining this run  : "
        f"{len(remaining):,}"
    )

    print(
        f"Checkpoint size     : "
        f"{CHECKPOINT_SIZE:,}"
    )

    print(
        f"Batch size          : "
        f"{BATCH_SIZE}"
    )

    print(
        f"Max length          : "
        f"{MAX_LENGTH}"
    )

    print()

    # -------------------------------------------------------------
    # NOTHING TO PROCESS
    # -------------------------------------------------------------

    if remaining.empty:

        print(
            "No remaining reviews to process."
        )

        return

    # -------------------------------------------------------------
    # LOAD MODEL
    # -------------------------------------------------------------

    print(
        "Loading sentiment model..."
    )

    tokenizer, model = (
        load_sentiment_model()
    )

    print(
        "Model loaded."
    )

    print()

    # -------------------------------------------------------------
    # PROCESS CHECKPOINTS
    # -------------------------------------------------------------

    total_to_process = len(
        remaining
    )

    processed_this_run = 0

    checkpoint_number = 1

    while not remaining.empty:

        # ---------------------------------------------------------
        # NEVER OVERWRITE AN EXISTING CHECKPOINT
        # ---------------------------------------------------------

        while checkpoint_path(
            checkpoint_number
        ).exists():

            checkpoint_number += 1

        # ---------------------------------------------------------
        # TAKE NEXT CHECKPOINT
        # ---------------------------------------------------------

        checkpoint = remaining.head(
            CHECKPOINT_SIZE
        ).copy()

        print(
            f"Checkpoint "
            f"{checkpoint_number}"
        )

        print(
            f"Reviews in checkpoint : "
            f"{len(checkpoint):,}"
        )

        start_time = (
            time.perf_counter()
        )

        # ---------------------------------------------------------
        # INFERENCE
        # ---------------------------------------------------------

        predictions = predict_sentiment(
            texts=checkpoint[
                "review_text"
            ].astype(str).tolist(),

            tokenizer=tokenizer,

            model=model,

            batch_size=BATCH_SIZE,

            max_length=MAX_LENGTH,
        )

        # ---------------------------------------------------------
        # ROW COUNT VALIDATION
        # ---------------------------------------------------------

        if len(predictions) != len(
            checkpoint
        ):

            raise SentimentInferenceRunnerError(
                "Prediction row count does not "
                "match checkpoint row count."
            )

        # ---------------------------------------------------------
        # BUILD OUTPUT
        # ---------------------------------------------------------

        output = pd.DataFrame(
            {
                "review_id": checkpoint[
                    "review_id"
                ].tolist(),

                "order_id": checkpoint[
                    "order_id"
                ].tolist(),

                "review_key": checkpoint[
                    "review_key"
                ].tolist(),

                "sentiment": predictions[
                    "sentiment"
                ].tolist(),

                "sentiment_confidence": (
                    predictions[
                        "sentiment_confidence"
                    ].tolist()
                ),
            }
        )

        # ---------------------------------------------------------
        # VALIDATE OUTPUT BEFORE SAVING
        # ---------------------------------------------------------

        validate_predictions(
            output
        )

        # ---------------------------------------------------------
        # PERFORMANCE
        # ---------------------------------------------------------

        elapsed = (
            time.perf_counter()
            - start_time
        )

        reviews_per_second = (
            len(checkpoint) / elapsed
            if elapsed > 0
            else 0
        )

        # ---------------------------------------------------------
        # SAVE IMMUTABLE CHECKPOINT
        # ---------------------------------------------------------

        saved_path = save_checkpoint(
            output=output,
            checkpoint_number=checkpoint_number,
        )

        # ---------------------------------------------------------
        # UPDATE PROGRESS
        # ---------------------------------------------------------

        processed_keys = set(
            output[
                "review_key"
            ].astype(str)
        )

        remaining = remaining[
            ~remaining[
                "review_key"
            ].isin(
                processed_keys
            )
        ].copy()

        processed_this_run += len(
            output
        )

        # ---------------------------------------------------------
        # REPORT
        # ---------------------------------------------------------

        print(
            f"Saved                : "
            f"{saved_path}"
        )

        print(
            f"Elapsed seconds      : "
            f"{elapsed:.2f}"
        )

        print(
            f"Reviews/sec          : "
            f"{reviews_per_second:.2f}"
        )

        print(
            "Sentiments           : "
            f"{output['sentiment'].value_counts().to_dict()}"
        )

        print(
            f"Progress this run    : "
            f"{processed_this_run:,} / "
            f"{total_to_process:,}"
        )

        print()

        checkpoint_number += 1

    # -------------------------------------------------------------
    # SUCCESS
    # -------------------------------------------------------------

    print(
        "SENTIMENT INFERENCE RUN: SUCCESS"
    )


if __name__ == "__main__":
    main()