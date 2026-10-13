from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.nlp.complaint_detection import (
    OUTPUT_COLUMNS,
    detect_complaints,
    load_complaint_registry,
)


INPUT_PATH = Path(
    "data/processed/nlp/customer_voice.parquet"
)

OUTPUT_DIR = Path(
    "outputs/customer_voice"
)

CHECKPOINT_PATH = OUTPUT_DIR / "complaint_checkpoints.parquet"

FINAL_OUTPUT_PATH = Path(
    "data/processed/nlp/customer_voice_complaints.parquet"
)

REGISTRY_PATH = Path(
    "configs/complaint_evidence.yaml"
)

CHUNK_SIZE = 1000


def load_input() -> pd.DataFrame:
    df = pd.read_parquet(INPUT_PATH)

    required_columns = {
        "review_id",
        "order_id",
        "customer_id",
        "review_text",
        "has_review_text",
    }

    missing = required_columns - set(df.columns)

    if missing:
        raise RuntimeError(
            f"Customer Voice input missing columns: "
            f"{sorted(missing)}"
        )

    text_df = df.loc[
        df["has_review_text"].eq(True)
    ].copy()

    duplicate_count = int(
        text_df.duplicated(
            subset=["review_id", "order_id"]
        ).sum()
    )

    if duplicate_count:
        raise RuntimeError(
            "Text-bearing input contains "
            f"{duplicate_count:,} duplicate review keys."
        )

    return text_df.reset_index(drop=True)


def load_checkpoint() -> pd.DataFrame:
    if not CHECKPOINT_PATH.exists():
        return pd.DataFrame(
            columns=[
                "review_id",
                "order_id",
                *OUTPUT_COLUMNS,
            ]
        )

    checkpoint = pd.read_parquet(
        CHECKPOINT_PATH
    )

    expected_columns = {
        "review_id",
        "order_id",
        *OUTPUT_COLUMNS,
    }

    missing = expected_columns - set(
        checkpoint.columns
    )

    if missing:
        raise RuntimeError(
            "Complaint checkpoint missing columns: "
            f"{sorted(missing)}"
        )

    duplicate_count = int(
        checkpoint.duplicated(
            subset=["review_id", "order_id"]
        ).sum()
    )

    if duplicate_count:
        raise RuntimeError(
            "Complaint checkpoint contains "
            f"{duplicate_count:,} duplicate review keys."
        )

    return checkpoint


def save_checkpoint(
    checkpoint: pd.DataFrame,
) -> None:
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    checkpoint.to_parquet(
        CHECKPOINT_PATH,
        index=False,
    )


def validate_checkpoint(
    checkpoint: pd.DataFrame,
    input_df: pd.DataFrame,
) -> None:

    input_keys = pd.MultiIndex.from_frame(
        input_df[
            ["review_id", "order_id"]
        ]
    )

    checkpoint_keys = pd.MultiIndex.from_frame(
        checkpoint[
            ["review_id", "order_id"]
        ]
    )

    if not checkpoint_keys.is_unique:
        raise RuntimeError(
            "Complaint checkpoint review keys "
            "are not unique."
        )

    unexpected = checkpoint_keys.difference(
        input_keys
    )

    if len(unexpected):
        raise RuntimeError(
            "Complaint checkpoint contains "
            f"{len(unexpected):,} unexpected keys."
        )

    null_flags = checkpoint[
        "complaint_flag"
    ].isna()

    null_themes = checkpoint[
        "complaint_theme"
    ].isna()

    ambiguous = (
        null_flags
        & checkpoint["complaint_confidence"].notna()
    )

    if ambiguous.any():
        bad = checkpoint.loc[
            ambiguous
            & checkpoint["complaint_theme"].ne(
                "unknown"
            )
        ]

        if len(bad):
            raise RuntimeError(
                "Ambiguous complaint records must "
                "use complaint_theme='unknown'."
            )

    confidence_columns = [
        "complaint_confidence",
        "complaint_theme_confidence",
    ]

    for column in confidence_columns:
        values = checkpoint[column].dropna()

        if ((values < 0) | (values > 1)).any():
            raise RuntimeError(
                f"{column} contains values outside [0, 1]."
            )

    allowed_themes = {
        "delivery",
        "product_quality",
        "wrong_or_missing_item",
        "seller_service",
        "order_or_transaction",
        "product_information",
        "packaging",
        "other",
        "unknown",
    }

    classified_themes = set(
        checkpoint["complaint_theme"]
        .dropna()
        .unique()
    )

    unexpected_themes = (
        classified_themes - allowed_themes
    )

    if unexpected_themes:
        raise RuntimeError(
            "Unexpected complaint themes: "
            f"{sorted(unexpected_themes)}"
        )


def process() -> None:

    print("=" * 70)
    print("COMPLAINT DETECTION — RESUMABLE PRODUCTION RUN")
    print("=" * 70)

    input_df = load_input()

    print(
        f"Text-bearing input reviews : "
        f"{len(input_df):,}"
    )

    registry = load_complaint_registry()

    checkpoint = load_checkpoint()

    completed_keys = set(
        zip(
            checkpoint["review_id"],
            checkpoint["order_id"],
        )
    )

    print(
        f"Existing checkpoint rows    : "
        f"{len(checkpoint):,}"
    )

    remaining = input_df.loc[
        ~input_df.apply(
            lambda row: (
                row["review_id"],
                row["order_id"],
            ) in completed_keys,
            axis=1,
        )
    ].copy()

    print(
        f"Remaining reviews           : "
        f"{len(remaining):,}"
    )

    if len(remaining):

        for start in range(
            0,
            len(remaining),
            CHUNK_SIZE,
        ):

            chunk = remaining.iloc[
                start:start + CHUNK_SIZE
            ].copy()

            result = detect_complaints(
                chunk,
                registry,
            )

            checkpoint_chunk = result[
                [
                    "review_id",
                    "order_id",
                    *OUTPUT_COLUMNS,
                ]
            ].copy()

            checkpoint = pd.concat(
                [
                    checkpoint,
                    checkpoint_chunk,
                ],
                ignore_index=True,
            )

            checkpoint = checkpoint.drop_duplicates(
                subset=[
                    "review_id",
                    "order_id",
                ],
                keep="last",
            )

            save_checkpoint(
                checkpoint
            )

            processed = min(
                start + CHUNK_SIZE,
                len(remaining),
            )

            print(
                f"Processed this run: "
                f"{processed:,}/{len(remaining):,} "
                f"| Total checkpoint: "
                f"{len(checkpoint):,}"
            )

    validate_checkpoint(
        checkpoint,
        input_df,
    )

    if len(checkpoint) != len(input_df):
        raise RuntimeError(
            "Checkpoint does not contain "
            "exactly one prediction for every "
            "text-bearing input review."
        )

    FINAL_OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    checkpoint.to_parquet(
        FINAL_OUTPUT_PATH,
        index=False,
    )

    print()
    print("=" * 70)
    print("COMPLAINT DETECTION COMPLETE")
    print("=" * 70)

    print(
        f"Input text-bearing reviews : "
        f"{len(input_df):,}"
    )

    print(
        f"Prediction rows            : "
        f"{len(checkpoint):,}"
    )

    print(
        "Complaints                 : "
        f"{int(checkpoint['complaint_flag'].eq(True).sum()):,}"
    )

    print(
        "Non-complaints             : "
        f"{int(checkpoint['complaint_flag'].eq(False).sum()):,}"
    )

    print(
        "Unknown                    : "
        f"{int(checkpoint['complaint_flag'].isna().sum()):,}"
    )

    print()
    print("Theme distribution:")

    print(
        checkpoint[
            "complaint_theme"
        ]
        .value_counts(
            dropna=False
        )
        .to_string()
    )

    print()
    print(
        f"Checkpoint: {CHECKPOINT_PATH}"
    )

    print(
        f"Final output: {FINAL_OUTPUT_PATH}"
    )


if __name__ == "__main__":
    process()