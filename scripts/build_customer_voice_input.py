from pathlib import Path

import pandas as pd

from src.nlp.customer_voice_input import (
    build_customer_voice_input,
)


WAREHOUSE_DIR = Path(
    "data/processed/warehouse"
)

OUTPUT_DIR = Path(
    "data/processed/nlp"
)

OUTPUT_PATH = OUTPUT_DIR / (
    "customer_voice_input.parquet"
)

FACT_REVIEW_PATH = (
    WAREHOUSE_DIR / "fact_review.parquet"
)

FACT_ORDER_PATH = (
    WAREHOUSE_DIR / "fact_order.parquet"
)


EXPECTED_REVIEWS = 99_224
EXPECTED_TEXT_REVIEWS = 42_687
EXPECTED_TEXTLESS_REVIEWS = 56_537


def main():

    print(
        "BUILDING CUSTOMER VOICE NLP INPUT"
    )
    print(
        "================================="
    )
    print()

    # -------------------------------------------------------------
    # Verify source files
    # -------------------------------------------------------------

    if not FACT_REVIEW_PATH.exists():
        raise FileNotFoundError(
            f"fact_review not found: "
            f"{FACT_REVIEW_PATH}"
        )

    if not FACT_ORDER_PATH.exists():
        raise FileNotFoundError(
            f"fact_order not found: "
            f"{FACT_ORDER_PATH}"
        )

    # -------------------------------------------------------------
    # Load warehouse sources
    # -------------------------------------------------------------

    fact_review = pd.read_parquet(
        FACT_REVIEW_PATH
    )

    fact_order = pd.read_parquet(
        FACT_ORDER_PATH
    )

    print(
        f"fact_review rows : "
        f"{len(fact_review):,}"
    )

    print(
        f"fact_order rows  : "
        f"{len(fact_order):,}"
    )

    # -------------------------------------------------------------
    # Build governed Customer Voice input
    # -------------------------------------------------------------

    result = build_customer_voice_input(
        fact_review=fact_review,
        fact_order=fact_order,
    )

    # -------------------------------------------------------------
    # Validate output
    # -------------------------------------------------------------

    duplicate_grain = int(
        result.duplicated(
            subset=[
                "review_id",
                "order_id",
            ]
        ).sum()
    )

    text_bearing = int(
        result["has_review_text"].sum()
    )

    textless = int(
        (~result["has_review_text"]).sum()
    )

    missing_customer_id = int(
        result["customer_id"].isna().sum()
    )

    print()
    print(
        "CUSTOMER VOICE INPUT"
    )
    print(
        "--------------------"
    )

    print(
        f"Rows                  : "
        f"{len(result):,}"
    )

    print(
        f"Columns               : "
        f"{len(result.columns)}"
    )

    print(
        f"Duplicate review grain: "
        f"{duplicate_grain:,}"
    )

    print(
        f"Text-bearing reviews  : "
        f"{text_bearing:,}"
    )

    print(
        f"Textless reviews      : "
        f"{textless:,}"
    )

    print(
        f"Customer IDs missing  : "
        f"{missing_customer_id:,}"
    )

    if len(result) != EXPECTED_REVIEWS:
        raise RuntimeError(
            f"Unexpected row count: "
            f"{len(result):,} != "
            f"{EXPECTED_REVIEWS:,}"
        )

    if duplicate_grain:
        raise RuntimeError(
            "Customer Voice input contains duplicate "
            "review_id + order_id keys."
        )

    if missing_customer_id:
        raise RuntimeError(
            "Customer Voice input contains NULL customer_id."
        )

    if text_bearing != EXPECTED_TEXT_REVIEWS:
        raise RuntimeError(
            f"Unexpected text-bearing count: "
            f"{text_bearing:,} != "
            f"{EXPECTED_TEXT_REVIEWS:,}"
        )

    if textless != EXPECTED_TEXTLESS_REVIEWS:
        raise RuntimeError(
            f"Unexpected textless count: "
            f"{textless:,} != "
            f"{EXPECTED_TEXTLESS_REVIEWS:,}"
        )

    # -------------------------------------------------------------
    # Persist
    # -------------------------------------------------------------

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    result.to_parquet(
        OUTPUT_PATH,
        index=False,
    )

    # -------------------------------------------------------------
    # Persisted-file validation
    # -------------------------------------------------------------

    persisted = pd.read_parquet(
        OUTPUT_PATH
    )

    persisted_duplicate_grain = int(
        persisted.duplicated(
            subset=[
                "review_id",
                "order_id",
            ]
        ).sum()
    )

    if len(persisted) != EXPECTED_REVIEWS:
        raise RuntimeError(
            "Persisted Customer Voice input has "
            "incorrect row count."
        )

    if persisted_duplicate_grain:
        raise RuntimeError(
            "Persisted Customer Voice input contains "
            "duplicate review grain."
        )

    if persisted["customer_id"].isna().any():
        raise RuntimeError(
            "Persisted Customer Voice input contains "
            "NULL customer_id."
        )

    persisted_text_bearing = int(
        persisted["has_review_text"].sum()
    )

    if persisted_text_bearing != EXPECTED_TEXT_REVIEWS:
        raise RuntimeError(
            "Persisted Customer Voice input has "
            "incorrect text-bearing count."
        )

    print()
    print(
        f"OUTPUT: {OUTPUT_PATH}"
    )

    print()
    print(
        "PERSISTED CUSTOMER VOICE INPUT VALIDATION"
    )
    print(
        "-----------------------------------------"
    )

    print(
        f"Rows                  : "
        f"{len(persisted):,}"
    )

    print(
        f"Duplicate review grain: "
        f"{persisted_duplicate_grain:,}"
    )

    print(
        f"Text-bearing reviews  : "
        f"{persisted_text_bearing:,}"
    )

    print(
        f"Customer IDs missing  : "
        f"{persisted['customer_id'].isna().sum():,}"
    )

    print()
    print(
        "CUSTOMER VOICE INPUT BUILD: SUCCESS"
    )


if __name__ == "__main__":
    main()