from __future__ import annotations

import pandas as pd


REVIEW_REQUIRED_COLUMNS = [
    "review_id",
    "order_id",
    "review_score",
    "review_creation_date",
    "review_answer_timestamp",
    "review_comment_title",
    "review_comment_message",
]

ORDER_REQUIRED_COLUMNS = [
    "order_id",
    "customer_id",
]


class CustomerVoiceInputError(Exception):
    """Raised when the customer voice NLP input contract is violated."""


def build_customer_voice_input(
    fact_review: pd.DataFrame,
    fact_order: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build the governed Customer Voice NLP input.

    Grain:
        one row per review_id + order_id

    The review remains at review grain.

    customer_id is brought from fact_order through the
    many-reviews-to-one-order relationship.
    """

    # -------------------------------------------------------------
    # Validate review source
    # -------------------------------------------------------------

    missing_review_columns = [
        column
        for column in REVIEW_REQUIRED_COLUMNS
        if column not in fact_review.columns
    ]

    if missing_review_columns:
        raise CustomerVoiceInputError(
            "fact_review is missing required columns: "
            f"{missing_review_columns}"
        )

    # -------------------------------------------------------------
    # Validate order source
    # -------------------------------------------------------------

    missing_order_columns = [
        column
        for column in ORDER_REQUIRED_COLUMNS
        if column not in fact_order.columns
    ]

    if missing_order_columns:
        raise CustomerVoiceInputError(
            "fact_order is missing required columns: "
            f"{missing_order_columns}"
        )

    # -------------------------------------------------------------
    # Validate review grain
    # -------------------------------------------------------------

    review = fact_review[
        REVIEW_REQUIRED_COLUMNS
    ].copy()

    review_duplicate_count = int(
        review.duplicated(
            subset=[
                "review_id",
                "order_id",
            ]
        ).sum()
    )

    if review_duplicate_count:
        raise CustomerVoiceInputError(
            "Review grain violation: "
            f"{review_duplicate_count} duplicate "
            "review_id + order_id rows found."
        )

    # -------------------------------------------------------------
    # Validate order grain
    # -------------------------------------------------------------

    order = fact_order[
        ORDER_REQUIRED_COLUMNS
    ].copy()

    order_duplicate_count = int(
        order["order_id"].duplicated().sum()
    )

    if order_duplicate_count:
        raise CustomerVoiceInputError(
            "Order grain violation: "
            f"{order_duplicate_count} duplicate order_id "
            "rows found."
        )

    # -------------------------------------------------------------
    # Join customer context
    # -------------------------------------------------------------

    result = review.merge(
        order,
        on="order_id",
        how="left",
        validate="many_to_one",
        indicator=True,
    )

    # Every review must map to an order.
    missing_order_mapping = int(
        (result["_merge"] != "both").sum()
    )

    if missing_order_mapping:
        raise CustomerVoiceInputError(
            "Customer Voice review-to-order mapping failed: "
            f"{missing_order_mapping} review rows do not map "
            "to fact_order."
        )

    result = result.drop(
        columns=["_merge"]
    )

    # -------------------------------------------------------------
    # Customer ID must be present
    # -------------------------------------------------------------

    missing_customer_id = int(
        result["customer_id"].isna().sum()
    )

    if missing_customer_id:
        raise CustomerVoiceInputError(
            "Customer Voice input contains "
            f"{missing_customer_id} reviews without customer_id."
        )

    # -------------------------------------------------------------
    # Build combined review text
    # -------------------------------------------------------------

    result["review_text"] = (
        result["review_comment_title"].fillna("")
        + " "
        + result["review_comment_message"].fillna("")
    )

    result["review_text"] = (
        result["review_text"]
        .str.replace(
            r"\s+",
            " ",
            regex=True,
        )
        .str.strip()
    )

    result["has_review_text"] = (
        result["review_text"].ne("")
    )

    # -------------------------------------------------------------
    # Final column order
    # -------------------------------------------------------------

    final_columns = [
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

    result = result[
        final_columns
    ].copy()

    # -------------------------------------------------------------
    # Final grain validation
    # -------------------------------------------------------------

    final_duplicate_count = int(
        result.duplicated(
            subset=[
                "review_id",
                "order_id",
            ]
        ).sum()
    )

    if final_duplicate_count:
        raise CustomerVoiceInputError(
            "Final Customer Voice input grain violation: "
            f"{final_duplicate_count} duplicate "
            "review_id + order_id rows found."
        )

    return result