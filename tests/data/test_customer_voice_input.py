import pandas as pd
import pytest

from src.nlp.customer_voice_input import (
    CustomerVoiceInputError,
    build_customer_voice_input,
)


def make_fact_review(
    review_ids,
    order_ids,
    titles,
    messages,
):
    return pd.DataFrame(
        {
            "review_id": review_ids,
            "order_id": order_ids,
            "review_score": [5] * len(review_ids),
            "review_creation_date": [
                "2018-01-01"
            ] * len(review_ids),
            "review_answer_timestamp": [
                "2018-01-02"
            ] * len(review_ids),
            "review_comment_title": titles,
            "review_comment_message": messages,
        }
    )


def make_fact_order(
    order_ids,
    customer_ids,
):
    return pd.DataFrame(
        {
            "order_id": order_ids,
            "customer_id": customer_ids,
        }
    )


def test_customer_voice_input_preserves_review_grain():

    fact_review = make_fact_review(
        review_ids=["r1", "r2"],
        order_ids=["o1", "o2"],
        titles=["Ótimo", None],
        messages=[
            "Produto muito bom",
            "Não chegou",
        ],
    )

    fact_order = make_fact_order(
        order_ids=["o1", "o2"],
        customer_ids=["c1", "c2"],
    )

    result = build_customer_voice_input(
        fact_review=fact_review,
        fact_order=fact_order,
    )

    assert len(result) == 2

    assert (
        result[
            ["review_id", "order_id"]
        ]
        .duplicated()
        .sum()
        == 0
    )

    assert list(result["customer_id"]) == [
        "c1",
        "c2",
    ]

    assert (
        bool(result.loc[0, "has_review_text"])
        is True
    )

    assert (
        bool(result.loc[1, "has_review_text"])
        is True
    )


def test_customer_voice_input_rejects_duplicate_review_grain():

    fact_review = make_fact_review(
        review_ids=["r1", "r1"],
        order_ids=["o1", "o1"],
        titles=["Good", "Good"],
        messages=["Nice", "Nice"],
    )

    fact_order = make_fact_order(
        order_ids=["o1"],
        customer_ids=["c1"],
    )

    with pytest.raises(
        CustomerVoiceInputError,
        match="Review grain violation",
    ):
        build_customer_voice_input(
            fact_review=fact_review,
            fact_order=fact_order,
        )


def test_customer_voice_input_distinguishes_missing_text():

    fact_review = make_fact_review(
        review_ids=["r1"],
        order_ids=["o1"],
        titles=[None],
        messages=[None],
    )

    fact_order = make_fact_order(
        order_ids=["o1"],
        customer_ids=["c1"],
    )

    result = build_customer_voice_input(
        fact_review=fact_review,
        fact_order=fact_order,
    )

    assert len(result) == 1

    assert (
        bool(result.loc[0, "has_review_text"])
        is False
    )

    assert result.loc[0, "review_text"] == ""

    assert result.loc[0, "customer_id"] == "c1"


def test_customer_voice_input_rejects_missing_order_mapping():

    fact_review = make_fact_review(
        review_ids=["r1"],
        order_ids=["o_missing"],
        titles=["Good"],
        messages=["Nice"],
    )

    fact_order = make_fact_order(
        order_ids=["o1"],
        customer_ids=["c1"],
    )

    with pytest.raises(
        CustomerVoiceInputError,
        match="review-to-order mapping failed",
    ):
        build_customer_voice_input(
            fact_review=fact_review,
            fact_order=fact_order,
        )


def test_customer_voice_input_rejects_duplicate_order_grain():

    fact_review = make_fact_review(
        review_ids=["r1"],
        order_ids=["o1"],
        titles=["Good"],
        messages=["Nice"],
    )

    fact_order = make_fact_order(
        order_ids=["o1", "o1"],
        customer_ids=["c1", "c2"],
    )

    with pytest.raises(
        CustomerVoiceInputError,
        match="Order grain violation",
    ):
        build_customer_voice_input(
            fact_review=fact_review,
            fact_order=fact_order,
        )