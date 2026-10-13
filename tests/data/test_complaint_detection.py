import pandas as pd
import pytest

from src.nlp.complaint_detection import (
    ComplaintDetectionError,
    classify_review,
    detect_complaints,
    load_complaint_registry,
)


REGISTRY_PATH = (
    "configs/complaint_evidence.yaml"
)


@pytest.fixture
def registry():
    return load_complaint_registry(
        REGISTRY_PATH
    )


def test_registry_loads(registry):

    assert registry["version"] == "1.0"

    assert len(
        registry["taxonomy"]
    ) == 9

    assert (
        "delivery"
        in registry["taxonomy"]
    )


def test_textless_review_is_unclassified(
    registry,
):

    result = classify_review(
        None,
        REGISTRY_PATH,
    )

    assert result["complaint_flag"] is None

    assert (
        result["complaint_confidence"]
        is None
    )

    assert (
        result["complaint_theme"]
        is None
    )

    assert (
        result["complaint_detection_method"]
        == "unclassified"
    )


def test_explicit_delivery_problem_is_complaint(
    registry,
):

    result = classify_review(
        "Não chegou meu produto",
        REGISTRY_PATH,
    )

    assert (
        result["complaint_flag"]
        is True
    )

    assert (
        result["complaint_theme"]
        == "delivery"
    )

    assert (
        result["complaint_confidence"]
        >= 0.70
    )


def test_explicit_product_problem_is_complaint(
    registry,
):

    result = classify_review(
        "O produto veio quebrado e não funciona",
        REGISTRY_PATH,
    )

    assert (
        result["complaint_flag"]
        is True
    )

    assert (
        result["complaint_theme"]
        == "product_quality"
    )


def test_wrong_item_is_complaint(
    registry,
):

    result = classify_review(
        "Recebi outro produto, veio errado",
        REGISTRY_PATH,
    )

    assert (
        result["complaint_flag"]
        is True
    )

    assert (
        result["complaint_theme"]
        == "wrong_or_missing_item"
    )


def test_negative_language_alone_is_not_automatically_complaint(
    registry,
):

    result = classify_review(
        "Não gostei",
        REGISTRY_PATH,
    )

    # The phrase is explicitly negative, but there is no concrete
    # operational/product/service problem.
    assert (
        result["complaint_flag"]
        is not True
    )


def test_praise_is_not_complaint(
    registry,
):

    result = classify_review(
        "Muito bom. Recomendo.",
        REGISTRY_PATH,
    )

    assert (
        result["complaint_flag"]
        is False
    )


def test_negated_problem_is_not_complaint(
    registry,
):

    result = classify_review(
        "Não tive problema e chegou no prazo",
        REGISTRY_PATH,
    )

    assert (
        result["complaint_flag"]
        is False
    )


def test_customer_voice_dataframe_preserves_grain(
    registry,
):

    df = pd.DataFrame(
        {
            "review_id": [
                "r1",
                "r2",
                "r3",
            ],
            "order_id": [
                "o1",
                "o2",
                "o3",
            ],
            "customer_id": [
                "c1",
                "c2",
                "c3",
            ],
            "review_text": [
                "Não chegou meu produto",
                "Muito bom. Recomendo.",
                "",
            ],
            "has_review_text": [
                True,
                True,
                False,
            ],
        }
    )

    result = detect_complaints(
        df,
        registry,
    )

    assert len(result) == 3

    assert (
        result[
            ["review_id", "order_id"]
        ]
        .duplicated()
        .sum()
        == 0
    )

    assert (
        result.loc[
            0,
            "complaint_flag",
        ]
        is True
    )

    assert (
        result.loc[
            0,
            "complaint_theme",
        ]
        == "delivery"
    )

    assert (
        result.loc[
            1,
            "complaint_flag",
        ]
        is False
    )

    assert pd.isna(
        result.loc[
            2,
            "complaint_flag",
        ]
    )

    assert pd.isna(
        result.loc[
            2,
            "complaint_theme",
        ]
    )


def test_duplicate_review_grain_is_rejected(
    registry,
):

    df = pd.DataFrame(
        {
            "review_id": [
                "r1",
                "r1",
            ],
            "order_id": [
                "o1",
                "o1",
            ],
            "review_text": [
                "Não chegou",
                "Não chegou",
            ],
            "has_review_text": [
                True,
                True,
            ],
        }
    )

    with pytest.raises(
        ComplaintDetectionError,
        match="duplicate review keys",
    ):
        detect_complaints(
            df,
            registry,
        )