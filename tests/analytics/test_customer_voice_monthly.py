import pandas as pd

from src.marts.customer_voice_monthly import (
    OUTPUT_COLUMNS,
    build_customer_voice_monthly,
)


CUSTOMER_VOICE_PATH = "data/processed/nlp/customer_voice.parquet"
COMPLAINT_PATH = "data/processed/nlp/customer_voice_complaints.parquet"


def test_customer_voice_monthly_contract():
    df = build_customer_voice_monthly(
        CUSTOMER_VOICE_PATH,
        COMPLAINT_PATH,
    )

    assert list(df.columns) == OUTPUT_COLUMNS
    assert len(df.columns) == 10


def test_customer_voice_monthly_grain():
    df = build_customer_voice_monthly(
        CUSTOMER_VOICE_PATH,
        COMPLAINT_PATH,
    )

    assert df["month"].duplicated().sum() == 0
    assert len(df) == 23


def test_customer_voice_monthly_reconciliation():
    df = build_customer_voice_monthly(
        CUSTOMER_VOICE_PATH,
        COMPLAINT_PATH,
    )

    assert df["review_count"].sum() == 99_224
    assert df["text_review_count"].sum() == 42_687
    assert df["complaint_volume"].sum() == 4_667


def test_customer_voice_monthly_no_null_output():
    df = build_customer_voice_monthly(
        CUSTOMER_VOICE_PATH,
        COMPLAINT_PATH,
    )

    assert df[OUTPUT_COLUMNS].isna().sum().sum() == 0


def test_customer_voice_monthly_date_range():
    df = build_customer_voice_monthly(
        CUSTOMER_VOICE_PATH,
        COMPLAINT_PATH,
    )

    assert df["month"].min() == pd.Timestamp("2016-10-01")
    assert df["month"].max() == pd.Timestamp("2018-08-01")