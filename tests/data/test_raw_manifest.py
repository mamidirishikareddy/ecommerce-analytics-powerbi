from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
MANIFEST_PATH = PROJECT_ROOT / "outputs" / "raw_data_manifest.csv"


def test_review_row_count_uses_parsed_csv_records():
    df = pd.read_csv(MANIFEST_PATH)

    review_rows = int(
        df.loc[
            df["file_name"] == "olist_order_reviews_dataset.csv",
            "row_count",
        ].iloc[0]
    )

    assert review_rows == 99_224


def test_manifest_row_counts_are_positive():
    df = pd.read_csv(MANIFEST_PATH)

    assert (df["row_count"] > 0).all()
    assert (df["column_count"] > 0).all()
    assert (df["file_size_bytes"] > 0).all()
