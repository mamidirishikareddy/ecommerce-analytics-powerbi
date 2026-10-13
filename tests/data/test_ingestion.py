from pathlib import Path

import pandas as pd
import pytest

from src.ingestion.load_raw import load_csv


RAW_DIR = Path("data/raw")


def test_load_orders_csv():
    path = RAW_DIR / "olist_orders_dataset.csv"

    df = load_csv(path)

    assert isinstance(df, pd.DataFrame)
    assert len(df) == 99_441
    assert list(df.columns) == [
        "order_id",
        "customer_id",
        "order_status",
        "order_purchase_timestamp",
        "order_approved_at",
        "order_delivered_carrier_date",
        "order_delivered_customer_date",
        "order_estimated_delivery_date",
    ]


def test_missing_file_raises_error():
    with pytest.raises(FileNotFoundError):
        load_csv(RAW_DIR / "does_not_exist.csv")


def test_non_csv_file_raises_error(tmp_path):
    file_path = tmp_path / "sample.txt"
    file_path.write_text("test")

    with pytest.raises(ValueError):
        load_csv(file_path)
