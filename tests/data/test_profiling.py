from pathlib import Path

import pandas as pd
import pytest

from src.profiling.profile_dataset import profile_dataset


RAW_DIR = Path("data/raw")


def test_profile_orders_dataset():
    result = profile_dataset(RAW_DIR / "olist_orders_dataset.csv")

    assert result["file_name"] == "olist_orders_dataset.csv"
    assert result["row_count"] == 99_441
    assert result["column_count"] == 8
    assert result["duplicate_row_count"] == 0

    profile = result["column_profile"]

    assert isinstance(profile, pd.DataFrame)
    assert len(profile) == 8
    assert set(
        [
            "column_name",
            "dtype",
            "row_count",
            "missing_count",
            "missing_pct",
            "unique_count",
        ]
    ).issubset(profile.columns)


def test_profile_missing_file_raises_error():
    with pytest.raises(FileNotFoundError):
        profile_dataset(RAW_DIR / "does_not_exist.csv")


def test_profile_non_csv_file_raises_error(tmp_path):
    file_path = tmp_path / "sample.txt"
    file_path.write_text("test")

    with pytest.raises(ValueError):
        profile_dataset(file_path)
