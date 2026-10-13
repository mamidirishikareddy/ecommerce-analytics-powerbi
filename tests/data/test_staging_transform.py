from pathlib import Path

import pandas as pd
import pytest
import yaml

from src.staging.transform import (
    _convert_column,
    load_staging_mapping,
    transform_dataset,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
MAPPING_PATH = PROJECT_ROOT / "configs" / "staging_mapping.yaml"


def load_mapping():
    return yaml.safe_load(
        MAPPING_PATH.read_text(encoding="utf-8")
    )


def test_load_staging_mapping():
    config = load_staging_mapping(MAPPING_PATH)

    assert config["version"] == "1.0"
    assert len(config["datasets"]) == 9


def test_transform_dataset_renames_and_converts():
    definition = {
        "source_file": "test.csv",
        "staging_name": "stg_test",
        "columns": {
            "name": {
                "target": "name",
                "dtype": "string",
            },
            "count": {
                "target": "count",
                "dtype": "integer",
            },
            "amount": {
                "target": "amount",
                "dtype": "decimal",
            },
            "event_time": {
                "target": "event_time",
                "dtype": "datetime",
            },
        },
    }

    source = pd.DataFrame(
        {
            "name": ["  Alice  ", "Bob"],
            "count": ["1", "2"],
            "amount": ["10.5", "20.0"],
            "event_time": [
                "2024-01-01 10:00:00",
                "2024-01-02 11:00:00",
            ],
        }
    )

    result = transform_dataset(
        source,
        definition,
    )

    assert list(result.columns) == [
        "name",
        "count",
        "amount",
        "event_time",
    ]

    assert result.loc[0, "name"] == "Alice"
    assert str(result["count"].dtype) == "Int64"
    assert str(result["amount"].dtype) == "Float64"
    assert pd.api.types.is_datetime64_any_dtype(
        result["event_time"]
    )


def test_transform_dataset_preserves_nulls():
    definition = {
        "source_file": "test.csv",
        "staging_name": "stg_test",
        "columns": {
            "value": {
                "target": "value",
                "dtype": "integer",
            },
        },
    }

    source = pd.DataFrame(
        {
            "value": [1, None, 3],
        }
    )

    result = transform_dataset(
        source,
        definition,
    )

    assert result["value"].isna().sum() == 1
    assert str(result["value"].dtype) == "Int64"


def test_transform_dataset_rejects_missing_source_column():
    definition = {
        "source_file": "test.csv",
        "staging_name": "stg_test",
        "columns": {
            "missing": {
                "target": "missing",
                "dtype": "string",
            },
        },
    }

    source = pd.DataFrame(
        {
            "value": ["A"],
        }
    )

    with pytest.raises(
        KeyError,
        match="Missing source columns",
    ):
        transform_dataset(
            source,
            definition,
        )


def test_convert_column_rejects_non_integer_values():
    series = pd.Series(
        ["1", "2.5"],
        name="count",
    )

    with pytest.raises(
        ValueError,
        match="Non-integer value",
    ):
        _convert_column(
            series,
            "integer",
        )


def test_mapping_target_names_are_preserved_for_products():
    config = load_mapping(
        )