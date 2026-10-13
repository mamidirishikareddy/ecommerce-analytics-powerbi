from pathlib import Path

import pandas as pd
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]

RAW_DIR = PROJECT_ROOT / "data" / "raw"
MAPPING_PATH = PROJECT_ROOT / "configs" / "staging_mapping.yaml"

SUPPORTED_DTYPES = {
    "string",
    "integer",
    "decimal",
    "datetime",
}


def load_mapping():
    with MAPPING_PATH.open(
        "r",
        encoding="utf-8",
    ) as file:
        return yaml.safe_load(file)["datasets"]


def test_staging_mapping_declares_nine_datasets():
    mapping = load_mapping()

    assert len(mapping) == 9


def test_all_raw_columns_are_mapped_exactly_once():
    mapping = load_mapping()

    failures = []

    for name, definition in mapping.items():
        raw_columns = set(
            pd.read_csv(
                RAW_DIR / definition["source_file"],
                nrows=0,
            ).columns
        )

        mapped_columns = set(
            definition["columns"]
        )

        if raw_columns != mapped_columns:
            failures.append(
                {
                    "dataset": name,
                    "raw_not_mapped": sorted(
                        raw_columns - mapped_columns
                    ),
                    "mapping_not_in_raw": sorted(
                        mapped_columns - raw_columns
                    ),
                }
            )

    assert failures == []


def test_target_column_names_are_unique():
    mapping = load_mapping()

    failures = []

    for name, definition in mapping.items():
        targets = [
            column_definition["target"]
            for column_definition in definition["columns"].values()
        ]

        duplicates = {
            target
            for target in targets
            if targets.count(target) > 1
        }

        if duplicates:
            failures.append(
                {
                    "dataset": name,
                    "duplicate_targets": sorted(duplicates),
                }
            )

    assert failures == []


def test_all_declared_dtypes_are_supported():
    mapping = load_mapping()

    invalid = []

    for name, definition in mapping.items():
        for source_column, column_definition in (
            definition["columns"].items()
        ):
            dtype = column_definition["dtype"]

            if dtype not in SUPPORTED_DTYPES:
                invalid.append(
                    {
                        "dataset": name,
                        "column": source_column,
                        "dtype": dtype,
                    }
                )

    assert invalid == []


def test_every_mapping_has_source_target_and_dtype():
    mapping = load_mapping()

    failures = []

    for dataset, definition in mapping.items():
        for source_column, column_definition in (
            definition["columns"].items()
        ):
            required = {
                "target",
                "dtype",
            }

            missing = required - set(column_definition)

            if missing:
                failures.append(
                    {
                        "dataset": dataset,
                        "column": source_column,
                        "missing": sorted(missing),
                    }
                )

    assert failures == []