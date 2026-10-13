from pathlib import Path

import pandas as pd
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]

RAW_DIR = PROJECT_ROOT / "data" / "raw"
CONTRACT_PATH = (
    PROJECT_ROOT
    / "docs"
    / "data_dictionary"
    / "staging_contract.yaml"
)


def load_contract():
    with CONTRACT_PATH.open(
        "r",
        encoding="utf-8",
    ) as file:
        return yaml.safe_load(file)["staging_contract"]


def test_staging_contract_declares_nine_datasets():
    contract = load_contract()

    assert len(contract["datasets"]) == 9


def test_all_declared_source_files_exist():
    contract = load_contract()

    missing_files = [
        name
        for name, definition in contract["datasets"].items()
        if not (
            RAW_DIR / definition["source_file"]
        ).exists()
    ]

    assert missing_files == []


def test_declared_natural_keys_exist_in_raw_schema():
    contract = load_contract()

    failures = []

    for name, definition in contract["datasets"].items():
        natural_key = definition["natural_key"]

        if not natural_key:
            continue

        columns = set(
            pd.read_csv(
                RAW_DIR / definition["source_file"],
                nrows=0,
            ).columns
        )

        missing_columns = set(natural_key) - columns

        if missing_columns:
            failures.append(
                {
                    "dataset": name,
                    "missing_columns": sorted(
                        missing_columns
                    ),
                }
            )

    assert failures == []


def test_geolocation_has_no_declared_single_column_natural_key():
    contract = load_contract()

    assert contract["datasets"]["geolocation"]["natural_key"] == []


def test_staging_contract_has_required_sections():
    contract = load_contract()

    assert "purpose" in contract
    assert "allowed_transformations" in contract
    assert "prohibited_transformations" in contract
    assert "datasets" in contract