from __future__ import annotations

from pathlib import Path

import pandas as pd
import yaml


SUPPORTED_DTYPES = {
    "string",
    "integer",
    "decimal",
    "datetime",
}


def load_staging_mapping(config_path: Path) -> dict:
    """Load and validate the staging mapping specification."""
    if not config_path.exists():
        raise FileNotFoundError(
            f"Staging mapping not found: {config_path}"
        )

    if config_path.suffix.lower() not in {".yaml", ".yml"}:
        raise ValueError(
            f"Expected a YAML mapping file: {config_path}"
        )

    with config_path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    if not isinstance(config, dict):
        raise ValueError("Staging mapping must be a mapping.")

    if "version" not in config:
        raise ValueError("Staging mapping is missing 'version'.")

    if "datasets" not in config:
        raise ValueError("Staging mapping is missing 'datasets'.")

    if not isinstance(config["datasets"], dict):
        raise ValueError("'datasets' must be a mapping.")

    for dataset_name, definition in config["datasets"].items():
        if not isinstance(definition, dict):
            raise ValueError(
                f"Definition for {dataset_name} must be a mapping."
            )

        required = {
            "source_file",
            "staging_name",
            "columns",
        }

        missing = required - set(definition)

        if missing:
            raise ValueError(
                f"{dataset_name} is missing: {sorted(missing)}"
            )

        for source_column, column_definition in (
            definition["columns"].items()
        ):
            if not isinstance(column_definition, dict):
                raise ValueError(
                    f"Column definition for "
                    f"{dataset_name}.{source_column} must be a mapping."
                )

            required_column_fields = {
                "target",
                "dtype",
            }

            missing_fields = (
                required_column_fields
                - set(column_definition)
            )

            if missing_fields:
                raise ValueError(
                    f"{dataset_name}.{source_column} is missing: "
                    f"{sorted(missing_fields)}"
                )

            if column_definition["dtype"] not in SUPPORTED_DTYPES:
                raise ValueError(
                    f"Unsupported dtype "
                    f"'{column_definition['dtype']}' for "
                    f"{dataset_name}.{source_column}."
                )

    return config


def _convert_column(
    series: pd.Series,
    dtype: str,
) -> pd.Series:
    """Convert one staging column to its governed target dtype."""
    if dtype == "string":
        return (
            series
            .astype("string")
            .str.strip()
        )

    if dtype == "integer":
        numeric = pd.to_numeric(
            series,
            errors="raise",
        )

        if numeric.notna().any():
            non_integer = (
                numeric.dropna() % 1 != 0
            )

            if non_integer.any():
                raise ValueError(
                    f"Non-integer value found in column "
                    f"'{series.name}'."
                )

        return numeric.astype("Int64")

    if dtype == "decimal":
        return pd.to_numeric(
            series,
            errors="raise",
        ).astype("Float64")

    if dtype == "datetime":
        return pd.to_datetime(
            series,
            errors="raise",
        )

    raise ValueError(
        f"Unsupported staging dtype: {dtype}"
    )


def transform_dataset(
    dataframe: pd.DataFrame,
    dataset_definition: dict,
) -> pd.DataFrame:
    """Transform one raw DataFrame according to the staging mapping."""
    source_columns = list(
        dataset_definition["columns"]
    )

    missing_columns = [
        column
        for column in source_columns
        if column not in dataframe.columns
    ]

    if missing_columns:
        raise KeyError(
            f"Missing source columns: {missing_columns}"
        )

    target_columns = [
        dataset_definition["columns"][column]["target"]
        for column in source_columns
    ]

    if len(target_columns) != len(set(target_columns)):
        raise ValueError(
            "Target staging column names must be unique."
        )

    staging = dataframe[source_columns].copy()

    renamed_columns = {
        source_column: dataset_definition["columns"][
            source_column
        ]["target"]
        for source_column in source_columns
    }

    staging = staging.rename(
        columns=renamed_columns
    )

    for source_column in source_columns:
        target_column = dataset_definition["columns"][
            source_column
        ]["target"]

        target_dtype = dataset_definition["columns"][
            source_column
        ]["dtype"]

        staging[target_column] = _convert_column(
            staging[target_column],
            target_dtype,
        )

    return staging


def transform_csv(
    source_path: Path,
    dataset_definition: dict,
) -> pd.DataFrame:
    """Load a raw CSV and transform it into its staging representation."""
    if not source_path.exists():
        raise FileNotFoundError(
            f"Source dataset not found: {source_path}"
        )

    dataframe = pd.read_csv(source_path)

    return transform_dataset(
        dataframe,
        dataset_definition,
    )


def transform_named_dataset(
    raw_dir: Path,
    mapping_config: dict,
    dataset_name: str,
) -> pd.DataFrame:
    """Transform one named dataset using the governed mapping."""
    datasets = mapping_config["datasets"]

    if dataset_name not in datasets:
        raise KeyError(
            f"Dataset not found in staging mapping: {dataset_name}"
        )

    definition = datasets[dataset_name]

    return transform_csv(
        raw_dir / definition["source_file"],
        definition,
    )