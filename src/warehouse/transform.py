from pathlib import Path

import pandas as pd
import yaml


class WarehouseTransformationError(Exception):
    """Raised when a warehouse transformation cannot be completed."""


def load_warehouse_mapping(mapping_path: str | Path) -> dict:
    """
    Load and validate the warehouse transformation mapping.
    """
    path = Path(mapping_path)

    if not path.exists():
        raise WarehouseTransformationError(
            f"Warehouse mapping file not found: {path}"
        )

    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    if not isinstance(config, dict):
        raise WarehouseTransformationError(
            "Warehouse mapping configuration must be a mapping."
        )

    if "warehouse_mapping" not in config:
        raise WarehouseTransformationError(
            "Missing 'warehouse_mapping' section."
        )

    return config["warehouse_mapping"]


def _load_staging_dataset(
    staging_dir: str | Path,
    staging_name: str,
) -> pd.DataFrame:
    """
    Load one staged Parquet dataset.
    """
    path = Path(staging_dir) / f"{staging_name}.parquet"

    if not path.exists():
        raise WarehouseTransformationError(
            f"Staging dataset not found: {path}"
        )

    return pd.read_parquet(path)


def _validate_source_columns(
    df: pd.DataFrame,
    source_columns: list[str],
    table_name: str,
) -> None:
    """
    Ensure all required source columns exist.
    """
    missing = [
        column
        for column in source_columns
        if column not in df.columns
    ]

    if missing:
        raise WarehouseTransformationError(
            f"{table_name}: missing source columns: {missing}"
        )


def _transform_direct_mapping(
    df: pd.DataFrame,
    table_name: str,
    mapping: dict,
) -> pd.DataFrame:
    """
    Perform explicit one-to-one source-to-warehouse column mapping.

    No joins, aggregations, deduplication, imputation, or business
    calculations are performed here.
    """
    columns = mapping.get("columns", {})

    if not columns:
        raise WarehouseTransformationError(
            f"{table_name}: direct_mapping requires column mappings."
        )

    source_columns = list(columns.values())

    _validate_source_columns(
        df=df,
        source_columns=source_columns,
        table_name=table_name,
    )

    transformed = df[source_columns].copy()

    rename_map = {
        source_column: target_column
        for target_column, source_column in columns.items()
    }

    transformed = transformed.rename(columns=rename_map)

    expected_columns = list(columns.keys())

    if list(transformed.columns) != expected_columns:
        raise WarehouseTransformationError(
            f"{table_name}: transformed columns do not match mapping."
        )

    return transformed


def _transform_generated_calendar(
    table_name: str,
    mapping: dict,
    start_date: str | pd.Timestamp,
    end_date: str | pd.Timestamp,
) -> pd.DataFrame:
    """
    Generate the governed calendar dimension.
    """
    if mapping.get("transformation") != "generated_calendar":
        raise WarehouseTransformationError(
            f"{table_name}: unsupported generated transformation."
        )

    dates = pd.date_range(
        start=pd.Timestamp(start_date).normalize(),
        end=pd.Timestamp(end_date).normalize(),
        freq="D",
    )

    transformed = pd.DataFrame({"date": dates})

    transformed["year"] = transformed["date"].dt.year
    transformed["quarter"] = transformed["date"].dt.quarter
    transformed["month"] = transformed["date"].dt.month
    transformed["month_name"] = transformed["date"].dt.month_name()
    transformed["day"] = transformed["date"].dt.day
    transformed["day_of_week"] = transformed["date"].dt.dayofweek + 1
    transformed["day_name"] = transformed["date"].dt.day_name()
    transformed["week_of_year"] = (
        transformed["date"].dt.isocalendar().week.astype("int64")
    )

    expected_columns = mapping.get("generated_columns", [])

    if list(transformed.columns) != expected_columns:
        raise WarehouseTransformationError(
            f"{table_name}: generated calendar columns do not "
            "match mapping."
        )

    return transformed


def transform_warehouse_table(
    table_name: str,
    mapping: dict,
    staging_dir: str | Path,
    calendar_start_date: str | pd.Timestamp | None = None,
    calendar_end_date: str | pd.Timestamp | None = None,
) -> pd.DataFrame:
    """
    Transform one governed warehouse table.
    """
    transformation = mapping.get("transformation")

    if transformation == "direct_mapping":
        staging_name = mapping.get("source_staging")

        if not staging_name:
            raise WarehouseTransformationError(
                f"{table_name}: source_staging is required."
            )

        df = _load_staging_dataset(
            staging_dir=staging_dir,
            staging_name=staging_name,
        )

        return _transform_direct_mapping(
            df=df,
            table_name=table_name,
            mapping=mapping,
        )

    if transformation == "generated_calendar":
        if calendar_start_date is None or calendar_end_date is None:
            raise WarehouseTransformationError(
                f"{table_name}: calendar start and end dates "
                "are required."
            )

        return _transform_generated_calendar(
            table_name=table_name,
            mapping=mapping,
            start_date=calendar_start_date,
            end_date=calendar_end_date,
        )

    raise WarehouseTransformationError(
        f"{table_name}: unsupported transformation "
        f"'{transformation}'."
    )


def transform_warehouse(
    mapping_path: str | Path,
    staging_dir: str | Path,
    calendar_start_date: str | pd.Timestamp | None = None,
    calendar_end_date: str | pd.Timestamp | None = None,
) -> dict[str, pd.DataFrame]:
    """
    Transform all warehouse tables defined in the mapping.
    """
    mapping = load_warehouse_mapping(mapping_path)

    warehouse_tables = {
        **mapping["facts"],
        **mapping["dimensions"],
    }

    transformed_tables: dict[str, pd.DataFrame] = {}

    for table_name, table_mapping in warehouse_tables.items():
        transformed_tables[table_name] = transform_warehouse_table(
            table_name=table_name,
            mapping=table_mapping,
            staging_dir=staging_dir,
            calendar_start_date=calendar_start_date,
            calendar_end_date=calendar_end_date,
        )

    return transformed_tables