from pathlib import Path

import pandas as pd
import yaml


SCHEMA_PATH = Path("docs/data_dictionary/warehouse_schema.yaml")
MAPPING_PATH = Path("configs/warehouse_mapping.yaml")
STAGING_DIR = Path("data/staging")


EXPECTED_SOURCE_ROW_COUNTS = {
    "fact_order": 99441,
    "fact_order_item": 112650,
    "fact_payment": 103886,
    "fact_review": 99224,
    "dim_customer": 99441,
    "dim_product": 32951,
    "dim_seller": 3095,
    "dim_category": 71,
    "dim_geography": 1000163,
}


def load_yaml(path):
    return yaml.safe_load(
        path.read_text(encoding="utf-8")
    )


def load_staging_counts(mapping):
    counts = {}

    for group in ("facts", "dimensions"):
        for table_name, table in mapping[group].items():
            source_staging = table.get("source_staging")

            if not source_staging:
                continue

            path = STAGING_DIR / f"{source_staging}.parquet"

            if not path.exists():
                raise AssertionError(
                    f"{table_name}: staging file not found: {path}"
                )

            counts[table_name] = len(
                pd.read_parquet(path)
            )

    return counts


def validate_columns(table_name, df, schema_table):
    expected_columns = list(schema_table["columns"].keys())
    actual_columns = list(df.columns)

    assert actual_columns == expected_columns, (
        f"{table_name}: column mismatch\n"
        f"Expected: {expected_columns}\n"
        f"Actual:   {actual_columns}"
    )


def validate_row_count(table_name, df, mapping_table):
    source_staging = mapping_table.get("source_staging")

    if not source_staging:
        return

    expected = EXPECTED_SOURCE_ROW_COUNTS[table_name]

    assert len(df) == expected, (
        f"{table_name}: row count mismatch: "
        f"expected {expected}, got {len(df)}"
    )


def validate_nullability(table_name, df, schema_table):
    for column_name, column in schema_table["columns"].items():
        if column.get("nullable") is False:
            null_count = int(df[column_name].isna().sum())

            assert null_count == 0, (
                f"{table_name}.{column_name}: "
                f"{null_count} null values found in "
                "non-nullable column"
            )


def validate_types(table_name, df, schema_table):
    for column_name, column in schema_table["columns"].items():
        dtype = column["dtype"]
        series = df[column_name]

        if dtype == "string":
            assert pd.api.types.is_string_dtype(series), (
                f"{table_name}.{column_name}: "
                f"expected string, got {series.dtype}"
            )

        elif dtype == "integer":
            assert pd.api.types.is_integer_dtype(series), (
                f"{table_name}.{column_name}: "
                f"expected integer, got {series.dtype}"
            )

        elif dtype == "decimal":
            assert pd.api.types.is_numeric_dtype(series), (
                f"{table_name}.{column_name}: "
                f"expected numeric decimal, got {series.dtype}"
            )

            assert not pd.api.types.is_integer_dtype(series), (
                f"{table_name}.{column_name}: "
                f"expected decimal, got integer dtype"
            )

        elif dtype == "datetime":
            assert pd.api.types.is_datetime64_any_dtype(series), (
                f"{table_name}.{column_name}: "
                f"expected datetime, got {series.dtype}"
            )

        elif dtype == "date":
            assert pd.api.types.is_datetime64_any_dtype(series), (
                f"{table_name}.{column_name}: "
                f"expected date-compatible datetime, got {series.dtype}"
            )

        else:
            raise AssertionError(
                f"{table_name}.{column_name}: "
                f"unsupported schema dtype '{dtype}'"
            )


def validate_grain(table_name, df, schema_table):
    business_key = schema_table.get("business_key")

    if business_key is None:
        business_key = [
            column_name
            for column_name, column in schema_table["columns"].items()
            if column.get("role") == "business_key"
        ]

    if not business_key:
        return

    duplicate_count = int(
        df.duplicated(
            subset=business_key,
            keep=False
        ).sum()
    )

    assert duplicate_count == 0, (
        f"{table_name}: business key {business_key} "
        f"has {duplicate_count} rows participating in duplicates"
    )


def validate_warehouse_output(tables, schema, mapping):
    schema_tables = {
        **schema["facts"],
        **schema["dimensions"],
    }

    mapping_tables = {
        **mapping["facts"],
        **mapping["dimensions"],
    }

    expected_tables = set(schema_tables)

    assert set(tables) == expected_tables, (
        f"Warehouse table coverage mismatch: "
        f"expected {sorted(expected_tables)}, "
        f"got {sorted(tables)}"
    )

    assert set(mapping_tables) == expected_tables, (
        "Warehouse mapping coverage does not match schema"
    )

    for table_name in sorted(expected_tables):
        df = tables[table_name]
        schema_table = schema_tables[table_name]
        mapping_table = mapping_tables[table_name]

        validate_columns(
            table_name,
            df,
            schema_table,
        )

        validate_row_count(
            table_name,
            df,
            mapping_table,
        )

        validate_nullability(
            table_name,
            df,
            schema_table,
        )

        validate_types(
            table_name,
            df,
            schema_table,
        )

        validate_grain(
            table_name,
            df,
            schema_table,
        )

    print("WAREHOUSE TABLES VALIDATED:", len(expected_tables))
    print("WAREHOUSE OUTPUT VALIDATION: PASS")


if __name__ == "__main__":
    from src.warehouse.transform import transform_warehouse

    schema = load_yaml(SCHEMA_PATH)["warehouse_schema"]
    mapping = load_yaml(MAPPING_PATH)["warehouse_mapping"]

    tables = transform_warehouse(
        mapping_path=MAPPING_PATH,
        staging_dir=STAGING_DIR,
        calendar_start_date="2016-01-01",
        calendar_end_date="2020-04-09",
    )

    validate_warehouse_output(
        tables=tables,
        schema=schema,
        mapping=mapping,
    )