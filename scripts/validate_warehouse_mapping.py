import yaml
from pathlib import Path


SCHEMA_PATH = Path("docs/data_dictionary/warehouse_schema.yaml")
MAPPING_PATH = Path("configs/warehouse_mapping.yaml")


schema = yaml.safe_load(
    SCHEMA_PATH.read_text(encoding="utf-8")
)["warehouse_schema"]

mapping = yaml.safe_load(
    MAPPING_PATH.read_text(encoding="utf-8")
)["warehouse_mapping"]


schema_tables = {
    **schema["facts"],
    **schema["dimensions"],
}

mapping_tables = {
    **mapping["facts"],
    **mapping["dimensions"],
}


errors = []


# ---------------------------------------------------------------------
# Table coverage
# ---------------------------------------------------------------------

schema_table_names = set(schema_tables)
mapping_table_names = set(mapping_tables)

missing_tables = schema_table_names - mapping_table_names
extra_tables = mapping_table_names - schema_table_names

for table_name in sorted(missing_tables):
    errors.append(
        f"Missing mapping for warehouse table: {table_name}"
    )

for table_name in sorted(extra_tables):
    errors.append(
        f"Mapping contains unknown warehouse table: {table_name}"
    )


# ---------------------------------------------------------------------
# Table-level metadata
# ---------------------------------------------------------------------

for table_name in sorted(schema_table_names & mapping_table_names):

    schema_table = schema_tables[table_name]
    mapping_table = mapping_tables[table_name]

    # Grain
    if mapping_table.get("grain") != schema_table.get("grain"):
        errors.append(
            f"{table_name}: grain mismatch"
        )

    # Business key
    schema_business_key = schema_table.get("business_key")

    if schema_business_key is None:
        schema_business_key = [
            column_name
            for column_name, column in schema_table["columns"].items()
            if column.get("role") == "business_key"
        ]

    mapping_business_key = mapping_table.get("business_key", [])

    if schema_business_key != mapping_business_key:
        errors.append(
            f"{table_name}: business key mismatch: "
            f"schema={schema_business_key}, "
            f"mapping={mapping_business_key}"
        )

    # Source staging
    schema_source = schema_table.get("source_staging")
    mapping_source = mapping_table.get("source_staging")

    if schema_source != mapping_source:
        errors.append(
            f"{table_name}: source staging mismatch: "
            f"schema={schema_source}, "
            f"mapping={mapping_source}"
        )


# ---------------------------------------------------------------------
# Column coverage and lineage
# ---------------------------------------------------------------------

for table_name in sorted(schema_table_names & mapping_table_names):

    schema_table = schema_tables[table_name]
    mapping_table = mapping_tables[table_name]

    schema_columns = schema_table["columns"]
    mapping_columns = mapping_table.get("columns", {})

    # Generated columns are identified by the absence of source_column
    schema_generated_columns = {
        column_name
        for column_name, column in schema_columns.items()
        if column.get("source_column") is None
    }

    # Source-backed columns only
    schema_source_columns = {
        column_name
        for column_name, column in schema_columns.items()
        if column.get("source_column") is not None
    }

    mapping_column_names = set(mapping_columns)

    # ---------------------------------------------------------------
    # Source-backed column coverage
    # ---------------------------------------------------------------

    missing_source_columns = (
        schema_source_columns - mapping_column_names
    )

    extra_mapping_columns = (
        mapping_column_names - schema_source_columns
    )

    for column_name in sorted(missing_source_columns):
        errors.append(
            f"{table_name}: missing mapping for source-backed "
            f"column '{column_name}'"
        )

    for column_name in sorted(extra_mapping_columns):
        errors.append(
            f"{table_name}: mapping contains unknown or generated "
            f"column '{column_name}'"
        )

    # ---------------------------------------------------------------
    # Source lineage
    # ---------------------------------------------------------------

    for column_name in sorted(
        schema_source_columns & mapping_column_names
    ):

        schema_column = schema_columns[column_name]

        schema_source_column = schema_column.get("source_column")
        mapping_source_column = mapping_columns[column_name]

        if mapping_source_column != schema_source_column:
            errors.append(
                f"{table_name}.{column_name}: source column mismatch: "
                f"schema={schema_source_column}, "
                f"mapping={mapping_source_column}"
            )

    # ---------------------------------------------------------------
    # Generated column coverage
    # ---------------------------------------------------------------

    mapping_generated_columns = set(
        mapping_table.get("generated_columns", [])
    )

    if schema_generated_columns != mapping_generated_columns:
        errors.append(
            f"{table_name}: generated column mismatch: "
            f"schema={sorted(schema_generated_columns)}, "
            f"mapping={sorted(mapping_generated_columns)}"
        )


# ---------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------

print("WAREHOUSE TABLES IN SCHEMA:", len(schema_table_names))
print("WAREHOUSE TABLES IN MAPPING:", len(mapping_table_names))
print("MAPPING ERRORS:", len(errors))

for error in errors:
    print("ERROR:", error)

assert not errors, "Warehouse mapping validation failed"

print("WAREHOUSE MAPPING VALIDATION: PASS")