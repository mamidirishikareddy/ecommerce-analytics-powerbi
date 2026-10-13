import yaml
from pathlib import Path


project_root = Path(__file__).resolve().parents[1]

warehouse_schema = yaml.safe_load(
    (project_root / "docs" / "data_dictionary" / "warehouse_schema.yaml")
    .read_text(encoding="utf-8")
)["warehouse_schema"]

staging_mapping = yaml.safe_load(
    (project_root / "configs" / "staging_mapping.yaml")
    .read_text(encoding="utf-8")
)["datasets"]

staging = {
    dataset["staging_name"]: dataset
    for dataset in staging_mapping.values()
}

errors = []
source_backed_columns = 0
generated_columns = 0

for group_name in ("facts", "dimensions"):
    for table_name, table in warehouse_schema[group_name].items():
        source_staging = table.get("source_staging")

        for column_name, column in table["columns"].items():
            source_column = column.get("source_column")

            if source_column is None:
                generated_columns += 1
                continue

            source_backed_columns += 1

            if isinstance(source_staging, list):
                errors.append(
                    f"{table_name}: source-backed columns cannot use "
                    f"an empty/generated staging source"
                )
                continue

            if source_staging not in staging:
                errors.append(
                    f"{table_name}.{column_name}: "
                    f"staging dataset '{source_staging}' not found"
                )
                continue

            staging_columns = {
                mapping["target"]
                for mapping in staging[source_staging]["columns"].values()
            }

            if source_column not in staging_columns:
                errors.append(
                    f"{table_name}.{column_name}: "
                    f"source column '{source_column}' not found "
                    f"in {source_staging}"
                )

print("SOURCE-BACKED COLUMNS CHECKED:", source_backed_columns)
print("GENERATED COLUMNS:", generated_columns)
print("LINEAGE ERRORS:", len(errors))

for error in errors:
    print("ERROR:", error)

assert not errors, "Warehouse column lineage validation failed"

print("COLUMN LINEAGE VALIDATION: PASS")
