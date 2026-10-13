import yaml
from pathlib import Path


schema = yaml.safe_load(
    Path("docs/data_dictionary/warehouse_schema.yaml")
    .read_text(encoding="utf-8")
)["warehouse_schema"]

tables = {
    **schema["facts"],
    **schema["dimensions"],
}

errors = []
business_keys = 0
foreign_keys = 0

for table_name, table in tables.items():

    columns = table["columns"]

    for column_name, column in columns.items():

        if column.get("role") == "business_key":
            business_keys += 1

        if column.get("role") == "foreign_key":
            foreign_keys += 1

            reference = column.get("references")

            if not reference:
                errors.append(
                    f"{table_name}.{column_name}: foreign key has no reference"
                )
                continue

            try:
                target_table, target_column = reference.split(".", 1)
            except ValueError:
                errors.append(
                    f"{table_name}.{column_name}: invalid reference '{reference}'"
                )
                continue

            if target_table not in tables:
                errors.append(
                    f"{table_name}.{column_name}: "
                    f"target table '{target_table}' does not exist"
                )
                continue

            if target_column not in tables[target_table]["columns"]:
                errors.append(
                    f"{table_name}.{column_name}: "
                    f"target column '{reference}' does not exist"
                )

print("WAREHOUSE TABLES:", len(tables))
print("BUSINESS KEY COLUMNS:", business_keys)
print("FOREIGN KEY COLUMNS:", foreign_keys)
print("KEY/REFERENCE ERRORS:", len(errors))

for error in errors:
    print("ERROR:", error)

assert not errors, "Warehouse key/reference validation failed"

print("WAREHOUSE KEY/REFERENCE VALIDATION: PASS")
