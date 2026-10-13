import yaml
from pathlib import Path
import pandas as pd


schema = yaml.safe_load(
    Path("docs/data_dictionary/warehouse_schema.yaml")
    .read_text(encoding="utf-8")
)["warehouse_schema"]

staging = {
    "stg_orders": "data/staging/stg_orders.parquet",
    "stg_order_items": "data/staging/stg_order_items.parquet",
    "stg_payments": "data/staging/stg_payments.parquet",
    "stg_reviews": "data/staging/stg_reviews.parquet",
    "stg_customers": "data/staging/stg_customers.parquet",
    "stg_products": "data/staging/stg_products.parquet",
    "stg_sellers": "data/staging/stg_sellers.parquet",
    "stg_geolocation": "data/staging/stg_geolocation.parquet",
    "stg_category_translation": "data/staging/stg_category_translation.parquet",
}

errors = []
checked = 0
nullable_columns = 0
non_nullable_columns = 0

for group_name in ("facts", "dimensions"):
    for table_name, table in schema[group_name].items():

        source_staging = table.get("source_staging")

        if not isinstance(source_staging, str):
            continue

        df = pd.read_parquet(staging[source_staging])

        for column_name, column in table["columns"].items():

            source_column = column.get("source_column")

            if source_column is None:
                continue

            checked += 1
            declared_nullable = column["nullable"]
            actual_nulls = int(df[source_column].isna().sum())

            if declared_nullable:
                nullable_columns += 1
            else:
                non_nullable_columns += 1

            if not declared_nullable and actual_nulls > 0:
                errors.append(
                    f"{table_name}.{column_name}: "
                    f"declared nullable=false but staged data contains "
                    f"{actual_nulls} nulls"
                )

print("SOURCE-BACKED COLUMNS CHECKED:", checked)
print("DECLARED NULLABLE COLUMNS:", nullable_columns)
print("DECLARED NON-NULLABLE COLUMNS:", non_nullable_columns)
print("NULLABILITY ERRORS:", len(errors))

for error in errors:
    print("ERROR:", error)

assert not errors, "Warehouse nullability validation failed"

print("WAREHOUSE NULLABILITY VALIDATION: PASS")
