import yaml
from pathlib import Path
import pandas as pd
from pandas.api.types import (
    is_datetime64_any_dtype,
    is_integer_dtype,
    is_numeric_dtype,
    is_string_dtype,
)


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

def dtype_matches(series, declared):
    actual = series.dtype

    if declared == "string":
        return is_string_dtype(actual)

    if declared == "integer":
        return is_integer_dtype(actual)

    if declared == "decimal":
        return is_numeric_dtype(actual) and not is_integer_dtype(actual)

    if declared == "datetime":
        return is_datetime64_any_dtype(actual)

    return False


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

            declared_dtype = column["dtype"]
            actual_dtype = df[source_column].dtype

            if not dtype_matches(df[source_column], declared_dtype):
                errors.append(
                    f"{table_name}.{column_name}: "
                    f"declared={declared_dtype}, "
                    f"actual={actual_dtype}"
                )


print("SOURCE-BACKED COLUMNS CHECKED:", checked)
print("TYPE ERRORS:", len(errors))

for error in errors:
    print("ERROR:", error)

assert not errors, "Warehouse physical type validation failed"

print("WAREHOUSE PHYSICAL TYPE VALIDATION: PASS")
