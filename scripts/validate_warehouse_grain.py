import yaml
from pathlib import Path
import pandas as pd


WAREHOUSE_SCHEMA_PATH = Path(
    "docs/data_dictionary/warehouse_schema.yaml"
)

STAGING_DIR = Path("data/staging")


expected_grains = {
    "fact_order": ["order_id"],
    "fact_order_item": ["order_id", "order_item_id"],
    "fact_payment": ["order_id", "payment_sequential"],
    "fact_review": ["review_id", "order_id"],
    "dim_customer": ["customer_id"],
    "dim_product": ["product_id"],
    "dim_seller": ["seller_id"],
    "dim_category": ["product_category_name"],
}


schema = yaml.safe_load(
    WAREHOUSE_SCHEMA_PATH.read_text(encoding="utf-8")
)["warehouse_schema"]


tables = {
    **schema["facts"],
    **schema["dimensions"],
}


errors = []


for table_name, key_columns in expected_grains.items():

    if table_name not in tables:
        errors.append(
            f"{table_name}: table not found in warehouse schema"
        )
        continue

    table = tables[table_name]

    parquet_name = f"stg_{table_name.replace('fact_', '').replace('dim_', '')}.parquet"

    # Explicit staging-file mapping
    staging_files = {
        "fact_order": "stg_orders.parquet",
        "fact_order_item": "stg_order_items.parquet",
        "fact_payment": "stg_payments.parquet",
        "fact_review": "stg_reviews.parquet",
        "dim_customer": "stg_customers.parquet",
        "dim_product": "stg_products.parquet",
        "dim_seller": "stg_sellers.parquet",
        "dim_category": "stg_category_translation.parquet",
    }

    parquet_path = STAGING_DIR / staging_files[table_name]

    if not parquet_path.exists():
        errors.append(
            f"{table_name}: staging file not found: {parquet_path}"
        )
        continue

    df = pd.read_parquet(parquet_path)

    # ------------------------------------------------------------------
    # Validate that expected grain columns exist in the physical data
    # ------------------------------------------------------------------
    missing_columns = [
        column
        for column in key_columns
        if column not in df.columns
    ]

    if missing_columns:
        errors.append(
            f"{table_name}: expected grain columns missing "
            f"{missing_columns}"
        )
        continue

    # ------------------------------------------------------------------
    # Validate table-level business_key metadata.
    #
    # Composite business keys are declared at table level because a
    # column can simultaneously participate in a composite business key
    # and act as a foreign key.
    #
    # Example:
    # fact_order_item:
    #   business_key:
    #     - order_id
    #     - order_item_id
    #
    # If no table-level business_key exists, fall back to the
    # column-level role: business_key metadata.
    # ------------------------------------------------------------------
    schema_keys = table.get("business_key")

    if schema_keys is None:
        schema_keys = [
            column_name
            for column_name, column in table["columns"].items()
            if column.get("role") == "business_key"
        ]

    if schema_keys != key_columns:
        errors.append(
            f"{table_name}: schema business keys {schema_keys} "
            f"do not match expected grain {key_columns}"
        )

    # ------------------------------------------------------------------
    # Validate physical grain uniqueness
    # ------------------------------------------------------------------
    duplicate_count = int(
        df.duplicated(
            subset=key_columns,
            keep=False
        ).sum()
    )

    if duplicate_count > 0:
        errors.append(
            f"{table_name}: grain columns {key_columns} "
            f"contain {duplicate_count} rows participating in duplicates"
        )


print("GRAINS CHECKED:", len(expected_grains))
print("GRAIN ERRORS:", len(errors))

for error in errors:
    print("ERROR:", error)

assert not errors, "Source grain preservation validation failed"

print("SOURCE GRAIN PRESERVATION VALIDATION: PASS")