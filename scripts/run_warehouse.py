from pathlib import Path

from src.warehouse.persistence import save_all_warehouse_tables
from src.warehouse.transform import transform_warehouse


MAPPING_PATH = Path("configs/warehouse_mapping.yaml")
STAGING_DIR = Path("data/staging")
OUTPUT_DIR = Path("data/processed/warehouse")


def main() -> None:
    tables = transform_warehouse(
        mapping_path=MAPPING_PATH,
        staging_dir=STAGING_DIR,
        calendar_start_date="2016-01-01",
        calendar_end_date="2020-04-09",
    )

    saved_paths = save_all_warehouse_tables(
        tables=tables,
        output_dir=OUTPUT_DIR,
    )

    print("WAREHOUSE TABLES PERSISTED:", len(saved_paths))

    for table_name, path in saved_paths.items():
        print(
            f"{table_name}: "
            f"{len(tables[table_name])} rows -> {path}"
        )


if __name__ == "__main__":
    main()