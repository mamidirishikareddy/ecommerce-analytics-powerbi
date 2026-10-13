import pandas as pd
import pytest

from src.warehouse.transform import (
    WarehouseTransformationError,
    transform_warehouse_table,
)


def test_direct_mapping_preserves_row_count(tmp_path):
    staging_dir = tmp_path / "staging"
    staging_dir.mkdir()

    source = pd.DataFrame(
        {
            "order_id": ["o1", "o2"],
            "customer_id": ["c1", "c2"],
            "order_status": ["delivered", "shipped"],
        }
    )

    source.to_parquet(
        staging_dir / "stg_orders.parquet",
        index=False,
    )

    mapping = {
        "source_staging": "stg_orders",
        "grain": "one row per order",
        "business_key": ["order_id"],
        "transformation": "direct_mapping",
        "generated_columns": [],
        "columns": {
            "order_id": "order_id",
            "customer_id": "customer_id",
            "order_status": "order_status",
        },
    }

    result = transform_warehouse_table(
        table_name="fact_order",
        mapping=mapping,
        staging_dir=staging_dir,
    )

    assert len(result) == len(source)
    assert list(result.columns) == [
        "order_id",
        "customer_id",
        "order_status",
    ]


def test_direct_mapping_raises_for_missing_source_column(tmp_path):
    staging_dir = tmp_path / "staging"
    staging_dir.mkdir()

    source = pd.DataFrame(
        {
            "order_id": ["o1"],
        }
    )

    source.to_parquet(
        staging_dir / "stg_orders.parquet",
        index=False,
    )

    mapping = {
        "source_staging": "stg_orders",
        "grain": "one row per order",
        "business_key": ["order_id"],
        "transformation": "direct_mapping",
        "generated_columns": [],
        "columns": {
            "order_id": "order_id",
            "customer_id": "customer_id",
        },
    }

    with pytest.raises(WarehouseTransformationError):
        transform_warehouse_table(
            table_name="fact_order",
            mapping=mapping,
            staging_dir=staging_dir,
        )


def test_generated_calendar():
    mapping = {
        "source_staging": [],
        "grain": "one row per calendar date",
        "business_key": ["date"],
        "transformation": "generated_calendar",
        "generated_columns": [
            "date",
            "year",
            "quarter",
            "month",
            "month_name",
            "day",
            "day_of_week",
            "day_name",
            "week_of_year",
        ],
        "columns": {},
    }

    result = transform_warehouse_table(
        table_name="dim_date",
        mapping=mapping,
        staging_dir="data/staging",
        calendar_start_date="2024-01-01",
        calendar_end_date="2024-01-03",
    )

    assert len(result) == 3

    assert list(result.columns) == [
        "date",
        "year",
        "quarter",
        "month",
        "month_name",
        "day",
        "day_of_week",
        "day_name",
        "week_of_year",
    ]

    assert result["date"].min() == pd.Timestamp("2024-01-01")
    assert result["date"].max() == pd.Timestamp("2024-01-03")


def test_unsupported_transformation(tmp_path):
    mapping = {
        "source_staging": "stg_orders",
        "transformation": "unsupported",
        "columns": {},
    }

    with pytest.raises(WarehouseTransformationError):
        transform_warehouse_table(
            table_name="fact_order",
            mapping=mapping,
            staging_dir=tmp_path,
        )