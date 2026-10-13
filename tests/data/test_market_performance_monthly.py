from pathlib import Path

import pandas as pd

from src.marts.market_performance_monthly import (
    build_market_performance_monthly,
)


def _write_warehouse_fixture(tmp_path: Path) -> Path:
    warehouse_dir = tmp_path / "warehouse"
    warehouse_dir.mkdir()

    fact_order = pd.DataFrame(
        {
            "order_id": ["o1", "o2", "o3"],
            "customer_id": ["c1", "c1", "c2"],
            "order_purchase_timestamp": [
                "2018-01-10 10:00:00",
                "2018-01-20 10:00:00",
                "2018-01-15 10:00:00",
            ],
        }
    )

    fact_order_item = pd.DataFrame(
        {
            "order_id": ["o1", "o1", "o2", "o3"],
            "price": [100.0, 20.0, 50.0, 80.0],
            "freight_value": [10.0, 2.0, 5.0, 8.0],
        }
    )

    dim_customer = pd.DataFrame(
        {
            "customer_id": ["c1", "c2"],
            "customer_unique_id": ["u1", "u2"],
            "customer_zip_code_prefix": [1000, 2000],
        }
    )

    dim_date = pd.DataFrame(
        {
            "date": pd.to_datetime(
                [
                    "2018-01-01",
                    "2018-02-01",
                ]
            )
        }
    )

    fact_order.to_parquet(
        warehouse_dir / "fact_order.parquet",
        index=False,
    )

    fact_order_item.to_parquet(
        warehouse_dir / "fact_order_item.parquet",
        index=False,
    )

    dim_customer.to_parquet(
        warehouse_dir / "dim_customer.parquet",
        index=False,
    )

    dim_date.to_parquet(
        warehouse_dir / "dim_date.parquet",
        index=False,
    )

    return warehouse_dir


def test_market_monthly_grain(tmp_path):
    warehouse_dir = _write_warehouse_fixture(tmp_path)

    mart = build_market_performance_monthly(
        warehouse_dir
    )

    assert len(mart) == 2

    assert not mart.duplicated(
        [
            "month",
            "customer_zip_code_prefix",
        ]
    ).any()


def test_market_monthly_metrics(tmp_path):
    warehouse_dir = _write_warehouse_fixture(tmp_path)

    mart = build_market_performance_monthly(
        warehouse_dir
    )

    market_1000 = mart[
        mart["customer_zip_code_prefix"] == 1000
    ].iloc[0]

    assert market_1000["order_count"] == 2
    assert market_1000["item_line_count"] == 3
    assert market_1000["customer_count"] == 1

    assert market_1000["revenue"] == 170.0
    assert market_1000["freight_value"] == 17.0

    assert market_1000["avg_order_value"] == 85.0
    assert market_1000["revenue_per_customer"] == 170.0
    assert market_1000["avg_freight_per_order"] == 8.5


def test_market_monthly_population_reconciliation(tmp_path):
    warehouse_dir = _write_warehouse_fixture(tmp_path)

    mart = build_market_performance_monthly(
        warehouse_dir
    )

    assert mart["revenue"].sum() == 250.0
    assert mart["freight_value"].sum() == 25.0