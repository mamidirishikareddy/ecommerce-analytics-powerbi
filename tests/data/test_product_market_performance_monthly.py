from pathlib import Path

import pandas as pd
import pytest

from src.marts.product_market_performance_monthly import (
    ProductMarketPerformanceMartError,
    build_product_market_performance_monthly,
)


def _write_warehouse_fixture(tmp_path: Path) -> Path:
    warehouse_dir = tmp_path / "warehouse"
    warehouse_dir.mkdir()

    fact_order = pd.DataFrame(
        {
            "order_id": ["o1", "o2", "o3"],
            "customer_id": ["c1", "c1", "c2"],
            "order_purchase_timestamp": [
                "2024-01-10 10:00:00",
                "2024-01-15 11:00:00",
                "2024-01-20 12:00:00",
            ],
        }
    )

    fact_order_item = pd.DataFrame(
        {
            "order_id": ["o1", "o1", "o2", "o3"],
            "order_item_id": [1, 2, 1, 1],
            "product_id": ["p1", "p1", "p2", "p1"],
            "price": [100.0, 20.0, 50.0, 80.0],
            "freight_value": [10.0, 5.0, 7.0, 8.0],
        }
    )

    dim_customer = pd.DataFrame(
        {
            "customer_id": ["c1", "c2"],
            "customer_unique_id": ["u1", "u2"],
            "customer_zip_code_prefix": [1000, 2000],
        }
    )

    dim_product = pd.DataFrame(
        {
            "product_id": ["p1", "p2"],
            "product_category_name": [
                "category_a",
                "category_b",
            ],
        }
    )

    dim_category = pd.DataFrame(
        {
            "product_category_name": [
                "category_a",
                "category_b",
            ],
            "product_category_name_english": [
                "Category A",
                "Category B",
            ],
        }
    )

    dim_date = pd.DataFrame(
        {
            "date": pd.to_datetime(
                [
                    "2024-01-01",
                    "2024-01-10",
                    "2024-01-15",
                    "2024-01-20",
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

    dim_product.to_parquet(
        warehouse_dir / "dim_product.parquet",
        index=False,
    )

    dim_category.to_parquet(
        warehouse_dir / "dim_category.parquet",
        index=False,
    )

    dim_date.to_parquet(
        warehouse_dir / "dim_date.parquet",
        index=False,
    )

    return warehouse_dir


def test_product_market_grain_and_columns(tmp_path):
    warehouse_dir = _write_warehouse_fixture(tmp_path)

    mart = build_product_market_performance_monthly(
        warehouse_dir
    )

    expected_columns = [
        "month",
        "product_id",
        "customer_zip_code_prefix",
        "product_category_name",
        "product_category_name_english",
        "order_count",
        "item_line_count",
        "customer_count",
        "revenue",
        "freight_value",
        "avg_order_value",
        "revenue_per_customer",
        "market_contribution_to_product_revenue_pct",
    ]

    assert list(mart.columns) == expected_columns

    assert not mart.duplicated(
        subset=[
            "month",
            "product_id",
            "customer_zip_code_prefix",
        ]
    ).any()


def test_product_market_metrics(tmp_path):
    warehouse_dir = _write_warehouse_fixture(tmp_path)

    mart = build_product_market_performance_monthly(
        warehouse_dir
    )

    p1_zip_1000 = mart[
        (mart["product_id"] == "p1")
        & (mart["customer_zip_code_prefix"] == 1000)
    ].iloc[0]

    assert p1_zip_1000["order_count"] == 1
    assert p1_zip_1000["item_line_count"] == 2
    assert p1_zip_1000["customer_count"] == 1
    assert p1_zip_1000["revenue"] == pytest.approx(120.0)
    assert p1_zip_1000["freight_value"] == pytest.approx(15.0)
    assert p1_zip_1000["avg_order_value"] == pytest.approx(120.0)
    assert p1_zip_1000["revenue_per_customer"] == pytest.approx(120.0)


def test_market_contribution_sums_to_100_percent(tmp_path):
    warehouse_dir = _write_warehouse_fixture(tmp_path)

    mart = build_product_market_performance_monthly(
        warehouse_dir
    )

    product_month_sums = (
        mart.groupby(
            [
                "month",
                "product_id",
            ]
        )["market_contribution_to_product_revenue_pct"]
        .sum()
    )

    assert product_month_sums.between(
        99.999999,
        100.000001,
    ).all()


def test_category_translation_is_preserved(tmp_path):
    warehouse_dir = _write_warehouse_fixture(tmp_path)

    mart = build_product_market_performance_monthly(
        warehouse_dir
    )

    p1 = mart[mart["product_id"] == "p1"]

    assert (
        p1["product_category_name"]
        == "category_a"
    ).all()

    assert (
        p1["product_category_name_english"]
        == "Category A"
    ).all()


def test_item_revenue_and_freight_reconcile(tmp_path):
    warehouse_dir = _write_warehouse_fixture(tmp_path)

    mart = build_product_market_performance_monthly(
        warehouse_dir
    )

    source_items = pd.read_parquet(
        warehouse_dir / "fact_order_item.parquet"
    )

    assert mart["revenue"].sum() == pytest.approx(
        source_items["price"].sum()
    )

    assert mart["freight_value"].sum() == pytest.approx(
        source_items["freight_value"].sum()
    )


def test_missing_category_translation_is_preserved(tmp_path):
    warehouse_dir = _write_warehouse_fixture(tmp_path)

    dim_category = pd.read_parquet(
        warehouse_dir / "dim_category.parquet"
    )

    dim_category.loc[
        dim_category["product_category_name"] == "category_a",
        "product_category_name_english",
    ] = pd.NA

    dim_category.to_parquet(
        warehouse_dir / "dim_category.parquet",
        index=False,
    )

    mart = build_product_market_performance_monthly(
        warehouse_dir
    )

    p1 = mart[mart["product_id"] == "p1"]

    assert p1[
        "product_category_name_english"
    ].isna().all()