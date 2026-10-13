import pandas as pd

from src.marts.product_performance_daily import (
    build_product_performance_daily,
)


def _write_warehouse_fixture(tmp_path):
    pd.DataFrame(
        {
            "order_id": ["o1", "o2"],
            "customer_id": ["c1", "c2"],
            "order_status": ["delivered", "delivered"],
            "order_purchase_timestamp": pd.to_datetime(
                [
                    "2024-01-01 10:00:00",
                    "2024-01-01 12:00:00",
                ]
            ),
            "order_approved_at": pd.to_datetime(
                [
                    "2024-01-01 10:01:00",
                    "2024-01-01 12:01:00",
                ]
            ),
            "order_delivered_carrier_date": pd.to_datetime(
                [
                    "2024-01-02",
                    "2024-01-02",
                ]
            ),
            "order_delivered_customer_date": pd.to_datetime(
                [
                    "2024-01-03",
                    "2024-01-03",
                ]
            ),
            "order_estimated_delivery_date": pd.to_datetime(
                [
                    "2024-01-05",
                    "2024-01-05",
                ]
            ),
        }
    ).to_parquet(
        tmp_path / "fact_order.parquet",
        index=False,
    )

    pd.DataFrame(
        {
            "order_id": ["o1", "o1", "o2"],
            "order_item_id": [1, 2, 1],
            "product_id": ["p1", "p1", "p2"],
            "seller_id": ["s1", "s1", "s2"],
            "shipping_limit_date": pd.to_datetime(
                [
                    "2024-01-02",
                    "2024-01-02",
                    "2024-01-02",
                ]
            ),
            "price": [100.0, 50.0, 80.0],
            "freight_value": [10.0, 5.0, 8.0],
        }
    ).to_parquet(
        tmp_path / "fact_order_item.parquet",
        index=False,
    )

    pd.DataFrame(
        {
            "product_id": ["p1", "p2"],
            "product_category_name": ["cat1", "cat2"],
            "product_name_length": [10, 12],
            "product_description_length": [100, 120],
            "product_photos_qty": [2, 3],
            "product_weight_g": [1000.0, 1200.0],
            "product_length_cm": [10.0, 12.0],
            "product_height_cm": [5.0, 6.0],
            "product_width_cm": [5.0, 6.0],
        }
    ).to_parquet(
        tmp_path / "dim_product.parquet",
        index=False,
    )

    pd.DataFrame(
        {
            "product_category_name": ["cat1", "cat2"],
            "product_category_name_english": [
                "Category One",
                "Category Two",
            ],
        }
    ).to_parquet(
        tmp_path / "dim_category.parquet",
        index=False,
    )

    pd.DataFrame(
        {
            "customer_id": ["c1", "c2"],
            "customer_unique_id": ["u1", "u2"],
            "customer_zip_code_prefix": [1000, 2000],
            "customer_city": ["A", "B"],
            "customer_state": ["X", "Y"],
        }
    ).to_parquet(
        tmp_path / "dim_customer.parquet",
        index=False,
    )


def test_product_performance_daily_grain(tmp_path):
    _write_warehouse_fixture(tmp_path)

    mart = build_product_performance_daily(tmp_path)

    assert len(mart) == 2

    assert list(mart.columns) == [
        "date",
        "product_id",
        "product_category_name",
        "product_category_name_english",
        "item_line_count",
        "order_count",
        "customer_count",
        "revenue",
        "freight_value",
        "gross_merchandise_value",
        "avg_item_price",
    ]

    assert not mart.duplicated(
        subset=["date", "product_id"]
    ).any()


def test_revenue_and_freight_are_not_fanned_out(tmp_path):
    _write_warehouse_fixture(tmp_path)

    mart = build_product_performance_daily(tmp_path)

    p1 = mart.loc[
        mart["product_id"] == "p1"
    ].iloc[0]

    assert p1["item_line_count"] == 2
    assert p1["order_count"] == 1
    assert p1["customer_count"] == 1
    assert p1["revenue"] == 150.0
    assert p1["freight_value"] == 15.0
    assert p1["gross_merchandise_value"] == 150.0
    assert p1["avg_item_price"] == 75.0


def test_customer_count_uses_customer_unique_id(tmp_path):
    _write_warehouse_fixture(tmp_path)

    mart = build_product_performance_daily(tmp_path)

    assert mart["customer_count"].sum() == 2


def test_category_translation_is_preserved(tmp_path):
    _write_warehouse_fixture(tmp_path)

    mart = build_product_performance_daily(tmp_path)

    p1 = mart.loc[
        mart["product_id"] == "p1"
    ].iloc[0]

    assert p1["product_category_name"] == "cat1"
    assert p1["product_category_name_english"] == "Category One"