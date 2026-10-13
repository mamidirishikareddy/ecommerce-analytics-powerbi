import pandas as pd

from src.marts.customer_performance_monthly import (
    build_customer_performance_monthly,
)


def _write_warehouse(tmp_path):

    pd.DataFrame(
        {
            "order_id": ["o1", "o2", "o3"],
            "customer_id": ["c1", "c1", "c2"],
            "order_purchase_timestamp": pd.to_datetime(
                [
                    "2024-01-10",
                    "2024-02-10",
                    "2024-01-15",
                ]
            ),
        }
    ).to_parquet(
        tmp_path / "fact_order.parquet",
        index=False,
    )

    pd.DataFrame(
        {
            "order_id": ["o1", "o1", "o2", "o3"],
            "price": [100.0, 20.0, 50.0, 80.0],
            "freight_value": [10.0, 2.0, 5.0, 8.0],
        }
    ).to_parquet(
        tmp_path / "fact_order_item.parquet",
        index=False,
    )

    pd.DataFrame(
        {
            "customer_id": ["c1", "c2"],
            "customer_unique_id": ["u1", "u2"],
        }
    ).to_parquet(
        tmp_path / "dim_customer.parquet",
        index=False,
    )

    pd.DataFrame(
        {
            "date": pd.to_datetime(
                [
                    "2024-01-01",
                    "2024-02-01",
                ]
            )
        }
    ).to_parquet(
        tmp_path / "dim_date.parquet",
        index=False,
    )


def test_customer_month_grain(tmp_path):

    _write_warehouse(tmp_path)

    mart = build_customer_performance_monthly(
        tmp_path
    )

    assert len(mart) == 3

    assert (
        mart[
            [
                "month",
                "customer_unique_id",
            ]
        ]
        .duplicated()
        .sum()
        == 0
    )


def test_customer_status(tmp_path):

    _write_warehouse(tmp_path)

    mart = build_customer_performance_monthly(
        tmp_path
    )

    u1 = mart[
        mart["customer_unique_id"] == "u1"
    ].sort_values("month")

    assert list(u1["customer_status"]) == [
        "NEW",
        "REPEAT",
    ]


def test_customer_revenue_and_aov(tmp_path):

    _write_warehouse(tmp_path)

    mart = build_customer_performance_monthly(
        tmp_path
    )

    january_u1 = mart[
        (mart["customer_unique_id"] == "u1")
        & (mart["month"] == pd.Timestamp("2024-01-01"))
    ].iloc[0]

    assert january_u1["order_count"] == 1
    assert january_u1["item_line_count"] == 2
    assert january_u1["revenue"] == 120.0
    assert january_u1["freight_value"] == 12.0
    assert january_u1["avg_order_value"] == 120.0