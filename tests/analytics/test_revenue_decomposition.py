
from pathlib import Path

import pandas as pd
import pytest

from src.analytics.revenue_decomposition import (
    RevenueDecompositionError,
    build_revenue_decomposition,
)


def _write_mart(
    base_dir: Path,
    filename: str,
    dataframe: pd.DataFrame,
) -> None:
    dataframe.to_parquet(base_dir / filename, index=False)


def _build_test_marts(tmp_path: Path) -> Path:
    marts_dir = tmp_path / "marts"
    marts_dir.mkdir()

    product = pd.DataFrame(
        {
            "date": pd.to_datetime(
                [
                    "2024-01-05",
                    "2024-01-20",
                    "2024-02-05",
                    "2024-02-20",
                ]
            ),
            "product_id": ["p1", "p2", "p1", "p2"],
            "revenue": [100.0, 100.0, 150.0, 150.0],
            "freight_value": [10.0, 10.0, 15.0, 15.0],
        }
    )

    customer = pd.DataFrame(
        {
            "month": pd.to_datetime(
                [
                    "2024-01-01",
                    "2024-01-01",
                    "2024-02-01",
                    "2024-02-01",
                    "2024-02-01",
                ]
            ),
            "customer_unique_id": [
                "c1",
                "c2",
                "c1",
                "c2",
                "c3",
            ],
        }
    )

    market = pd.DataFrame(
        {
            "month": pd.to_datetime(
                [
                    "2024-01-01",
                    "2024-01-01",
                    "2024-02-01",
                    "2024-02-01",
                ]
            ),
            "customer_zip_code_prefix": [
                1000,
                2000,
                1000,
                2000,
            ],
            "order_count": [1, 1, 2, 1],
        }
    )

    product_market = pd.DataFrame(
        {
            "month": pd.to_datetime(
                [
                    "2024-01-01",
                    "2024-01-01",
                ]
            ),
            "product_id": ["p1", "p2"],
            "customer_zip_code_prefix": [1000, 2000],
            "revenue": [100.0, 100.0],
        }
    )

    _write_mart(
        marts_dir,
        "mart_product_performance_daily.parquet",
        product,
    )

    _write_mart(
        marts_dir,
        "mart_customer_performance_monthly.parquet",
        customer,
    )

    _write_mart(
        marts_dir,
        "mart_market_performance_monthly.parquet",
        market,
    )

    _write_mart(
        marts_dir,
        "mart_product_market_performance_monthly.parquet",
        product_market,
    )

    return marts_dir


def test_revenue_decomposition_grain_and_columns(tmp_path):
    marts_dir = _build_test_marts(tmp_path)

    result = build_revenue_decomposition(marts_dir)

    assert len(result) == 2
    assert result["month"].is_unique

    assert list(result.columns) == [
        "month",
        "is_partial_month",
        "revenue",
        "freight_value",
        "order_count",
        "customer_count",
        "avg_order_value",
        "revenue_per_customer",
        "revenue_change",
        "order_count_change",
        "avg_order_value_change",
        "order_volume_effect",
        "avg_order_value_effect",
    ]


def test_revenue_metrics_and_bridge(tmp_path):
    marts_dir = _build_test_marts(tmp_path)

    result = build_revenue_decomposition(marts_dir)

    jan = result.iloc[0]
    feb = result.iloc[1]

    assert jan["revenue"] == pytest.approx(200.0)
    assert jan["freight_value"] == pytest.approx(20.0)
    assert jan["order_count"] == 2
    assert jan["customer_count"] == 2
    assert jan["avg_order_value"] == pytest.approx(100.0)
    assert jan["revenue_per_customer"] == pytest.approx(100.0)

    assert feb["revenue"] == pytest.approx(300.0)
    assert feb["freight_value"] == pytest.approx(30.0)
    assert feb["order_count"] == 3
    assert feb["customer_count"] == 3
    assert feb["avg_order_value"] == pytest.approx(100.0)
    assert feb["revenue_per_customer"] == pytest.approx(100.0)

    assert feb["revenue_change"] == pytest.approx(100.0)
    assert feb["order_count_change"] == pytest.approx(1.0)
    assert feb["avg_order_value_change"] == pytest.approx(0.0)

    assert feb["order_volume_effect"] == pytest.approx(100.0)
    assert feb["avg_order_value_effect"] == pytest.approx(0.0)

    assert (
        feb["order_volume_effect"]
        + feb["avg_order_value_effect"]
    ) == pytest.approx(feb["revenue_change"])


def test_partial_final_month_remains_in_output(tmp_path):
    marts_dir = _build_test_marts(tmp_path)

    result = build_revenue_decomposition(marts_dir)

    assert result["month"].tolist() == [
        pd.Timestamp("2024-01-01"),
        pd.Timestamp("2024-02-01"),
    ]

    assert "is_partial_month" in result.columns
    assert result["is_partial_month"].dtype == bool
    assert not result["is_partial_month"].any()


def test_missing_mart_fails(tmp_path):
    with pytest.raises(RevenueDecompositionError):
        build_revenue_decomposition(tmp_path)
