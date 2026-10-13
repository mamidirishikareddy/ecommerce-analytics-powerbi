import pandas as pd

from src.marts.persistence import save_mart


def test_save_mart(tmp_path):
    df = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-01"]),
            "product_id": ["p1"],
            "revenue": [100.0],
        }
    )

    path = save_mart(
        df=df,
        mart_name="mart_product_performance_daily",
        output_dir=tmp_path,
    )

    assert path.exists()

    loaded = pd.read_parquet(path)

    pd.testing.assert_frame_equal(
        loaded,
        df,
    )