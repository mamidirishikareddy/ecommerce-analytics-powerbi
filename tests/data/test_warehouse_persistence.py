import pandas as pd

from src.warehouse.persistence import (
    WarehousePersistenceError,
    save_all_warehouse_tables,
    save_warehouse_table,
)


def test_save_warehouse_table(tmp_path):
    df = pd.DataFrame(
        {
            "id": ["a", "b"],
            "value": [10, 20],
        }
    )

    path = save_warehouse_table(
        df=df,
        table_name="fact_test",
        output_dir=tmp_path,
    )

    assert path.exists()
    assert path.name == "fact_test.parquet"

    loaded = pd.read_parquet(path)

    pd.testing.assert_frame_equal(
        loaded,
        df,
    )


def test_save_all_warehouse_tables(tmp_path):
    tables = {
        "fact_order": pd.DataFrame(
            {"order_id": ["o1", "o2"]}
        ),
        "dim_product": pd.DataFrame(
            {"product_id": ["p1", "p2"]}
        ),
    }

    paths = save_all_warehouse_tables(
        tables=tables,
        output_dir=tmp_path,
    )

    assert set(paths) == {
        "fact_order",
        "dim_product",
    }

    assert all(
        path.exists()
        for path in paths.values()
    )


def test_persistence_creates_output_directory(tmp_path):
    output_dir = tmp_path / "warehouse"

    df = pd.DataFrame(
        {"id": ["a"]}
    )

    path = save_warehouse_table(
        df=df,
        table_name="test_table",
        output_dir=output_dir,
    )

    assert output_dir.exists()
    assert path.exists()