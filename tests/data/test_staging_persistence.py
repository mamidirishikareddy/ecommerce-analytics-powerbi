import pandas as pd

from src.staging.persistence import (
    save_all_staged_datasets,
    save_staged_dataset,
)


def test_save_staged_dataset(tmp_path):
    dataframe = pd.DataFrame(
        {
            "id": [1, 2],
            "value": ["A", "B"],
        }
    )

    output_path = tmp_path / "stg_test.parquet"

    returned_path = save_staged_dataset(
        dataframe,
        output_path,
    )

    assert returned_path == output_path
    assert output_path.exists()

    saved = pd.read_parquet(output_path)

    pd.testing.assert_frame_equal(
        saved,
        dataframe,
    )


def test_save_all_staged_datasets(tmp_path):
    staged = {
        "orders": pd.DataFrame(
            {"order_id": ["O1"]}
        ),
        "products": pd.DataFrame(
            {"product_id": ["P1"]}
        ),
    }

    mapping = {
        "datasets": {
            "orders": {
                "staging_name": "stg_orders",
            },
            "products": {
                "staging_name": "stg_products",
            },
        }
    }

    paths = save_all_staged_datasets(
        staged,
        tmp_path,
        mapping,
    )

    assert set(paths) == {
        "orders",
        "products",
    }

    assert paths["orders"].name == "stg_orders.parquet"
    assert paths["products"].name == "stg_products.parquet"

    assert paths["orders"].exists()
    assert paths["products"].exists()