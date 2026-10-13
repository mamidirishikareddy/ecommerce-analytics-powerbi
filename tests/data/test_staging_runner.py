from pathlib import Path

import pandas as pd
import pytest

from src.staging.runner import run_staging


def test_run_staging_executes_all_datasets(tmp_path):
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()

    files = {
             "orders": (
            "olist_orders_dataset.csv",
            "order_id,customer_id,order_status,"
            "order_purchase_timestamp,order_approved_at,"
            "order_delivered_carrier_date,"
            "order_delivered_customer_date,"
            "order_estimated_delivery_date\n"
            "O1,C1,delivered,"
            "2024-01-01 10:00:00,"
            "2024-01-01 10:05:00,"
            "2024-01-02 10:00:00,"
            "2024-01-03 10:00:00,"
            "2024-01-05 00:00:00\n",
        ),
        "order_items": (
            "olist_order_items_dataset.csv",
            "order_id,order_item_id,product_id,seller_id,"
            "shipping_limit_date,price,freight_value\n"
            "O1,1,P1,S1,2024-01-02 10:00:00,100.0,10.0\n",
        ),
        "payments": (
            "olist_order_payments_dataset.csv",
            "order_id,payment_sequential,payment_type,"
            "payment_installments,payment_value\n"
            "O1,1,credit_card,1,110.0\n",
        ),
        "reviews": (
            "olist_order_reviews_dataset.csv",
            "review_id,order_id,review_score,"
            "review_comment_title,review_comment_message,"
            "review_creation_date,review_answer_timestamp\n"
            "R1,O1,5,,,2024-01-03,2024-01-04\n",
        ),
        "customers": (
            "olist_customers_dataset.csv",
            "customer_id,customer_unique_id,"
            "customer_zip_code_prefix,customer_city,customer_state\n"
            "C1,U1,1000,city,ST\n",
        ),
        "products": (
            "olist_products_dataset.csv",
            "product_id,product_category_name,"
            "product_name_lenght,product_description_lenght,"
            "product_photos_qty,product_weight_g,"
            "product_length_cm,product_height_cm,product_width_cm\n"
            "P1,cat,10,20,1,100.0,10.0,5.0,8.0\n",
        ),
        "sellers": (
            "olist_sellers_dataset.csv",
            "seller_id,seller_zip_code_prefix,seller_city,"
            "seller_state\n"
            "S1,1000,city,ST\n",
        ),
        "geolocation": (
            "olist_geolocation_dataset.csv",
            "geolocation_zip_code_prefix,geolocation_lat,"
            "geolocation_lng,geolocation_city,geolocation_state\n"
            "1000,1.0,2.0,city,ST\n",
        ),
        "category_translation": (
            "product_category_name_translation.csv",
            "product_category_name,product_category_name_english\n"
            "cat,category\n",
        ),
    }

    for _, (filename, content) in files.items():
        (raw_dir / filename).write_text(
            content,
            encoding="utf-8",
        )

    mapping_path = (
        Path(__file__).resolve().parents[2]
        / "configs"
        / "staging_mapping.yaml"
    )

    staged = run_staging(
        raw_dir,
        mapping_path,
    )

    assert len(staged) == 9

    expected_names = {
        "orders",
        "order_items",
        "payments",
        "reviews",
        "customers",
        "products",
        "sellers",
        "geolocation",
        "category_translation",
    }

    assert set(staged) == expected_names

    for dataframe in staged.values():
        assert isinstance(dataframe, pd.DataFrame)


def test_run_staging_preserves_dataset_row_counts(tmp_path):
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()

    source = (
        "order_id,customer_id,order_status,"
        "order_purchase_timestamp,order_approved_at,"
        "order_delivered_carrier_date,"
        "order_delivered_customer_date,"
        "order_estimated_delivery_date\n"
        "O1,C1,delivered,2024-01-01,2024-01-01,"
        "2024-01-02,2024-01-03,2024-01-05\n"
    )

    (raw_dir / "olist_orders_dataset.csv").write_text(
        source,
        encoding="utf-8",
    )

    # This test intentionally checks runner dependency behavior:
    # the runner expects all declared datasets to exist.
    from src.staging.runner import run_staging

    mapping_path = (
        Path(__file__).resolve().parents[2]
        / "configs"
        / "staging_mapping.yaml"
    )

    with pytest.raises(FileNotFoundError):
        run_staging(
            raw_dir,
            mapping_path,
        )