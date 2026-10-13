from pathlib import Path

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]

CONTRACT_PATH = (
    PROJECT_ROOT
    / "docs"
    / "data_dictionary"
    / "warehouse_contract.yaml"
)


def load_contract():
    with CONTRACT_PATH.open(
        "r",
        encoding="utf-8",
    ) as file:
        return yaml.safe_load(file)["warehouse_contract"]


def test_warehouse_contract_has_expected_facts():
    warehouse = load_contract()

    assert set(warehouse["facts"]) == {
        "fact_order",
        "fact_order_item",
        "fact_payment",
        "fact_review",
    }


def test_warehouse_contract_has_expected_dimensions():
    warehouse = load_contract()

    assert set(warehouse["dimensions"]) == {
        "dim_date",
        "dim_customer",
        "dim_product",
        "dim_seller",
        "dim_category",
        "dim_geography",
    }


def test_all_facts_have_grain_and_business_key():
    warehouse = load_contract()

    for name, definition in warehouse["facts"].items():
        assert definition["grain"]
        assert "business_key" in definition


def test_all_dimensions_have_grain_and_business_key():
    warehouse = load_contract()

    for name, definition in warehouse["dimensions"].items():
        assert definition["grain"]
        assert "business_key" in definition


def test_expected_relationships_are_declared():
    warehouse = load_contract()

    assert set(warehouse["retained_relationships"]) == {
        "order_customer",
        "order_item_order",
        "order_item_product",
        "order_item_seller",
        "payment_order",
        "review_order",
    }


def test_warehouse_design_constraints_exist():
    warehouse = load_contract()

    assert len(warehouse["design_constraints"]) == 8
