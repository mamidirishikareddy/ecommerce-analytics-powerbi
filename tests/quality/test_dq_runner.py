import pandas as pd
import pytest

from src.quality.dq_runner import (
    SUPPORTED_EVALUATORS,
    execute_rule,
    load_execution_config,
    prepare_derived_datasets,
)


def test_supported_evaluators_are_defined():
    expected = {
        "allowed_values",
        "conditional_value",
        "minimum",
        "not_null",
        "range",
        "reconciliation",
        "reference_integrity",
        "temporal_order",
        "unique",
    }

    assert SUPPORTED_EVALUATORS == expected


def test_load_execution_config():
    config = load_execution_config(
        __import__("pathlib").Path("configs/dq_execution.yaml")
    )

    assert config["version"] == "1.0"
    assert len(config["datasets"]) == 10
    assert len(config["rules"]) == 44


def test_prepare_derived_datasets():
    orders = pd.DataFrame(
        {
            "order_id": ["O1", "O2"],
            "order_purchase_timestamp": pd.to_datetime(
                ["2024-01-01", "2024-01-02"]
            ),
        }
    )

    order_items = pd.DataFrame(
        {
            "order_id": ["O1", "O1", "O2"],
            "shipping_limit_date": pd.to_datetime(
                ["2024-01-03", "2024-01-04", "2024-01-05"]
            ),
            "price": [100.0, 50.0, 80.0],
            "freight_value": [10.0, 5.0, 8.0],
        }
    )

    payments = pd.DataFrame(
        {
            "order_id": ["O1", "O2"],
            "payment_value": [165.0, 88.0],
        }
    )

    datasets = {
        "orders": orders,
        "order_items": order_items,
        "payments": payments,
    }

    derived = prepare_derived_datasets(datasets)

    assert set(derived) == {
        "order_item_temporal",
        "order_reconciliation",
    }

    assert len(derived["order_item_temporal"]) == 3
    assert len(derived["order_reconciliation"]) == 2


def test_execute_rule_not_null():
    df = pd.DataFrame(
        {
            "value": [1, 2, None],
        }
    )

    result = execute_rule(
        "TEST-001",
        {
            "evaluator": "not_null",
            "dataset": "test",
            "column": "value",
        },
        {"test": df},
    )

    assert result["rule_id"] == "TEST-001"
    assert result["status"] == "FAIL"
    assert result["affected_records"] == 1


def test_execute_rule_unique():
    df = pd.DataFrame(
        {
            "id": ["A", "B", "B"],
        }
    )

    result = execute_rule(
        "TEST-002",
        {
            "evaluator": "unique",
            "dataset": "test",
            "columns": ["id"],
        },
        {"test": df},
    )

    assert result["status"] == "FAIL"
    assert result["affected_records"] == 2


def test_load_execution_config_rejects_unknown_evaluator(tmp_path):
    config_path = tmp_path / "invalid.yaml"

    config_path.write_text(
        """
version: "1.0"
datasets: {}
rules:
  TEST-001:
    evaluator: unsupported_evaluator
""",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Unsupported evaluator"):
        load_execution_config(config_path)


def test_run_dq_executes_all_governed_rules():
    from pathlib import Path

    from src.quality.dq_runner import run_dq

    results = run_dq(
        Path("data/raw"),
        Path("configs/dq_execution.yaml"),
    )

    assert len(results) == 44
    assert set(results["status"]) <= {"PASS", "FAIL"}
    assert results["rule_id"].is_unique
    assert results["affected_records"].ge(0).all()
    assert results["total_records"].ge(0).all()