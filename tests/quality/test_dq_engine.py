from pathlib import Path

import pytest

import pandas as pd

from src.quality.dq_engine import (
    evaluate_allowed_values,
    evaluate_conditional_value,
    evaluate_minimum,
    evaluate_not_null,
    evaluate_range,
    evaluate_reference_integrity,
    evaluate_unique,
    load_rules,
    evaluate_temporal_order,
    evaluate_reconciliation,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RULES_PATH = PROJECT_ROOT / "configs" / "dq_rules.yaml"


def test_load_rules_returns_all_governed_rules():
    rules = load_rules(RULES_PATH)

    assert len(rules) == 44


def test_rule_ids_are_unique():
    rules = load_rules(RULES_PATH)
    rule_ids = [rule["rule_id"] for rule in rules]

    assert len(rule_ids) == len(set(rule_ids))


def test_all_rules_have_required_fields():
    rules = load_rules(RULES_PATH)
    required_fields = {
        "rule_id",
        "domain",
        "name",
        "severity",
        "description",
    }

    for rule in rules:
        assert required_fields.issubset(rule)


def test_missing_rule_registry_raises_error(tmp_path):
    missing_path = tmp_path / "missing.yaml"

    with pytest.raises(FileNotFoundError):
        load_rules(missing_path)

def test_evaluate_not_null_passes_complete_column():
    df = pd.DataFrame({"value": [1, 2, 3]})

    result = evaluate_not_null(
        df,
        rule_id="TEST-001",
        column="value",
    )

    assert result["status"] == "PASS"
    assert result["affected_records"] == 0
    assert result["affected_pct"] == 0.0


def test_evaluate_not_null_fails_incomplete_column():
    df = pd.DataFrame({"value": [1, None, 3]})

    result = evaluate_not_null(
        df,
        rule_id="TEST-002",
        column="value",
    )

    assert result["status"] == "FAIL"
    assert result["affected_records"] == 1
    assert result["affected_pct"] > 0

def test_evaluate_not_null_raises_for_missing_column():
    df = pd.DataFrame({"value": [1, 2, 3]})

    with pytest.raises(KeyError):
        evaluate_not_null(
            df,
            rule_id="TEST-003",
            column="missing_column",
        )

def test_evaluate_unique_passes_unique_column():
    df = pd.DataFrame({"id": [1, 2, 3]})

    result = evaluate_unique(
        df,
        rule_id="TEST-004",
        columns=["id"],
    )

    assert result["status"] == "PASS"
    assert result["affected_records"] == 0


def test_evaluate_unique_fails_duplicate_column():
    df = pd.DataFrame({"id": [1, 2, 2, 3]})

    result = evaluate_unique(
        df,
        rule_id="TEST-005",
        columns=["id"],
    )

    assert result["status"] == "FAIL"
    assert result["affected_records"] == 2
    assert result["affected_pct"] == 50.0


def test_evaluate_unique_supports_composite_key():
    df = pd.DataFrame(
        {
            "order_id": ["A", "A", "B"],
            "item_id": [1, 2, 1],
        }
    )

    result = evaluate_unique(
        df,
        rule_id="TEST-006",
        columns=["order_id", "item_id"],
    )

    assert result["status"] == "PASS"
def test_evaluate_unique_passes_unique_column():
    df = pd.DataFrame({"id": [1, 2, 3]})

    result = evaluate_unique(
        df,
        rule_id="TEST-004",
        columns=["id"],
    )

    assert result["status"] == "PASS"
    assert result["affected_records"] == 0


def test_evaluate_unique_fails_duplicate_column():
    df = pd.DataFrame({"id": [1, 2, 2, 3]})

    result = evaluate_unique(
        df,
        rule_id="TEST-005",
        columns=["id"],
    )

    assert result["status"] == "FAIL"
    assert result["affected_records"] == 2
    assert result["affected_pct"] == 50.0


def test_evaluate_unique_supports_composite_key():
    df = pd.DataFrame(
        {
            "order_id": ["A", "A", "B"],
            "item_id": [1, 2, 1],
        }
    )

    result = evaluate_unique(
        df,
        rule_id="TEST-006",
        columns=["order_id", "item_id"],
    )

    assert result["status"] == "PASS"


def test_evaluate_unique_raises_for_missing_column():
    df = pd.DataFrame({"id": [1, 2, 3]})

    with pytest.raises(KeyError):
        evaluate_unique(
            df,
            rule_id="TEST-007",
            columns=["missing_column"],
        )
def test_evaluate_range_passes_valid_values():
    df = pd.DataFrame({"score": [1, 3, 5]})

    result = evaluate_range(
        df,
        rule_id="TEST-008",
        column="score",
        minimum=1,
        maximum=5,
    )

    assert result["status"] == "PASS"
    assert result["affected_records"] == 0


def test_evaluate_range_fails_out_of_range_values():
    df = pd.DataFrame({"score": [1, 3, 5, 0, 6]})

    result = evaluate_range(
        df,
        rule_id="TEST-009",
        column="score",
        minimum=1,
        maximum=5,
    )

    assert result["status"] == "FAIL"
    assert result["affected_records"] == 2


def test_evaluate_range_ignores_null_values():
    df = pd.DataFrame({"score": [1, None, 5]})

    result = evaluate_range(
        df,
        rule_id="TEST-010",
        column="score",
        minimum=1,
        maximum=5,
    )

    assert result["status"] == "PASS"
    assert result["affected_records"] == 0


def test_evaluate_range_raises_for_missing_column():
    df = pd.DataFrame({"score": [1, 2, 3]})

    with pytest.raises(KeyError):
        evaluate_range(
            df,
            rule_id="TEST-011",
            column="missing_column",
            minimum=1,
            maximum=5,
        )

def test_evaluate_minimum_passes_valid_values():
    df = pd.DataFrame({"value": [0, 1, 2]})

    result = evaluate_minimum(
        df,
        rule_id="TEST-012",
        column="value",
        minimum=0,
    )

    assert result["status"] == "PASS"
    assert result["affected_records"] == 0


def test_evaluate_minimum_fails_values_below_threshold():
    df = pd.DataFrame({"value": [0, 1, -1, -5]})

    result = evaluate_minimum(
        df,
        rule_id="TEST-013",
        column="value",
        minimum=0,
    )

    assert result["status"] == "FAIL"
    assert result["affected_records"] == 2


def test_evaluate_minimum_ignores_null_values():
    df = pd.DataFrame({"value": [0, None, 2]})

    result = evaluate_minimum(
        df,
        rule_id="TEST-014",
        column="value",
        minimum=0,
    )

    assert result["status"] == "PASS"
    assert result["affected_records"] == 0


def test_evaluate_minimum_raises_for_missing_column():
    df = pd.DataFrame({"value": [0, 1, 2]})

    with pytest.raises(KeyError):
        evaluate_minimum(
            df,
            rule_id="TEST-015",
            column="missing_column",
            minimum=0,
        )

def test_reference_integrity_passes_valid_keys():
    child_df = pd.DataFrame({"parent_id": ["A", "B", "C"]})
    parent_df = pd.DataFrame({"id": ["A", "B", "C"]})

    result = evaluate_reference_integrity(
        child_df,
        parent_df,
        rule_id="TEST-016",
        child_column="parent_id",
        parent_column="id",
    )

    assert result["status"] == "PASS"
    assert result["affected_records"] == 0


def test_reference_integrity_fails_orphan_keys():
    child_df = pd.DataFrame({"parent_id": ["A", "B", "X"]})
    parent_df = pd.DataFrame({"id": ["A", "B", "C"]})

    result = evaluate_reference_integrity(
        child_df,
        parent_df,
        rule_id="TEST-017",
        child_column="parent_id",
        parent_column="id",
    )

    assert result["status"] == "FAIL"
    assert result["affected_records"] == 1


def test_reference_integrity_ignores_null_foreign_keys():
    child_df = pd.DataFrame({"parent_id": ["A", None, "B"]})
    parent_df = pd.DataFrame({"id": ["A", "B"]})

    result = evaluate_reference_integrity(
        child_df,
        parent_df,
        rule_id="TEST-018",
        child_column="parent_id",
        parent_column="id",
    )

    assert result["status"] == "PASS"
    assert result["affected_records"] == 0


def test_reference_integrity_raises_for_missing_column():
    child_df = pd.DataFrame({"parent_id": ["A", "B"]})
    parent_df = pd.DataFrame({"id": ["A", "B"]})

    with pytest.raises(KeyError):
        evaluate_reference_integrity(
            child_df,
            parent_df,
            rule_id="TEST-019",
            child_column="missing_column",
            parent_column="id",
        )
def test_evaluate_allowed_values_passes_valid_domain():
    df = pd.DataFrame(
        {"status": ["delivered", "shipped", "canceled"]}
    )

    result = evaluate_allowed_values(
        df,
        rule_id="TEST-020",
        column="status",
        allowed_values={"delivered", "shipped", "canceled"},
    )

    assert result["status"] == "PASS"
    assert result["affected_records"] == 0


def test_evaluate_allowed_values_fails_unknown_value():
    df = pd.DataFrame(
        {"status": ["delivered", "shipped", "unknown"]}
    )

    result = evaluate_allowed_values(
        df,
        rule_id="TEST-021",
        column="status",
        allowed_values={"delivered", "shipped", "canceled"},
    )

    assert result["status"] == "FAIL"
    assert result["affected_records"] == 1


def test_evaluate_allowed_values_ignores_null_values():
    df = pd.DataFrame(
        {"status": ["delivered", None, "shipped"]}
    )

    result = evaluate_allowed_values(
        df,
        rule_id="TEST-022",
        column="status",
        allowed_values={"delivered", "shipped"},
    )

    assert result["status"] == "PASS"
    assert result["affected_records"] == 0


def test_evaluate_allowed_values_raises_for_missing_column():
    df = pd.DataFrame({"status": ["delivered"]})

    with pytest.raises(KeyError):
        evaluate_allowed_values(
            df,
            rule_id="TEST-023",
            column="missing_column",
            allowed_values={"delivered"},
        )
def test_evaluate_temporal_order_passes_valid_sequence():
    df = pd.DataFrame(
        {
            "purchase": pd.to_datetime(
                ["2024-01-01", "2024-01-02"]
            ),
            "approved": pd.to_datetime(
                ["2024-01-01", "2024-01-03"]
            ),
        }
    )

    result = evaluate_temporal_order(
        df,
        rule_id="TEST-024",
        earlier_column="purchase",
        later_column="approved",
    )

    assert result["status"] == "PASS"
    assert result["affected_records"] == 0


def test_evaluate_temporal_order_fails_invalid_sequence():
    df = pd.DataFrame(
        {
            "purchase": pd.to_datetime(
                ["2024-01-01", "2024-01-05"]
            ),
            "approved": pd.to_datetime(
                ["2024-01-02", "2024-01-04"]
            ),
        }
    )

    result = evaluate_temporal_order(
        df,
        rule_id="TEST-025",
        earlier_column="purchase",
        later_column="approved",
    )

    assert result["status"] == "FAIL"
    assert result["affected_records"] == 1


def test_evaluate_temporal_order_ignores_missing_timestamps():
    df = pd.DataFrame(
        {
            "purchase": pd.to_datetime(
                ["2024-01-01", None]
            ),
            "approved": pd.to_datetime(
                ["2024-01-02", "2024-01-03"]
            ),
        }
    )

    result = evaluate_temporal_order(
        df,
        rule_id="TEST-026",
        earlier_column="purchase",
        later_column="approved",
    )

    assert result["status"] == "PASS"
    assert result["affected_records"] == 0


def test_evaluate_temporal_order_raises_for_missing_column():
    df = pd.DataFrame(
        {
            "purchase": pd.to_datetime(
                ["2024-01-01"]
            )
        }
    )

    with pytest.raises(KeyError):
        evaluate_temporal_order(
            df,
            rule_id="TEST-027",
            earlier_column="purchase",
            later_column="approved",
        )
def test_evaluate_reconciliation_passes_within_tolerance():
    df = pd.DataFrame(
        {
            "expected": [100.00, 200.00, 300.005],
            "actual": [100.00, 200.01, 300.00],
        }
    )

    result = evaluate_reconciliation(
        df,
        rule_id="TEST-028",
        expected_column="expected",
        actual_column="actual",
        tolerance=0.01,
    )

    assert result["status"] == "PASS"
    assert result["affected_records"] == 0


def test_evaluate_reconciliation_fails_outside_tolerance():
    df = pd.DataFrame(
        {
            "expected": [100.00, 200.00, 300.00],
            "actual": [100.00, 205.00, 300.00],
        }
    )

    result = evaluate_reconciliation(
        df,
        rule_id="TEST-029",
        expected_column="expected",
        actual_column="actual",
        tolerance=0.01,
    )

    assert result["status"] == "FAIL"
    assert result["affected_records"] == 1


def test_evaluate_reconciliation_ignores_incomplete_pairs():
    df = pd.DataFrame(
        {
            "expected": [100.00, None, 300.00],
            "actual": [100.00, 200.00, None],
        }
    )

    result = evaluate_reconciliation(
        df,
        rule_id="TEST-030",
        expected_column="expected",
        actual_column="actual",
        tolerance=0.01,
    )

    assert result["status"] == "PASS"
    assert result["total_records"] == 1
    assert result["affected_records"] == 0


def test_evaluate_reconciliation_raises_for_missing_column():
    df = pd.DataFrame(
        {"expected": [100.00] }
    )

    with pytest.raises(KeyError):
        evaluate_reconciliation(
            df,
            rule_id="TEST-031",
            expected_column="expected",
            actual_column="actual",
            tolerance=0.01,
        )
def test_evaluate_conditional_value_passes_when_condition_is_satisfied():
    df = pd.DataFrame(
        {
            "transition_count": [0, 0, 1, 2],
            "evidence_level": ["NONE", "NONE", "LOW", "MEDIUM"],
        }
    )

    result = evaluate_conditional_value(
        df,
        rule_id="TEST-COND-001",
        condition_column="transition_count",
        condition_value=0,
        target_column="evidence_level",
        expected_value="NONE",
    )

    assert result["status"] == "PASS"
    assert result["total_records"] == 2
    assert result["affected_records"] == 0
    assert result["affected_pct"] == 0.0

def test_evaluate_conditional_value_supports_greater_than_or_equal():
    df = pd.DataFrame(
        {
            "transition_count": [0, 1, 2, 3],
            "evidence_level": ["NONE", "LOW", "MEDIUM", "HIGH"],
        }
    )

    result = evaluate_conditional_value(
        df,
        rule_id="TEST-COND-002",
        condition_column="transition_count",
        condition_value=1,
        condition_operator="greater_than_or_equal",
        target_column="evidence_level",
        expected_value="LOW",
    )

    assert result["status"] == "FAIL"
    assert result["total_records"] == 3
    assert result["affected_records"] == 2
    assert result["affected_pct"] == pytest.approx(66.6666666667)
def test_evaluate_conditional_value_rejects_unsupported_operator():
    df = pd.DataFrame(
        {
            "transition_count": [0, 1],
            "evidence_level": ["NONE", "LOW"],
        }
    )

    with pytest.raises(ValueError):
        evaluate_conditional_value(
            df,
            rule_id="TEST-COND-003",
            condition_column="transition_count",
            condition_value=1,
            condition_operator="unsupported_operator",
            target_column="evidence_level",
            expected_value="LOW",
        )