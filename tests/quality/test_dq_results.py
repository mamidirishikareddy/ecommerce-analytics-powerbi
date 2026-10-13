from pathlib import Path

import pandas as pd
import pytest

from src.quality.dq_results import (
    GOVERNANCE_COLUMNS,
    RESULT_COLUMNS,
    enrich_results_with_rule_metadata,
    save_results,
    validate_results,
)


def valid_results():
    return pd.DataFrame(
        {
            "rule_id": ["R1", "R2"],
            "status": ["PASS", "FAIL"],
            "total_records": [100, 100],
            "affected_records": [0, 2],
            "affected_pct": [0.0, 2.0],
            "evaluator": ["not_null", "unique"],
            "dataset": ["orders", "orders"],
        }
    )


def test_validate_results_accepts_valid_results():
    results = valid_results()

    validate_results(results)


def test_validate_results_rejects_missing_columns():
    results = valid_results().drop(columns=["status"])

    with pytest.raises(ValueError, match="Missing DQ result columns"):
        validate_results(results)


def test_validate_results_rejects_duplicate_rule_ids():
    results = valid_results()
    results.loc[1, "rule_id"] = "R1"

    with pytest.raises(
        ValueError,
        match="rule_id values must be unique",
    ):
        validate_results(results)


def test_validate_results_rejects_invalid_status():
    results = valid_results()
    results.loc[0, "status"] = "UNKNOWN"

    with pytest.raises(
        ValueError,
        match="Invalid DQ statuses",
    ):
        validate_results(results)


def test_validate_results_rejects_negative_counts():
    results = valid_results()
    results.loc[0, "affected_records"] = -1

    with pytest.raises(
        ValueError,
        match="affected_records",
    ):
        validate_results(results)


def test_save_results(tmp_path):
    results = valid_results()
    output_path = tmp_path / "dq_results.csv"

    returned_path = save_results(
        results,
        output_path,
    )

    assert returned_path == output_path
    assert output_path.exists()

    saved = pd.read_csv(output_path)

    assert list(saved.columns) == RESULT_COLUMNS
    assert len(saved) == 2


def test_enrich_results_with_rule_metadata(tmp_path):
    rules_path = tmp_path / "dq_rules.yaml"

    rules_path.write_text(
        """
version: "1.0"
rules:
  - rule_id: R1
    domain: completeness
    name: test_not_null
    severity: critical
    description: Test completeness rule
  - rule_id: R2
    domain: uniqueness
    name: test_unique
    severity: warning
    description: Test uniqueness rule
""",
        encoding="utf-8",
    )

    enriched = enrich_results_with_rule_metadata(
        valid_results(),
        rules_path,
    )

    assert list(enriched.columns) == (
        RESULT_COLUMNS + GOVERNANCE_COLUMNS
    )

    assert enriched.loc[0, "domain"] == "completeness"
    assert enriched.loc[0, "severity"] == "critical"
    assert enriched.loc[1, "domain"] == "uniqueness"
    assert enriched.loc[1, "severity"] == "warning"


def test_enrich_results_rejects_unknown_rule_id(tmp_path):
    rules_path = tmp_path / "dq_rules.yaml"

    rules_path.write_text(
        """
version: "1.0"
rules:
  - rule_id: R1
    domain: completeness
    name: test_not_null
    severity: critical
    description: Test completeness rule
""",
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError,
        match="missing from the governed registry",
    ):
        enrich_results_with_rule_metadata(
            valid_results(),
            rules_path,
        )


def test_save_results_preserves_governance_metadata(tmp_path):
    results = valid_results().assign(
        domain=["completeness", "uniqueness"],
        name=["test_not_null", "test_unique"],
        severity=["critical", "warning"],
        description=[
            "Test completeness rule",
            "Test uniqueness rule",
        ],
    )

    output_path = tmp_path / "dq_results.csv"

    save_results(
        results,
        output_path,
    )

    saved = pd.read_csv(output_path)

    assert list(saved.columns) == (
        RESULT_COLUMNS + GOVERNANCE_COLUMNS
    )
    assert saved.loc[0, "severity"] == "critical"
    assert saved.loc[1, "domain"] == "uniqueness"