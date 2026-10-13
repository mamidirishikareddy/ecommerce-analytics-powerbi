import json

import pandas as pd
import pytest

from src.quality.dq_summary import (
    SUMMARY_COLUMNS,
    build_summary,
    save_summary,
)


def valid_results():
    return pd.DataFrame(
        {
            "rule_id": ["R1", "R2", "R3"],
            "status": ["PASS", "FAIL", "FAIL"],
            "severity": ["critical", "warning", "critical"],
            "affected_records": [0, 2, 3],
        }
    )


def test_build_summary():
    summary = build_summary(
        valid_results(),
        execution_timestamp_utc="2026-09-27T04:30:00+00:00",
    )

    assert list(summary.keys()) == SUMMARY_COLUMNS
    assert summary["execution_timestamp_utc"] == (
        "2026-09-27T04:30:00+00:00"
    )
    assert summary["rules_executed"] == 3
    assert summary["pass_count"] == 1
    assert summary["fail_count"] == 2
    assert summary["critical_failure_count"] == 1
    assert summary["warning_failure_count"] == 1
    assert summary["affected_records_sum_across_rules"] == 5


def test_build_summary_generates_timestamp():
    summary = build_summary(valid_results())

    assert summary["execution_timestamp_utc"]
    assert summary["execution_timestamp_utc"].endswith("+00:00")


def test_build_summary_rejects_missing_columns():
    results = valid_results().drop(columns=["severity"])

    with pytest.raises(
        ValueError,
        match="Missing DQ result columns",
    ):
        build_summary(results)


def test_build_summary_rejects_duplicate_rule_ids():
    results = valid_results()
    results.loc[1, "rule_id"] = "R1"

    with pytest.raises(
        ValueError,
        match="rule_id values must be unique",
    ):
        build_summary(results)


def test_save_summary(tmp_path):
    summary = build_summary(
        valid_results(),
        execution_timestamp_utc="2026-09-27T04:30:00+00:00",
    )

    output_path = tmp_path / "dq_baseline_summary.json"

    returned_path = save_summary(
        summary,
        output_path,
    )

    assert returned_path == output_path
    assert output_path.exists()

    saved = json.loads(
        output_path.read_text(encoding="utf-8")
    )

    assert saved == summary