from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import json

import pandas as pd


SUMMARY_COLUMNS = [
    "execution_timestamp_utc",
    "rules_executed",
    "pass_count",
    "fail_count",
    "critical_failure_count",
    "warning_failure_count",
    "affected_records_sum_across_rules",
]


def build_summary(
    results: pd.DataFrame,
    execution_timestamp_utc: str | None = None,
) -> dict:
    """Build a machine-readable summary of a governed DQ execution."""
    required_columns = {
        "rule_id",
        "status",
        "severity",
        "affected_records",
    }

    missing_columns = required_columns - set(results.columns)

    if missing_columns:
        raise ValueError(
            f"Missing DQ result columns: {sorted(missing_columns)}"
        )

    if results["rule_id"].duplicated().any():
        raise ValueError("DQ result rule_id values must be unique.")

    timestamp = execution_timestamp_utc

    if timestamp is None:
        timestamp = (
            datetime.now(timezone.utc)
            .replace(microsecond=0)
            .isoformat()
        )

    return {
        "execution_timestamp_utc": timestamp,
        "rules_executed": int(len(results)),
        "pass_count": int((results["status"] == "PASS").sum()),
        "fail_count": int((results["status"] == "FAIL").sum()),
        "critical_failure_count": int(
            (
                (results["status"] == "FAIL")
                & (results["severity"] == "critical")
            ).sum()
        ),
        "warning_failure_count": int(
            (
                (results["status"] == "FAIL")
                & (results["severity"] == "warning")
            ).sum()
        ),
        "affected_records_sum_across_rules": int(
            results.loc[
                results["status"] == "FAIL",
                "affected_records",
            ].sum()
        ),
    }


def save_summary(
    summary: dict,
    output_path: Path,
) -> Path:
    """Persist a DQ baseline summary as JSON."""
    missing_fields = [
        field for field in SUMMARY_COLUMNS
        if field not in summary
    ]

    if missing_fields:
        raise ValueError(
            f"Missing DQ summary fields: {missing_fields}"
        )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with output_path.open("w", encoding="utf-8") as file:
        json.dump(
            summary,
            file,
            indent=2,
        )

    return output_path