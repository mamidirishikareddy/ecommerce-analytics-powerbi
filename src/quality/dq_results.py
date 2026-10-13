from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.quality.dq_engine import load_rules


RESULT_COLUMNS = [
    "rule_id",
    "status",
    "total_records",
    "affected_records",
    "affected_pct",
    "evaluator",
    "dataset",
]

GOVERNANCE_COLUMNS = [
    "domain",
    "name",
    "severity",
    "description",
]


def validate_results(results: pd.DataFrame) -> None:
    """Validate the structure of a governed DQ result set."""
    missing_columns = [
        column
        for column in RESULT_COLUMNS
        if column not in results.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing DQ result columns: {missing_columns}"
        )

    if results["rule_id"].duplicated().any():
        raise ValueError("DQ result rule_id values must be unique.")

    invalid_statuses = set(results["status"]) - {"PASS", "FAIL"}

    if invalid_statuses:
        raise ValueError(
            f"Invalid DQ statuses: {sorted(invalid_statuses)}"
        )

    if results["affected_records"].lt(0).any():
        raise ValueError(
            "affected_records cannot contain negative values."
        )

    if results["total_records"].lt(0).any():
        raise ValueError(
            "total_records cannot contain negative values."
        )


def enrich_results_with_rule_metadata(
    results: pd.DataFrame,
    rules_path: Path,
) -> pd.DataFrame:
    """Attach governed rule metadata to executed DQ results."""
    validate_results(results)

    rules = load_rules(rules_path)

    metadata = pd.DataFrame(
        rules,
        columns=[
            "rule_id",
            *GOVERNANCE_COLUMNS,
        ],
    )

    result_rule_ids = set(results["rule_id"])
    registry_rule_ids = set(metadata["rule_id"])

    missing_metadata = sorted(
        result_rule_ids - registry_rule_ids
    )

    if missing_metadata:
        raise ValueError(
            "DQ results contain rule IDs missing from the "
            f"governed registry: {missing_metadata}"
        )

    enriched = results.merge(
        metadata,
        on="rule_id",
        how="left",
        validate="one_to_one",
    )

    if enriched[GOVERNANCE_COLUMNS].isna().any().any():
        raise ValueError(
            "Governance metadata is incomplete for one or more DQ results."
        )

    return enriched[
        RESULT_COLUMNS + GOVERNANCE_COLUMNS
    ]


def save_results(
    results: pd.DataFrame,
    output_path: Path,
) -> Path:
    """Validate and persist governed DQ results as CSV."""
    validate_results(results)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    columns_to_save = RESULT_COLUMNS + [
        column
        for column in GOVERNANCE_COLUMNS
        if column in results.columns
    ]

    results[columns_to_save].to_csv(
        output_path,
        index=False,
    )

    return output_path