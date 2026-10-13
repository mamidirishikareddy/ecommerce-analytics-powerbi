
"""
Data-quality execution runner.

This module:
1. Loads governed DQ execution configuration.
2. Loads configured CSV and Parquet datasets.
3. Prepares derived datasets required by DQ checks.
4. Executes each governed DQ rule through the appropriate evaluator.
5. Adds execution metadata to each rule result.
6. Returns normalized rule-level results.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import yaml

from src.quality.dq_engine import (
    evaluate_allowed_values,
    evaluate_conditional_value,
    evaluate_minimum,
    evaluate_not_null,
    evaluate_range,
    evaluate_reconciliation,
    evaluate_reference_integrity,
    evaluate_temporal_order,
    evaluate_unique,
)
from src.quality.dq_preparation import (
    build_order_item_temporal_frame,
    build_order_reconciliation_frame,
)

SUPPORTED_EVALUATORS = {
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


def load_execution_config(config_path: str | Path) -> dict:
    """Load and validate the DQ execution configuration."""
    config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(
            f"Execution configuration not found: {config_path}"
        )

    if config_path.suffix.lower() not in {".yaml", ".yml"}:
        raise ValueError(
            f"Expected a YAML configuration file: {config_path}"
        )

    with config_path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    if not isinstance(config, dict):
        raise ValueError(
            "Execution configuration must be a mapping."
        )

    required_sections = {"version", "datasets", "rules"}
    missing_sections = required_sections - set(config)

    if missing_sections:
        raise ValueError(
            f"Missing configuration sections: {sorted(missing_sections)}"
        )

    if not isinstance(config["datasets"], dict):
        raise ValueError("'datasets' must be a mapping.")

    if not isinstance(config["rules"], dict):
        raise ValueError("'rules' must be a mapping.")

    for rule_id, rule in config["rules"].items():
        if not isinstance(rule, dict):
            raise ValueError(
                f"Execution definition for {rule_id} must be a mapping."
            )

        if "evaluator" not in rule:
            raise ValueError(
                f"Execution definition for {rule_id} "
                "is missing 'evaluator'."
            )

        evaluator = rule["evaluator"]

        if evaluator not in SUPPORTED_EVALUATORS:
            raise ValueError(
                f"Unsupported evaluator '{evaluator}' "
                f"for rule {rule_id}."
            )

    return config


def load_datasets(
    raw_dir: str | Path,
    dataset_config: dict,
) -> dict[str, pd.DataFrame]:
    """Load configured CSV and Parquet datasets."""
    datasets: dict[str, pd.DataFrame] = {}

    raw_dir = Path(raw_dir)
    project_root = raw_dir.resolve().parent

    for dataset_name, dataset_definition in dataset_config.items():
        if not isinstance(dataset_definition, dict):
            raise ValueError(
                f"Dataset definition for {dataset_name} must be a mapping."
            )

        if "file" not in dataset_definition:
            raise ValueError(
                f"Dataset definition for {dataset_name} "
                "is missing 'file'."
            )

        configured_path = Path(dataset_definition["file"])

        if configured_path.is_absolute():
            file_path = configured_path
        else:
            raw_candidate = raw_dir / configured_path
            project_candidate = project_root / configured_path

            if raw_candidate.exists():
                file_path = raw_candidate
            else:
                file_path = project_candidate

        if not file_path.exists():
            raise FileNotFoundError(
                f"Configured dataset '{dataset_name}' not found: "
                f"{file_path}"
            )

        suffix = file_path.suffix.lower()

        if suffix == ".csv":
            datasets[dataset_name] = pd.read_csv(file_path)
        elif suffix == ".parquet":
            datasets[dataset_name] = pd.read_parquet(file_path)
        else:
            raise ValueError(
                f"Unsupported file format for dataset '{dataset_name}': "
                f"{suffix}"
            )

    return datasets


def prepare_derived_datasets(
    datasets: dict[str, pd.DataFrame],
) -> dict[str, pd.DataFrame]:
    """Build explicitly declared derived datasets required by DQ rules."""
    derived = {}

    derived["order_item_temporal"] = build_order_item_temporal_frame(
        datasets["orders"],
        datasets["order_items"],
    )

    derived["order_reconciliation"] = build_order_reconciliation_frame(
        datasets["order_items"],
        datasets["payments"],
    )

    return derived


def execute_rule(
    rule_id: str,
    rule: dict,
    datasets: dict[str, pd.DataFrame],
) -> dict:
    """Execute one configured DQ rule."""
    evaluator = rule["evaluator"]

    if evaluator == "not_null":
        return evaluate_not_null(
            datasets[rule["dataset"]],
            rule_id,
            rule["column"],
        )

    if evaluator == "unique":
        columns = rule.get("columns")

        if columns is None:
            column = rule.get("column")
            if column is None:
                raise ValueError(
                    f"Unique rule {rule_id} requires 'column' or 'columns'."
                )
            columns = [column]

        elif isinstance(columns, str):
            columns = [columns]

        return evaluate_unique(
            datasets[rule["dataset"]],
            rule_id,
            columns,
        )

    if evaluator == "minimum":
        return evaluate_minimum(
            datasets[rule["dataset"]],
            rule_id,
            rule["column"],
            rule["minimum"],
        )

    if evaluator == "range":
        minimum = rule.get("min_value")
        maximum = rule.get("max_value")

        if minimum is None and maximum is None:
            raise ValueError(
                f"Range rule {rule_id} must define min_value or max_value."
            )

        df = datasets[rule["dataset"]]
        column = rule["column"]

        if column not in df.columns:
            raise KeyError(f"Column not found: {column}")

        invalid_mask = df[column].notna()

        if minimum is not None:
            invalid_mask &= df[column] >= minimum

        if maximum is not None:
            invalid_mask &= df[column] <= maximum

        # Values outside the permitted bounds are invalid.
        invalid_mask = df[column].notna() & ~invalid_mask

        affected_records = int(invalid_mask.sum())
        total_records = len(df)

        return {
            "rule_id": rule_id,
            "status": "PASS" if affected_records == 0 else "FAIL",
            "total_records": total_records,
            "affected_records": affected_records,
            "affected_pct": (
                affected_records / total_records * 100
                if total_records
                else 0.0
            ),
        }

    if evaluator == "reference_integrity":
        return evaluate_reference_integrity(
            datasets[rule["child_dataset"]],
            datasets[rule["parent_dataset"]],
            rule_id,
            rule["child_column"],
            rule["parent_column"],
        )

    if evaluator == "temporal_order":
        earlier_column = rule.get(
            "earlier_column",
            rule.get("start_column"),
        )
        later_column = rule.get(
            "later_column",
            rule.get("end_column"),
        )

        if earlier_column is None or later_column is None:
            raise ValueError(
                f"Temporal rule {rule_id} must define "
                "'earlier_column'/'later_column' or "
                "'start_column'/'end_column'."
            )

        return evaluate_temporal_order(
            datasets[rule["dataset"]],
            rule_id,
            earlier_column,
            later_column,
        )

    if evaluator == "reconciliation":
        # Original financial reconciliation format.
        if (
            "dataset" in rule
            and "expected_column" in rule
            and "actual_column" in rule
        ):
            return evaluate_reconciliation(
                datasets[rule["dataset"]],
                rule_id,
                rule["expected_column"],
                rule["actual_column"],
                rule.get("tolerance", 0.0),
            )

        raise ValueError(
            f"Unsupported reconciliation configuration for rule {rule_id}. "
            "Expected dataset, expected_column, and actual_column."
        )

    if evaluator == "allowed_values":
        return evaluate_allowed_values(
            datasets[rule["dataset"]],
            rule_id,
            rule["column"],
            set(rule["allowed_values"]),
        )

    if evaluator == "conditional_value":
        return evaluate_conditional_value(
            datasets[rule["dataset"]],
            rule_id,
            rule["condition_column"],
            rule["condition_value"],
            rule["target_column"],
            rule["expected_value"],
            rule.get("operator", "equals"),
        )

    raise ValueError(
        f"Unsupported evaluator '{evaluator}' for rule {rule_id}."
    )


def run_dq(
    raw_dir: str | Path,
    execution_config_path: str | Path,
) -> pd.DataFrame:
    """Execute the governed DQ rule set and return structured results."""
    config = load_execution_config(execution_config_path)

    datasets = load_datasets(
        raw_dir,
        config["datasets"],
    )

    derived_datasets = prepare_derived_datasets(datasets)

    all_datasets = {
        **datasets,
        **derived_datasets,
    }

    results = []

    for rule_id, rule in config["rules"].items():
        result = execute_rule(
            rule_id,
            rule,
            all_datasets,
        )

        result["evaluator"] = rule["evaluator"]
        result["dataset"] = rule.get(
            "dataset",
            rule.get("child_dataset", ""),
        )

        results.append(result)

    return pd.DataFrame(results)
