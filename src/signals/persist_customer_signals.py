"""
Orchestration for governed data-quality execution and persistence.

This module provides the production entry point for:
1. Executing the governed DQ rules.
2. Enriching results with rule-governance metadata.
3. Persisting detailed DQ results.
4. Building and persisting the machine-readable DQ summary.
"""

from __future__ import annotations

from pathlib import Path

from src.quality.dq_results import (
    enrich_results_with_rule_metadata,
    save_results,
)
from src.quality.dq_runner import run_dq
from src.quality.dq_summary import (
    build_summary,
    save_summary,
)


def persist_dq_execution(
    raw_data_dir: str | Path,
    execution_config_path: str | Path,
    rules_path: str | Path,
    output_dir: str | Path,
) -> tuple[Path, Path]:
    """
    Execute governed DQ rules and persist the resulting artifacts.

    Parameters
    ----------
    raw_data_dir:
        Directory containing the raw source datasets.

    execution_config_path:
        Path to the DQ execution configuration defining the datasets
        and evaluators to execute.

    rules_path:
        Path to the governed DQ rule registry containing rule metadata.

    output_dir:
        Directory where the DQ result and summary artifacts will
        be persisted.

    Returns
    -------
    tuple[Path, Path]
        Paths to:
        1. The enriched DQ results CSV.
        2. The DQ summary JSON.
    """
    raw_data_dir = Path(raw_data_dir)
    execution_config_path = Path(execution_config_path)
    rules_path = Path(rules_path)
    output_dir = Path(output_dir)

    # Step 1: Execute the governed DQ rules.
    results = run_dq(
        raw_data_dir,
        execution_config_path,
    )

    # Step 2: Attach governance metadata from the rule registry.
    enriched_results = enrich_results_with_rule_metadata(
        results,
        rules_path,
    )

    # Step 3: Persist detailed DQ results.
    results_path = save_results(
        enriched_results,
        output_dir / "dq_results.csv",
    )

    # Step 4: Build the machine-readable execution summary.
    summary = build_summary(
        enriched_results,
    )

    # Step 5: Persist the execution summary.
    summary_path = save_summary(
        summary,
        output_dir / "dq_summary.json",
    )

    return results_path, summary_path