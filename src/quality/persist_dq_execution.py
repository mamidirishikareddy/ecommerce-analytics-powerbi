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
        Path to the DQ execution configuration.

    rules_path:
        Path to the governed DQ rule registry.

    output_dir:
        Directory where DQ artifacts will be persisted.

    Returns
    -------
    tuple[Path, Path]
        Paths to the persisted DQ results CSV and summary JSON.
    """
    raw_data_dir = Path(raw_data_dir)
    execution_config_path = Path(execution_config_path)
    rules_path = Path(rules_path)
    output_dir = Path(output_dir)

    # Execute the governed DQ rules.
    results = run_dq(
        raw_data_dir,
        execution_config_path,
    )

    # Attach governance metadata to the executed results.
    enriched_results = enrich_results_with_rule_metadata(
        results,
        rules_path,
    )

    # Persist detailed DQ results.
    results_path = save_results(
        enriched_results,
        output_dir / "dq_results.csv",
    )

    # Build the machine-readable DQ execution summary.
    summary = build_summary(
        enriched_results,
    )

    # Persist the summary.
    summary_path = save_summary(
        summary,
        output_dir / "dq_summary.json",
    )

    return results_path, summary_path