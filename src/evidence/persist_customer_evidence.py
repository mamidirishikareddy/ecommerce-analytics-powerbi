"""
Persistence for the governed customer evidence dataset.

This module builds the customer evidence assessment and persists it
using the project's shared mart persistence utility.
"""

from __future__ import annotations

from pathlib import Path

from src.evidence.customer_evidence import build_customer_evidence
from src.marts.persistence import save_mart


def persist_customer_evidence(
    signals_path: str | Path,
    output_dir: str | Path,
) -> Path:
    """
    Build and persist the governed customer evidence dataset.

    Parameters
    ----------
    signals_path:
        Path to the persisted customer signal mart.

    output_dir:
        Directory where the evidence mart will be persisted.

    Returns
    -------
    Path
        Path to the persisted customer evidence mart.
    """
    evidence = build_customer_evidence(
        signals_path,
    )

    return save_mart(
        evidence,
        mart_name="mart_customer_evidence",
        output_dir=output_dir,
    )