from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.staging.transform import (
    load_staging_mapping,
    transform_csv,
)


def run_staging(
    raw_dir: Path,
    mapping_path: Path,
) -> dict[str, pd.DataFrame]:
    """Execute the governed staging transformation for all datasets."""
    mapping = load_staging_mapping(mapping_path)

    staged: dict[str, pd.DataFrame] = {}

    for dataset_name, definition in mapping["datasets"].items():
        source_path = raw_dir / definition["source_file"]

        staged[dataset_name] = transform_csv(
            source_path,
            definition,
        )

    return staged