from __future__ import annotations

from pathlib import Path

import pandas as pd


def save_staged_dataset(
    dataframe: pd.DataFrame,
    output_path: Path,
) -> Path:
    """Persist one validated staging dataset as Parquet."""
    if not isinstance(dataframe, pd.DataFrame):
        raise TypeError("dataframe must be a pandas DataFrame.")

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    dataframe.to_parquet(
        output_path,
        index=False,
        compression="snappy",
    )

    return output_path


def save_all_staged_datasets(
    staged: dict[str, pd.DataFrame],
    output_dir: Path,
    mapping: dict,
) -> dict[str, Path]:
    """Persist all governed staging datasets as Parquet."""
    saved_paths: dict[str, Path] = {}

    for dataset_name, dataframe in staged.items():
        definition = mapping["datasets"][dataset_name]

        output_path = (
            output_dir
            / f"{definition['staging_name']}.parquet"
        )

        saved_paths[dataset_name] = save_staged_dataset(
            dataframe,
            output_path,
        )

    return saved_paths