from pathlib import Path

import pandas as pd


class MartPersistenceError(Exception):
    """Raised when analytical mart persistence fails."""


def save_mart(
    df: pd.DataFrame,
    mart_name: str,
    output_dir: str | Path,
) -> Path:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    path = output_path / f"{mart_name}.parquet"

    try:
        df.to_parquet(
            path,
            index=False,
            compression="snappy",
        )
    except Exception as exc:
        raise MartPersistenceError(
            f"Failed to persist mart '{mart_name}': {exc}"
        ) from exc

    if not path.exists():
        raise MartPersistenceError(
            f"Mart file was not created: {path}"
        )

    return path