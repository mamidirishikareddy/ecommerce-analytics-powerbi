from pathlib import Path

import pandas as pd


class WarehousePersistenceError(Exception):
    """Raised when warehouse persistence fails."""


def save_warehouse_table(
    df: pd.DataFrame,
    table_name: str,
    output_dir: str | Path,
) -> Path:
    """
    Persist one validated warehouse table as Snappy-compressed Parquet.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    path = output_path / f"{table_name}.parquet"

    try:
        df.to_parquet(
            path,
            index=False,
            compression="snappy",
        )
    except Exception as exc:
        raise WarehousePersistenceError(
            f"Failed to persist warehouse table '{table_name}': {exc}"
        ) from exc

    if not path.exists():
        raise WarehousePersistenceError(
            f"Warehouse table was not created: {path}"
        )

    return path


def save_all_warehouse_tables(
    tables: dict[str, pd.DataFrame],
    output_dir: str | Path,
) -> dict[str, Path]:
    """
    Persist all warehouse tables.
    """
    saved_paths = {}

    for table_name, df in tables.items():
        saved_paths[table_name] = save_warehouse_table(
            df=df,
            table_name=table_name,
            output_dir=output_dir,
        )

    return saved_paths