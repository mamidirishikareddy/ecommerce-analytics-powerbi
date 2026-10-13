from pathlib import Path

import pandas as pd


def load_csv(file_path: Path) -> pd.DataFrame:
    """Load a raw CSV file without modifying its contents."""
    if not file_path.exists():
        raise FileNotFoundError(f"Raw file not found: {file_path}")

    if file_path.suffix.lower() != ".csv":
        raise ValueError(f"Expected a CSV file: {file_path}")

    return pd.read_csv(file_path)
