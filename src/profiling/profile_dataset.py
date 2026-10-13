from pathlib import Path

import pandas as pd


def profile_dataset(file_path: Path) -> dict:
    """Profile a raw CSV dataset without modifying it."""
    if not file_path.exists():
        raise FileNotFoundError(f"Dataset not found: {file_path}")

    if file_path.suffix.lower() != ".csv":
        raise ValueError(f"Expected a CSV file: {file_path}")

    df = pd.read_csv(file_path)

    column_profile = pd.DataFrame(
        {
            "column_name": df.columns,
            "dtype": df.dtypes.astype(str).values,
            "row_count": len(df),
            "missing_count": df.isna().sum().values,
            "missing_pct": (df.isna().mean() * 100).values,
            "unique_count": df.nunique(dropna=True).values,
        }
    )

    return {
        "file_name": file_path.name,
        "row_count": len(df),
        "column_count": len(df.columns),
        "duplicate_row_count": int(df.duplicated().sum()),
        "column_profile": column_profile,
    }
