from pathlib import Path
import hashlib

import pandas as pd


RAW_DIR = Path("data/raw")
OUTPUT_PATH = Path("outputs/raw_data_manifest.csv")

rows = []

for file_path in sorted(RAW_DIR.glob("*.csv")):
    df = pd.read_csv(file_path)

    rows.append(
        {
            "file_name": file_path.name,
            "row_count": len(df),
            "column_count": len(df.columns),
            "file_size_bytes": file_path.stat().st_size,
            "sha256": hashlib.sha256(file_path.read_bytes()).hexdigest(),
        }
    )

manifest = pd.DataFrame(rows)
manifest.to_csv(OUTPUT_PATH, index=False)

print("MANIFEST REBUILT")
print(manifest.to_string(index=False))
