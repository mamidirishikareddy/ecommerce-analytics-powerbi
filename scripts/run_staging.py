from pathlib import Path

import yaml

from src.staging.persistence import save_all_staged_datasets
from src.staging.runner import run_staging


PROJECT_ROOT = Path(__file__).resolve().parents[1]

RAW_DIR = PROJECT_ROOT / "data" / "raw"
MAPPING_PATH = PROJECT_ROOT / "configs" / "staging_mapping.yaml"
OUTPUT_DIR = PROJECT_ROOT / "data" / "staging"


def main() -> None:
    mapping = yaml.safe_load(
        MAPPING_PATH.read_text(encoding="utf-8")
    )

    staged = run_staging(
        raw_dir=RAW_DIR,
        mapping_path=MAPPING_PATH,
    )

    saved = save_all_staged_datasets(
        staged=staged,
        output_dir=OUTPUT_DIR,
        mapping=mapping,
    )

    print(f"Datasets staged: {len(staged)}")

    for dataset_name, output_path in saved.items():
        print(
            f"{dataset_name}: "
            f"{len(staged[dataset_name])} rows -> "
            f"{output_path}"
        )


if __name__ == "__main__":
    main()