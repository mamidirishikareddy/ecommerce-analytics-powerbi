from pathlib import Path

from src.quality.persist_dq_execution import persist_dq_execution


PROJECT_ROOT = Path(__file__).resolve().parents[1]

RAW_DIR = PROJECT_ROOT / "data" / "raw"
EXECUTION_CONFIG = PROJECT_ROOT / "configs" / "dq_execution.yaml"
RULES_CONFIG = PROJECT_ROOT / "configs" / "dq_rules.yaml"

OUTPUT_DIR = PROJECT_ROOT / "data" / "processed" / "quality"


def main() -> None:
    results_path, summary_path = persist_dq_execution(
        raw_data_dir=RAW_DIR,
        execution_config_path=EXECUTION_CONFIG,
        rules_path=RULES_CONFIG,
        output_dir=OUTPUT_DIR,
    )

    print(f"Results written to: {results_path}")
    print(f"Summary written to: {summary_path}")


if __name__ == "__main__":
    main()
