from pathlib import Path

from src.quality.dq_results import (
    enrich_results_with_rule_metadata,
    save_results,
)
from src.quality.dq_runner import run_dq
from src.quality.dq_summary import (
    build_summary,
    save_summary,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]

RAW_DIR = PROJECT_ROOT / "data" / "raw"
EXECUTION_CONFIG = PROJECT_ROOT / "configs" / "dq_execution.yaml"
RULES_CONFIG = PROJECT_ROOT / "configs" / "dq_rules.yaml"

OUTPUT_DIR = PROJECT_ROOT / "outputs" / "data_quality"

RESULTS_OUTPUT_PATH = OUTPUT_DIR / "dq_results.csv"
SUMMARY_OUTPUT_PATH = OUTPUT_DIR / "dq_baseline_summary.json"


def main() -> None:
    results = run_dq(
        raw_dir=RAW_DIR,
        execution_config_path=EXECUTION_CONFIG,
    )

    governed_results = enrich_results_with_rule_metadata(
        results,
        RULES_CONFIG,
    )

    save_results(
        governed_results,
        RESULTS_OUTPUT_PATH,
    )

    summary = build_summary(governed_results)

    save_summary(
        summary,
        SUMMARY_OUTPUT_PATH,
    )

    print(
        f"Rules executed: {summary['rules_executed']}"
    )
    print(
        f"PASS: {summary['pass_count']}"
    )
    print(
        f"FAIL: {summary['fail_count']}"
    )
    print(
        f"Critical failures: "
        f"{summary['critical_failure_count']}"
    )
    print(
        f"Warning failures: "
        f"{summary['warning_failure_count']}"
    )
    print(
        "Affected records summed across failed rules: "
        f"{summary['affected_records_sum_across_rules']}"
    )
    print(
        f"Results written to: {RESULTS_OUTPUT_PATH}"
    )
    print(
        f"Baseline summary written to: {SUMMARY_OUTPUT_PATH}"
    )


if __name__ == "__main__":
    main()