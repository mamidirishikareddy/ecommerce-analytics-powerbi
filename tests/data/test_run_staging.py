from pathlib import Path

import pandas as pd
import yaml

import importlib.util

SCRIPT_PATH = (
    Path(__file__).resolve().parents[2]
    / "scripts"
    / "run_staging.py"
)

SPEC = importlib.util.spec_from_file_location(
    "run_staging_script",
    SCRIPT_PATH,
)

run_staging_script = importlib.util.module_from_spec(SPEC)

assert SPEC.loader is not None

SPEC.loader.exec_module(run_staging_script)

def test_run_staging_main_orchestrates_staging(monkeypatch, tmp_path):
    raw_dir = tmp_path / "raw"
    output_dir = tmp_path / "staging"
    mapping_path = tmp_path / "staging_mapping.yaml"

    raw_dir.mkdir()

    mapping = {
        "version": "1.0",
        "datasets": {
            "orders": {
                "source_file": "olist_orders_dataset.csv",
                "staging_name": "stg_orders",
                "columns": {
                    "order_id": {
                        "target": "order_id",
                        "dtype": "string",
                    },
                },
            }
        },
    }

    mapping_path.write_text(
        yaml.safe_dump(mapping),
        encoding="utf-8",
    )

    staged = {
        "orders": pd.DataFrame(
            {"order_id": ["O1", "O2"]}
        )
    }

    saved_paths = {
        "orders": output_dir / "stg_orders.parquet"
    }

    monkeypatch.setattr(
        run_staging_script,
        "RAW_DIR",
        raw_dir,
    )
    monkeypatch.setattr(
        run_staging_script,
        "MAPPING_PATH",
        mapping_path,
    )
    monkeypatch.setattr(
        run_staging_script,
        "OUTPUT_DIR",
        output_dir,
    )

    calls = {}

    def fake_run_staging(
        raw_dir,
        mapping_path,
    ):
        calls["run_staging"] = (
            raw_dir,
            mapping_path,
        )
        return staged

    def fake_save_all_staged_datasets(
        staged,
        output_dir,
        mapping,
    ):
        calls["save"] = (
            staged,
            output_dir,
            mapping,
        )
        return saved_paths

    monkeypatch.setattr(
        run_staging_script,
        "run_staging",
        fake_run_staging,
    )
    monkeypatch.setattr(
        run_staging_script,
        "save_all_staged_datasets",
        fake_save_all_staged_datasets,
    )

    run_staging_script.main()

    assert calls["run_staging"] == (
        raw_dir,
        mapping_path,
    )

    assert calls["save"][0] is staged
    assert calls["save"][1] == output_dir
    assert calls["save"][2]["version"] == "1.0"

    assert set(saved_paths) == {"orders"}