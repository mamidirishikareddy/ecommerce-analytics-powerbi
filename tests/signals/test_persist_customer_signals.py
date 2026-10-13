
"""Persistence for governed customer behavioral signals."""

from pathlib import Path

from src.signals.customer_signals import build_customer_signals


def persist_customer_signals(
    marts_dir: str | Path,
    output_dir: str | Path,
) -> Path:
    """Build customer signals and persist them as a Parquet mart."""
    marts_dir = Path(marts_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    signals = build_customer_signals(marts_dir)
    output_path = output_dir / "mart_customer_signals.parquet"
    signals.to_parquet(output_path, index=False)

    return output_path
