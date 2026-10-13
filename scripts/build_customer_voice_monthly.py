from pathlib import Path

from src.marts.customer_voice_monthly import build_customer_voice_monthly


PROJECT_ROOT = Path(__file__).resolve().parents[1]

CUSTOMER_VOICE_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "nlp"
    / "customer_voice.parquet"
)

COMPLAINT_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "nlp"
    / "customer_voice_complaints.parquet"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "marts"
    / "customer_voice_monthly.parquet"
)


def main() -> None:
    print("=" * 70)
    print("CUSTOMER VOICE MONTHLY MART")
    print("=" * 70)

    print(f"Customer Voice input : {CUSTOMER_VOICE_PATH}")
    print(f"Complaint input      : {COMPLAINT_PATH}")
    print(f"Output               : {OUTPUT_PATH}")

    if not CUSTOMER_VOICE_PATH.exists():
        raise FileNotFoundError(
            f"Customer Voice input not found: {CUSTOMER_VOICE_PATH}"
        )

    if not COMPLAINT_PATH.exists():
        raise FileNotFoundError(
            f"Complaint input not found: {COMPLAINT_PATH}"
        )

    mart = build_customer_voice_monthly(
        customer_voice_path=CUSTOMER_VOICE_PATH,
        complaint_path=COMPLAINT_PATH,
    )

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    mart.to_parquet(
        OUTPUT_PATH,
        index=False,
        engine="pyarrow",
    )

    print()
    print("STATUS: PASS")
    print(f"Rows: {len(mart):,}")
    print(f"Columns: {len(mart.columns)}")
    print(f"Date range: {mart['month'].min()} → {mart['month'].max()}")
    print(f"Output: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()