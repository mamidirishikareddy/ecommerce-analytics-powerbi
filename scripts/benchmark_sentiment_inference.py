from pathlib import Path
from time import perf_counter

import pandas as pd

from src.nlp.sentiment_inference import (
    load_sentiment_model,
    predict_sentiment,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "nlp"
    / "customer_voice_input.parquet"
)


def main() -> None:
    df = pd.read_parquet(INPUT_PATH)

    sample = df.loc[
        df["has_review_text"],
        "review_text",
    ].head(500).tolist()

    tokenizer, model = load_sentiment_model()

    start = perf_counter()

    result = predict_sentiment(
        sample,
        tokenizer,
        model,
        batch_size=16,
        max_length=256,
    )

    elapsed = perf_counter() - start

    print("SENTIMENT INFERENCE BENCHMARK")
    print(f"REVIEWS: {len(sample):,}")
    print(f"BATCH SIZE: 16")
    print(f"MAX LENGTH: 256")
    print(f"ELAPSED SECONDS: {elapsed:.2f}")
    print(
        f"REVIEWS PER SECOND: "
        f"{len(sample) / elapsed:.2f}"
    )
    print(
        f"ESTIMATED MINUTES FOR 42,687: "
        f"{(42687 / (len(sample) / elapsed)) / 60:.2f}"
    )

    assert len(result) == 500
    assert result["sentiment"].notna().all()
    assert result["sentiment_confidence"].notna().all()

    print("BENCHMARK: PASS")


if __name__ == "__main__":
    main()