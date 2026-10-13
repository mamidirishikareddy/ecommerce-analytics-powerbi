from pathlib import Path

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
        ["review_id", "order_id", "review_text"],
    ].head(100).reset_index(drop=True)

    tokenizer, model = load_sentiment_model()

    predictions = predict_sentiment(
        sample["review_text"].tolist(),
        tokenizer,
        model,
        batch_size=16,
        max_length=256,
    )

    result = pd.concat(
        [sample, predictions],
        axis=1,
    )

    print("REAL SENTIMENT INFERENCE TEST")
    print(f"INPUT REVIEWS: {len(sample):,}")
    print(f"OUTPUT ROWS: {len(result):,}")
    print(
        "SENTIMENT COUNTS:"
    )
    print(
        result["sentiment"].value_counts().to_string()
    )
    print()
    print(
        "CONFIDENCE RANGE: "
        f"{result['sentiment_confidence'].min():.4f} "
        f"to "
        f"{result['sentiment_confidence'].max():.4f}"
    )
    print()
    print("SAMPLE PREDICTIONS:")
    print(
        result[
            [
                "review_id",
                "sentiment",
                "sentiment_confidence",
                "review_text",
            ]
        ]
        .head(12)
        .to_string(index=False)
    )

    assert len(result) == 100
    assert result["sentiment"].notna().all()
    assert result["sentiment_confidence"].notna().all()
    assert result["sentiment_confidence"].between(
        0.0, 1.0
    ).all()

    print()
    print("REAL SENTIMENT INFERENCE TEST: PASS")


if __name__ == "__main__":
    main()