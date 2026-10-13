import pandas as pd

from src.nlp.sentiment_inference import predict_sentiment


class FakeConfig:
    id2label = {
        0: "negative",
        1: "neutral",
        2: "positive",
    }


class FakeModel:
    config = FakeConfig()


class FakeTokenizer:
    def __call__(
        self,
        batch,
        padding=True,
        truncation=True,
        max_length=256,
        return_tensors="pt",
    ):
        return {
            "input_ids": None,
            "attention_mask": None,
        }


def test_empty_input_returns_empty_result():
    result = predict_sentiment(
        [],
        FakeTokenizer(),
        FakeModel(),
    )

    assert isinstance(result, pd.DataFrame)
    assert list(result.columns) == [
        "sentiment",
        "sentiment_confidence",
    ]
    assert len(result) == 0