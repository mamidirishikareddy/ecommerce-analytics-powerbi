from pathlib import Path

import pandas as pd
import torch
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
)


MODEL_ID = "clapAI/roberta-base-multilingual-sentiment"


class SentimentInferenceError(Exception):
    """Raised when sentiment inference fails."""


def load_sentiment_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_ID)

    model.eval()

    return tokenizer, model


def predict_sentiment(
    texts: list[str],
    tokenizer,
    model,
    batch_size: int = 16,
    max_length: int = 256,
) -> pd.DataFrame:

    if not texts:
        return pd.DataFrame(
            columns=[
                "sentiment",
                "sentiment_confidence",
            ]
        )

    labels = model.config.id2label

    sentiments = []
    confidences = []

    for start in range(0, len(texts), batch_size):
        batch = texts[start:start + batch_size]

        inputs = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )

        with torch.no_grad():
            logits = model(**inputs).logits

        probabilities = torch.softmax(logits, dim=-1)
        predictions = probabilities.argmax(dim=-1)

        for row_index in range(len(batch)):
            prediction = int(predictions[row_index])

            sentiments.append(
                labels[prediction]
            )

            confidences.append(
                float(
                    probabilities[
                        row_index,
                        prediction
                    ]
                )
            )

    return pd.DataFrame(
        {
            "sentiment": sentiments,
            "sentiment_confidence": confidences,
        }
    )