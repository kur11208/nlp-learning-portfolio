"""CSV → data checks/cleaning → TF-IDF/NB classification → evaluation.

A small, educational pipeline. A single hold-out score on a small dataset
is not a reliable estimate of generalization performance.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline


def inspect_and_clean(csv_path: str | Path) -> tuple[pd.DataFrame, dict]:
    """Inspect a review CSV and drop missing/duplicate training examples."""
    df = pd.read_csv(csv_path)
    required = {"review_text", "label"}
    if not required.issubset(df.columns):
        raise ValueError(f"CSV needs columns: {sorted(required)}")

    report = {
        "shape_before": tuple(df.shape),
        "missing_per_column": df.isna().sum().to_dict(),
        "exact_duplicates": int(df.duplicated().sum()),
        "labels_before": df["label"].value_counts().to_dict(),
    }
    cleaned = df.dropna(subset=["review_text", "label"]).drop_duplicates().copy()
    cleaned = cleaned[
        cleaned["review_text"].astype(str).str.strip().ne("")
        & cleaned["label"].astype(str).str.strip().ne("")
    ].copy()
    report["shape_after"] = tuple(cleaned.shape)
    report["labels_after"] = cleaned["label"].value_counts().to_dict()
    return cleaned, report


def train_and_evaluate(
    csv_path: str | Path, *, test_size: float = 0.25, random_state: int = 42
) -> tuple[dict, list[tuple[str, str, str]], dict]:
    """Train on the training partition only, and score held-out reviews."""
    cleaned, report = inspect_and_clean(csv_path)
    if set(cleaned["label"].unique()) != {"positive", "negative"}:
        raise ValueError("This example expects positive and negative labels.")

    train_texts, test_texts, train_labels, test_labels = train_test_split(
        cleaned["review_text"],
        cleaned["label"],
        test_size=test_size,
        random_state=random_state,
        stratify=cleaned["label"],
    )
    pipe = Pipeline([
        ("tfidf", TfidfVectorizer()),
        ("model", MultinomialNB()),
    ])
    pipe.fit(train_texts, train_labels)
    predictions = pipe.predict(test_texts)
    metrics = {
        "accuracy": float(accuracy_score(test_labels, predictions)),
        "precision_positive": float(precision_score(test_labels, predictions, pos_label="positive", zero_division=0)),
        "recall_positive": float(recall_score(test_labels, predictions, pos_label="positive", zero_division=0)),
        "f1_positive": float(f1_score(test_labels, predictions, pos_label="positive", zero_division=0)),
    }
    examples = list(zip(test_texts.tolist(), test_labels.tolist(), predictions.tolist()))
    return metrics, examples, report


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze and classify review CSVs")
    parser.add_argument("csv", type=Path, help="Path to a CSV with review_text and label columns")
    args = parser.parse_args()
    metrics, examples, report = train_and_evaluate(args.csv)
    print("Dataset report:", report)
    print("Test metrics:", metrics)
    print("Test examples (review | actual | predicted):")
    for text, actual, predicted in examples:
        print(f"{text} | {actual} | {predicted}")


if __name__ == "__main__":
    main()
