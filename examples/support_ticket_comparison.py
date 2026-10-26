"""Compare count and TF-IDF features on synthetic support ticket categories.

Run from the repository root:
    python -m examples.support_ticket_comparison path/to/tickets.csv

This is an educational example, not a production routing system.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline


CATEGORIES = ["billing", "technical"]


def inspect_tickets(csv_path: str | Path) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    """Return raw profiling, cleaned modeling rows, and response-time averages.

    Keep separate cleaning paths: missing message/category does not invalidate a
    row for response-time aggregation if channel and response time are present.
    """
    raw = pd.read_csv(csv_path)
    required = {"message", "category", "channel", "response_minutes"}
    if not required.issubset(raw.columns):
        raise ValueError(f"CSV requires: {sorted(required)}")

    profile = {
        "raw_shape": tuple(raw.shape),
        "missing_per_column": raw.isna().sum().to_dict(),
        "duplicate_rows": int(raw.duplicated().sum()),
        "category_counts_including_missing": raw["category"].value_counts(dropna=False).to_dict(),
    }

    for_model = raw.dropna(subset=["message", "category"]).drop_duplicates()
    for_model = for_model.loc[
        for_model["message"].astype(str).str.strip().ne("")
        & for_model["category"].astype(str).str.strip().ne("")
    ].copy()
    for_stats = raw.drop_duplicates().dropna(subset=["channel", "response_minutes"])
    channel_means = for_stats.groupby("channel")["response_minutes"].mean()
    return profile, for_model, channel_means


def compare_models(
    csv_path: str | Path, *, test_size: float = 0.25, random_state: int = 3
) -> tuple[dict, dict, pd.Series]:
    """Evaluate BoW and TF-IDF on the same stratified hold-out rows."""
    profile, cleaned, channel_means = inspect_tickets(csv_path)
    if set(cleaned["category"].unique()) != set(CATEGORIES):
        raise ValueError("Expected both 'billing' and 'technical' categories")

    train_texts, test_texts, train_labels, test_labels = train_test_split(
        cleaned["message"],
        cleaned["category"],
        test_size=test_size,
        random_state=random_state,
        stratify=cleaned["category"],
    )
    results = {}
    for name, vectorizer in [
        ("BoW", CountVectorizer()),
        ("TF-IDF", TfidfVectorizer()),
    ]:
        pipeline = Pipeline([
            ("features", vectorizer),
            ("classifier", MultinomialNB()),
        ])
        pipeline.fit(train_texts, train_labels)
        prediction = pipeline.predict(test_texts)
        results[name] = {
            "accuracy": float(accuracy_score(test_labels, prediction)),
            "confusion_matrix": confusion_matrix(
                test_labels, prediction, labels=CATEGORIES
            ).tolist(),
            "test_count": len(test_labels),
        }
    return profile, results, channel_means


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare two NLP baselines")
    parser.add_argument("csv", type=Path, help="CSV path containing ticket columns")
    args = parser.parse_args()
    profile, results, channel_means = compare_models(args.csv)
    print("Raw CSV:", profile)
    print("Mean response minutes by channel:")
    print(channel_means.to_string())
    print("Confusion matrix order (actual rows, predicted columns):", CATEGORIES)
    for name, result in results.items():
        print(f"{name}: accuracy={result['accuracy']:.3f}, n={result['test_count']}")
        print(result["confusion_matrix"])


if __name__ == "__main__":
    main()
