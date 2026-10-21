"""Leakage-aware classification of small example reviews.

This is a learning exercise on synthetic data, not a production model.
"""

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

from examples.text_preprocessing import normalize_text


TEXTS = [
    "  GREAT product!!!  ",
    "Excellent service 100!",
    "I LOVE this item.",
    "Amazing quality!!",
    "Very happy 2026",
    "Wonderful experience!",
    "Good product.",
    "Really satisfied!!",
    "  BAD product!!!  ",
    "Terrible service 100!",
    "I HATE this item.",
    "Awful quality!!",
    "Very disappointed 2026",
    "Horrible experience!",
    "Poor product.",
    "Really unhappy!!",
]
LABELS = ["positive"] * 8 + ["negative"] * 8


def run_experiment() -> tuple[dict[str, float], list[tuple[str, str, str]]]:
    """Fit a training-only TF-IDF model; evaluate on held-out examples."""
    cleaned_texts = [normalize_text(text) for text in TEXTS]
    train_texts, test_texts, train_labels, test_labels = train_test_split(
        cleaned_texts,
        LABELS,
        test_size=0.25,
        random_state=42,
        stratify=LABELS,
    )
    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer()),
        ("classifier", MultinomialNB()),
    ])
    pipeline.fit(train_texts, train_labels)
    train_predictions = pipeline.predict(train_texts)
    test_predictions = pipeline.predict(test_texts)
    metrics = {
        "train_accuracy": float(accuracy_score(train_labels, train_predictions)),
        "test_accuracy": float(accuracy_score(test_labels, test_predictions)),
        "test_precision": float(precision_score(test_labels, test_predictions, pos_label="positive", zero_division=0)),
        "test_recall": float(recall_score(test_labels, test_predictions, pos_label="positive", zero_division=0)),
        "test_f1": float(f1_score(test_labels, test_predictions, pos_label="positive", zero_division=0)),
    }
    examples = [
        (text, truth, prediction)
        for text, truth, prediction in zip(test_texts, test_labels, test_predictions)
    ]
    return metrics, examples


if __name__ == "__main__":
    metrics, examples = run_experiment()
    for name, score in metrics.items():
        print(f"{name}: {score:.2f}")
    print("\nTest examples (text | actual | predicted):")
    for text, truth, predicted in examples:
        print(f"{text} | {truth} | {predicted}")
