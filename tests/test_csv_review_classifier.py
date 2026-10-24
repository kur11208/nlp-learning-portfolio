"""Tests of CSV profiling, cleaning, and leakage-aware classification."""

import pandas as pd
import pytest

from examples.csv_review_classifier import inspect_and_clean, train_and_evaluate


@pytest.fixture
def review_csv(tmp_path):
    examples = [
        ("Great product", "positive"), ("Excellent service", "positive"),
        ("Love this feature", "positive"), ("Fast reply", "positive"),
        ("Helpful team", "positive"), ("Nice design", "positive"),
        ("Wonderful support", "positive"), ("Good update", "positive"),
        ("Bad product", "negative"), ("Terrible service", "negative"),
        ("Hate this feature", "negative"), ("Slow reply", "negative"),
        ("Unhelpful team", "negative"), ("Poor design", "negative"),
        ("Awful support", "negative"), ("Broken update", "negative"),
    ]
    records = [
        {"review_id": i, "review_text": text, "label": label, "channel": "web"}
        for i, (text, label) in enumerate(examples, start=1)
    ]
    records.append(records[0].copy())  # one identical row
    records.append({"review_id": 17, "review_text": None, "label": "positive", "channel": "app"})
    records.append({"review_id": 18, "review_text": "Unknown", "label": None, "channel": "web"})
    path = tmp_path / "reviews.csv"
    pd.DataFrame(records).to_csv(path, index=False)
    return path


def test_csv_inspection_and_cleaning(review_csv):
    cleaned, report = inspect_and_clean(review_csv)
    assert report["shape_before"] == (19, 4)
    assert report["missing_per_column"]["review_text"] == 1
    assert report["missing_per_column"]["label"] == 1
    assert report["exact_duplicates"] == 1
    assert report["shape_after"] == (16, 4)
    assert cleaned["label"].value_counts().to_dict() == {"positive": 8, "negative": 8}


def test_csv_pipeline_metrics_and_reproducibility(review_csv):
    result = train_and_evaluate(review_csv)
    metrics, examples, report = result
    assert len(examples) == 4
    assert len(metrics) == 4
    assert all(0 <= score <= 1 for score in metrics.values())
    assert report["shape_after"] == (16, 4)
    assert train_and_evaluate(review_csv) == result


def test_csv_requires_expected_columns(tmp_path):
    path = tmp_path / "bad.csv"
    pd.DataFrame({"text": ["great"], "label": ["positive"]}).to_csv(path, index=False)
    with pytest.raises(ValueError, match="review_text"):
        inspect_and_clean(path)
