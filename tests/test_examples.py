"""Small regression tests for the reproducible study examples."""

import pytest

from examples.pandas_review_analysis import summarize_reviews
from examples.review_classifier import run_experiment
from examples.text_preprocessing import normalize_text, word_frequencies
from examples.tfidf_search import search_documents


def test_preprocessing_keeps_word_boundaries():
    assert normalize_text("  Hello!!!  AI   2026 ") == "hello ai"
    assert word_frequencies(["Python Python!", "PYTHON."])["python"] == 3


def test_tfidf_search_handles_multiple_queries():
    documents = ["cat drinks milk", "dog eats meat", "cat eats fish"]
    results = search_documents(documents, ["cat eats milk", "dog eats meat"])
    assert len(results) == 2
    assert results[1][1] == "dog eats meat"
    assert results[1][2] == pytest.approx(1.0)
    assert 0.0 <= results[0][2] <= 1.0


def test_tfidf_search_rejects_empty_documents():
    with pytest.raises(ValueError):
        search_documents([], ["hello"])


def test_pandas_review_summary():
    result = summarize_reviews()
    assert result.loc["positive", "review_count"] == 2
    assert result.loc["negative", "review_count"] == 2
    assert result.loc["positive", "average_score"] == pytest.approx(4.5)


def test_pipeline_classification_reproducible():
    metrics, examples = run_experiment()
    assert len(examples) == 4
    assert len(metrics) == 5
    assert all(0.0 <= value <= 1.0 for value in metrics.values())
    assert run_experiment() == (metrics, examples)
