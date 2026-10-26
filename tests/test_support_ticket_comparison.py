"""Regression tests for model comparison with separate cleaning objectives."""

import pandas as pd
import pytest

from examples.support_ticket_comparison import compare_models, inspect_tickets


@pytest.fixture
def support_csv(tmp_path):
    billing = [
        "Refund my purchase", "Please send the invoice", "Card charged twice",
        "Where is the bill", "Subscription price is wrong", "Cancel monthly payment",
        "Apply account credit", "Billing address change", "Tax invoice missing",
        "I need a refund", "Update payment details", "The plan costs too much",
    ]
    technical = [
        "The app keeps crashing", "Cannot sign in", "Page fails to load",
        "An API error occurred", "Login not working", "The tool is frozen",
        "Dashboard loads slowly", "Software installation failed", "Network keeps disconnecting",
        "The server timed out", "Cannot reset my password", "Application error again",
    ]
    records = []
    for i, (message, category) in enumerate(
        [(text, "billing") for text in billing] + [(text, "technical") for text in technical], 1
    ):
        records.append({
            "ticket_id": i,
            "message": message,
            "category": category,
            "channel": "web" if i % 2 else "app",
            "response_minutes": 10.0 if i % 2 else 20.0,
        })
    records.append(records[0].copy())
    records.append({
        "ticket_id": 200, "message": None, "category": "billing",
        "channel": "web", "response_minutes": 30.0,
    })
    records.append({
        "ticket_id": 201, "message": "Unlabeled", "category": None,
        "channel": "app", "response_minutes": 40.0,
    })
    records.append({
        "ticket_id": 202, "message": "Help with pricing", "category": "billing",
        "channel": "web", "response_minutes": None,
    })
    path = tmp_path / "tickets.csv"
    pd.DataFrame(records).to_csv(path, index=False)
    return path


def test_objective_specific_cleaning(support_csv):
    profile, model_df, means = inspect_tickets(support_csv)
    assert profile["raw_shape"] == (28, 5)
    assert profile["duplicate_rows"] == 1
    assert profile["missing_per_column"]["response_minutes"] == 1
    assert profile["missing_per_column"]["category"] == 1
    assert model_df.shape[0] == 25
    assert model_df["response_minutes"].isna().sum() == 1  # kept for NLP
    assert means.loc["web"] == pytest.approx((12 * 10.0 + 30.0) / 13)
    assert means.loc["app"] == pytest.approx((12 * 20.0 + 40.0) / 13)


def test_comparison_uses_shared_split_and_is_reproducible(support_csv):
    result = compare_models(support_csv)
    _, results, _ = result
    assert set(results) == {"BoW", "TF-IDF"}
    assert results["BoW"]["test_count"] == results["TF-IDF"]["test_count"]
    assert results["BoW"]["test_count"] == 7
    for data in results.values():
        assert 0.0 <= data["accuracy"] <= 1.0
        assert len(data["confusion_matrix"]) == 2
        assert sum(map(sum, data["confusion_matrix"])) == data["test_count"]
    profile2, results2, means2 = compare_models(support_csv)
    assert profile2 == result[0]
    assert results2 == result[1]
    pd.testing.assert_series_equal(means2, result[2])


def test_missing_required_csv_columns(tmp_path):
    path = tmp_path / "no_channel.csv"
    pd.DataFrame({"message": ["hi"], "category": ["technical"]}).to_csv(path, index=False)
    with pytest.raises(ValueError, match="requires"):
        inspect_tickets(path)
