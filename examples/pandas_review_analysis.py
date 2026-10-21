"""Basic pandas workflow: missing values, duplicates, and group aggregates."""

import pandas as pd


REVIEWS = [
    {"text": "  Good service! ", "label": "positive", "score": 5},
    {"text": "Bad quality.", "label": "negative", "score": 2},
    {"text": "  Good service! ", "label": "positive", "score": 5},
    {"text": "Great support", "label": "positive", "score": 4},
    {"text": None, "label": "negative", "score": 1},
]


def summarize_reviews(records: list[dict] | None = None) -> pd.DataFrame:
    """Return per-label counts and average scores after cleaning."""
    df = pd.DataFrame(REVIEWS if records is None else records).copy()
    required = {"text", "label", "score"}
    if not required.issubset(df.columns):
        raise ValueError(f"records must contain columns {sorted(required)}")
    df["text"] = df["text"].fillna("").str.strip().str.lower()
    df = df.drop_duplicates(subset=["text", "label", "score"])
    summary = df.groupby("label").agg(
        review_count=("text", "size"), average_score=("score", "mean")
    )
    return summary.sort_index()


if __name__ == "__main__":
    print(summarize_reviews().to_string())
