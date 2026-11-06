"""面接デモ用アダプター。既存の文章分類ロジックは変更せず、画面用結果に変換する。"""

from __future__ import annotations

from pathlib import Path

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

from examples.support_ticket_comparison import CATEGORIES, inspect_tickets

ROOT = Path(__file__).resolve().parent
SAMPLE_CSV = ROOT / "demo_data" / "sample_tickets.csv"
SPLIT_SEED = 3
TEST_SIZE = 0.25


def build_bundle(csv_path: Path = SAMPLE_CSV) -> tuple[dict, dict]:
    """学習用サンプルを同じ訓練/テスト分割で比較し、表示データと学習済みモデルを作る。"""
    profile, cleaned, _ = inspect_tickets(csv_path)
    if set(cleaned["category"].unique()) != set(CATEGORIES):
        raise ValueError("Demo requires exactly billing and technical categories")

    # 学習データだけで語彙とIDFを学習する。テストデータをfit()に入れない。
    train_x, test_x, train_y, test_y = train_test_split(
        cleaned["message"], cleaned["category"],
        test_size=TEST_SIZE, random_state=SPLIT_SEED, stratify=cleaned["category"],
    )

    report = {
        "dataset": "固定の架空英語問い合わせ（デモ専用・実績値ではありません）",
        "sample_count": int(len(cleaned)),
        "train_count": int(len(train_x)),
        "test_count": int(len(test_x)),
        "seed": SPLIT_SEED,
        "labels": CATEGORIES,
        "models": {},
    }
    fitted = {}
    for name, vectorizer in (
        ("BoW", CountVectorizer()),
        ("TF-IDF", TfidfVectorizer()),
    ):
        # 既存の support_ticket_comparison.py と同じモデル構成に統一する。
        pipeline = Pipeline([
            ("features", vectorizer),
            ("classifier", MultinomialNB()),
        ])
        pipeline.fit(train_x, train_y)
        predicted = pipeline.predict(test_x)
        scores = classification_report(
            test_y, predicted, labels=CATEGORIES, output_dict=True, zero_division=0,
        )
        mistakes = [
            {"text": str(text), "actual": str(actual), "predicted": str(guess)}
            for text, actual, guess in zip(test_x, test_y, predicted)
            if actual != guess
        ]
        report["models"][name] = {
            "accuracy": float(accuracy_score(test_y, predicted)),
            "precision_macro": float(scores["macro avg"]["precision"]),
            "recall_macro": float(scores["macro avg"]["recall"]),
            "f1_macro": float(scores["macro avg"]["f1-score"]),
            "confusion_matrix": confusion_matrix(test_y, predicted, labels=CATEGORIES).tolist(),
            "mistakes": mistakes[:8],
        }
        fitted[name] = pipeline
    return report, fitted


def predict_message(models: dict, message: str) -> dict:
    """自由入力を、訓練済みの2つのモデルで推論する。これは正解保証ではない。"""
    if not isinstance(message, str) or not message.strip():
        raise ValueError("英文の問い合わせを入力してください")
    if len(message) > 500:
        raise ValueError("入力は500文字以内にしてください")
    return {name: str(model.predict([message])[0]) for name, model in models.items()}
