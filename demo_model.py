"""NLPの学習内容を使って、デモ画面に結果を渡すコード（AI整備版）。"""

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


def build_bundle(csv_path=SAMPLE_CSV):
    """同じデータを使ってBoWとTF-IDFを比較する。"""
    _, df, _ = inspect_tickets(csv_path)

    if set(df["category"].unique()) != set(CATEGORIES):
        raise ValueError("Demo requires exactly billing and technical categories")

    # 先に分割する。ベクトル化のfitは訓練データだけに行う。
    train_x, test_x, train_y, test_y = train_test_split(
        df["message"],
        df["category"],
        test_size=TEST_SIZE,
        random_state=SPLIT_SEED,
        stratify=df["category"],
    )

    report = {
        "dataset": "固定の架空英語問い合わせ（デモ専用・実績値ではありません）",
        "sample_count": int(len(df)),
        "train_count": int(len(train_x)),
        "test_count": int(len(test_x)),
        "seed": SPLIT_SEED,
        "labels": CATEGORIES,
        "models": {},
    }
    models = {}

    # 分割方法と分類器を揃え、ベクトル化の方法だけ変える。
    for name, vectorizer in [
        ("BoW", CountVectorizer()),
        ("TF-IDF", TfidfVectorizer()),
    ]:
        pipe = Pipeline([
            ("features", vectorizer),
            ("classifier", MultinomialNB()),
        ])

        pipe.fit(train_x, train_y)
        pred = pipe.predict(test_x)

        scores = classification_report(
            test_y, pred,
            labels=CATEGORIES,
            output_dict=True,
            zero_division=0,
        )

        # 正解と予測が違った文章を取り出す。
        mistakes = []
        for text, actual, guess in zip(test_x, test_y, pred):
            if actual != guess:
                mistakes.append({
                    "text": str(text),
                    "actual": str(actual),
                    "predicted": str(guess),
                })

        report["models"][name] = {
            "accuracy": float(accuracy_score(test_y, pred)),
            "precision_macro": float(scores["macro avg"]["precision"]),
            "recall_macro": float(scores["macro avg"]["recall"]),
            "f1_macro": float(scores["macro avg"]["f1-score"]),
            "confusion_matrix": confusion_matrix(
                test_y, pred, labels=CATEGORIES
            ).tolist(),
            "mistakes": mistakes[:8],
        }
        models[name] = pipe

    return report, models


def predict_message(models, message):
    """画面で入力された英文を2つのモデルで分類する。"""
    if not isinstance(message, str) or not message.strip():
        raise ValueError("英文の問い合わせを入力してください")
    if len(message) > 500:
        raise ValueError("入力は500文字以内にしてください")

    result = {}
    for name, pipe in models.items():
        pred = pipe.predict([message])
        result[name] = str(pred[0])
    return result
