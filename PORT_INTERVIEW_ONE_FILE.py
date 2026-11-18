
from __future__ import annotations

import argparse
import errno
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import socket
import webbrowser

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline


# ===== 共通設定 =======================================================
ROOT = Path(__file__).resolve().parent
CSV_PATH = ROOT / "demo_data" / "sample_tickets.csv"
WEB_DIR = ROOT / "demo_web"
CATEGORIES = ["billing", "technical"]  # 請求・支払い／技術的な問題
TEST_SIZE = 0.25                      # 48件のうち25%=12件をテスト
RANDOM_SEED = 3                       # 分割を毎回同じにして再現性を保つ


# ===== STEP 1：CSVを読み込んで、学習に使えるデータを残す =============
def load_data(csv_path=CSV_PATH):
    """欠損行・重複行・空文字の問い合わせを除く。"""
    df = pd.read_csv(csv_path)

    # もとのデモと同じCSV列の存在を確認する。
    required = {"message", "category", "channel", "response_minutes"}
    if not required.issubset(df.columns):
        raise ValueError(f"CSVには {sorted(required)} の列が必要です")

    # message=問い合わせ文、category=正解ラベル。
    # 欠損値や重複した行を除き、空白しかない文章も除外する。
    df = df.dropna(subset=["message", "category"]).drop_duplicates()
    df = df.loc[
        df["message"].astype(str).str.strip().ne("")
        & df["category"].astype(str).str.strip().ne("")
    ].copy()

    if set(df["category"].unique()) != set(CATEGORIES):
        raise ValueError("billing と technical の2カテゴリが必要です")
    return df


# ===== STEP 2〜4：データ分割 → 学習 → 評価 → 誤分類分析 =================
def build_bundle(csv_path=CSV_PATH):
    df = load_data(csv_path)

    # STEP 2【学習用とテスト用を先に分ける】
    # 75%=36件で学習、25%=12件で未知データへの予測を評価。
    # stratifyで請求と技術の割合を同じにする。
    # 分割後にベクトル化のfitを行い、テストデータからの情報漏れを防ぐ。
    train_x, test_x, train_y, test_y = train_test_split(
        df["message"],
        df["category"],
        test_size=TEST_SIZE,
        random_state=RANDOM_SEED,
        stratify=df["category"],
    )

    report = {
        "dataset": "固定の架空英語問い合わせ（デモ専用・実績値ではありません）",
        "sample_count": len(df),
        "train_count": len(train_x),
        "test_count": len(test_x),
        "seed": RANDOM_SEED,
        "labels": CATEGORIES,
        "models": {},
    }
    models = {}

    # STEP 3【文章の数値化と、機械学習モデルの学習】
    # BoW：各単語の出現回数を数える。
    # TF-IDF：単語の出現頻度と他の文章への登場状況から重み付けする。
    # 分類器はどちらも同じ MultinomialNB（多項ナイーブベイズ）。
    # ⇒ 数値化の方法だけを変えて、同じ条件で比較できる。
    for name, vectorizer in [
        ("BoW", CountVectorizer()),
        ("TF-IDF", TfidfVectorizer()),
    ]:
        pipeline = Pipeline([
            ("features", vectorizer),           # 文章 → 数字
            ("classifier", MultinomialNB()),    # 数字 → カテゴリの予測
        ])

        pipeline.fit(train_x, train_y)           # 学習（fit）
        predicted = pipeline.predict(test_x)     # 未知の12件を予測（predict）

        # STEP 4【予測がどれくらい正しいかを評価】
        # Accuracy：全体の正解率
        # Precision：そのカテゴリと予測したものが正しかった割合
        # Recall：実際にそのカテゴリのものを見つけられた割合
        # F1：PrecisionとRecallの調和平均
        # Macro：カテゴリごとに計算した値を単純平均する
        metrics = classification_report(
            test_y,
            predicted,
            labels=CATEGORIES,
            output_dict=True,
            zero_division=0,
        )

        # 【間違えた文章の中身を見る】
        # 正解ラベルと予測ラベルが違ったものだけ記録する。
        mistakes = []
        for text, actual, guess in zip(test_x, test_y, predicted):
            if actual != guess:
                mistakes.append({
                    "text": str(text),
                    "actual": str(actual),
                    "predicted": str(guess),
                })

        report["models"][name] = {
            "accuracy": float(accuracy_score(test_y, predicted)),
            "precision_macro": float(metrics["macro avg"]["precision"]),
            "recall_macro": float(metrics["macro avg"]["recall"]),
            "f1_macro": float(metrics["macro avg"]["f1-score"]),
            "confusion_matrix": confusion_matrix(
                test_y, predicted, labels=CATEGORIES
            ).tolist(),
            "mistakes": mistakes[:8],
        }
        models[name] = pipeline  # あとでWeb画面から新しい文章を分類できる

    return report, models


# ===== STEP 5：学習済みモデルで「新しい文章」を分類する ================
def predict_message(models, message):
    """選択した英文を2つのモデルに渡し、予測ラベルを返す。"""
    if not isinstance(message, str) or not message.strip():
        raise ValueError("英文の問い合わせを入力してください")
    if len(message) > 500:
        raise ValueError("入力は500文字以内にしてください")

    results = {}
    for name, pipeline in models.items():
        # 1件でもリストに入れてpredictする。
        # Pipelineが文章の数値化→カテゴリ分類まで自動で行う。
        results[name] = str(pipeline.predict([message])[0])
    return results


# ===== STEP 6：Web画面とPythonをJSONでつなぐ ========================
# ブラウザ：JavaScript → POST /api/predict → Pythonで推論 → JSONで返却。
# 画面用HTML / CSS / JavaScriptは既存のdemo_web/をそのまま使う。
# ※このファイルにNLPのコードとAPI側のコードをまとめている。
def create_server(port=8765, bundle=None):
    report, models = bundle if bundle is not None else build_bundle()

    # GET /api/report：Accuracy等の評価結果をブラウザに渡す。
    pages = {
        "/": ((WEB_DIR / "index.html").read_bytes(), "text/html; charset=utf-8"),
        "/style.css": ((WEB_DIR / "style.css").read_bytes(), "text/css; charset=utf-8"),
        "/app.js": ((WEB_DIR / "app.js").read_bytes(), "text/javascript; charset=utf-8"),
        "/api/report": (
            json.dumps(report, ensure_ascii=False).encode("utf-8"),
            "application/json; charset=utf-8",
        ),
    }

    class Handler(BaseHTTPRequestHandler):
        def reply(self, status, data, mime):
            self.send_response(status)
            self.send_header("Content-Type", mime)
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header(
                "Content-Security-Policy",
                "default-src 'self'; script-src 'self'; style-src 'self'; "
                "connect-src 'self'; object-src 'none'; frame-ancestors 'none'",
            )
            self.end_headers()
            self.wfile.write(data)

        def send_json(self, status, obj):
            body = json.dumps(obj, ensure_ascii=False).encode("utf-8")
            self.reply(status, body, "application/json; charset=utf-8")

        def do_GET(self):
            page = pages.get(self.path)
            if page is None:
                self.send_error(404)
                return
            self.reply(200, *page)

        def do_POST(self):
            # POST /api/predict：画面から来た文章を分類する。
            if self.path != "/api/predict":
                self.send_error(404)
                return
            if self.headers.get("Content-Type", "").split(";")[0].strip().lower() != "application/json":
                self.send_json(415, {"error": "JSON形式で送信してください"})
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
            except ValueError:
                self.send_json(400, {"error": "不正なContent-Lengthです"})
                return
            if length <= 0 or length > 4096:
                self.send_json(413, {"error": "送信データが大きすぎるか空です"})
                return
            try:
                payload = json.loads(self.rfile.read(length))
                if not isinstance(payload, dict):
                    raise ValueError("JSONオブジェクトを送信してください")
                predictions = predict_message(models, payload.get("message"))
            except (ValueError, UnicodeDecodeError, TypeError):
                self.send_json(400, {"error": "500文字以内の英語の問い合わせを入力してください"})
                return

            self.send_json(200, {
                "predictions": predictions,
                "note": "架空データで学習した分類例。正確性を保証しません。",
            })

        def log_message(self, format, *args):
            pass  # 面接中、アクセスログを大量に出さない

    return ThreadingHTTPServer(("127.0.0.1", port), Handler)


# ===== 起動部分：プログラム全体の入口 ================================
def main():
    parser = argparse.ArgumentParser(description="NLP面接用：全処理1ファイル")
    parser.add_argument("--check", action="store_true", help="評価結果を出す")
    parser.add_argument("--no-browser", action="store_true")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    if not 0 <= args.port <= 65535:
        parser.error("--port must be 0..65535")

    bundle = build_bundle()  # 起動時にデータを読み込んで2モデルを学習
    if args.check:
        report, models = bundle
        for name, stats in report["models"].items():
            print(
                f"{name}: Accuracy={stats['accuracy']:.1%}, "
                f"Macro F1={stats['f1_macro']:.1%}, "
                f"誤分類={len(stats['mistakes'])}件"
            )
        return

    candidates = range(8765, 8775) if args.port == 8765 else [args.port]
    for port in candidates:
        try:
            server = create_server(port, bundle)
            break
        except OSError as exc:
            if exc.errno not in (errno.EADDRINUSE, errno.EACCES):
                raise
    else:
        parser.error("利用可能なポートがありません。--port 8888 などを指定してください")

    url = f"http://127.0.0.1:{server.server_address[1]}/"
    print(f"Webデモ: {url}", flush=True)
    if not args.no_browser:
        webbrowser.open(url)

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
