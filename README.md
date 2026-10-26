# NLP Learning Portfolio | Pythonで学ぶ自然言語処理

[![Python tests](https://github.com/kur11208/nlp-learning-portfolio/actions/workflows/tests.yml/badge.svg)](https://github.com/kur11208/nlp-learning-portfolio/actions/workflows/tests.yml)


Pythonでのデータ前処理から、単語頻度集計、TF-IDFによる類似文検索、機械学習によるテキスト分類・評価までを、**段階的に学習・実装した記録**です。2026年10月の技術面接対策をきっかけに、毎日のコーディング演習と修正履歴を整理しました。

> **位置付け**：学習用の小規模な再現可能デモです。実サービス向けの精度や大規模データでの性能は主張しません。練習時のコードをもとに、対外的に読みやすい構成へリファクタリングしテストを追加しています。

## 実装したこと

| テーマ | 技術 | 実装のポイント |
|---|---|---|
| 前処理・単語頻度 | Python, `re`, `Counter` | 大小文字・数字・句読点・連続空白の正規化、単語集計 |
| 文章検索 | `TfidfVectorizer`, `cosine_similarity` | 追加クエリは学習済みベクトライザの `transform()` で変換し、類似度最大の文を返す |
| データ集計 | pandas | 欠損値、重複削除、ラベル別の集計 |
| 2表の結合・点数分析 | pandas, NumPy | left結合→欠損・重複削除→部門平均→2次元配列でaxis別平均・条件抽出 |
| CSVからの文章分類 | pandas, scikit-learn | CSVの状態確認、欠損・重複の除去、学習/テスト分割、4指標による評価 |
| BoWとTF-IDFの比較 | CountVectorizer, TfidfVectorizer, MultinomialNB | 同一の層化分割でモデルを比較し、Accuracy・混同行列を評価。集計用と分類用の前処理を区別 |
| テキスト分類 | `train_test_split`, `Pipeline`, `MultinomialNB` | 訓練データだけで特徴量を学習し、未知のテストデータで評価 |
| 評価 | Accuracy, Precision, Recall, F1 | 学習データとテストデータの差、クラス指定、少数データの限界を確認 |

## 動かし方

Python 3.11以降を想定しています。

```bash
python -m venv .venv
# Windows PowerShell: .\.venv\Scripts\Activate.ps1
# macOS / Linux: source .venv/bin/activate
python -m pip install -r requirements.txt

python -m examples.text_preprocessing
python -m examples.tfidf_search
python -m examples.pandas_review_analysis
python -m examples.review_classifier
# review_text と label を含むCSVを指定する場合:
python -m examples.csv_review_classifier path/to/reviews.csv
# message, category, channel, response_minutes を持つ問い合わせCSVを指定:
python -m examples.support_ticket_comparison path/to/tickets.csv
python -m pytest -q
```

## 実行例

```text
query: cat eats milk
match: cat drinks milk (cosine similarity: 0.669)
query: dog eats meat
match: dog eats meat (cosine similarity: 1.000)
```

16件の小さなレビューを使った文章分類では、ある再現可能な分割（`random_state=42`）に対し、**Train Accuracy 1.00 / Test Accuracy 0.25** となりました。これは良い精度ではなく、データが少ない・未知の表現に弱いといった限界を議論するための観察です。単一の分割だけで過学習と断定することもできません。

初期の5テストに加え、10/22には表結合・欠損/重複削除・NumPy集計を検証する3テストを追加しました。10/24にはCSVの品質確認から文章分類までの通し演習を独立した例に整理し、CSV検査・分類処理・入力列検証の3テストを追加しました（追加3件はローカル実行で成功）。10/26には問い合わせデータの集計とBoW/TF-IDF比較、および3件のテストを追加（3件をローカルで実行して成功）。GitHub ActionsでPythonテストを実行する構成です。

## リポジトリ内の構成

```text
.
├── README.md
├── LEARNING_JOURNEY.md
├── requirements.txt
├── examples/
│   ├── text_preprocessing.py
│   ├── tfidf_search.py
│   ├── pandas_review_analysis.py
│   ├── review_classifier.py
│   ├── csv_review_classifier.py
│   └── support_ticket_comparison.py
└── tests/
    └── test_examples.py
```

## CSVレビュー分類演習（10/24）

`examples/csv_review_classifier.py` は `review_text` と `label`（`positive` / `negative`）列を含むCSVを受け取り、元データの欠損・完全重複を確認してから学習・テストへ分割し、TF-IDF + MultinomialNBのPipelineを学習・評価します。`review_id` のような識別子はモデル入力に使いません。サンプル用CSVはテスト時に生成するため、利用する際は自身のCSVファイルを指定してください。

**限界**：小規模な人工データへの単一hold-out評価は不安定です。実際の学習演習（別の36行の教材CSV）では、32行にクレンジングした後、8行のテストで4指標がいずれも0.0となるケースを観察しました。この結果は学習者のローカル実行ログに基づき、公開コードのデータと同一ではなく、このリポジトリだけで再現できる数値ではありません。評価結果の改善は未検証です。

## 問い合わせ分類のBoW / TF-IDF比較（10/26）

`examples/support_ticket_comparison.py` では、`message`、`category`、`channel`、`response_minutes` を持つ教材CSVを読み込みます。文章分類用の前処理（`message`、`category` の欠損削除）と応答時間集計用の前処理を分離し、集計に使える行を不必要に捨てません。

その上で、`CountVectorizer` + `MultinomialNB` と `TfidfVectorizer` + `MultinomialNB` を、**同一の学習・テスト分割**で比較します。混同行列のクラス順は `billing`、`technical` で固定しています。

演習時に用いた102行の架空データでは、文章分類に98行を使用し、各モデルのテストデータ25行で**Accuracy 0.84、混同行列 `[[12, 1], [3, 9]]`** という結果を本人の実行ログとこちらの環境で確認しました。ただし、その102行CSVはこの公開リポジトリには同梱していません。公開したテストでは独立した合成データを生成するので、**同じ0.84が出ることをテストの要件とはしていません**。1回の分割で差が出なかったことは、モデルの一般的な優劣を意味しません。

公開コードは対話演習で本人が書いた構成を元に、AI支援で関数化・入力列の検証・自動テストを追加したものです。

## 技術的に意識したこと

- **データリーク回避**：学習・テストを分割してから `Pipeline.fit(train_texts, train_labels)` を実施。テストデータには `predict()` のみ実行する。
- **再現性**：分割に固定シードを設定し、簡単な自動テストを追加する。
- **指標の意味**：AccuracyだけでなくPrecision・Recall・F1を表示し、正例ラベルを明示する。
- **評価の限界**：教材データは小規模・人工的。公開ベンチマークや外部データを使う評価は今後の課題。

## 継続的なアップデート

新しく学んだ技術は、**実装 → テスト → 課題や改善点の説明**を1セットとして、このリポジトリへ追加していきます。

- 新規のPythonコードは `examples/` に追加
- 仕様変更やバグ修正では `tests/` の確認項目も更新
- 技術的な判断と限界はREADMEや学習記録に明記
- 何が増えたかは [CHANGELOG.md](CHANGELOG.md) に時系列で記録

公開向けコードは学習内容を整理したもので、学習時に自分で書いた部分とAI支援による編集・追加を区別して記載します。個人用の日報や選考情報は公開しません。

## 学習記録・今後の改善

[LEARNING_JOURNEY.md](LEARNING_JOURNEY.md) に10月の進展と解決した課題をまとめています。次の課題は、NumPy/pandasを含む初見のデータ分析、複数データ分割での検証、説明しながら時間内に書く演習です。

**学習方法**：対話型AIからフィードバックを受けながら本人が演習コードを書き、試行錯誤した内容を整理しています。公開向けのサンプルは学習コードをもとに構成・テストを追加したもので、当初の演習答案そのものではありません。
