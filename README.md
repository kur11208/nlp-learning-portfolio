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

ローカルで `pytest` の**5テスト成功**を確認しています。テスト項目は前処理、複数クエリ検索、空データの扱い、pandasの集計、分類処理の再現性です。

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
│   └── review_classifier.py
└── tests/
    └── test_examples.py
```

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
