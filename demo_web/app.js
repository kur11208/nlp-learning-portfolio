"use strict";
const $ = (id) => document.getElementById(id);
const format = (x) => (100 * x).toFixed(1) + "%";
const label = (x) => x === "billing" ? "請求・支払い（billing）" : x === "technical" ? "技術的な問題（technical）" : x;

function el(tag, cls, text) {
  const node = document.createElement(tag);
  if (cls) node.className = cls;
  if (text !== undefined) node.textContent = String(text);
  return node;
}

function renderReport(report) {
  $("meta").textContent = report.dataset + " ｜ 全" + report.sample_count + "件、訓練" + report.train_count + "件、テスト" + report.test_count + "件 ｜ split seed=" + report.seed;
  const holder = $("metrics");
  holder.replaceChildren();
  Object.entries(report.models).forEach(([name, details]) => {
    const card = el("article", "item");
    card.append(el("h3", "", name));
    const stats = el("div", "stats");
    [["Accuracy", details.accuracy], ["Macro Precision", details.precision_macro], ["Macro Recall", details.recall_macro], ["Macro F1", details.f1_macro]].forEach(([title, value]) => {
      const part = el("div", "metric", title);
      part.append(el("strong", "", format(value)));
      stats.append(part);
    });
    card.append(stats);
    const desc = el("p", "subtle", "混同行列：行＝正解、列＝予測（billing → technical）");
    card.append(desc);
    const table = el("table", "matrix");
    const head = el("tr");
    ["実際 ／ 予測", ...report.labels].forEach((v) => head.append(el("th", "", v)));
    table.append(head);
    report.labels.forEach((name, index) => {
      const row = el("tr");
      row.append(el("th", "", name));
      details.confusion_matrix[index].forEach((v) => row.append(el("td", "", v)));
      table.append(row);
    });
    card.append(table);
    holder.append(card);
  });
  const errors = $("errors");
  errors.replaceChildren();
  Object.entries(report.models).forEach(([name, details]) => {
    const section = el("div", "item");
    section.append(el("h3", "", name + "：誤分類 " + details.mistakes.length + "件（最大8件表示）"));
    if (!details.mistakes.length) {
      section.append(el("p", "subtle", "この固定テスト分割では誤分類なし。新しいデータで正しいとは限りません。"));
    }
    details.mistakes.forEach((example) => {
      const wrap = el("div", "error");
      wrap.append(el("strong", "", example.text));
      wrap.append(el("span", "", "正解: " + label(example.actual) + " ／ 予測: " + label(example.predicted)));
      section.append(wrap);
    });
    errors.append(section);
  });
}

async function classify() {
  const button = $("predict");
  button.disabled = true;
  $("prediction-status").textContent = "分類中…";
  try {
    const response = await fetch("/api/predict", {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify({message: $("message").value})
    });
    const result = await response.json();
    if (!response.ok) throw new Error(result.error || "分類に失敗しました");
    const area = $("prediction-results");
    area.replaceChildren();
    Object.entries(result.predictions).forEach(([name, resultLabel]) => {
      const item = el("div", "item");
      item.append(el("h3", "", name));
      item.append(el("div", "big", label(resultLabel)));
      area.append(item);
    });
    $("prediction-status").textContent = result.note;
  } catch (err) {
    $("prediction-status").textContent = err.message;
  } finally {
    button.disabled = false;
  }
}

async function loadReport() {
  try {
    const response = await fetch("/api/report");
    if (!response.ok) throw new Error("評価情報を取得できません");
    renderReport(await response.json());
  } catch (err) {
    $("meta").textContent = "読み込みエラー: " + err.message;
  }
}

$("predict").addEventListener("click", classify);
document.querySelectorAll(".example").forEach((button) =>
  button.addEventListener("click", () => { $("message").value = button.dataset.text; classify(); })
);
loadReport();
