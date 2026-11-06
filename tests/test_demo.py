"""NLP面接デモの再現性・入力検証・読み取り専用HTTPのテスト。

ここでは分類精度の優秀さではなく、デモの動作を検証する。
"""

from __future__ import annotations

from http.client import HTTPConnection
import json
import threading

import pytest

from demo import create_server
from demo_model import build_bundle, predict_message


@pytest.fixture(scope="module")
def bundle():
    return build_bundle()


def test_report_is_deterministic_and_uses_equal_splits(bundle):
    report, models = bundle
    assert report == build_bundle()[0]
    assert report["sample_count"] == 48
    assert report["train_count"] + report["test_count"] == 48
    assert set(report["models"]) == {"BoW", "TF-IDF"}
    assert set(models) == {"BoW", "TF-IDF"}
    assert report["labels"] == ["billing", "technical"]
    for metrics in report["models"].values():
        assert all(
            0 <= metrics[name] <= 1
            for name in ("accuracy", "precision_macro", "recall_macro", "f1_macro")
        )
        matrix = metrics["confusion_matrix"]
        assert len(matrix) == 2
        assert all(len(row) == 2 for row in matrix)
        assert sum(map(sum, matrix)) == report["test_count"]
        assert len(metrics["mistakes"]) <= 8


def test_prediction_uses_both_existing_baselines(bundle):
    _, models = bundle
    result = predict_message(models, "Please check my invoice and refund")
    assert set(result) == {"BoW", "TF-IDF"}
    assert set(result.values()) <= {"billing", "technical"}


@pytest.mark.parametrize("message", ["", "  ", None, 42, "x" * 501])
def test_prediction_rejects_bad_input(bundle, message):
    with pytest.raises(ValueError):
        predict_message(bundle[1], message)


@pytest.fixture(scope="module")
def http_demo(bundle):
    server = create_server(0, bundle)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield server.server_address[1]
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=3)


def request(port, method, path, body=None, headers=None):
    conn = HTTPConnection("127.0.0.1", port, timeout=5)
    try:
        conn.request(method, path, body=body, headers=headers or {})
        response = conn.getresponse()
        return response.status, response.read(), dict(response.getheaders())
    finally:
        conn.close()


def test_demo_http_serves_only_expected_assets(http_demo):
    for path in ["/", "/style.css", "/app.js", "/api/report"]:
        status, body, headers = request(http_demo, "GET", path)
        assert status == 200 and body
        assert "no-store" == headers.get("Cache-Control")
        assert "default-src 'self'" in headers.get("Content-Security-Policy", "")
    status, _, _ = request(http_demo, "GET", "/../../etc/passwd")
    assert status == 404


def test_demo_http_predict_and_rejects_unexpected_calls(http_demo):
    data = json.dumps({"message": "The app crashes while logging in"}).encode("utf-8")
    status, body, _ = request(http_demo, "POST", "/api/predict", data, {"Content-Type": "application/json"})
    assert status == 200
    result = json.loads(body)
    assert set(result["predictions"]) == {"BoW", "TF-IDF"}

    status, _, _ = request(http_demo, "POST", "/api/predict", b'{"message":""}', {"Content-Type": "application/json"})
    assert status == 400
    status, _, _ = request(http_demo, "POST", "/api/predict", data, {"Content-Type": "text/plain"})
    assert status == 415
    status, _, _ = request(http_demo, "POST", "/api/orders", data, {"Content-Type": "application/json"})
    assert status == 404
