"""NLP技術面接用ローカルデモ。外部APIや外部サーバーは使わない。"""

from __future__ import annotations

import argparse
import errno
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import socket
import webbrowser

from demo_model import build_bundle, predict_message

ROOT = Path(__file__).resolve().parent
WEB = ROOT / "demo_web"


class LocalServer(ThreadingHTTPServer):
    allow_reuse_address = False

    def server_bind(self):
        if hasattr(socket, "SO_EXCLUSIVEADDRUSE"):
            self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
        super().server_bind()


def create_server(port: int = 8765, bundle: tuple[dict, dict] | None = None) -> ThreadingHTTPServer:
    """起動時に学習を完了し、画面への読み取りアクセスと推論だけを許す。"""
    report, models = bundle if bundle is not None else build_bundle()
    pages = {
        "/": ((WEB / "index.html").read_bytes(), "text/html; charset=utf-8"),
        "/style.css": ((WEB / "style.css").read_bytes(), "text/css; charset=utf-8"),
        "/app.js": ((WEB / "app.js").read_bytes(), "text/javascript; charset=utf-8"),
        "/api/report": (json.dumps(report, ensure_ascii=False).encode("utf-8"), "application/json; charset=utf-8"),
    }

    class Handler(BaseHTTPRequestHandler):
        def reply(self, status: int, data: bytes, mime: str):
            self.send_response(status)
            self.send_header("Content-Type", mime)
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Content-Security-Policy", "default-src 'self'; script-src 'self'; style-src 'self'; connect-src 'self'; object-src 'none'; frame-ancestors 'none'")
            self.end_headers()
            self.wfile.write(data)

        def send_json(self, status: int, obj: dict):
            self.reply(status, json.dumps(obj, ensure_ascii=False).encode("utf-8"), "application/json; charset=utf-8")

        def do_GET(self):
            page = pages.get(self.path)
            if page is None:
                self.send_error(404)
                return
            self.reply(200, *page)

        def do_POST(self):
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
                result = predict_message(models, payload.get("message"))
            except (ValueError, UnicodeDecodeError, TypeError):
                self.send_json(400, {"error": "500文字以内の英語の問い合わせを入力してください"})
                return
            self.send_json(200, {"predictions": result, "note": "架空データで学習した分類例。正確性を保証しません。"})

        def log_message(self, format, *args):
            # 標準出力にアクセス内容を大量に表示しない。
            pass

    return LocalServer(("127.0.0.1", port), Handler)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8765, help="local HTTP port")
    parser.add_argument("--no-browser", action="store_true")
    parser.add_argument("--check", action="store_true", help="compute demo and verify deterministic output")
    args = parser.parse_args()
    if not 0 <= args.port <= 65535:
        parser.error("--port must be 0..65535")

    bundle = build_bundle()
    if args.check:
        # 画面は開かず、同じ入力と設定から結果が再現するか確認する。
        second, _ = build_bundle()
        if bundle[0] != second:
            raise RuntimeError("NLP demo is not deterministic")
        print(json.dumps(bundle[0], ensure_ascii=False, indent=2))
        return 0

    candidates = range(8765, 8775) if args.port == 8765 else [args.port]
    server = None
    for candidate in candidates:
        try:
            server = create_server(candidate, bundle)
            break
        except OSError as exc:
            if exc.errno not in (errno.EADDRINUSE, errno.EACCES):
                raise
    if server is None:
        parser.error("No available local port. Try --port 8888")

    url = f"http://127.0.0.1:{server.server_address[1]}/"
    print(f"NLP interview demo: {url}", flush=True)
    print("Press Ctrl+C to stop. Synthetic examples only; no external API.", flush=True)
    if not args.no_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
