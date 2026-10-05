"""Local Responses and Anthropic Messages adapters for the WorkBuddy gateway.

The conversion logic is from the adjacent MIT-licensed workbuddy2api project;
see *_adapter.py and LICENSE in this directory.
"""

import json
import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import Request, urlopen

from anthropic_adapter import AnthropicStreamConverter, anthropic_request_to_chat
from responses_adapter import ResponsesStreamConverter, responses_request_to_chat


UPSTREAM = os.environ.get("WB2API_BASE_URL", "http://wb2api:7863").rstrip("/")
MAX_BODY = 8 * 1024 * 1024


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def _send(self, status, payload, content_type="application/json; charset=utf-8"):
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(payload)))
        self.send_header("Connection", "close")
        self.end_headers()
        self.wfile.write(payload)
        self.close_connection = True

    def _error(self, status, message, anthropic=False):
        error = {"message": message, "type": "invalid_request_error"}
        payload = json.dumps({"type": "error", "error": error} if anthropic else {"error": error}).encode("utf-8")
        self._send(status, payload)

    def _upstream_request(self, path, data=None):
        headers = {}
        auth = self.headers.get("Authorization")
        if not auth and self.headers.get("x-api-key"):
            auth = "Bearer " + self.headers["x-api-key"]
        if auth:
            headers["Authorization"] = auth
        if data is not None:
            headers["Content-Type"] = "application/json"
        return Request(UPSTREAM + path, data=data, headers=headers)

    def _open_upstream(self, request, anthropic=False):
        try:
            return urlopen(request, timeout=360)
        except HTTPError as exc:
            raw = exc.read()
            if anthropic:
                try:
                    upstream_error = json.loads(raw).get("error", {})
                    message = upstream_error.get("message") or raw.decode("utf-8", "replace")
                except (ValueError, AttributeError):
                    message = raw.decode("utf-8", "replace")
                self._error(exc.code, message[:1000], anthropic=True)
            else:
                self._send(exc.code, raw, exc.headers.get("Content-Type", "application/json"))
        except URLError as exc:
            self._error(502, f"WorkBuddy gateway unavailable: {exc.reason}", anthropic=anthropic)
        return None

    def do_GET(self):
        if self.path == "/bridge/healthz":
            self._send(200, b'{"service":"workbuddy2api-protocol-bridge"}')
            return
        if self.path in ("/", "/v1", "/v1/"):
            self._send(200, b'{"service":"workbuddy2api-protocol-bridge"}')
            return
        if self.path not in ("/v1/models", "/status", "/healthz"):
            self._error(404, "Unknown endpoint")
            return
        upstream = self._open_upstream(self._upstream_request(self.path))
        if upstream is None:
            return
        with upstream:
            self._send(
                upstream.status,
                upstream.read(),
                upstream.headers.get("Content-Type", "application/json"),
            )

    def do_POST(self):
        path = urlsplit(self.path).path
        is_anthropic = path in ("/v1/messages", "/v1/v1/messages", "/v1/messages/count_tokens", "/v1/v1/messages/count_tokens")
        if path != "/v1/responses" and not is_anthropic:
            self._error(404, "Unknown endpoint", anthropic=is_anthropic)
            return
        try:
            size = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            self._error(400, "Invalid Content-Length", anthropic=is_anthropic)
            return
        if size <= 0 or size > MAX_BODY:
            self._error(413 if size > MAX_BODY else 400, "Invalid request body size", anthropic=is_anthropic)
            return
        try:
            body = json.loads(self.rfile.read(size))
        except (UnicodeDecodeError, json.JSONDecodeError):
            self._error(400, "Invalid JSON body", anthropic=is_anthropic)
            return
        if not isinstance(body, dict):
            self._error(400, "Expected a JSON object", anthropic=is_anthropic)
            return

        if path.endswith("/count_tokens"):
            auth_check = self._open_upstream(self._upstream_request("/status"), anthropic=True)
            if auth_check is not None:
                auth_check.close()
                self._send(200, b'{"input_tokens":0}')
            return

        stream = body.get("stream") is True
        if is_anthropic and not body.get("messages"):
            self._error(400, "messages is required", anthropic=True)
            return
        try:
            chat = anthropic_request_to_chat(body) if is_anthropic else responses_request_to_chat(body)
        except (TypeError, ValueError) as exc:
            self._error(400, f"Request conversion failed: {exc}", anthropic=is_anthropic)
            return
        # CC Switch model labels may append a context badge. The gateway only
        # recognizes the bare model ID advertised by /v1/models.
        if is_anthropic and isinstance(chat.get("model"), str) and chat["model"].lower().endswith("[1m]"):
            chat["model"] = chat["model"][:-4]
        chat["stream_options"] = {"include_usage": True}
        payload = json.dumps(chat, ensure_ascii=False).encode("utf-8")
        upstream = self._open_upstream(
            self._upstream_request("/v1/chat/completions", data=payload),
            anthropic=is_anthropic,
        )
        if upstream is None:
            return

        converter = (
            AnthropicStreamConverter(model=chat.get("model", "unknown"))
            if is_anthropic
            else ResponsesStreamConverter(model=body.get("model", "unknown"))
        )
        with upstream:
            if stream:
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream; charset=utf-8")
                self.send_header("Cache-Control", "no-cache")
                self.send_header("Connection", "close")
                self.end_headers()
                self.close_connection = True
            try:
                for raw_line in upstream:
                    line = raw_line.decode("utf-8", "replace")
                    if line.strip() == "data: [DONE]":
                        break
                    events = converter.feed_line(line)
                    if stream and events:
                        self.wfile.write(events.encode("utf-8"))
                        self.wfile.flush()
                if stream:
                    self.wfile.write(converter.finish().encode("utf-8"))
                    self.wfile.flush()
                else:
                    result = converter.build_message() if is_anthropic else converter.get_nonstream_response()
                    self._send(200, json.dumps(result, ensure_ascii=False).encode("utf-8"))
            except (BrokenPipeError, ConnectionResetError):
                pass


if __name__ == "__main__":
    ThreadingHTTPServer(("0.0.0.0", 7864), Handler).serve_forever()
