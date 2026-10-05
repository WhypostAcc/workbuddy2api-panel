import json
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.request import Request, urlopen

import bridge
from anthropic_adapter import AnthropicStreamConverter, anthropic_request_to_chat
from responses_adapter import ResponsesStreamConverter, responses_request_to_chat


class ResponsesBridgeTests(unittest.TestCase):
    def test_claude_tool_round_trip(self):
        chat = anthropic_request_to_chat({
            "model": "cn:deepseek-v4.1-flash",
            "messages": [
                {"role": "assistant", "content": [{"type": "tool_use", "id": "tool_1", "name": "Bash", "input": {"command": "pwd"}}]},
                {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "tool_1", "content": "C:/repo"}]},
            ],
            "tools": [{"name": "Bash", "input_schema": {"type": "object"}}],
        })
        self.assertEqual(chat["messages"][0]["tool_calls"][0]["function"]["name"], "Bash")
        self.assertEqual(chat["messages"][1]["tool_call_id"], "tool_1")
        self.assertEqual(chat["tools"][0]["function"]["name"], "Bash")

        # Anthropic may put text and several tool_result blocks in one user
        # content array. Tool results must be emitted before the text message;
        # otherwise the Chat history becomes assistant -> user -> tool and
        # DeepSeek rejects it as 11148.
        mixed = anthropic_request_to_chat({
            "model": "cn:deepseek-v4.1-flash",
            "messages": [
                {"role": "assistant", "content": [
                    {"type": "tool_use", "id": "tool_1", "name": "Bash", "input": {"command": "a"}},
                    {"type": "tool_use", "id": "tool_2", "name": "Bash", "input": {"command": "b"}},
                ]},
                {"role": "user", "content": [
                    {"type": "tool_result", "tool_use_id": "tool_2", "content": "B"},
                    {"type": "text", "text": "continue"},
                    {"type": "tool_result", "tool_use_id": "tool_1", "content": "A"},
                    {"type": "tool_result", "tool_use_id": "tool_1", "content": "duplicate"},
                ]},
            ],
        })
        self.assertEqual(
            [(m["role"], m.get("tool_call_id")) for m in mixed["messages"]],
            [("assistant", None), ("tool", "tool_2"), ("tool", "tool_1"), ("user", None)],
        )

        converter = AnthropicStreamConverter(model="cn:deepseek-v4.1-flash")
        chunk = {"choices": [{"delta": {"tool_calls": [{"index": 0, "id": "tool_2", "function": {"name": "Bash", "arguments": '{"command":"pwd"}'}}]}, "finish_reason": "tool_calls"}]}
        events = converter.feed_line("data: " + json.dumps(chunk)) + converter.finish()
        self.assertIn("event: content_block_start", events)
        self.assertIn("event: message_stop", events)
        self.assertEqual(converter.build_message()["stop_reason"], "tool_use")

    def test_codex_tool_round_trip(self):
        request = {
            "model": "cn:glm-5.2",
            "instructions": "You are a coding agent.",
            "input": [
                {"role": "user", "content": [{"type": "input_text", "text": "List files"}]},
                {"type": "function_call", "call_id": "call_123", "name": "shell", "arguments": "{}"},
                {"type": "function_call_output", "call_id": "call_123", "output": "a.txt"},
            ],
            "tools": [{"type": "function", "name": "shell", "parameters": {"type": "object"}}],
            "reasoning": {"effort": "medium"},
        }
        chat = responses_request_to_chat(request)
        self.assertEqual(chat["messages"][0], {"role": "system", "content": request["instructions"]})
        self.assertEqual(chat["messages"][-1], {"role": "tool", "tool_call_id": "call_123", "content": "a.txt"})
        self.assertEqual(chat["tools"][0]["function"]["name"], "shell")
        self.assertEqual(chat["reasoning_effort"], "medium")

        converter = ResponsesStreamConverter(model="cn:glm-5.2")
        chunk = {"choices": [{"delta": {"tool_calls": [{"index": 0, "id": "call_456", "function": {"name": "shell", "arguments": '{"command":"pwd"}'}}]}}]}
        events = converter.feed_line("data: " + json.dumps(chunk)) + converter.finish()
        self.assertIn("event: response.function_call_arguments.delta", events)
        self.assertIn("event: response.completed", events)
        output = converter.get_nonstream_response()["output"]
        self.assertEqual(output[0]["call_id"], "call_456")
        self.assertEqual(output[0]["arguments"], '{"command":"pwd"}')

    def test_text_stream(self):
        converter = ResponsesStreamConverter(model="cn:glm-5.2")
        events = converter.feed_line('data: {"choices":[{"delta":{"content":"hello"}}]}')
        events += converter.finish()
        self.assertIn("event: response.created", events)
        self.assertIn("event: response.output_text.delta", events)
        self.assertIn("event: response.completed", events)
        self.assertEqual(converter.get_nonstream_response()["output"][0]["content"][0]["text"], "hello")

    def test_http_stream_and_authorization(self):
        observed = {}

        class FakeGateway(BaseHTTPRequestHandler):
            def do_POST(self):
                observed["path"] = self.path
                observed["authorization"] = self.headers.get("Authorization")
                observed["body"] = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                payload = (
                    'data: {"choices":[{"delta":{"content":"hello"}}]}\n\n'
                    'data: {"choices":[{"delta":{},"finish_reason":"stop"}]}\n\n'
                    'data: [DONE]\n\n'
                ).encode()
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)

            def log_message(self, *_):
                pass

        gateway = ThreadingHTTPServer(("127.0.0.1", 0), FakeGateway)
        adapter = ThreadingHTTPServer(("127.0.0.1", 0), bridge.Handler)
        old_upstream = bridge.UPSTREAM
        bridge.UPSTREAM = f"http://127.0.0.1:{gateway.server_port}"
        threads = [threading.Thread(target=s.serve_forever, daemon=True) for s in (gateway, adapter)]
        for thread in threads:
            thread.start()
        try:
            body = json.dumps({"model": "glm-5.2", "input": "hi", "stream": True}).encode()
            request = Request(
                f"http://127.0.0.1:{adapter.server_port}/v1/responses",
                data=body,
                headers={"Content-Type": "application/json", "Authorization": "Bearer test-token"},
            )
            with urlopen(request, timeout=5) as response:
                self.assertEqual(response.status, 200)
                events = response.read().decode()
            self.assertEqual(observed["path"], "/v1/chat/completions")
            self.assertEqual(observed["authorization"], "Bearer test-token")
            self.assertEqual(observed["body"]["messages"], [{"role": "user", "content": "hi"}])
            self.assertIn("event: response.output_text.delta", events)
            self.assertIn("event: response.completed", events)
        finally:
            bridge.UPSTREAM = old_upstream
            for server in (gateway, adapter):
                server.shutdown()
                server.server_close()
            for thread in threads:
                thread.join(timeout=2)

    def test_claude_messages_route_and_model_label(self):
        observed = {}

        class FakeGateway(BaseHTTPRequestHandler):
            def do_POST(self):
                observed["authorization"] = self.headers.get("Authorization")
                observed["body"] = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                payload = (
                    'data: {"choices":[{"delta":{"content":"你好"}}]}\n\n'
                    'data: {"choices":[{"delta":{},"finish_reason":"stop"}]}\n\n'
                    'data: [DONE]\n\n'
                ).encode()
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)

            def log_message(self, *_):
                pass

        gateway = ThreadingHTTPServer(("127.0.0.1", 0), FakeGateway)
        adapter = ThreadingHTTPServer(("127.0.0.1", 0), bridge.Handler)
        old_upstream = bridge.UPSTREAM
        bridge.UPSTREAM = f"http://127.0.0.1:{gateway.server_port}"
        threads = [threading.Thread(target=s.serve_forever, daemon=True) for s in (gateway, adapter)]
        for thread in threads:
            thread.start()
        try:
            for stream in (False, True):
                body = json.dumps({
                    "model": "cn:deepseek-v4.1-flash[1M]",
                    "messages": [{"role": "user", "content": "你好"}],
                    "max_tokens": 128,
                    "stream": stream,
                }).encode()
                request = Request(
                    f"http://127.0.0.1:{adapter.server_port}/v1/v1/messages?beta=true",
                    data=body,
                    headers={"Content-Type": "application/json", "x-api-key": "test-token"},
                )
                with urlopen(request, timeout=5) as response:
                    self.assertEqual(response.status, 200)
                    reply = response.read().decode()
                if stream:
                    self.assertIn("event: message_start", reply)
                    self.assertIn("event: message_stop", reply)
                else:
                    self.assertEqual(json.loads(reply)["content"][0]["text"], "你好")
                self.assertEqual(observed["authorization"], "Bearer test-token")
                self.assertEqual(observed["body"]["model"], "cn:deepseek-v4.1-flash")
        finally:
            bridge.UPSTREAM = old_upstream
            for server in (gateway, adapter):
                server.shutdown()
                server.server_close()
            for thread in threads:
                thread.join(timeout=2)


if __name__ == "__main__":
    unittest.main()
