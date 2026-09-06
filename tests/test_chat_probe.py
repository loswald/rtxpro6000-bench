import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import tempfile
import threading
import unittest

from gates.chat_probe import (
    assess_response, item_seed, load_prompts, make_request, run_item,
    run_probe, validate_request_config,
)


def response(final="391", reasoning="calculation", finish="stop", tokens=10):
    return {"choices": [{"message": {"content": final, "reasoning_content": reasoning},
                         "finish_reason": finish}], "usage": {"completion_tokens": tokens}}


class ProbeTests(unittest.TestCase):
    def test_expected_answer_in_reasoning_cannot_rescue_wrong_or_empty_final(self):
        wrong = assess_response(response("392", "17*23 is 391"), "391", 8192)
        self.assertEqual(wrong["verdict"], "wrong")
        self.assertTrue(wrong["expected_found_in_reasoning"])
        self.assertEqual(assess_response(response("", "391"), "391", 8192)["verdict"], "empty")

    def test_caps_and_reasoning_loops_are_separate_from_final_repetition(self):
        result = assess_response(response("391", "one two three four five six " * 20, "length", 8192), "391", 8192)
        self.assertEqual(result["verdict"], "degenerate")
        self.assertTrue(result["capped"])
        self.assertTrue(result["reasoning_repetition"]["flag"])
        self.assertFalse(result["final_repetition"]["flag"])
        self.assertEqual(assess_response(response("391", "", "length", 8192), "391", 8192)["verdict"], "truncated")

    def test_explicit_config_and_item_seeds_are_preserved(self):
        root = Path(__file__).parents[1]
        config = json.loads((root / "gates/profiles/qwen38fn_vendor.json").read_text())
        validate_request_config(config)
        prompts, source_hash = load_prompts(root / "box/quality20.py")
        self.assertEqual(len(prompts), 20)
        self.assertEqual(len(source_hash), 64)
        seed = item_seed(1234, 0, prompts[0][0])
        self.assertEqual(seed, item_seed(1234, 0, prompts[0][0]))
        self.assertNotEqual(seed, item_seed(1234, 1, prompts[1][0]))
        request = make_request("m", prompts[0][0], config, seed)
        for key, value in config.items():
            self.assertEqual(request[key], value)
        with self.assertRaises(ValueError):
            validate_request_config({**config, "seed": 999})

    def test_http_error_preserves_exact_request_and_raw_response(self):
        received = []
        raw = b'{"error":{"message":"test rejection"}}'

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                received.append(self.rfile.read(int(self.headers["Content-Length"])))
                self.send_response(400)
                self.end_headers()
                self.wfile.write(raw)

            def log_message(self, *args):
                pass

        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            with tempfile.TemporaryDirectory() as tmp:
                directory = Path(tmp)
                request = make_request("m", "test", {"temperature": 1, "top_p": 0.95, "max_tokens": 8192}, 42)
                result = run_item(0, "test", "answer", request, f"http://127.0.0.1:{server.server_port}/v1/chat/completions", directory, 2)
                self.assertEqual(result["verdict"], "error")
                self.assertEqual(result["http_status"], 400)
                self.assertEqual(result["response"]["error"]["message"], "test rejection")
                self.assertEqual((directory / "00/response.raw").read_bytes(), raw)
                self.assertEqual((directory / "00/request.json").read_bytes(), received[0])
                self.assertEqual(json.loads(received[0]), request)
        finally:
            server.shutdown()
            server.server_close()
            thread.join()

    def test_existing_output_directory_is_never_replaced(self):
        with tempfile.TemporaryDirectory() as tmp:
            existing = Path(tmp)
            marker = existing / "greedy-result.json"
            marker.write_text("immutable")
            with self.assertRaises(FileExistsError):
                run_probe(model="m", base_url="http://127.0.0.1:1", out_dir=existing,
                          request_config={"temperature": 1, "top_p": 0.95, "max_tokens": 8192},
                          prompts_source=Path(__file__).parents[1] / "box/quality20.py")
            self.assertEqual(marker.read_text(), "immutable")


if __name__ == "__main__":
    unittest.main()
