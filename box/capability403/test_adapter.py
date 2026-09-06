import asyncio
import inspect
import json
from pathlib import Path
import sys
import tempfile
import unittest
from types import SimpleNamespace

from run import CAPS, CONFIG, RequestLedger, SUITE, body_from_call, canonical_hash, validate_urls, verify_suite


class AdapterTests(unittest.TestCase):
    def test_exact_historical_cases(self):
        verified = verify_suite()
        self.assertEqual(403, verified["case_count"])
        self.assertEqual("efb50b88d5b0aaae7f92d527bf05c761c0da6c9c153af5bde45e1add2ed3735b", verified["historical_manifest_digest"])
        self.assertEqual({"math": 32768, "code": 20480, "knowledge": 20480, "ifeval": 16384, "tools": 8192, "longctx": 6144}, CAPS)

    def test_request_body_preserves_sampling_and_extra_overrides(self):
        sys.path.insert(0, str(SUITE))
        import common
        messages = [{"role": "user", "content": "test"}]
        body = body_from_call(SimpleNamespace(model="m"), common.ChatClient.chat, messages,
                              {"max_tokens": 32768, "temperature": 1.0, "top_p": 0.95, "extra_body": {"seed": 42}})
        self.assertEqual({"model": "m", "messages": messages, "max_tokens": 32768, "temperature": 1.0,
                          "top_p": 0.95, "stream": False, "seed": 42}, body)

    def test_remote_and_credential_urls_rejected(self):
        for url in ["http://example.com:8000", "http://127.0.0.1:8000/path", "http://secret@127.0.0.1:8000"]:
            with self.assertRaises(ValueError):
                validate_urls(url)
        validate_urls("http://127.0.0.1:8000,http://[::1]:8001")

    def test_replay_requires_all_cases_and_valid_body_hash(self):
        with tempfile.TemporaryDirectory() as temp:
            replay = Path(temp) / "replay.jsonl"
            replay.write_text(json.dumps({"id": "a", "turn": 0, "body": {}, "sha256": "wrong"}) + "\n")
            with self.assertRaises(ValueError):
                RequestLedger(Path(temp) / "out.jsonl", {"a": "math"}, replay)
            replay.write_text(json.dumps({"id": "a", "turn": 0, "body": {}, "sha256": canonical_hash({})}) + "\n")
            with self.assertRaises(ValueError):
                RequestLedger(Path(temp) / "out.jsonl", {"a": "math", "b": "code"}, replay)

    def test_replay_freezes_long_context_and_rejects_sampling_drift(self):
        async def scenario(temp):
            class Client:
                model = "m"
                async def chat(self, messages, *, route_key=0, max_tokens=2048, temperature=0.0, top_p=1.0,
                               seed=20260903, tools=None, tool_choice=None, extra_body=None, model=None):
                    return extra_body
            body = body_from_call(Client(), Client.chat, [{"role": "user", "content": "baseline prompt"}], {})
            replay = Path(temp) / "replay.jsonl"
            replay.write_text(json.dumps({"id": "a", "turn": 0, "family": "longctx", "body": body, "sha256": canonical_hash(body)}) + "\n")
            ledger = RequestLedger(Path(temp) / "out.jsonl", {"a": "longctx"}, replay)
            async def run_one(w):
                return await Client().chat([{"role": "user", "content": "changed calibration"}])
            common, runner = SimpleNamespace(ChatClient=Client), SimpleNamespace(run_one=run_one)
            ledger.install(common, runner)
            self.assertEqual(body, await runner.run_one(SimpleNamespace(id="a", family="longctx")))
            self.assertTrue(ledger.complete())
            ledger.context.set(["a", "longctx", 0])
            with self.assertRaisesRegex(ValueError, "settings differ"):
                await Client().chat(body["messages"], temperature=1.0)
        with tempfile.TemporaryDirectory() as temp:
            asyncio.run(scenario(temp))


if __name__ == "__main__":
    unittest.main()
