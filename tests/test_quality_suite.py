"""Local synthetic tests only: these are not measured model quality results."""
import contextlib
import copy
import hashlib
import http.server
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import threading
import types
import unittest
from unittest import mock
import urllib.request


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("quality_suite", ROOT / "gates" / "quality_suite.py")
q = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(q)


def fixture_suite(n=200):
    return {"schema_version": 1, "purpose": "accuracy",
            "identity": {"model_id": "synthetic-fixture-not-a-model", "model_revision": "a" * 40,
                         "tokenizer_revision": "b" * 40, "prompt_format_sha256": "c" * 64},
            "seed": 1234, "limit": n, "concurrency": 8, "max_gen_toks": 32768,
            "generation": {"temperature": 0, "top_p": 1, "chat_template_kwargs": {"enable_thinking": True}, "until": []},
            "think_end_token": "</think>", "fewshot_as_multiturn": True, "system_instruction": None,
            "tasks": [{"name": "synthetic_math", "capability": "math", "metric": "exact_match,none", "num_fewshot": 0},
                      {"name": "synthetic_logic", "capability": "logic", "metric": "exact_match,none", "num_fewshot": 0}]}


def request(index, suite):
    return {"model": "fixture-alias", "messages": [{"role": "user", "content": f"synthetic question {index}"}],
            "seed": suite["seed"], "max_tokens": suite["max_gen_toks"], "temperature": 0, "top_p": 1,
            "chat_template_kwargs": suite["generation"]["chat_template_kwargs"], "stop": []}


def response():
    return {"choices": [{"index": 0, "message": {"content": "synthetic answer", "reasoning": "retained synthetic reasoning"},
                         "finish_reason": "stop"}], "usage": {"completion_tokens": 120, "prompt_tokens": 12}}


def raw_result(task, suite):
    samples = [{"doc_id": i, "doc_hash": q.digest(["doc", i]), "prompt_hash": q.digest(["prompt", i]),
                "target_hash": q.digest(["target", i]), "target": "synthetic answer", "arguments": [[f"question {i}", {}]],
                "resps": [["synthetic answer"]], "filtered_resps": ["synthetic answer"],
                "filter": "none", "exact_match": 1} for i in range(suite["limit"])]
    name = task["name"]
    return {"samples": {name: samples}, "results": {name: {task["metric"]: 1.0}},
            "configs": {name: {"output_type": "generate_until", "num_fewshot": 0}},
            "versions": {name: 1}, "n-samples": {name: {"original": len(samples), "effective": len(samples)}}}


def fixture_run(n=200):
    suite = fixture_suite(n)
    run = {"schema_version": 1, "status": "complete", "issues": [], "suite": suite,
           "suite_sha256": q.digest(suite), "harness_version": "synthetic", "harness_sha256": "h" * 64,
           "runner_sha256": "r" * 64, "launch_sha256": "l" * 64, "tasks": {}}
    for task in suite["tasks"]:
        raw = raw_result(task, suite)
        metadata = {k: raw[k][task["name"]] for k in ("configs", "versions", "n-samples")}
        records = [{"request": request(i, suite), "response": response(), "status": 200} for i in range(n)]
        run["tasks"][task["name"]] = {"task": task, "metadata": metadata, "metadata_sha256": q.digest(metadata),
                                       "samples": q.selected_samples(raw, task), "requests": records,
                                       "accuracy": 1.0, "issues": []}
    return run


class ComparisonTests(unittest.TestCase):
    def setUp(self):
        self.baseline = fixture_run()
        self.candidate = copy.deepcopy(self.baseline)

    def compare(self, margin=0):
        return q.compare_runs(self.baseline, self.candidate, margin)

    def test_identical_small_sample_is_inconclusive_at_zero_margin(self):
        result = self.compare()
        self.assertEqual(result["status"], "inconclusive")
        self.assertFalse(result["pass"])
        for task in result["tasks"].values():
            self.assertEqual(task["losses"], 0)
            self.assertLess(task["delta_lower_bound"], 0)

    def test_identical_samples_can_support_predeclared_nonzero_margin(self):
        self.assertTrue(self.compare(margin=0.03)["pass"])

    def test_capability_regression_cannot_be_hidden_by_other_tasks(self):
        entry = self.candidate["tasks"]["synthetic_logic"]
        entry["samples"][0].update(score=0, exact_match=0)
        entry["accuracy"] = .995
        result = self.compare()
        self.assertEqual(result["status"], "observed_regression")
        self.assertEqual(result["tasks"]["synthetic_logic"]["losses"], 1)

    def test_seed_budget_template_reasoning_model_and_sampling_mismatch(self):
        modifications = [lambda s: s.update(seed=7), lambda s: s.update(max_gen_toks=128),
                         lambda s: s["generation"].update(temperature=0.7),
                         lambda s: s["generation"].update(chat_template_kwargs={"enable_thinking": False}),
                         lambda s: s["identity"].update(model_id="different"),
                         lambda s: s["identity"].update(prompt_format_sha256="f" * 64)]
        for change in modifications:
            with self.subTest(change=change):
                self.candidate = copy.deepcopy(self.baseline)
                change(self.candidate["suite"])
                self.candidate["suite_sha256"] = q.digest(self.candidate["suite"])
                self.assertEqual(self.compare()["status"], "invalid")

    def test_bad_finishes_empty_answers_missing_requests_and_changed_prompts_fail(self):
        changes = [lambda t: t["requests"][0]["response"]["choices"][0].update(finish_reason="length"),
                   lambda t: t["requests"][0]["response"]["choices"][0].pop("finish_reason"),
                   lambda t: t["requests"][0]["response"]["choices"][0]["message"].update(content=None),
                   lambda t: t["requests"].pop(),
                   lambda t: t["requests"][0]["request"]["messages"][0].update(content="changed prompt"),
                   lambda t: t["requests"][0]["request"].update(max_tokens=128),
                   lambda t: t["requests"][0].update(status=500)]
        for change in changes:
            with self.subTest(change=change):
                self.candidate = copy.deepcopy(self.baseline)
                change(self.candidate["tasks"]["synthetic_math"])
                self.assertEqual(self.compare(margin=0.03)["status"], "invalid")

    def test_sample_provenance_and_completeness_are_required(self):
        changes = [lambda t: t["samples"][0].pop("prompt_hash"),
                   lambda t: t["samples"][0].update(target_hash="different"),
                   lambda t: t["samples"].append(copy.deepcopy(t["samples"][0])),
                   lambda t: t["samples"][0].pop("resps"),
                   lambda t: t.update(metadata_sha256="stale")]
        for change in changes:
            with self.subTest(change=change):
                self.candidate = copy.deepcopy(self.baseline)
                change(self.candidate["tasks"]["synthetic_math"])
                self.assertEqual(self.compare()["status"], "invalid")

    def test_reordered_samples_requests_and_aliases_pair_correctly(self):
        for task in self.candidate["tasks"].values():
            task["samples"].reverse()
            task["requests"].reverse()
            for record in task["requests"]:
                record["request"]["model"] = "new-alias-same-checkpoint"
        self.assertEqual(self.compare()["status"], "inconclusive")

    def test_smoke_and_missing_task_never_pass(self):
        self.candidate["suite"]["purpose"] = "smoke"
        self.candidate["suite_sha256"] = q.digest(self.candidate["suite"])
        self.assertEqual(self.compare()["status"], "invalid")
        self.candidate = copy.deepcopy(self.baseline)
        self.candidate["tasks"].pop("synthetic_math")
        self.assertEqual(self.compare()["status"], "invalid")

    def test_exact_upper_bound_has_nonzero_uncertainty_with_no_losses(self):
        self.assertAlmostEqual(q.binomial_upper(0, 200, .05), 1 - .05 ** (1 / 200))
        self.assertAlmostEqual(q.binomial_upper(1, 2, .05), .95 ** .5)
        self.assertEqual(q.binomial_upper(200, 200, .05), 1)

    def test_harness_scores_and_provenance_must_exist(self):
        suite = fixture_suite(2)
        task = suite["tasks"][0]
        for field in ("resps", "prompt_hash", "arguments", "target"):
            raw = raw_result(task, suite)
            raw["samples"][task["name"]][0].pop(field)
            with self.assertRaises(ValueError):
                q.selected_samples(raw, task)
        raw = raw_result(task, suite)
        raw["results"][task["name"]][task["metric"]] = .5
        with self.assertRaises(ValueError):
            q.selected_samples(raw, task)

    def test_incomplete_counts_or_unscored_edits_never_pass(self):
        entry = self.candidate["tasks"]["synthetic_math"]
        entry["samples"][0]["score"] = 0
        self.assertEqual(self.compare()["status"], "invalid")
        self.candidate = copy.deepcopy(self.baseline)
        entry = self.candidate["tasks"]["synthetic_math"]
        entry["metadata"]["n-samples"]["effective"] = 1
        entry["metadata_sha256"] = q.digest(entry["metadata"])
        self.assertEqual(self.compare()["status"], "invalid")


class RunnerTests(unittest.TestCase):
    def test_loopback_proxy_and_harness_integration_preserve_raw_evidence(self):
        seen = []

        class Upstream(http.server.BaseHTTPRequestHandler):
            def log_message(self, *_args):
                pass

            def do_POST(self):
                seen.append(json.loads(self.rfile.read(int(self.headers["Content-Length"]))))
                body = json.dumps(response()).encode()
                self.send_response(200)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

        server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Upstream)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            with tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                suite = fixture_suite(2)
                q.write(root / "suite.json", suite)
                q.write(root / "launch.json", {"engine": "synthetic fixture"})
                calls = []

                def fake_evaluate(**kwargs):
                    calls.append(kwargs)
                    task = next(t for t in suite["tasks"] if t["name"] == kwargs["tasks"][0])
                    for i in range(suite["limit"]):
                        body = json.dumps(request(i, suite)).encode()
                        req = urllib.request.Request(kwargs["model_args"]["base_url"], data=body, headers={"Content-Type": "application/json"})
                        with urllib.request.urlopen(req, timeout=5) as resp:
                            self.assertEqual(json.load(resp), response())
                    return raw_result(task, suite)

                fake_package = types.SimpleNamespace(simple_evaluate=fake_evaluate, __file__=q.__file__)
                args = types.SimpleNamespace(suite=str(root / "suite.json"), url=f"http://127.0.0.1:{server.server_port}",
                                             served_model="fixture-alias", launch=str(root / "launch.json"), label="fixture",
                                             out=str(root / "run"), timeout=5)
                with mock.patch.dict("sys.modules", {"lm_eval": fake_package}), mock.patch.object(q.importlib.metadata, "version", return_value="fixture"), contextlib.redirect_stdout(io.StringIO()):
                    self.assertEqual(q.run_suite(args), 0)
                manifest = q.load(root / "run" / "quality.json")
                self.assertEqual(manifest["status"], "complete")
                self.assertEqual(len(seen), 4)
                for call in calls:
                    self.assertEqual(call["gen_kwargs"]["max_gen_toks"], 32768)
                    self.assertTrue(call["apply_chat_template"])
                    self.assertFalse(call["confirm_run_unsafe_code"])
                    self.assertIsNone(call["model_args"]["tokenizer_backend"])
                for task in suite["tasks"]:
                    capture = (root / "run" / f"{task['name']}.http.jsonl").read_text()
                    self.assertIn("retained synthetic reasoning", capture)
                    self.assertIn("raw_response", capture)
                    self.assertTrue((root / "run" / f"{task['name']}.lm_eval.json").is_file())
                self.assertEqual(q.compare_runs(manifest, manifest)["status"], "inconclusive")
                with self.assertRaises(FileExistsError):
                    q.run_suite(args)
        finally:
            server.shutdown()
            thread.join()
            server.server_close()


if __name__ == "__main__":
    unittest.main()
