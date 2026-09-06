import importlib.util
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, HTTPServer

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("cache_control", ROOT / "bench/cache_control.py")
cache = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cache)


class CacheTests(unittest.TestCase):
    def test_empty_http_200_is_not_verified(self):
        for body in ("", "null", "true", '{"success": false}', '{"success": "true"}'):
            self.assertFalse(cache.acknowledged("vllm", body))
        self.assertTrue(cache.acknowledged("vllm", '{"success": true}'))

    def test_reset_does_not_abort_requests(self):
        seen = []

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                seen.append(self.path)
                self.send_response(200)
                self.end_headers()
                self.wfile.write(b'{"success": true}')

            def log_message(self, *args):
                pass

        server = HTTPServer(("127.0.0.1", 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            self.assertTrue(cache.reset("vllm", server.server_port)["verified"])
            self.assertEqual(seen, ["/reset_prefix_cache"])
        finally:
            server.shutdown()
            server.server_close()
            thread.join()

    def test_sglang_failure_is_not_success(self):
        self.assertFalse(cache.acknowledged("sglang", "Flush cache failed."))
        self.assertTrue(cache.acknowledged("sglang", "Cache flushed.\n Please check backend logs."))


class HarnessTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        git_bash = Path("C:/Program Files/Git/bin/bash.exe")
        cls.bash = str(git_bash) if git_bash.exists() else shutil.which("bash")
        if not cls.bash:
            raise unittest.SkipTest("bash is needed for harness dry-run validation")

    def run_script(self, script, *args, **overrides):
        with tempfile.TemporaryDirectory(prefix="bench-controls-") as temp:
            wrapper = Path(temp) / "python3"
            # Actual executable path, shell-quoted, without a dependency on Store aliases.
            python = sys.executable.replace("\\", "/").replace("'", "'\"'\"'")
            wrapper.write_text(f"#!/usr/bin/env bash\nexec '{python}' \"$@\"\n", encoding="utf-8")
            wrapper.chmod(0o755)
            env = os.environ.copy()
            env.update({"PATH": temp + os.pathsep + env.get("PATH", ""),
                        "BENCH_SKIP_ROOT_ENV": "1", "RESULTS_ROOT": temp,
                        "MODELS_DIR": temp, "BENCH_SEED": "1234"})
            env.update(overrides)
            result = subprocess.run([self.bash, str(ROOT / script), *args], cwd=ROOT,
                                    env=env, text=True, capture_output=True, timeout=30)
            return result

    def test_dp4ep4_native_checkpoint_and_kernel_flags(self):
        result = self.run_script("bench/launch.sh", "ds4flash_dp4ep4", "--dry-run")
        self.assertEqual(result.returncode, 0, result.stderr)
        for value in ("DeepSeek-V4-Flash-0731", "--tensor-parallel-size 1", "--data-parallel-size 4",
                      "--enable-expert-parallel", "--kv-cache-dtype fp8", "--kernel-config.linear_backend b12x"):
            self.assertIn(value, result.stdout)
        self.assertNotIn("--speculative-config", result.stdout)

    def test_glm_vendor_control_retains_model_layout_and_parser(self):
        result = self.run_script("bench/launch.sh", "glm53flash_nvfp4_dp4_control", "--dry-run")
        self.assertEqual(result.returncode, 0, result.stderr)
        for value in ("python3 -m vllm.entrypoints.openai.api_server", "--model RedHatAI/GLM-5.3-Flash-NVFP4",
                      "--tensor-parallel-size 1", "--data-parallel-size 4", "--max-num-seqs 192",
                      "--reasoning-parser glm45", "--block-size 1024", "--kv-cache-dtype auto"):
            self.assertIn(value, result.stdout)

    def test_same_inputs_across_layouts_distinct_across_points(self):
        seeds = []
        for cell in ("ds4flash_tp4", "ds4flash_dp4ep4"):
            result = self.run_script("bench/sweep.sh", cell, "router judge", "1 4", "--dry-run")
            self.assertEqual(result.returncode, 0, result.stderr)
            seeds.append(re.findall(r"--seed (\d+)", result.stdout))
        self.assertEqual(len(seeds[0]), 4)
        self.assertEqual(len(set(seeds[0])), 4)
        self.assertEqual(seeds[0], seeds[1])

    def test_cap_below_concurrency_rejected(self):
        result = self.run_script("bench/sweep.sh", "ds4flash_dp4ep4", "router", "512",
                                 "--dry-run", NUM_PROMPTS_CAP="64")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("cannot reach requested concurrency", result.stderr)

    def test_shared_prefix_uses_total_input_length_without_double_counting(self):
        result = self.run_script("bench/sweep.sh", "ds4flash_dp4ep4", "promptopt", "64", "--dry-run")
        self.assertEqual(result.returncode, 0, result.stderr)
        for value in ("--random-input-len 512", "--random-prefix-len 3072",
                      "--random-output-len 256", "in_len=3584", "--num-prompts 512"):
            self.assertIn(value, result.stdout)

    def test_precision_override_is_kept(self):
        result = self.run_script("bench/launch.sh", "ds4flash_tp4", "--dry-run",
                                 KV_CACHE_DTYPE="bfloat16", LINEAR_BACKEND="cutlass")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--kv-cache-dtype bfloat16", result.stdout)
        self.assertIn("--kernel-config.linear_backend cutlass", result.stdout)


if __name__ == "__main__":
    unittest.main()
