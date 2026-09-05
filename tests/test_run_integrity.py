"""Regression cases for partial runs, cache contamination and missing shards."""
import json
import math
import pathlib
import subprocess
import sys
import tempfile
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from bench.run_integrity import validate_run
from bench.summarise import aggregate, discover


def metadata(**updates):
    result = {"ports": "8000", "bench_exit_code": 0, "num_prompts": 256,
              "cache_policy": "reset", "cache_reset_verified": 1,
              "ignore_eos": 1, "output_length_mode": "fixed", "out_len": 128,
              "shape": "router", "concurrency": 64}
    result.update(updates)
    return result


def shard(**updates):
    result = {"completed": 256, "num_prompts": 256, "failed": 0,
              "total_output_tokens": 32768, "total_input_tokens": 262144,
              "duration": 32, "request_throughput": 8, "output_throughput": 1024,
              "total_token_throughput": 9216}
    result.update(updates)
    return result


class IntegrityTests(unittest.TestCase):
    def test_complete_run_is_eligible(self):
        result = validate_run([shard()], metadata())
        self.assertTrue(result["valid"])
        self.assertEqual(result["cache_status"], "verified")

    def test_partial_requests_with_zero_client_rc_are_invalid(self):
        result = validate_run([shard(completed=976, num_prompts=2048, failed=1072,
                                     total_output_tokens=976 * 128)], metadata(num_prompts=2048))
        self.assertEqual(result["status"], "invalid")
        self.assertFalse(result["cost_eligible"])

    def test_short_outputs_are_invalid_even_when_every_request_completed(self):
        result = validate_run([shard(total_output_tokens=32767)], metadata())
        self.assertFalse(result["valid"])
        self.assertTrue(any("expected exactly 32768" in x for x in result["errors"]))

    def test_missing_shard_is_invalid(self):
        result = validate_run([shard(_file="run__p8000.json")], metadata(ports="8000,8001", num_prompts=512))
        self.assertTrue(any("shard count mismatch" in x for x in result["errors"]))

    def test_duplicate_shard_cannot_replace_missing_port(self):
        result = validate_run([shard(_file="run__p8000.json"), shard(_file="run__p8000.json")],
                              metadata(ports="8000,8001", num_prompts=512))
        self.assertIn("duplicate benchmark ports", result["errors"])

    def test_wrong_port_is_invalid(self):
        result = validate_run([shard(replica_port=8001)], metadata())
        self.assertIn("unexpected benchmark ports: 8001", result["errors"])

    def test_complete_replicas_with_recorded_identity_are_valid(self):
        result = validate_run([shard(replica_port="8000"), shard(replica_port="8001")],
                              metadata(ports=[8000, 8001], num_prompts=512))
        self.assertTrue(result["valid"])

    def test_unidentified_replica_ports_are_unknown(self):
        result = validate_run([shard(), shard()], metadata(ports="8000,8001", num_prompts=512))
        self.assertEqual(result["status"], "unknown")

    def test_missing_metadata_does_not_invent_a_pass(self):
        result = validate_run([shard()], {})
        self.assertEqual(result["status"], "unknown")
        self.assertFalse(result["headline_eligible"])

    def test_uncontrolled_cache_is_unknown_not_proven_contamination(self):
        result = validate_run([shard()], metadata(cache_policy="uncontrolled", cache_reset_verified=0))
        self.assertEqual(result["status"], "unknown")
        self.assertEqual(result["cache_status"], "unknown")

    def test_failed_cache_reset_is_invalid(self):
        result = validate_run([shard()], metadata(cache_reset_verified=0))
        self.assertEqual(result["cache_status"], "invalid")

    def test_disabled_cache_needs_explicit_evidence(self):
        result = validate_run([shard()], metadata(cache_policy="disabled", prefix_cache_enabled=False))
        self.assertTrue(result["valid"])
        result = validate_run([shard()], metadata(cache_policy="disabled"))
        self.assertFalse(result["valid"])

    def test_client_and_embedded_errors_block_pass(self):
        self.assertFalse(validate_run([shard()], metadata(bench_exit_code=1))["valid"])
        self.assertFalse(validate_run([shard(errors=["Internal Server Error"])], metadata())["valid"])

    def test_missing_failure_count_is_unknown(self):
        result_shard = shard()
        result_shard.pop("failed")
        self.assertEqual(validate_run([result_shard], metadata())["status"], "unknown")

    def test_nonfinite_throughput_is_invalid(self):
        for value in (math.inf, math.nan, -1, None):
            with self.subTest(value=value):
                self.assertFalse(validate_run([shard(output_throughput=value)], metadata())["valid"])

    def test_variable_lengths_require_total_expected_output(self):
        variable = metadata(out_len=-1, output_length_mode="variable")
        self.assertEqual(validate_run([shard()], variable)["status"], "unknown")
        variable["expected_total_output_tokens"] = 32768
        self.assertTrue(validate_run([shard()], variable)["valid"])

    def test_no_shards_is_invalid(self):
        self.assertEqual(validate_run([], metadata())["status"], "invalid")

    def test_summary_preserves_diagnostics_but_suppresses_cost(self):
        result = aggregate("test", "run", {"meta": metadata(), "bench": [shard(failed=1)]}, 4)
        self.assertEqual(result["out_tok_s"], 1024)
        self.assertFalse(result["headline_eligible"])
        self.assertIsNone(result["cost_per_1m_out_usd"])
        self.assertIsNone(result["cost_per_1m_total_usd"])

    def test_summary_missing_hardware_flags_stays_unknown(self):
        result = aggregate("test", "run", {"meta": metadata(), "bench": [shard()]}, 4)
        self.assertIsNone(result["p2p_disabled"])
        self.assertIsNone(result["custom_allreduce"])
        self.assertIsNotNone(result["cost_per_1m_out_usd"])

    def test_discovery_keeps_corrupt_shards_and_ignores_integrity_output(self):
        with tempfile.TemporaryDirectory() as folder:
            cell = pathlib.Path(folder) / "test"
            cell.mkdir()
            run_id = "test__router__c64__20260905T120000"
            (cell / (run_id + ".meta.json")).write_text(json.dumps(metadata()), encoding="utf-8")
            (cell / (run_id + ".json")).write_text("{broken", encoding="utf-8")
            (cell / (run_id + ".integrity.json")).write_text('{"valid":false}', encoding="utf-8")
            result = discover(folder)["test"]["runs"][run_id]
            self.assertEqual(len(result["bench"]), 1)
            self.assertFalse(validate_run(result["bench"], result["meta"])["valid"])

    def test_cli_emits_readable_failure_for_missing_file(self):
        with tempfile.TemporaryDirectory() as folder:
            meta_path = pathlib.Path(folder) / "run.meta.json"
            meta_path.write_text(json.dumps(metadata()), encoding="utf-8")
            output_path = pathlib.Path(folder) / "run.integrity.json"
            result = subprocess.run([sys.executable, str(ROOT / "bench/run_integrity.py"),
                                     "--meta", str(meta_path), "--out", str(output_path),
                                     str(pathlib.Path(folder) / "missing.json")], capture_output=True, text=True)
            self.assertEqual(result.returncode, 1)
            output = json.loads(output_path.read_text(encoding="utf-8"))
            self.assertEqual(output["status"], "invalid")
            self.assertTrue(any("could not be read" in x for x in output["errors"]))

    def test_real_archived_failure_is_excluded(self):
        path = ROOT / "results/probe/ds_marlin_ep/ds_marlin_ep__router__c1024__p8000.json"
        if not path.exists():
            self.skipTest("optional archived campaign result absent")
        archived = json.loads(path.read_text(encoding="utf-8"))
        result = validate_run([archived], metadata(num_prompts=2048))
        self.assertEqual(result["completed"], 976)
        self.assertFalse(result["headline_eligible"])

    def test_summary_headline_excludes_faster_failed_run(self):
        with tempfile.TemporaryDirectory() as folder:
            cell = pathlib.Path(folder) / "test"
            cell.mkdir()
            for concurrency, result in [(64, shard()), (128, shard(failed=1, output_throughput=999999))]:
                run_id = "test__router__c%d__20260905T120000" % concurrency
                (cell / (run_id + ".meta.json")).write_text(
                    json.dumps(metadata(concurrency=concurrency)), encoding="utf-8")
                (cell / (run_id + ".json")).write_text(json.dumps(result), encoding="utf-8")
            process = subprocess.run([sys.executable, str(ROOT / "bench/summarise.py"),
                                      "--results-dir", folder, "--cost-per-hour", "4", "--quiet"],
                                     capture_output=True, text=True)
            self.assertEqual(process.returncode, 0, process.stderr)
            markdown = (pathlib.Path(folder) / "summary.md").read_text(encoding="utf-8")
            self.assertIn("1,024 @c64", markdown)
            self.assertNotIn("999,999 @c128", markdown)
            self.assertIn("999,999", markdown)  # diagnostic value remains reviewable
            output = json.loads((pathlib.Path(folder) / "summary.json").read_text(encoding="utf-8"))
            bad = next(row for row in output["rows"] if row["concurrency"] == 128)
            self.assertIsNone(bad["cost_per_1m_out_usd"])


if __name__ == "__main__":
    unittest.main()
