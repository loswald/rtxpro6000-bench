import ast
import hashlib
import json
import os
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

try:
    import torch
except ImportError:
    torch = None

from patches.glm_fa2_plan_audit import (
    EXPECTED_SOURCE_SHA256,
    RUNTIME_HELPER,
    compare_plan_counts,
    patched_source,
    validate_query_rows,
)


class QueryMetadataTest(unittest.TestCase):
    def validate(self, **changes):
        args = dict(planned_rows=2, actual_rows=2, num_reqs=2,
                    query_start_loc=[0, 1, 2], query_rows=4, topk_rows=4,
                    request_rows=4, max_rows=8)
        args.update(changes)
        return validate_query_rows(**args)

    def test_current_metadata_confirms_live_rows_and_permits_storage_padding(self):
        evidence = self.validate()
        self.assertFalse(evidence["metadata_errors"])
        self.assertEqual(evidence["active_rows"], 2)

    def test_stale_one_row_plan_cannot_hide_second_live_query(self):
        # The old count-only check calls the second, invalid row padding.
        self.assertTrue(compare_plan_counts([1, 128], [1, 129], 1, 128)["matches"])
        evidence = self.validate(planned_rows=1)
        self.assertTrue(evidence["metadata_errors"])
        self.assertEqual(evidence["active_rows"], 2)

    def test_global_metadata_exceeding_decode_callback_fails(self):
        self.assertTrue(self.validate(query_rows=1, topk_rows=1)["metadata_errors"])

    def test_short_unsorted_offset_or_wrong_endpoint_query_starts_fail(self):
        for qsl in ([0, 1], [1, 1, 2], [0, 2, 1], [0, 1, 3], [0, 1.0, 2]):
            with self.subTest(query_start_loc=qsl):
                self.assertTrue(self.validate(query_start_loc=qsl)["metadata_errors"])

    def test_no_plan_or_missing_current_metadata_fails(self):
        for field in ("planned_rows", "actual_rows", "num_reqs"):
            self.assertTrue(self.validate(**{field: None})["metadata_errors"])

    def test_storage_shortage_or_capacity_overflow_fails(self):
        for changes in ({"topk_rows": 3}, {"request_rows": 3}, {"max_rows": 3}):
            self.assertTrue(self.validate(**changes)["metadata_errors"])


class AuditFailureLoggingTest(unittest.TestCase):
    def test_stale_metadata_is_logged_before_converter_even_after_record_limit(self):
        class TensorShape:
            def __init__(self, shape, values=None):
                self.shape, self.ndim, self.dtype = shape, len(shape), "int32"
                self.values = values
            def cpu(self):
                return self
            def tolist(self):
                return self.values

        fake_torch = types.SimpleNamespace(
            cuda=types.SimpleNamespace(is_current_stream_capturing=lambda: False),
            distributed=types.SimpleNamespace(is_available=lambda: False),
        )
        namespace = {"torch": fake_torch, "compare_plan_counts": compare_plan_counts,
                     "validate_query_rows": validate_query_rows}
        exec(RUNTIME_HELPER, namespace)
        state = types.SimpleNamespace(_glm_audit_active_rows=1, _lens_cpu=[1, 128], max_tokens=8)
        metadata = types.SimpleNamespace(
            num_actual_tokens=2, num_reqs=2, block_size=1024,
            query_start_loc=TensorShape((3,), [0, 1, 2]),
            req_id_per_token=TensorShape((2,), [0, 1]),
            block_table=TensorShape((2, 1)),
        )
        config_module = types.ModuleType("vllm.config")
        config_module.get_current_vllm_config = lambda: types.SimpleNamespace(
            model_config=types.SimpleNamespace(enforce_eager=True))
        with tempfile.TemporaryDirectory() as tmp:
            marker = Path(tmp) / "active"
            marker.touch()
            environment = {"GLM_FA2_AUDIT_MARKER": str(marker), "GLM_FA2_AUDIT_LOG_DIR": tmp,
                           "GLM_FA2_AUDIT_MAX_RECORDS": "0", "GLM_FA2_AUDIT_MODE": "exact"}
            with patch.dict(os.environ, environment), patch.dict("sys.modules", {"vllm.config": config_module}):
                with self.assertRaisesRegex(RuntimeError, "Host plan rows differ"):
                    namespace["_glm_audit_before_convert"](
                        state, types.SimpleNamespace(), types.SimpleNamespace(), metadata,
                        TensorShape((2, 32, 512)), TensorShape((2, 32, 0)),
                        TensorShape((1, 1024, 512)), TensorShape((2, 128)),
                    )
            record = json.loads(next(Path(tmp).glob("*.jsonl")).read_text().strip())
            self.assertEqual(record["active_rows"], 2)
            self.assertEqual(record["planned_active_rows"], 1)
            self.assertEqual(record["phase"], "before_converter")
            self.assertFalse(record["exact_replan_applied"])
            self.assertTrue(record["failure_reasons"])


@unittest.skipIf(torch is None, "Torch is optional locally; run on the independent test node")
class TorchPreconvertTest(unittest.TestCase):
    def check(self, alter=None):
        namespace = {"torch": torch, "compare_plan_counts": compare_plan_counts,
                     "validate_query_rows": validate_query_rows}
        exec(RUNTIME_HELPER, namespace)
        device = torch.device("cpu")
        impl = types.SimpleNamespace(num_heads=32, kv_lora_rank=512, qk_rope_head_dim=0,
                                     scale=0.5, head_size=512, use_fp8_kv_cache=False)
        state = types.SimpleNamespace(
            _glm_audit_active_rows=2, _lens_cpu=torch.tensor([1, 1, 128], dtype=torch.int32),
            max_tokens=3, num_heads=32, topk_width=128, kv_lora_rank=512,
            qk_rope_head_dim=0, sm_scale=0.5, device=device, kv_dtype=torch.bfloat16,
        )
        metadata = types.SimpleNamespace(
            num_actual_tokens=2, num_reqs=2, block_size=64,
            query_start_loc=torch.tensor([0, 1, 2], dtype=torch.int32),
            req_id_per_token=torch.tensor([0, 1, 0], dtype=torch.int32),
            block_table=torch.tensor([[0, 2147483647], [1, -100]], dtype=torch.int32),
        )
        topk = torch.full((3, 128), -1, dtype=torch.int32)
        topk[:2, 0] = 0
        if alter:
            alter(metadata, topk)
        config_module = types.ModuleType("vllm.config")
        config_module.get_current_vllm_config = lambda: types.SimpleNamespace(
            model_config=types.SimpleNamespace(enforce_eager=True))
        with tempfile.TemporaryDirectory() as tmp:
            marker = Path(tmp) / "active"
            marker.touch()
            environment = {"GLM_FA2_AUDIT_MARKER": str(marker), "GLM_FA2_AUDIT_LOG_DIR": tmp,
                           "GLM_FA2_AUDIT_MAX_RECORDS": "0", "GLM_FA2_AUDIT_MODE": "exact"}
            with (patch.dict(os.environ, environment),
                  patch.dict("sys.modules", {"vllm.config": config_module}),
                  patch.object(torch.cuda, "is_current_stream_capturing", return_value=False)):
                try:
                    evidence = namespace["_glm_audit_before_convert"](
                        state, impl, types.SimpleNamespace(), metadata,
                        torch.zeros((3, 32, 512), dtype=torch.bfloat16),
                        torch.zeros((3, 32, 0), dtype=torch.bfloat16),
                        torch.zeros((2, 64, 512), dtype=torch.bfloat16), topk,
                    )
                    failure = None
                except RuntimeError as exc:
                    evidence, failure = None, str(exc)
            records = [json.loads(line) for path in Path(tmp).glob("*.jsonl")
                       for line in path.read_text().splitlines()]
        return evidence, failure, records

    def test_valid_sentinel_padding_and_unreferenced_bad_blocks_are_allowed(self):
        evidence, failure, _ = self.check()
        self.assertIsNone(failure)
        self.assertEqual(evidence["active_rows"], 2)
        self.assertEqual(evidence["physical_slot_capacity"], 128)

    def test_stale_but_in_range_request_id_is_rejected(self):
        def alter(metadata, topk):
            metadata.req_id_per_token[1] = 0
        _, failure, records = self.check(alter)
        self.assertIn("query_start_loc ownership", failure)
        self.assertEqual(len(records), 1)

    def test_referenced_negative_or_wrapping_physical_block_is_rejected(self):
        for invalid in (-1, 2, 67108864):
            with self.subTest(physical_block=invalid):
                def alter(metadata, topk):
                    metadata.block_table[1, 0] = invalid
                _, failure, records = self.check(alter)
                self.assertIn("outside cache before int32 slot conversion", failure)
                self.assertEqual(len(records), 1)

    def test_padding_request_id_is_rejected_if_it_reads_the_block_table(self):
        def alter(metadata, topk):
            metadata.req_id_per_token[2] = -1
            topk[2, 0] = 0
        _, failure, records = self.check(alter)
        self.assertIn("out-of-range request ID", failure)
        self.assertEqual(len(records), 1)

    def test_negative_sentinel_still_reads_block_zero_under_triton_division(self):
        for sentinel in (-1, -2):
            with self.subTest(sentinel=sentinel):
                def alter(metadata, topk):
                    metadata.req_id_per_token[2] = -1
                    topk[2].fill_(sentinel)
                _, failure, records = self.check(alter)
                self.assertIn("out-of-range request ID", failure)
                self.assertEqual(len(records), 1)

    def test_invalid_padding_request_is_inert_only_with_all_column_loads_masked(self):
        for sentinel in (-64, 128):
            with self.subTest(sentinel=sentinel):
                def alter(metadata, topk):
                    metadata.req_id_per_token[2] = -1
                    topk[2].fill_(sentinel)
                _, failure, _ = self.check(alter)
                self.assertIsNone(failure)

    def test_padded_query_can_have_inert_bad_physical_value_with_safe_table_address(self):
        def alter(metadata, topk):
            topk[2, 0] = 64
        _, failure, _ = self.check(alter)
        self.assertIsNone(failure)


@unittest.skipIf(torch is None, "Torch is optional locally; run on the independent test node")
class TorchConvertedSlotTest(unittest.TestCase):
    def run_audit(self, planned, actual, slots, mode="check"):
        namespace = {"torch": torch, "compare_plan_counts": compare_plan_counts,
                     "validate_query_rows": validate_query_rows}
        exec(RUNTIME_HELPER, namespace)
        replans = []
        state = types.SimpleNamespace(
            _lens_cpu=torch.tensor(planned, dtype=torch.int32), device=torch.device("cpu"),
            plan=lambda count, lens: replans.append((count, lens.tolist())),
        )
        evidence = {"call": 1, "active_rows": 2, "width": 4, "mode": mode,
                    "callback_query_rows": 3, "physical_slot_capacity": 16,
                    "exact_replan_applied": False}
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, {
                "GLM_FA2_AUDIT_LOG_DIR": tmp, "GLM_FA2_AUDIT_MAX_RECORDS": "0"}):
            try:
                namespace["_glm_audit_before_fa2_run"](
                    state, torch.tensor(actual, dtype=torch.int32),
                    torch.tensor(slots, dtype=torch.int32), evidence)
                failure = None
            except RuntimeError as exc:
                failure = str(exc)
            records = [json.loads(line) for path in Path(tmp).glob("*.jsonl")
                       for line in path.read_text().splitlines()]
        return evidence, replans, failure, records

    def test_upper_physical_slot_is_rejected_before_attention(self):
        evidence, replans, failure, records = self.run_audit(
            [1, 1, 4], [1, 1, 0], [[0, -1, -1, -1], [16, -1, -1, -1], [-1] * 4])
        self.assertTrue(failure)
        self.assertTrue(evidence["out_of_bounds_in_live_prefix"])
        self.assertFalse(replans)
        self.assertEqual(len(records), 1)

    def test_invalid_second_live_prefix_is_not_padding(self):
        evidence, _, failure, _ = self.run_audit(
            [1, 1, 4], [1, 1, 0], [[0, -1, -1, -1], [-1] * 4, [-1] * 4])
        self.assertTrue(failure)
        self.assertTrue(evidence["negative_in_live_prefix"])

    def test_exact_mode_only_replans_counts_and_retains_padding(self):
        evidence, replans, failure, records = self.run_audit(
            [2, 2, 4], [1, 2, 0], [[0, -1, -1, -1], [14, 15, -1, -1], [-1] * 4], "exact")
        self.assertIsNone(failure)
        self.assertEqual(replans, [(2, [1, 2])])
        self.assertTrue(evidence["exact_replan_applied"])
        self.assertEqual(len(records), 1)

    def test_equal_zero_count_live_row_fails_both_modes(self):
        for mode in ("check", "exact"):
            with self.subTest(mode=mode):
                evidence, replans, failure, records = self.run_audit(
                    [1, 0, 4], [1, 0, 0], [[0, -1, -1, -1], [-1] * 4, [-1] * 4], mode)
                self.assertTrue(failure)
                self.assertEqual(evidence["zero_count_active_rows"], [1])
                self.assertFalse(replans)
                self.assertEqual(len(records), 1)

    def test_minus_two_is_not_the_converter_tail_sentinel(self):
        evidence, _, failure, _ = self.run_audit(
            [1, 1, 4], [1, 1, 0], [[0, -2, -1, -1], [15, -1, -1, -1], [-1] * 4])
        self.assertTrue(failure)
        self.assertTrue(evidence["invalid_sentinel_in_live_tail"])


class SparsePlanCountsTest(unittest.TestCase):
    def test_sentinel_padding_is_not_a_live_length_mismatch(self):
        result = compare_plan_counts([2048, 2049, 2176, 2176], [2048, 2049, 0, 0], 2, 2176)
        self.assertTrue(result["matches"])
        self.assertEqual(result["padding_rows"], 2)
        self.assertEqual(result["planned"], [2048, 2049])

    def test_dummy_nonzero_padding_is_reported_without_changing_live_plan(self):
        result = compare_plan_counts([3, 2176], [3, 2176], 1, 2176)
        self.assertTrue(result["matches"])
        self.assertEqual(result["padding_nonzero_counts"], 1)

    def test_overscheduled_live_row_is_a_mismatch(self):
        result = compare_plan_counts([2049], [2048], 1, 2176)
        self.assertFalse(result["matches"])
        self.assertEqual(result["mismatched_rows"], [0])

    def test_underscheduled_live_row_is_also_a_mismatch(self):
        self.assertFalse(compare_plan_counts([2048], [2051], 1, 2176)["matches"])

    def test_all_pool_remainders_and_short_context(self):
        counts = [1, 2047, 2048, 2049, 2050, 2051]
        self.assertTrue(compare_plan_counts(counts, counts, len(counts), 2176)["matches"])

    def test_zero_live_count_is_visible_to_exact_mode(self):
        result = compare_plan_counts([1, 2176], [0, 0], 1, 2176)
        self.assertEqual(result["zero_count_active_rows"], [0])

    def test_bad_count_rejected_even_if_equal(self):
        self.assertFalse(compare_plan_counts([2177], [2177], 1, 2176)["matches"])

    def test_active_rows_cannot_exceed_storage(self):
        with self.assertRaises(ValueError):
            compare_plan_counts([1, 2], [1], 2, 2176)

    def test_zero_query_batch_is_inert(self):
        self.assertTrue(compare_plan_counts([2176], [0], 0, 2176)["matches"])


class CapturedSourcePatchTest(unittest.TestCase):
    def test_unknown_source_is_never_patched(self):
        with self.assertRaisesRegex(ValueError, "Source hash mismatch"):
            patched_source(b"# not the pinned backend\n")

    def test_pinned_copy_compiles_and_retains_original_index_operations(self):
        source_path = Path(__file__).parent / "fixtures/glm_fa2_baseline.py"
        source = source_path.read_bytes()
        self.assertEqual(hashlib.sha256(source).hexdigest(), EXPECTED_SOURCE_SHA256)
        output = patched_source(source)
        ast.parse(output.decode("utf-8"))
        self.assertEqual(source_path.read_bytes(), source)
        text = output.decode("utf-8")
        self.assertIn('topk_slots.reshape(-1).clamp_(min=0).to(torch.int32)', text)
        self.assertIn('return_valid_counts=True,', text)
        self.assertIn('if not marker or not _glm_audit_os.path.isfile(marker):', text)
        self.assertIn('not config.model_config.enforce_eager', text)
        self.assertIn('state.plan(int(active), actual_cpu[:active])', text)
        self.assertLess(text.index('audit_evidence = _glm_audit_before_convert('),
                        text.index('        topk_slots, valid_counts = triton_convert_req_index_to_global_index('))
        self.assertIn('physical_slot_capacity', text)


if __name__ == "__main__":
    unittest.main()
