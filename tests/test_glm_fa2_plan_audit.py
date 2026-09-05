import ast
import hashlib
import unittest
from pathlib import Path

from patches.glm_fa2_plan_audit import (
    EXPECTED_SOURCE_SHA256,
    compare_plan_counts,
    patched_source,
)


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
        source_path = Path(__file__).parents[1] / (
            "results/live_20260905_host1/glmimg/usr/local/lib/python3.12/"
            "dist-packages/vllm/v1/attention/backends/mla/flashinfer_mla_sparse_sm90.py"
        )
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


if __name__ == "__main__":
    unittest.main()
