"""Patch guards plus optional real-PyTorch CPU tests. No model/GPU measurements."""
import ast
import importlib.util
import json
from pathlib import Path
import tempfile
import types
import unittest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("qwen_ple_patch", ROOT / "patches/qwen_ple_opaque_lookup.py")
patch = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(patch)
SOURCE = ROOT / "results/live_20260905_host2/baseline_bundle/usr/local/lib/python3.12/dist-packages/vllm/models/qwen4_exp/nvidia/ple_layer.py"

try:
    import torch
except ImportError:
    torch = None


def get_ast_method(source, class_name, method_name):
    cls = next(n for n in ast.parse(source).body if isinstance(n, ast.ClassDef) and n.name == class_name)
    return next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == method_name)


class SourcePatchTests(unittest.TestCase):
    def test_both_audited_sources_patch_without_modifying_scale_or_loader(self):
        for path in (SOURCE, SOURCE.with_name(SOURCE.name + ".orig")):
            with self.subTest(path=path):
                original = path.read_bytes()
                changed = patch.transform(original)
                for cls, method in (("Qwen4ExpPLELayer", "_dequantize_embeddings"),
                                    ("Qwen4ExpNGramEmbedding", "load_weights"),
                                    ("Qwen4ExpPLEFp8EmbeddingMethod", "embedding"),
                                    ("Qwen4ExpPLEFp8EmbeddingMethod", "process_weights_after_loading")):
                    self.assertEqual(ast.dump(get_ast_method(original.decode(), cls, method)),
                                     ast.dump(get_ast_method(changed.decode(), cls, method)))
                self.assertEqual(original.count(b'VLLM_QWEN4EXP_PLE_FP8'), changed.count(b'VLLM_QWEN4EXP_PLE_FP8'))
                self.assertIn(b'compilation_config.splitting_ops.append(_ple_op)', changed)
                self.assertIn(b'get_forward_context().no_compile_layers[layer_name]', changed)
                self.assertNotIn(b'@torch.compiler.disable', changed)

    def test_unknown_source_fails_closed(self):
        with self.assertRaisesRegex(ValueError, "Unaudited source"):
            patch.transform(SOURCE.read_bytes() + b"\n# different engine revision\n")

    def test_apply_is_audited_idempotent_and_revert_preserves_loader_fix(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "ple_layer.py"
            original = SOURCE.read_bytes()
            target.write_bytes(original)
            self.assertEqual(patch.patch_file(target)["status"], "compatible")
            self.assertEqual(target.read_bytes(), original)
            audit = patch.patch_file(target, "apply")
            self.assertEqual(audit["before_sha256"], patch.sha256(original))
            self.assertEqual(audit["after_sha256"], patch.sha256(target.read_bytes()))
            self.assertEqual(patch.patch_file(target, "apply")["status"], "already_applied")
            self.assertEqual(patch.patch_file(target, "revert")["status"], "reverted")
            self.assertEqual(target.read_bytes(), original)
            self.assertTrue(patch.audit_paths(target)[0].is_file())

    def test_intervening_edits_prevent_revert(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "ple_layer.py"
            target.write_bytes(SOURCE.read_bytes())
            patch.patch_file(target, "apply")
            target.write_bytes(target.read_bytes() + b"\n# another fix\n")
            with self.assertRaisesRegex(ValueError, "changed after application"):
                patch.patch_file(target, "revert")

    def test_custom_op_has_no_weight_scale_or_table_argument(self):
        op = next(n for n in ast.parse(patch.OP_SOURCE).body if isinstance(n, ast.FunctionDef) and n.name == patch.OP_NAME)
        self.assertEqual([a.arg for a in op.args.args],
                         ["input_ids", "query_start_loc", "ngram_context", "output", "layer_name"])
        fake = next(n for n in ast.parse(patch.OP_SOURCE).body if isinstance(n, ast.FunctionDef) and n.name == patch.OP_NAME + "_fake")
        self.assertFalse(any(isinstance(n, ast.Call) for n in ast.walk(fake)))


@unittest.skipIf(torch is None, "Optional CPU numerical/compiler tests require PyTorch; patch guard tests still run")
class TorchCpuTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.layers = {}
        cls.forward_context = types.SimpleNamespace(no_compile_layers=cls.layers)
        namespace = {"torch": torch, "get_forward_context": lambda: cls.forward_context}
        functions = [n for n in ast.parse(patch.OP_SOURCE).body if isinstance(n, ast.FunctionDef)]
        exec(compile(ast.Module(body=functions, type_ignores=[]), "qwen_ple_patch_test", "exec"), namespace)
        cls.op_impl = staticmethod(namespace[patch.OP_NAME])
        cls.op = torch.library.custom_op("qwen_ple_test::lookup_with_output", mutates_args=("output",))(namespace[patch.OP_NAME])
        cls.op.register_fake(namespace[patch.OP_NAME + "_fake"])
        dequantize = get_ast_method(SOURCE.read_text(encoding="utf-8"), "Qwen4ExpPLELayer", "_dequantize_embeddings")
        ns = {"torch": torch, "is_fp8": lambda t: t.dtype in (torch.float8_e4m3fn, torch.float8_e5m2)}
        exec(compile(ast.Module(body=[dequantize], type_ignores=[]), "original_dequantize", "exec"), ns)
        cls.dequantize = staticmethod(ns["_dequantize_embeddings"])

    def make_layer(self, dtype, scale):
        table = torch.tensor([[0., 1., -2., .125], [3., -4., 5., .5],
                              [10., 0., -.25, 7.], [-5., 2., 1., -1.]], dtype=dtype)
        weight_scale = torch.tensor([scale], dtype=torch.float32)

        class Embedding:
            embedding_dim = 4

            def __call__(self, ids, query_start_loc, ngram_context):
                # Small fixture stands in for original n-gram ID generation;
                # tensor lookup and the upstream dequantization are real torch.
                return torch.nn.functional.embedding(ids, table)

        layer = types.SimpleNamespace(ple_embedding=Embedding())
        layer._get_embedding_weight_scale = lambda: weight_scale
        layer._dequantize_embeddings = types.MethodType(type(self).dequantize, layer)
        self.layers["fixture"] = layer
        return layer, table, weight_scale

    def test_exact_fp8_rows_scales_repeats_and_empty_batch(self):
        for dtype in (torch.float8_e4m3fn, torch.float8_e5m2, torch.bfloat16):
            for scale in (.00390625, .7, 12.75):
                with self.subTest(dtype=dtype, scale=scale):
                    layer, table, weight_scale = self.make_layer(dtype, scale)
                    before = table.view(torch.uint8).clone()
                    scale_before = weight_scale.clone()
                    for ids in (torch.tensor([3, 0, 3, 1]), torch.empty(0, dtype=torch.long)):
                        output = torch.empty((ids.shape[0], 4), dtype=torch.bfloat16)
                        expected = layer._dequantize_embeddings(layer.ple_embedding(ids, None, None), output.dtype)
                        self.op(ids, torch.tensor([0, len(ids)]), torch.empty(0, dtype=torch.long), output, "fixture")
                        self.assertTrue(torch.equal(output, expected))
                    self.assertTrue(torch.equal(table.view(torch.uint8), before))
                    self.assertTrue(torch.equal(weight_scale, scale_before))

    def test_missing_fp8_scale_and_dtype_mismatch_fail(self):
        layer, _, _ = self.make_layer(torch.float8_e4m3fn, .7)
        layer._get_embedding_weight_scale = lambda: None
        with self.assertRaisesRegex(RuntimeError, "missing its global scale"):
            self.op_impl(torch.tensor([0]), torch.tensor([0, 1]), torch.tensor([]), torch.empty((1, 4)), "fixture")
        layer, _, _ = self.make_layer(torch.float32, .7)
        with self.assertRaisesRegex(RuntimeError, "shape/dtype"):
            self.op_impl(torch.tensor([0]), torch.tensor([0, 1]), torch.tensor([]), torch.empty((1, 4), dtype=torch.bfloat16), "fixture")

    def test_fullgraph_boundary_excludes_table_and_scale_inputs(self):
        layer, table, scale = self.make_layer(torch.float8_e4m3fn, .7)
        captures = []

        def backend(graph, inputs):
            captures.append((graph, inputs))
            return graph.forward

        op = type(self).op

        def lookup(ids, starts, context):
            output = torch.empty((ids.shape[0], 4), device=ids.device, dtype=torch.bfloat16)
            op(ids, starts, context, output, "fixture")
            return output

        compiled = torch.compile(lookup, backend=backend, fullgraph=True, dynamic=True)
        for ids in (torch.tensor([3, 0, 3, 1]), torch.tensor([2])):
            starts, context = torch.tensor([0, len(ids)]), torch.empty(0, dtype=torch.long)
            self.assertTrue(torch.equal(compiled(ids, starts, context), lookup(ids, starts, context)))
        self.assertTrue(captures)
        for graph, inputs in captures:
            self.assertTrue(any("qwen_ple_test.lookup_with_output" in str(node.target) for node in graph.graph.nodes))
            for value in inputs:
                if isinstance(value, torch.Tensor):
                    self.assertNotEqual(value.data_ptr(), table.data_ptr())
                    self.assertNotEqual(value.data_ptr(), scale.data_ptr())
                    self.assertEqual(value.dtype, torch.int64)
            self.assertFalse(any("embedding" in str(node.target) for node in graph.graph.nodes))


if __name__ == "__main__":
    unittest.main()
