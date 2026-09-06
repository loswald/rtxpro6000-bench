#!/usr/bin/env python3
"""Opt-in, source-pinned Qwen4Exp PLE compile-memory workaround.

Move the ORIGINAL PLE embedding lookup and output dequantization into a vLLM
splitting custom op. Only token/metadata tensors and a graph-owned output cross
the compile boundary; the giant table and its scale stay in the layer registry.
Enable at process startup with VLLM_QWEN4EXP_PLE_OPAQUE_LOOKUP=1 after --apply.
No checkpoint values, loader/scales, precision, hashing or TP math are changed.
"""
from __future__ import annotations

import argparse
import ast
import datetime as dt
import difflib
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import tempfile


ENV_FLAG = "VLLM_QWEN4EXP_PLE_OPAQUE_LOOKUP"
MARKER = "# QWEN_PLE_OPAQUE_LOOKUP_V1"
OP_NAME = "qwen4_exp_ple_lookup_with_output"
KNOWN_SOURCES = {
    "859ae689a7a74b8e4d8ea8c62b3479dbb214314f4fb168ebc2cb963ab3e4a664":
        "vLLM 0.28.1rc1.dev446+g798544433 original ple_layer.py",
    "78969d27e1feead35e2c9207d44c383c440b80e1bf563ab3dce50320409c195a":
        "same source with captured VLLM_QWEN4EXP_PLE_FP8 loader-selection fix",
}

INIT_ANCHOR = """        compilation_config = get_current_vllm_config().compilation_config
        if prefix in compilation_config.static_forward_context:
"""
INIT_PATCH = """        compilation_config = get_current_vllm_config().compilation_config
        # QWEN_PLE_OPAQUE_LOOKUP_V1: opt in per server process, before compilation.
        import os as _ple_os
        self._ple_opaque_lookup = (
            _ple_os.environ.get("VLLM_QWEN4EXP_PLE_OPAQUE_LOOKUP", "0") == "1"
        )
        if self._ple_opaque_lookup:
            # This op includes request-dependent n-gram ID generation. Like the
            # original ID op, it must run outside token-count-only CUDA graphs.
            _ple_op = "vllm::qwen4_exp_ple_lookup_with_output"
            if not isinstance(compilation_config.splitting_ops, list):
                raise RuntimeError("PLE lookup patch requires resolved splitting_ops")
            if _ple_op not in compilation_config.splitting_ops:
                compilation_config.splitting_ops.append(_ple_op)
        if prefix in compilation_config.static_forward_context:
"""
FORWARD_ANCHOR = """        embeddings = self.ple_embedding(input_ids, query_start_loc, ngram_context)
        embeddings = self._dequantize_embeddings(embeddings, hidden_states.dtype)
"""
FORWARD_PATCH = """        if self._ple_opaque_lookup:
            # The graph sees only the requested rows, never the full PLE weight
            # or its global scale as an autotuning input. Keep graph-owned output.
            embeddings = hidden_states.new_empty(
                (input_ids.shape[0], self.ple_embedding.embedding_dim)
            )
            torch.ops.vllm.qwen4_exp_ple_lookup_with_output(
                input_ids, query_start_loc, ngram_context, embeddings, self.prefix
            )
        else:
            embeddings = self.ple_embedding(input_ids, query_start_loc, ngram_context)
            embeddings = self._dequantize_embeddings(embeddings, hidden_states.dtype)
"""
REGISTER_ANCHOR = """direct_register_custom_op(
    op_name="qwen4_exp_compute_ple_ngram_ids",
"""
OP_SOURCE = '''def qwen4_exp_ple_lookup_with_output(
    input_ids: torch.Tensor,
    query_start_loc: torch.Tensor,
    ngram_context: torch.Tensor,
    output: torch.Tensor,
    layer_name: str,
) -> None:
    """Original lookup + scale, outside Inductor and piecewise CUDA graphs."""
    layer = get_forward_context().no_compile_layers[layer_name]
    embeddings = layer.ple_embedding(input_ids, query_start_loc, ngram_context)
    # Preserve the upstream cast-then-multiply order and global scale. Copy only
    # requested rows; no table conversion, clone, precision change or CPU offload.
    values = layer._dequantize_embeddings(embeddings, output.dtype)
    if values.shape != output.shape or values.dtype != output.dtype:
        raise RuntimeError("PLE lookup output shape/dtype differs from graph-owned output")
    output.copy_(values)


def qwen4_exp_ple_lookup_with_output_fake(
    input_ids: torch.Tensor,
    query_start_loc: torch.Tensor,
    ngram_context: torch.Tensor,
    output: torch.Tensor,
    layer_name: str,
) -> None:
    # Caller owns the small output, as for existing vLLM splitting ops.
    return


direct_register_custom_op(
    op_name="qwen4_exp_ple_lookup_with_output",
    op_func=qwen4_exp_ple_lookup_with_output,
    mutates_args=["output"],
    fake_impl=qwen4_exp_ple_lookup_with_output_fake,
)


'''


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def transform(data):
    """Return patched bytes only for an audited, byte-identical source."""
    source = data.decode("utf-8")
    if MARKER in source:
        raise ValueError("Source already contains this patch; use --check for its audit record")
    fingerprint = sha256(data)
    if fingerprint not in KNOWN_SOURCES:
        raise ValueError(f"Unaudited source SHA256 {fingerprint}; refusing to rewrite another vLLM version")
    for anchor, replacement in ((INIT_ANCHOR, INIT_PATCH),
                                (FORWARD_ANCHOR, FORWARD_PATCH),
                                (REGISTER_ANCHOR, OP_SOURCE + REGISTER_ANCHOR)):
        if source.count(anchor) != 1:
            raise ValueError("Expected exactly one audited source anchor")
        source = source.replace(anchor, replacement)
    ast.parse(source)  # Syntax validation does not import or initialize vLLM/CUDA.
    return source.encode("utf-8")


def atomic_write(path, data, mode=None):
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=path.name + ".", delete=False) as stream:
        temporary = Path(stream.name)
        stream.write(data)
    try:
        if mode is not None:
            os.chmod(temporary, mode)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def audit_paths(path):
    return (path.with_name(path.name + ".qwen_ple_opaque_lookup.orig"),
            path.with_name(path.name + ".qwen_ple_opaque_lookup.json"))


def inspect_target(path):
    data = path.read_bytes()
    backup, audit_path = audit_paths(path)
    if MARKER.encode() in data:
        if not backup.is_file() or not audit_path.is_file():
            raise ValueError("Patched source lacks its backup/audit record")
        audit = json.loads(audit_path.read_text(encoding="utf-8"))
        original = backup.read_bytes()
        if sha256(original) != audit.get("before_sha256") or sha256(data) != audit.get("after_sha256") or transform(original) != data:
            raise ValueError("Patched source or backup changed after application; refusing to overwrite it")
        return data, data, audit, True
    patched = transform(data)
    audit = {"patch": "qwen_ple_opaque_lookup_v1", "target": str(path.resolve()),
             "before_sha256": sha256(data), "after_sha256": sha256(patched),
             "source": KNOWN_SOURCES[sha256(data)], "enable_environment": f"{ENV_FLAG}=1",
             "backup": str(backup.resolve()), "status": "review_only"}
    return data, patched, audit, False


def patch_file(path, action="check"):
    path = Path(path).resolve()
    original, patched, audit, already = inspect_target(path)
    backup, audit_path = audit_paths(path)
    if action == "apply":
        if already:
            return {**audit, "status": "already_applied"}
        if backup.exists() or audit_path.exists():
            raise ValueError("Backup/audit path already exists; refusing to mix patch histories")
        # Save the precise current file, including the earlier loader fix if present.
        with backup.open("xb") as stream:
            stream.write(original)
        audit.update(status="applied", applied_utc=dt.datetime.now(dt.timezone.utc).isoformat())
        atomic_write(audit_path, (json.dumps(audit, indent=2) + "\n").encode("utf-8"))
        atomic_write(path, patched, path.stat().st_mode)
        return audit
    if action == "revert":
        if not already:
            raise ValueError("Source is not patched")
        atomic_write(path, backup.read_bytes(), path.stat().st_mode)
        # Preserve audit artifacts; reapplication requires a fresh deployment copy.
        return {**audit, "status": "reverted"}
    if action == "diff":
        return "".join(difflib.unified_diff(original.decode().splitlines(keepends=True),
                                           patched.decode().splitlines(keepends=True),
                                           fromfile=str(path), tofile=str(path) + ".patched"))
    return {**audit, "status": "already_applied" if already else "compatible"}


def locate_target():
    spec = importlib.util.find_spec("vllm")
    if spec is None or spec.origin is None:
        raise ValueError("vLLM not found; pass --target /path/to/ple_layer.py")
    return Path(spec.origin).parent / "models/qwen4_exp/nvidia/ple_layer.py"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--target", type=Path, help="Explicit source file; default locates installed vLLM without importing it")
    actions = ap.add_mutually_exclusive_group()
    for action in ("check", "diff", "apply", "revert"):
        actions.add_argument("--" + action, dest="action", action="store_const", const=action)
    ap.set_defaults(action="check")
    args = ap.parse_args(argv)
    try:
        result = patch_file(args.target or locate_target(), args.action)
        print(result if isinstance(result, str) else json.dumps(result, indent=2))
        return 0
    except (OSError, ValueError, SyntaxError) as exc:
        print(f"PLE patch refused: {exc}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
