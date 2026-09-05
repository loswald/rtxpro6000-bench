#!/usr/bin/env python3
"""Emit a diagnostic copy of the exact captured GLM FA2 backend.

This never edits the source. Runtime checks activate only after the marker
file exists and require --enforce-eager. See glm_fa2_plan_audit.md.
"""
from __future__ import annotations

import argparse
import hashlib
import inspect
import json
from pathlib import Path

EXPECTED_SOURCE_SHA256 = "7a19dafb16f1a2f9ac58992ce78e4d27b8f52edf08059c387d4f32d70d0edab3"


def compare_plan_counts(planned, actual, active_rows, width):
    """Compare live rows only; full-width lengths on zero-query padding are inert."""
    if width < 1 or active_rows < 0:
        raise ValueError("Invalid width or active row count")
    if active_rows > len(planned) or active_rows > len(actual):
        raise ValueError("Active rows exceed planned or actual row storage")
    live_planned = [int(x) for x in planned[:active_rows]]
    live_actual = [int(x) for x in actual[:active_rows]]
    invalid = [
        i for i, (p, a) in enumerate(zip(live_planned, live_actual))
        if not (0 <= p <= width and 0 <= a <= width)
    ]
    mismatch = [
        i for i, (p, a) in enumerate(zip(live_planned, live_actual)) if p != a
    ]
    return {
        "matches": not mismatch and not invalid,
        "active_rows": active_rows,
        "padding_rows": len(actual) - active_rows,
        "padding_nonzero_counts": sum(int(x) != 0 for x in actual[active_rows:]),
        "mismatched_rows": mismatch,
        "invalid_count_rows": invalid,
        "zero_count_active_rows": [i for i, x in enumerate(live_actual) if x == 0],
        "planned": live_planned,
        "actual": live_actual,
    }


RUNTIME_HELPER = '''
import json as _glm_audit_json
import os as _glm_audit_os
import time as _glm_audit_time

_GLM_FA2_AUDIT_CALLS = 0


def _glm_audit_before_fa2_run(state, valid_counts, slots, impl, layer, metadata):
    global _GLM_FA2_AUDIT_CALLS
    marker = _glm_audit_os.environ.get("GLM_FA2_AUDIT_MARKER", "")
    if not marker or not _glm_audit_os.path.isfile(marker):
        return
    from vllm.config import get_current_vllm_config
    config = get_current_vllm_config()
    if not config.model_config.enforce_eager or torch.cuda.is_current_stream_capturing():
        raise RuntimeError("GLM FA2 audit requires --enforce-eager; graph replay cannot be audited by Python")
    mode = _glm_audit_os.environ.get("GLM_FA2_AUDIT_MODE", "check")
    if mode not in ("check", "exact"):
        raise RuntimeError("GLM_FA2_AUDIT_MODE must be check or exact")
    log_dir = _glm_audit_os.environ.get("GLM_FA2_AUDIT_LOG_DIR", "")
    if not log_dir:
        raise RuntimeError("Set GLM_FA2_AUDIT_LOG_DIR before activating the audit marker")
    active = getattr(state, "_glm_audit_active_rows", None)
    if active is None or not hasattr(state, "_lens_cpu"):
        raise RuntimeError("GLM FA2 audit: no preceding host plan")
    width = int(slots.shape[1])
    if (state.num_heads != impl.num_heads or state.topk_width != width
            or state.kv_lora_rank != impl.kv_lora_rank
            or state.qk_rope_head_dim != impl.qk_rope_head_dim
            or state.sm_scale != impl.scale
            or state.device != slots.device):
        raise RuntimeError("GLM FA2 audit: process-wide wrapper geometry differs from current layer")
    expected_kv_dtype = torch.float8_e4m3fn if impl.use_fp8_kv_cache else torch.bfloat16
    if state.kv_dtype != expected_kv_dtype:
        raise RuntimeError("GLM FA2 audit: process-wide wrapper KV dtype differs from current layer")
    actual_cpu = valid_counts.reshape(-1).to(device="cpu", dtype=torch.int32)
    if len(actual_cpu) != slots.shape[0] or slots.shape[0] > state.max_tokens:
        raise RuntimeError("GLM FA2 audit: sparse row count or capacity is inconsistent")
    evidence = compare_plan_counts(
        state._lens_cpu.tolist(), actual_cpu.tolist(), int(active), width
    )
    # A valid count is useful only if conversion produced its promised compact
    # prefix. Check live rows before the existing negative-index clamp.
    live_slots = slots[:active]
    column = torch.arange(width, device=slots.device).unsqueeze(0)
    prefix = column < valid_counts.reshape(-1)[:active].unsqueeze(1)
    negative_prefix = bool(((live_slots < 0) & prefix).any().item())
    nonnegative_tail = bool(((live_slots >= 0) & ~prefix).any().item())
    _GLM_FA2_AUDIT_CALLS += 1
    dist = torch.distributed
    rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else None
    evidence.update({
        "time_unix": _glm_audit_time.time(),
        "pid": _glm_audit_os.getpid(), "rank": rank,
        "layer": str(getattr(layer, "layer_name", getattr(layer, "prefix", "unknown"))),
        "layer_object_id": id(layer), "call": _GLM_FA2_AUDIT_CALLS,
        "mode": mode, "width": width, "heads": int(impl.num_heads),
        "physical_block_size": int(metadata.block_size),
        "request_ids": metadata.req_id_per_token[:active].cpu().tolist(),
        "kv_dtype": str(state.kv_dtype),
        "negative_in_live_prefix": negative_prefix,
        "nonnegative_in_live_tail": nonnegative_tail,
        "exact_replan_applied": False,
    })
    invalid = bool(evidence["invalid_count_rows"]) or negative_prefix or nonnegative_tail
    failure = invalid or (mode == "check" and not evidence["matches"])
    if mode == "exact" and not evidence["matches"] and not invalid:
        # An all-masked live row needs a separate kernel contract. Do not treat
        # it as a harmless length correction. Padded zero-query rows are ignored.
        if evidence["zero_count_active_rows"]:
            failure = True
        else:
            state.plan(int(active), actual_cpu[:active])
            evidence["exact_replan_applied"] = True
    limit = int(_glm_audit_os.environ.get("GLM_FA2_AUDIT_MAX_RECORDS", "256"))
    if _GLM_FA2_AUDIT_CALLS <= limit or failure:
        _glm_audit_os.makedirs(log_dir, exist_ok=True)
        log_path = _glm_audit_os.path.join(log_dir, f"glm_fa2_audit.{_glm_audit_os.getpid()}.jsonl")
        with open(log_path, "a", encoding="utf-8") as handle:
            handle.write(_glm_audit_json.dumps(evidence, sort_keys=True) + "\\n")
    if failure:
        raise RuntimeError("GLM FA2 audit failed before attention; inspect the per-process JSONL evidence")

'''


def patched_source(source: bytes) -> bytes:
    observed = hashlib.sha256(source).hexdigest()
    if observed != EXPECTED_SOURCE_SHA256:
        raise ValueError(f"Source hash mismatch: expected {EXPECTED_SOURCE_SHA256}, got {observed}")
    text = source.decode("utf-8")
    replacements = [
        (
            '_FP8_KV_DTYPES = ("fp8", "fp8_e4m3")',
            inspect.getsource(compare_plan_counts) + "\n" + RUNTIME_HELPER
            + '_FP8_KV_DTYPES = ("fp8", "fp8_e4m3")',
        ),
        (
            '        self._lens_cpu[:num_tokens] = kv_lens.to(torch.int32)\n',
            '        self._lens_cpu[:num_tokens] = kv_lens.to(torch.int32)\n'
            '        self._glm_audit_active_rows = num_tokens\n',
        ),
        (
            '        state = _SM90_STATE\n        assert state is not None\n',
            '        state = _SM90_STATE\n        assert state is not None\n'
            '        _glm_audit_before_fa2_run(state, valid_counts, topk_slots, self, layer, attn_metadata)\n',
        ),
    ]
    for before, after in replacements:
        if text.count(before) != 1:
            raise ValueError(f"Expected exactly one source anchor: {before[:70]!r}")
        text = text.replace(before, after, 1)
    compile(text, "flashinfer_mla_sparse_sm90_audit.py", "exec")
    return text.encode("utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    source_path, output_path = args.source.resolve(), args.output.resolve()
    if source_path == output_path:
        parser.error("Source and output must differ; the baseline is never edited")
    if output_path.exists():
        parser.error("Output already exists; choose a new diagnostic copy")
    manifest_path = output_path.with_suffix(output_path.suffix + ".manifest.json")
    if manifest_path.exists():
        parser.error("Manifest already exists; choose a new diagnostic copy")
    source = source_path.read_bytes()
    try:
        output = patched_source(source)
    except ValueError as exc:
        parser.error(str(exc))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("xb") as handle:
        handle.write(output)
    manifest = {
        "source": str(source_path), "output": str(output_path),
        "source_sha256": hashlib.sha256(source).hexdigest(),
        "output_sha256": hashlib.sha256(output).hexdigest(),
        "requires_enforce_eager": True,
        "activation": "GLM_FA2_AUDIT_MARKER exists after startup",
        "changes_weights_topk_or_indices": False,
    }
    with manifest_path.open("x", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
        handle.write("\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
