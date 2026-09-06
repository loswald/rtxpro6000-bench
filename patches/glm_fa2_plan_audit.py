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


def validate_query_rows(planned_rows, actual_rows, num_reqs, query_start_loc,
                        query_rows, topk_rows, request_rows, max_rows):
    """Establish live rows from current metadata before treating storage as padding."""
    errors = []
    values = [planned_rows, actual_rows, num_reqs, query_rows, topk_rows,
              request_rows, max_rows]
    if any(not isinstance(x, int) or isinstance(x, bool) or x < 0 for x in values):
        return {"metadata_errors": ["Missing or invalid row metadata"], "active_rows": None}
    if not isinstance(query_start_loc, list) or len(query_start_loc) < num_reqs + 1:
        errors.append("query_start_loc does not contain num_reqs + 1 entries")
        qsl = []
    else:
        qsl = query_start_loc[:num_reqs + 1]
        if any(not isinstance(x, int) or isinstance(x, bool) or x < 0 for x in qsl):
            errors.append("query_start_loc has invalid entries")
        elif qsl[0] != 0 or any(a > b for a, b in zip(qsl, qsl[1:])):
            errors.append("query_start_loc must start at zero and be nondecreasing")
        elif qsl[-1] != actual_rows:
            errors.append("query_start_loc endpoint differs from num_actual_tokens")
    if planned_rows != actual_rows:
        errors.append("Host plan rows differ from current num_actual_tokens")
    if actual_rows > query_rows:
        errors.append("Current live batch exceeds callback query storage; possible decode-only callback")
    if query_rows != topk_rows:
        errors.append("Callback query rows differ from top-k row storage")
    if request_rows < query_rows:
        errors.append("Request-ID storage is shorter than converter query storage")
    if max(actual_rows, planned_rows, query_rows, topk_rows) > max_rows:
        errors.append("Current rows exceed wrapper capacity")
    return {
        "metadata_errors": errors, "active_rows": actual_rows,
        "planned_active_rows": planned_rows, "metadata_num_actual_tokens": actual_rows,
        "metadata_num_reqs": num_reqs, "query_start_loc": qsl,
        "callback_query_rows": query_rows, "topk_rows": topk_rows,
        "request_id_storage_rows": request_rows, "wrapper_max_rows": max_rows,
    }


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
_GLM_FA2_AUDIT_CORRECTIONS = 0


def _glm_audit_write(evidence, failure=False):
    evidence["correction_count"] = _GLM_FA2_AUDIT_CORRECTIONS
    limit = int(_glm_audit_os.environ.get("GLM_FA2_AUDIT_MAX_RECORDS", "256"))
    first_corrections = evidence.get("exact_replan_applied", False) and _GLM_FA2_AUDIT_CORRECTIONS <= 8
    if evidence["call"] <= limit or failure or first_corrections:
        log_dir = _glm_audit_os.environ["GLM_FA2_AUDIT_LOG_DIR"]
        _glm_audit_os.makedirs(log_dir, exist_ok=True)
        path = _glm_audit_os.path.join(log_dir, f"glm_fa2_audit.{_glm_audit_os.getpid()}.jsonl")
        with open(path, "a", encoding="utf-8") as handle:
            handle.write(_glm_audit_json.dumps(evidence, sort_keys=True) + "\\n")


def _glm_audit_fail(evidence, reason):
    evidence.setdefault("failure_reasons", []).append(reason)
    _glm_audit_write(evidence, failure=True)
    raise RuntimeError("GLM FA2 audit failed before attention: " + reason)


def _glm_audit_validate_launch():
    if _glm_audit_os.environ.get("GLM_FA2_AUDIT_MARKER", ""):
        from vllm.config import get_current_vllm_config
        if not get_current_vllm_config().model_config.enforce_eager:
            raise RuntimeError("GLM FA2 audit marker configured: launch must use --enforce-eager")


def _glm_audit_before_convert(state, impl, layer, metadata, q_nope, q_pe, cache, topk):
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
    _GLM_FA2_AUDIT_CALLS += 1
    dist = torch.distributed
    rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else None
    evidence = {
        "time_unix": _glm_audit_time.time(), "pid": _glm_audit_os.getpid(), "rank": rank,
        "layer": str(getattr(layer, "layer_name", getattr(layer, "prefix", "unknown"))),
        "layer_object_id": id(layer), "call": _GLM_FA2_AUDIT_CALLS, "mode": mode,
        "exact_replan_applied": False, "phase": "before_converter",
    }
    if state is None or not hasattr(state, "_lens_cpu"):
        _glm_audit_fail(evidence, "No preceding host plan")
    missing = [name for name in ("query_start_loc", "req_id_per_token", "block_table",
                                  "num_actual_tokens", "num_reqs", "block_size")
               if not hasattr(metadata, name)]
    if missing:
        _glm_audit_fail(evidence, "Missing current metadata fields: " + ", ".join(missing))
    tensors = {"q_nope": q_nope, "q_pe": q_pe, "cache": cache, "topk": topk,
               "query_start_loc": metadata.query_start_loc,
               "request_ids": metadata.req_id_per_token, "block_table": metadata.block_table}
    evidence["tensor_shapes"] = {name: list(t.shape) for name, t in tensors.items()}
    evidence["tensor_dtypes"] = {name: str(t.dtype) for name, t in tensors.items()}
    expected_dims = {"q_nope": 3, "q_pe": 3, "cache": 3, "topk": 2,
                     "query_start_loc": 1, "request_ids": 1, "block_table": 2}
    if any(t.ndim != expected_dims[name] for name, t in tensors.items()):
        _glm_audit_fail(evidence, "Unexpected metadata/query/cache tensor dimensions")
    qsl = metadata.query_start_loc.cpu().tolist()
    evidence.update(validate_query_rows(
        getattr(state, "_glm_audit_active_rows", None), metadata.num_actual_tokens,
        metadata.num_reqs, qsl, q_nope.shape[0], topk.shape[0],
        metadata.req_id_per_token.shape[0], state.max_tokens,
    ))
    if evidence["metadata_errors"]:
        _glm_audit_fail(evidence, "; ".join(evidence["metadata_errors"]))
    active, query_rows, width = evidence["active_rows"], q_nope.shape[0], topk.shape[1]
    evidence.update({"width": width, "heads": impl.num_heads,
                     "physical_block_size": metadata.block_size, "kv_dtype": str(state.kv_dtype)})
    if (tuple(q_nope.shape[1:]) != (impl.num_heads, impl.kv_lora_rank)
            or tuple(q_pe.shape) != (query_rows, impl.num_heads, impl.qk_rope_head_dim)
            or width < 1 or width % 128 != 0 or len(state._lens_cpu) < active):
        _glm_audit_fail(evidence, "Query, sparse width, or host-plan storage geometry mismatch")
    if (cache.shape[1] != metadata.block_size or metadata.block_size < 1
            or cache.shape[2] != impl.head_size
            or impl.head_size != impl.kv_lora_rank + impl.qk_rope_head_dim):
        _glm_audit_fail(evidence, "Physical cache geometry differs from current layer metadata")
    capacity = cache.shape[0] * cache.shape[1]
    evidence["physical_slot_capacity"] = capacity
    if capacity < 1 or capacity > 2147483647:
        _glm_audit_fail(evidence, "Physical cache does not fit signed int32 slot addressing")
    if topk.numel() > 2147483647 or metadata.block_table.numel() > 2147483647:
        _glm_audit_fail(evidence, "Contiguous converter matrices exceed signed int32 address arithmetic")
    if (state.num_heads != impl.num_heads or state.topk_width != width
            or state.kv_lora_rank != impl.kv_lora_rank
            or state.qk_rope_head_dim != impl.qk_rope_head_dim
            or state.sm_scale != impl.scale
            or state.device != topk.device):
        _glm_audit_fail(evidence, "Process-wide wrapper geometry differs from current layer")
    expected_kv_dtype = torch.float8_e4m3fn if impl.use_fp8_kv_cache else torch.bfloat16
    cache_dtypes = (torch.uint8, torch.float8_e4m3fn) if impl.use_fp8_kv_cache else (torch.bfloat16,)
    if (state.kv_dtype != expected_kv_dtype or cache.dtype not in cache_dtypes
            or q_nope.dtype != torch.bfloat16 or q_pe.dtype != torch.bfloat16
            or any(tensors[name].dtype != torch.int32 for name in ("request_ids", "block_table", "topk"))
            or metadata.query_start_loc.dtype not in (torch.int32, torch.int64)
            or any(t.device != state.device for t in tensors.values())):
        _glm_audit_fail(evidence, "Tensor dtype/device differs from current layer contract")
    reqs = metadata.req_id_per_token[:query_rows]
    evidence["request_ids"] = reqs[:active].cpu().tolist()
    if bool(((reqs[:active] < 0) | (reqs[:active] >= metadata.num_reqs)
             | (reqs[:active] >= metadata.block_table.shape[0])).any().item()):
        _glm_audit_fail(evidence, "Live request ID is outside current request/block-table rows")
    expected_reqs = torch.repeat_interleave(
        torch.arange(metadata.num_reqs, dtype=torch.int32, device=state.device),
        (metadata.query_start_loc[1:metadata.num_reqs + 1]
         - metadata.query_start_loc[:metadata.num_reqs]).to(torch.int64),
    )
    if bool((reqs[:active] != expected_reqs).any().item()):
        _glm_audit_fail(evidence, "Live request IDs disagree with current query_start_loc ownership")
    # Even padding participates in conversion. The converter's load mask uses
    # only block-column bounds, not token sign. Triton divides toward zero:
    # a -1 sentinel still has block column 0 and can read the block table.
    # Validate only referenced physical blocks before the converter multiplies
    # int32 block IDs. A bad large base could otherwise wrap to a plausible
    # in-range slot. Chunking bounds scratch memory; the input is never changed.
    for first in range(0, query_rows, 128):
        last = min(first + 128, query_rows)
        tokens64 = topk[first:last].to(torch.int64)
        columns64 = torch.div(tokens64, metadata.block_size, rounding_mode="trunc")
        reads = (columns64 >= 0) & (columns64 < metadata.block_table.shape[1])
        if not bool(reads.any().item()):
            continue
        chunk_reqs = reqs[first:last]
        bad_req = (chunk_reqs < 0) | (chunk_reqs >= metadata.block_table.shape[0])
        if bool((reads.any(dim=1) & bad_req).any().item()):
            _glm_audit_fail(evidence, "Converter would read a block table with an out-of-range request ID")
        live_in_chunk = max(0, min(last, active) - first)
        if not live_in_chunk:
            continue
        safe_reqs = chunk_reqs[:live_in_chunk].to(torch.int64).clamp(0, metadata.block_table.shape[0] - 1)
        safe_columns = columns64[:live_in_chunk].clamp(0, metadata.block_table.shape[1] - 1)
        physical_blocks = metadata.block_table[safe_reqs.unsqueeze(1), safe_columns]
        invalid_blocks = (physical_blocks < 0) | (physical_blocks >= cache.shape[0])
        if bool((reads[:live_in_chunk] & (tokens64[:live_in_chunk] >= 0) & invalid_blocks).any().item()):
            evidence["physical_block_check_query_start"] = first
            evidence["physical_block_check_query_end"] = last
            _glm_audit_fail(evidence, "Referenced physical block ID is outside cache before int32 slot conversion")
    return evidence


def _glm_audit_before_fa2_run(state, valid_counts, slots, evidence):
    global _GLM_FA2_AUDIT_CORRECTIONS
    if evidence is None:
        return
    active, width, mode = evidence["active_rows"], evidence["width"], evidence["mode"]
    evidence["phase"] = "before_attention"
    if (slots.ndim != 2 or tuple(slots.shape) != (evidence["callback_query_rows"], width)
            or valid_counts.ndim != 1 or valid_counts.shape[0] != slots.shape[0]
            or slots.dtype != torch.int32 or valid_counts.dtype != torch.int32
            or slots.device != state.device or valid_counts.device != state.device):
        _glm_audit_fail(evidence, "Converted sparse row count, capacity, dtype, or device is inconsistent")
    actual_cpu = valid_counts.reshape(-1).to(device="cpu", dtype=torch.int32)
    evidence.update(compare_plan_counts(
        state._lens_cpu.tolist(), actual_cpu.tolist(), int(active), width
    ))
    # A valid count is useful only if conversion produced its promised compact
    # prefix. Check live rows before the existing negative-index clamp.
    live_slots = slots[:active]
    column = torch.arange(width, device=slots.device).unsqueeze(0)
    prefix = column < valid_counts.reshape(-1)[:active].unsqueeze(1)
    negative_prefix = bool(((live_slots < 0) & prefix).any().item())
    nonnegative_tail = bool(((live_slots >= 0) & ~prefix).any().item())
    invalid_tail = bool(((live_slots != -1) & ~prefix).any().item())
    high_prefix = bool(((live_slots >= evidence["physical_slot_capacity"]) & prefix).any().item())
    evidence.update({
        "negative_in_live_prefix": negative_prefix,
        "nonnegative_in_live_tail": nonnegative_tail,
        "invalid_sentinel_in_live_tail": invalid_tail,
        "out_of_bounds_in_live_prefix": high_prefix,
    })
    invalid = (bool(evidence["invalid_count_rows"]) or bool(evidence["zero_count_active_rows"])
               or negative_prefix or invalid_tail or high_prefix)
    failure = invalid or (mode == "check" and not evidence["matches"])
    if mode == "exact" and not evidence["matches"] and not invalid:
        # An all-masked live row needs a separate kernel contract. Do not treat
        # it as a harmless length correction. Padded zero-query rows are ignored.
        if evidence["zero_count_active_rows"]:
            failure = True
        else:
            state.plan(int(active), actual_cpu[:active])
            evidence["exact_replan_applied"] = True
            _GLM_FA2_AUDIT_CORRECTIONS += 1
    if failure:
        _glm_audit_fail(evidence, "Converted counts, compact prefix, or physical slot bounds failed")
    _glm_audit_write(evidence)

'''


def patched_source(source: bytes) -> bytes:
    observed = hashlib.sha256(source).hexdigest()
    if observed != EXPECTED_SOURCE_SHA256:
        raise ValueError(f"Source hash mismatch: expected {EXPECTED_SOURCE_SHA256}, got {observed}")
    text = source.decode("utf-8")
    replacements = [
        (
            '_FP8_KV_DTYPES = ("fp8", "fp8_e4m3")',
            inspect.getsource(validate_query_rows) + "\n" + inspect.getsource(compare_plan_counts) + "\n" + RUNTIME_HELPER
            + '_FP8_KV_DTYPES = ("fp8", "fp8_e4m3")',
        ),
        (
            '        topk_indices = self.topk_indices_buffer[:num_tokens]\n',
            '        topk_indices = self.topk_indices_buffer[:num_tokens]\n'
            '        audit_evidence = _glm_audit_before_convert(\n'
            '            _SM90_STATE, self, layer, attn_metadata, q_nope, q_pe,\n'
            '            kv_c_and_k_pe_cache, topk_indices,\n'
            '        )\n',
        ),
        (
            '        from flashinfer.mla import BatchMLAPagedAttentionWrapper\n',
            '        _glm_audit_validate_launch()\n'
            '        from flashinfer.mla import BatchMLAPagedAttentionWrapper\n',
        ),
        (
            '        self._lens_cpu[:num_tokens] = kv_lens.to(torch.int32)\n',
            '        self._lens_cpu[:num_tokens] = kv_lens.to(torch.int32)\n'
            '        self._glm_audit_active_rows = num_tokens\n',
        ),
        (
            '        state = _SM90_STATE\n        assert state is not None\n',
            '        state = _SM90_STATE\n        assert state is not None\n'
            '        _glm_audit_before_fa2_run(state, valid_counts, topk_slots, audit_evidence)\n',
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
