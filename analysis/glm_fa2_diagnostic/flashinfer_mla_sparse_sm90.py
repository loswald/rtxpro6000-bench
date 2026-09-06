# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""FlashInfer sparse MLA backend for SM90 (Hopper) NoPE models.

Wraps FlashInfer's ``BatchMLAPagedAttentionWrapper`` (FA2/FA3 paths), which
as of FlashInfer 0.6.18 supports ``head_dim_kpe=0`` (GLM-5-Next's NoPE MLA)
and FP8 E4M3 KV caches on SM90 with in-kernel dequantization: the FP8 cache
is read directly (half the bf16 HBM traffic) and converted to BF16 in shared
memory, while queries stay BF16 (no query quantization).

Sparsity rides the same trick the FA-based sparse backend uses: with
``page_size=1`` the per-token top-k slot indices ARE the page table, so each
query token becomes one varlen batch row whose ``kv_indices`` slice is its
top-k row and whose ``kv_len`` is its valid count. Causality is already
encoded by the indexer's selection, so ``causal=False``.

CUDA-graph handling: ``plan()`` copies its inputs to host unconditionally,
so it must stay outside graph capture. A process-wide state object owns the
wrapper (created eagerly at impl construction with the model's head count
and KV dtype), reserved capture-stable device buffers, and the plan
parameters. The wrapper bakes the per-row ``kv_len`` into its int schedule
at plan() time — ``run()`` never reads the device-side buffer — so the
metadata builder replans every step (outside capture) with exact host-side
lengths derived from the batch's sequence lengths; a full-width schedule
would send the kernel past each row's valid count into the -1 tail of the
converted index buffer (illegal address). Per-step content (top-k slots)
is written into the reserved buffers by kernels inside the captured
forward, and captured runs read the refreshed plan buffers on replay.

KV cache format: plain contiguous E4M3 ``[num_blocks, block_size, 512]``
(uint8 storage) with a per-tensor ``k_scale``; BF16 caches also work. The
per-token x 128-channel-group ``ckv_scale_arr`` layout is supported by the
kernel but not wired yet (it needs a group-quantizing cache-write op).
"""

from typing import Any, ClassVar

import torch

from vllm.config import VllmConfig
from vllm.config.cache import CacheDType
from vllm.model_executor.layers.attention.sparse_mla_attention import (
    SparseMLACommonImpl,
)
from vllm.platforms.interface import DeviceCapability
from vllm.utils.flashinfer import has_flashinfer_sm90_nope_mla
from vllm.v1.attention.backend import (
    AttentionBackend,
    AttentionLayer,
    CommonAttentionMetadata,
    MLAAttentionImpl,
    MultipleOf,
)
from vllm.v1.attention.backends.mla.flashinfer_mla_sparse import (
    FlashInferMLASparseMetadata,
    FlashInferMLASparseMetadataBuilder,
)
from vllm.v1.attention.backends.mla.sparse_utils import (
    triton_convert_req_index_to_global_index,
)
from vllm.v1.attention.backends.utils import KVCacheLayoutType
from vllm.v1.kv_cache_interface import AttentionSpec

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
            handle.write(_glm_audit_json.dumps(evidence, sort_keys=True) + "\n")


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

_FP8_KV_DTYPES = ("fp8", "fp8_e4m3")
_WORKSPACE_BYTES = 128 * 1024 * 1024

# The BatchMLAPagedAttentionWrapper keeps its plan/schedule state inside the
# workspace between plan() and run() (across layers and steps, including CUDA
# graph replays). The shared step workspace is clobbered by the indexer and
# MoE kernels that run between MLA layers, so this must be a private buffer
# with a stable address.
_SM90_WORKSPACE: torch.Tensor | None = None


def _get_sm90_workspace(device: torch.device) -> torch.Tensor:
    global _SM90_WORKSPACE
    if _SM90_WORKSPACE is None:
        _SM90_WORKSPACE = torch.empty(
            _WORKSPACE_BYTES, dtype=torch.uint8, device=device
        )
    return _SM90_WORKSPACE


class FlashInferMLASparseSM90Backend(AttentionBackend):
    supported_dtypes: ClassVar[list[torch.dtype]] = [torch.bfloat16]
    supported_kv_cache_dtypes: ClassVar[list[CacheDType]] = [
        "auto",
        "bfloat16",
        "fp8",
        "fp8_e4m3",
    ]

    @staticmethod
    def get_supported_kernel_block_sizes() -> list[int | MultipleOf]:
        return [MultipleOf(64)]

    @staticmethod
    def get_name() -> str:
        return "FLASHINFER_MLA_SPARSE_SM90"

    @staticmethod
    def get_builder_cls() -> type["FlashInferMLASparseSM90Builder"]:
        return FlashInferMLASparseSM90Builder

    @staticmethod
    def get_impl_cls() -> type[MLAAttentionImpl]:
        return FlashInferMLASparseSM90Impl

    @classmethod
    def get_supported_head_sizes(cls) -> list[int]:
        # 512 = ckv 512 + kpe 0 (NoPE); 576 = ckv 512 + kpe 64.
        return [512, 576]

    @classmethod
    def is_mla(cls) -> bool:
        return True

    @classmethod
    def is_sparse(cls) -> bool:
        return True

    @classmethod
    def supports_compute_capability(cls, capability: DeviceCapability) -> bool:
        return capability.major in (9, 12)  # sm_120 port: FA2 path below

    @classmethod
    def supports_combination(
        cls,
        head_size: int,
        dtype: torch.dtype,
        kv_cache_dtype: CacheDType | None,
        block_size: int | None,
        use_mla: bool,
        has_sink: bool,
        use_sparse: bool,
        use_mm_prefix: bool,
        device_capability: DeviceCapability,
    ) -> str | None:
        if not has_flashinfer_sm90_nope_mla():
            return (
                "FLASHINFER_MLA_SPARSE_SM90 requires FlashInfer with SM90 "
                "MLA support (ckv_scale_arr in "
                "BatchMLAPagedAttentionWrapper.run, FlashInfer >= 0.6.18)"
            )
        from vllm.config import get_current_vllm_config

        vllm_config = get_current_vllm_config()
        if vllm_config.model_config is not None:
            hf = vllm_config.model_config.hf_text_config
            # The SM90 FA2/FA3 kernel covers ckv=512 with kpe in {0, 64}
            # (NoPE models and DeepSeek-style rope MLA alike).
            if getattr(hf, "kv_lora_rank", 512) != 512:
                return "FLASHINFER_MLA_SPARSE_SM90 requires kv_lora_rank=512"
            if getattr(hf, "qk_rope_head_dim", 0) not in (0, 64):
                return "FLASHINFER_MLA_SPARSE_SM90 requires qk_rope_head_dim in (0, 64)"
            if not hasattr(hf, "index_topk"):
                return "FLASHINFER_MLA_SPARSE_SM90 requires a sparse model"
        return None

    @staticmethod
    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
        cache_dtype_str: str = "auto",
    ) -> tuple[int, ...]:
        return (num_blocks, block_size, head_size)

    @classmethod
    def get_required_kv_cache_layout(cls) -> "KVCacheLayoutType | None":
        return "HND"


class _SM90State:
    """Process-wide wrapper, capture-stable buffers, and plan parameters.

    One instance serves every MLA layer: the plan depends only on the batch
    shape, not the layer. Created eagerly at the first impl construction
    (always before any graph capture), so the head count and KV dtype are
    known when planning.
    """

    def __init__(
        self,
        device: torch.device,
        num_heads: int,
        kv_dtype: torch.dtype,
        max_tokens: int,
        topk_width: int,
        kv_lora_rank: int,
        qk_rope_head_dim: int,
        sm_scale: float,
    ) -> None:
        _glm_audit_validate_launch()
        from flashinfer.mla import BatchMLAPagedAttentionWrapper

        float_workspace = _get_sm90_workspace(device)
        self.device = device
        self.num_heads = num_heads
        self.kv_dtype = kv_dtype
        self.max_tokens = max_tokens
        self.topk_width = topk_width
        self.kv_lora_rank = kv_lora_rank
        self.qk_rope_head_dim = qk_rope_head_dim
        self.sm_scale = sm_scale
        # User-reserved buffers: with use_cuda_graph=True plan() refreshes
        # these in place, so run()'s captured kernels always read them.
        self.kv_indices = torch.zeros(
            max_tokens * topk_width, dtype=torch.int32, device=device
        )
        self.kv_len_arr = torch.full(
            (max_tokens,), topk_width, dtype=torch.int32, device=device
        )
        self.wrapper = BatchMLAPagedAttentionWrapper(
            float_workspace,
            qo_indptr=torch.zeros(max_tokens + 1, dtype=torch.int32, device=device),
            kv_indptr=torch.zeros(max_tokens + 1, dtype=torch.int32, device=device),
            kv_indices=self.kv_indices,
            kv_len_arr=self.kv_len_arr,
            use_cuda_graph=True,
            backend=("fa2" if torch.cuda.get_device_capability(device)[0] == 12 else "fa3"),
        )

    def plan(self, num_tokens: int, kv_lens: torch.Tensor) -> None:
        """Replan with exact per-row KV lengths (CPU int32, ``[num_tokens]``).

        The wrapper bakes kv_len into its int schedule from host values;
        run() never reads the device kv_len_arr buffer. Scheduling with the
        full buffer width while kv_indices rows carry a -1 tail past each
        row's valid count makes the kernel compute ``-1 * ckv_stride_page``
        (illegal address), so the lengths must be exact at plan time and
        replanned every step as contexts grow. Must run outside CUDA graph
        capture: the in-place refreshed plan_info/indptr buffers are what
        captured runs read.
        """
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError(
                "FlashInferMLASparseSM90 plan() called inside CUDA graph "
                "capture; lengths must be planned host-side before capture."
            )
        # Cached CPU staging buffers, filled in place: plan() runs per step
        # (once per draft/verify metadata build), so per-call allocations and
        # device round trips are on the hot path. Passing CPU tensors lets
        # the wrapper's internal .to("cpu") no-op; its reserved-buffer
        # copy_ then performs the single H2D transfer per tensor.
        if getattr(self, "_arange_cpu", None) is None:
            self._arange_cpu = torch.arange(self.max_tokens + 1, dtype=torch.int32)
            self._qo_cpu = torch.empty(self.max_tokens + 1, dtype=torch.int32)
            self._kv_cpu = torch.empty(self.max_tokens + 1, dtype=torch.int32)
            self._lens_cpu = torch.full(
                (self.max_tokens,), self.topk_width, dtype=torch.int32
            )
        # use_cuda_graph=True makes the wrapper copy qo/kv indptr into its
        # fixed (max_tokens+1)-sized buffers with exact-size copy_, so the
        # indptr must always be full-size. Rows past num_tokens are padded
        # empty (qo_indptr flat at num_tokens) — zero-query rows read no q
        # and schedule no work. Padded rows keep the full width lens; the
        # value is never dereferenced.
        torch.clamp(self._arange_cpu, max=num_tokens, out=self._qo_cpu)
        torch.mul(self._qo_cpu, self.topk_width, out=self._kv_cpu)
        self._lens_cpu.fill_(self.topk_width)
        self._lens_cpu[:num_tokens] = kv_lens.to(torch.int32)
        self._glm_audit_active_rows = num_tokens
        self.wrapper.plan(
            self._qo_cpu,
            self._kv_cpu,
            self.kv_indices,
            self._lens_cpu,
            self.num_heads,
            self.kv_lora_rank,  # head_dim_ckv
            self.qk_rope_head_dim,  # 0 (NoPE) or 64 (rope MLA)
            1,  # page_size: top-k slots are the page table
            False,  # causal: encoded by the indexer's selection
            self.sm_scale,
            q_data_type=torch.bfloat16,
            kv_data_type=self.kv_dtype,
        )


_SM90_STATE: _SM90State | None = None


def _get_sm90_state() -> "_SM90State | None":
    return _SM90_STATE


class FlashInferMLASparseSM90Builder(FlashInferMLASparseMetadataBuilder):
    """Reuse the common sparse metadata (req ids, topk buffer access)."""

    metadata_cls = FlashInferMLASparseMetadata

    def __init__(
        self,
        kv_cache_spec: "AttentionSpec",
        layer_names: list[str],
        vllm_config: "VllmConfig",
        device: torch.device,
    ) -> None:
        super().__init__(kv_cache_spec, layer_names, vllm_config, device)
        # seq_lens_cpu_upper_bound is optimistic on decode rows under async
        # spec decode, so the fast sync-free path is only safe without it;
        # under async scheduling the exact (device) positions are used at
        # the cost of one D2H sync per metadata build.
        self._async_scheduling = bool(vllm_config.scheduler_config.async_scheduling)
        hf_config = vllm_config.model_config.hf_text_config
        self._index_topk = int(getattr(hf_config, "index_topk", 2048))
        self._index_kpool = int(getattr(hf_config, "index_kpool", 1))

    def _kv_lens_host(self, cam: CommonAttentionMetadata) -> tuple[int, torch.Tensor]:
        """Exact per-row KV lengths, host-side (the flashinfer wrapper bakes
        them into its schedule at plan time; there is no device-side path).

        A row for the j-th query token of request i attends
        ``seq_lens[i] - q_len[i] + j + 1`` tokens. The indexer's selection
        then bounds the valid count: contexts up to ``index_topk`` select
        everything (valid == context); longer contexts keep the top
        ``index_topk`` pool-expanded tokens plus the trailing incomplete
        pool (valid == ``index_topk + context % index_kpool``). Both match
        the count of non -1 entries the convert kernel produces.
        """
        num_reqs = cam.num_reqs
        qsl = cam.query_start_loc_cpu[: num_reqs + 1]
        num_rows = int(qsl[-1])
        if num_rows == 0:
            return 0, torch.zeros(0, dtype=torch.int32)
        # Row context == position + 1. Without async scheduling the
        # maintained host upper bound is exact, giving a sync-free path;
        # under async scheduling it is optimistic on decode rows, so fall
        # back to the exact device positions (one D2H sync per build). The
        # seq_lens derivation equals position + 1 because positions are
        # contiguous per request.
        sl_host = getattr(cam, "seq_lens_cpu_upper_bound", None)
        positions = getattr(cam, "positions", None)
        if not getattr(self, "_async_scheduling", False) and sl_host is not None:
            seq_lens = sl_host[:num_reqs].to(torch.int32)
            q_lens = qsl[1:] - qsl[:-1]
            first_pos = seq_lens - q_lens
            req_of_row = torch.repeat_interleave(
                torch.arange(num_reqs, dtype=torch.int64), q_lens.to(torch.int64)
            )
            rows = torch.arange(num_rows, dtype=torch.int32)
            ctx = (
                first_pos.to(torch.int64)[req_of_row]
                + rows.to(torch.int64)
                - qsl.to(torch.int64)[req_of_row]
                + 1
            )
        elif positions is not None and num_rows <= positions.shape[0]:
            ctx = positions[:num_rows].cpu().to(torch.int64) + 1
        else:
            seq_lens = cam.seq_lens[:num_reqs].cpu().to(torch.int32)
            q_lens = qsl[1:] - qsl[:-1]
            first_pos = seq_lens - q_lens
            req_of_row = torch.repeat_interleave(
                torch.arange(num_reqs, dtype=torch.int64), q_lens.to(torch.int64)
            )
            rows = torch.arange(num_rows, dtype=torch.int32)
            ctx = (
                first_pos.to(torch.int64)[req_of_row]
                + rows.to(torch.int64)
                - qsl.to(torch.int64)[req_of_row]
                + 1
            )
        topk = self._index_topk
        kpool = max(self._index_kpool, 1)
        lens = torch.where(ctx <= topk, ctx, topk + ctx % kpool)
        return num_rows, lens.to(torch.int32)

    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata: CommonAttentionMetadata,
        fast_build: bool = False,
    ) -> FlashInferMLASparseMetadata:
        metadata = super().build(common_prefix_len, common_attn_metadata, fast_build)
        # Replan every step outside any CUDA graph capture with this step's
        # exact per-row lengths; captured runs read the refreshed buffers.
        if _SM90_STATE is not None:
            num_rows, kv_lens = self._kv_lens_host(common_attn_metadata)
            _SM90_STATE.plan(num_rows, kv_lens)
        return metadata


class FlashInferMLASparseSM90Impl(SparseMLACommonImpl[FlashInferMLASparseMetadata]):
    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        num_kv_heads: int,
        alibi_slopes: list[float] | None,
        sliding_window: int | None,
        kv_cache_dtype: str,
        logits_soft_cap: float | None,
        attn_type: str,
        kv_sharing_target_layer_name: str | None,
        topk_indices_buffer: torch.Tensor | None = None,
        indexer: Any | None = None,
        **mla_args: Any,
    ) -> None:
        global _SM90_STATE
        if any([alibi_slopes, sliding_window, logits_soft_cap]):
            raise NotImplementedError(
                "FlashInferMLASparseSM90Impl does not support alibi, sliding "
                "window, or logits soft cap."
            )
        super().__init__(
            num_heads,
            head_size,
            scale,
            num_kv_heads,
            alibi_slopes,
            sliding_window,
            kv_cache_dtype,
            logits_soft_cap,
            attn_type,
            kv_sharing_target_layer_name,
            indexer=indexer,
            topk_indices_buffer=topk_indices_buffer,
            **mla_args,
        )
        assert self.topk_indices_buffer is not None
        self.supports_quant_query_input = False
        self.use_fp8_kv_cache = self.kv_cache_dtype in _FP8_KV_DTYPES
        if _SM90_STATE is None:
            from vllm.config import get_current_vllm_config

            assert topk_indices_buffer is not None
            max_tokens = (
                get_current_vllm_config().scheduler_config.max_num_batched_tokens
            )
            _SM90_STATE = _SM90State(
                topk_indices_buffer.device,
                num_heads,
                (torch.float8_e4m3fn if self.use_fp8_kv_cache else torch.bfloat16),
                max_tokens,
                topk_indices_buffer.shape[1],
                kv_lora_rank=self.kv_lora_rank,
                qk_rope_head_dim=self.qk_rope_head_dim,
                sm_scale=self.scale,
            )

    def forward_mqa(
        self,
        q: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
        kv_c_and_k_pe_cache: torch.Tensor,
        attn_metadata: FlashInferMLASparseMetadata,
        layer: AttentionLayer,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if not isinstance(q, tuple):
            raise NotImplementedError(
                "FlashInferMLASparseSM90Impl expects split (q_nope, q_rope)."
            )
        q_nope, q_rope = q
        num_tokens = q_rope.shape[0]
        # NoPE models hand a zero-width rope tensor through; rope MLA hands
        # the real 64-dim part. The kernel takes both as-is.
        q_pe = q_rope.reshape(num_tokens, self.num_heads, self.qk_rope_head_dim)

        assert self.topk_indices_buffer is not None
        topk_indices = self.topk_indices_buffer[:num_tokens]
        audit_evidence = _glm_audit_before_convert(
            _SM90_STATE, self, layer, attn_metadata, q_nope, q_pe,
            kv_c_and_k_pe_cache, topk_indices,
        )
        # return_valid_counts=True keeps the compacted-prefix layout: valid
        # entries at [0, valid_count), -1 past it — exactly the prefix the
        # planned per-row lengths address.
        topk_slots, valid_counts = triton_convert_req_index_to_global_index(
            attn_metadata.req_id_per_token[:num_tokens],
            attn_metadata.block_table,
            topk_indices,
            BLOCK_SIZE=attn_metadata.block_size,
            NUM_TOPK_TOKENS=topk_indices.shape[1],
            return_valid_counts=True,
        )
        state = _SM90_STATE
        assert state is not None
        _glm_audit_before_fa2_run(state, valid_counts, topk_slots, audit_evidence)
        # Refresh the kv_indices contents with in-graph device copies so
        # captured runs observe this step's top-k rows. Per-row lengths are
        # NOT refreshed here: the wrapper bakes them into its schedule at
        # plan() time (host-side), which the builder does every step before
        # capture/replay. Planned lengths are upper bounds of the valid count;
        # clamp the -1 tail to slot 0 so any over-scheduled read (e.g. the
        # capture dummy's degenerate rows) lands on a valid page instead of a
        # negative address. A no-op whenever plan is exact.
        width = topk_slots.shape[1]
        state.kv_indices[: num_tokens * width].copy_(
            topk_slots.reshape(-1).clamp_(min=0).to(torch.int32)
        )

        flat = (
            kv_c_and_k_pe_cache.view(torch.float8_e4m3fn)
            if self.use_fp8_kv_cache
            else kv_c_and_k_pe_cache
        ).reshape(-1, 1, self.head_size)
        ckv = flat[..., : self.kv_lora_rank]
        kpe = flat[..., self.kv_lora_rank :]

        scale_kwargs = (
            {"ckv_scale": float(layer._k_scale_float or 1.0), "kpe_scale": 1.0}
            if self.use_fp8_kv_cache
            else {}
        )
        out = state.wrapper.run(q_nope, q_pe, ckv, kpe, **scale_kwargs)
        return out, None
