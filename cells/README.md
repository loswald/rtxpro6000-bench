# cells/ — one file per serving cell

A cell is a bash fragment sourced by `bench/env.sh:load_cell` (see `_template.env` for
every key and for which script consumes which variable). `bench/launch.sh <cell>` renders
it into `vllm serve` command lines, `bench/sweep.sh <cell>` into `vllm bench serve` runs.
Everything below was checked on **2026-09-02** against the Hugging Face Hub
(`/api/models/<id>`), the vLLM recipes (recipes.vllm.ai) and vLLM **main**
(`0.28.1rc1.dev312+g41848caa6`, the build installed in place on the Vast box).

## The box these cells are written for

4x RTX PRO 6000 Blackwell **Server Edition**, 96 GB GDDR7 each, sm_120, PCIe Gen5 x16,
no NVLink. `nvidia-smi topo`: GPUs 0-1 PIX (same PCIe switch), 2-3 PIX, cross pairs NODE.
PCIe ACS is enabled on the host (not changeable from the container), so switch-local P2P is
redirected through the root complex; measured NCCL `all_reduce` busbw: pair 0-1 ~21 GB/s,
pair 0-2 ~38 GB/s, 4-GPU ring ~19 GB/s. P2P itself works (`can_device_access_peer` true,
NCCL transport P2P/CUMEM) — **no cell sets `NCCL_P2P_DISABLE`**. Custom all-reduce is off by
default in `bench/env.sh` (`CUSTOM_ALLREDUCE=1` for an A/B). Host: 192 threads, 1.5 TB RAM,
**390 GB disk (373 GB free at start)**. Models live as plain directories
`/workspace/models/<basename of the HF id>` (`hf download --local-dir`); cells keep `MODEL`
as the HF id and `load_cell` derives the local path.

### GPU layout rules (decision 3)

| layout | `GPU_IDS` | why |
|---|---|---|
| 4x TP1 replicas | `0,1,2,3` | one server per GPU, no cross-GPU traffic; **not** pessimistic. Declared as `REPLICAS=${REPLICAS:-4}` / `GPU_IDS=${GPU_IDS:-0,1,2,3}` so `REPLICAS=1 GPU_IDS=0 bench/launch.sh <x4 cell>` (and `gates/run_kv_diff.sh`, `KVDIFF_SINGLE_REPLICA=1`) can bring up a single replica |
| DP2 x TP2 (one server) | **`0,2,1,3`** | vLLM hands DP rank *i* the *i*-th TP-sized slice of `CUDA_VISIBLE_DEVICES` (`vllm/v1/engine/utils.py`: `start = local_dp_rank * world_size; user_assigned_gpu_ids[start:stop]`), so rank0 = physical 0,2 and rank1 = physical 1,3: both replicas pair **cross-switch** (~38 vs ~21 GB/s all-reduce). Pessimistic (dagger) |
| TP4 | `0,1,2,3` | all four GPUs, ACS-on PCIe ring ~19 GB/s. Pessimistic (dagger) |
| co-tenancy TP2 | `0,2` (`COTENANT_GPU_IDS` to override) | cross-switch pair; GPU 3 trains, GPU 1 idle |

Every sweep meta JSON carries `gpu_ids`, and `summarise.py` / `gates_summary.py` mark TP2/TP4
rows with a dagger (`acs_suspected=1`, `pessimistic_tp=1` from `results/hw/decisions.env`).

## Cells

Sizes are Hub `usedStorage` (2026-09-02). "Fit" is weights per GPU -> KV left at the cell's
`GPU_MEM_UTIL`; read `launch.json`'s `GPU KV cache size` line before sweeping.

| cell | model (Hub id, sha) | GB on disk | layout (GPU ids) | KV dtype | expected fit per GPU | dagger |
|---|---|---|---|---|---|---|
| `qwen38_27b_fp8_x4` | Qwen/Qwen3.8-27B-FP8 (017b9c7) | 31 | 4x TP1 (0,1,2,3), ports 8000-8003; `REPLICAS`/`GPU_IDS` env-overridable (gates/run_kv_diff.sh launches one replica) | fp8 | ~28 GB weights -> ~60 GB KV @0.92 | no |
| `gptoss120b_x4_marlin` | openai/gpt-oss-120b (b5c939d) | 65 (safetensors only; 196 with `original/`+`metal/`) | 4x TP1 (0,1,2,3) | fp8 | ~63 GB weights -> ~23 GB KV @0.92; agent shape saturates KV at C>=64 | no |
| `gptoss120b_x4_b12x` | same | shared | 4x TP1 (0,1,2,3) | fp8 | as marlin; `--moe-backend b12x` (needs the `b12x` package) | no |
| `gptoss120b_x4_ficutlass` | same | shared | 4x TP1 (0,1,2,3) | fp8 | as marlin; `flashinfer_cutlass` is expected to be rejected on sm_120 -> `launch.json.error_excerpt` is the result | no |
| `qwen38_27b_bf16_x4` | Qwen/Qwen3.8-27B (1d4bf0f) | 56 | 4x TP1 (0,1,2,3) | fp8 | ~55 GB weights -> ~33 GB KV @0.92; GSM8K reference | no |
| `qwen35_122b_fp8_tp2x2` | Qwen/Qwen3.5-122B-A10B-FP8 (a099dee) | 127 | DP2 x TP2 (**0,2,1,3**) | fp8 | ~61 GB weights -> ~27 GB KV @0.92 | yes |
| `cotenant_tp2_gpu01` | Qwen/Qwen3.5-122B-A10B-FP8 (or `COTENANT_MODEL=Qwen/Qwen3.8-27B-FP8`) | shared | TP2 (**0,2**), sleep mode on; trainer on GPU 3 | fp8 | as one replica of the cell above | yes |
| `ds4flash_tp4` | deepseek-ai/DeepSeek-V4-Flash-0731 (7872f01) | 167 | TP4 (0,1,2,3), EP on (recipe) | fp8_ds_mla (= recipe `fp8`) | ~42 GB weights -> ~45 GB KV @0.92 | yes |
| `ds4flash_tp2x2` | same | shared | DP2 x TP2 (**0,2,1,3**), EP off | fp8_ds_mla | ~84 GB weights -> ~7 GB KV @0.95: tight, OOM possible; tp4 is the reference | yes |
| `qwen38flashnext_fp8_tp4` | Qwen/Qwen3.8-Flash-Next-FP8 (236dfdf) | 186 | TP4 (0,1,2,3) | bfloat16 | ~31 GB main weights -> ~55 GB KV @0.90 **if** the 51 GB PLE table is CPU-offloaded; ~4 GB KV if it stays on GPU | yes |
| `qwen38flashnext_fp8_tp2x2` | same | shared | DP2 x TP2 (**0,2,1,3**) | bfloat16 | ~63 GB main -> ~23 GB KV @0.90 **only if** the PLE table is offloaded; 114 GB/GPU = OOM otherwise | yes |
| `glm53flash_fp8_tp4_loadtest` | zai-org/GLM-5.3-Flash (03eb536) | 328 | TP4 (0,1,2,3), `LOADTEST_ONLY=1` | fp8 (bf16 retry) | ~77 GiB weights -> ~14 GiB rest; **architecture not in vLLM main's registry** (only the recipe image knows it) | n/a |

## Order of execution on a 390 GB disk

Small models first (they bench while the big ones download), each model deleted right after
its last cell. Free space must be >= model size + ~20 GB before starting a download; never
overlap two downloads whose sum exceeds the free space. `du -sh /workspace/models/*` and
`df -h /workspace` before each step.

| step | run | download in background | disk after step (approx.) |
|---|---|---|---|
| 1 | `qwen38_27b_fp8_x4` (smoke + sweep + gsm8k + kv_diff) | gpt-oss-120b (65, `--exclude 'original/*' 'metal/*'`), Qwen3.8-27B bf16 (56) | 152 GB used |
| 2 | `gptoss120b_x4_marlin`, `gptoss120b_x4_b12x`, `gptoss120b_x4_ficutlass` (one launch attempt) | Qwen3.5-122B-A10B-FP8 (127) | 279 GB |
| 3 | `qwen38_27b_bf16_x4` (gsm8k reference for step 1) -> **delete** `Qwen3.8-27B` and `gpt-oss-120b` | Qwen3-8B (16, trainer for co-tenancy) | 174 GB |
| 4 | `qwen35_122b_fp8_tp2x2` (+ `ENABLE_EP=1 RUN_TAG=ep`), then `cotenant_tp2_gpu01` + `train/lora_cotenant.sh` + `train/sleep_wake.sh` -> **delete** `Qwen3.5-122B-A10B-FP8`, `Qwen3-8B`, `Qwen3.8-27B-FP8` | DeepSeek-V4-Flash-0731 (167) once >= 187 GB free | 167 GB |
| 5 | `ds4flash_tp4` (+ `ENABLE_EP=0 RUN_TAG=noep`), `ds4flash_tp2x2` -> **delete** `DeepSeek-V4-Flash-0731` | Qwen3.8-Flash-Next-FP8 (186) only after the delete (167 + 186 = 353 GB would leave < 20 GB) | 186 GB |
| 6 | `qwen38flashnext_fp8_tp4`, then `qwen38flashnext_fp8_tp2x2` -> **delete** | — | 0 |
| 7 | `glm53flash_fp8_tp4_loadtest` **only** with a build that registers `Glm5NextForConditionalGeneration` (recipe image); otherwise skip and record the registry check | GLM-5.3-Flash (328) alone | 328 GB |

Delete a model directory when disk is needed (the `.complete` marker from `onstart.sh` goes
with it, so a later `prefetch.sh`/`download_models.sh` re-downloads cleanly):

```bash
bench/prefetch.sh --delete qwen38_27b_bf16_x4 --yes   # removes $MODELS_DIR/<basename of the cell's MODEL>
rm -rf /workspace/models/Qwen3.8-27B            # the same by hand (basename of the HF id)
df -h /workspace
```

Re-download with `bench/prefetch.sh <cell>` (or `/workspace/tools/download_models.sh <key>`),
which writes `--local-dir /workspace/models/<basename>`. `bench/launch.sh <cell> --dry-run`
prints the exact `vllm serve` line(s) a cell resolves to without touching the GPUs or `results/`.

## Per-launch overrides that the cells honour

| variable | cells | effect |
|---|---|---|
| `KV_CACHE_DTYPE`, `MAX_MODEL_LEN`, `MAX_NUM_SEQS`, `MAX_NUM_BATCHED_TOKENS`, `GPU_MEM_UTIL`, `ENABLE_EP`, `RUN_TAG` | all | the usual `${VAR:-default}` knobs (`MAX_NUM_BATCHED_TOKENS=16384 RUN_TAG=mnbt16k`) |
| `REPLICAS`, `GPU_IDS` | the four x4 cells | `REPLICAS=1 GPU_IDS=0` = one replica on one GPU (what `gates/run_kv_diff.sh` does for its deterministic capture); one cell per process -- `load_cell` exports them |
| `MOE_BACKEND` | ds4flash_*, gptoss120b_* | `--moe-backend` value. MXFP4-valid spellings in vLLM main: `auto b12x marlin triton triton_unfused flashinfer_trtllm flashinfer_cutlass deep_gemm ...`; **`flashinfer_b12x` is not valid for MXFP4** (raises "moe_backend='flashinfer_b12x' is not supported for MXFP4 MoE") |
| `DS4_LEGACY_INDEXER_FLAG=0` | ds4flash_* | drop the recipe's `--attention_config.use_fp4_indexer_cache False` (deprecated no-op alias of `indexer_kv_dtype`, removal announced for v0.29) |
| `DS4_BENCH_TOKENIZER_MODE=deepseek_v4` | ds4flash_* | add `--tokenizer-mode deepseek_v4` to the bench client if the HF auto tokenizer fails. `vllm bench serve --tokenizer-mode` has **no argparse choices** and is resolved through `vllm/tokenizers/registry.py`, where `deepseek_v4` is registered (checked 2026-09-02); that the client path then loads the V4 tokenizer files cleanly is UNVERIFIED |
| `DS4_BENCH_TOKENIZER=<dir>` | ds4flash_* | override the bench client's `--tokenizer` with another local tokenizer directory (last occurrence wins) |
| `COTENANT_MODEL`, `COTENANT_ALIAS`, `COTENANT_GPU_IDS` | cotenant_tp2_gpu01 | lighter model (`Qwen/Qwen3.8-27B-FP8`), served name, GPU pair (`0,1` for a same-switch A/B) |
| `VLLM_PLE_CPU_OFFLOAD` | qwen38flashnext_* | exported `=1` by default (recipe); see the UNVERIFIED note below |
| `HOST_RAM_NEEDED_GB` | qwen38flashnext_* (204) | hint for launch.sh's host-RAM warning (never fatal): one ~51 GB PLE copy per TP worker if offloaded |
| `HEALTH_TIMEOUT` | all (glm53 loadtest: 5400) | seconds launch.sh waits for `/health`; `load_cell` applies the 7200 default |
| `ENABLE_SLEEP_MODE` | cotenant_tp2_gpu01 | `--enable-sleep-mode` (default 1) |
| `SPEC_CONFIG` | any | speculative decoding JSON; **off everywhere** in this campaign |

## What was verified against vLLM main (2026-09-02) and what was not

Verified (file references are in the cell headers): `--moe-backend` is a top-level flag
(`KernelConfig.moe_backend`); `--language-model-only`; `--kv-cache-dtype` values `fp8`,
`fp8_ds_mla`, `bfloat16`; `--tokenizer-mode deepseek_v4` (server literal + tokenizer
registry; the bench client's `--tokenizer-mode` is a plain `type=str` argument resolved through the
same registry, so `deepseek_v4` is accepted -- loading is UNVERIFIED); tool parsers `openai`, `deepseek_v4`,
`qwen3_xml`, `qwen3_coder`, `glm47`, `glm45`; reasoning parsers `deepseek_v4`, `qwen3`,
`glm45`, `openai_gptoss`; the dotted `--attention_config.use_fp4_indexer_cache False` form
(FlexibleArgumentParser repacks it as JSON) and that the field is a deprecated no-op alias;
the sm_120 sparse-MLA backend (`FLASHINFER_MLA_SPARSE_SM120`: kv dtypes
`auto|fp8|fp8_e4m3|fp8_ds_mla`, block sizes 64/256, `fp8` an alias of `fp8_ds_mla` for
DeepSeek-V4); DP rank -> GPU slice order; `Qwen4ExpForConditionalGeneration` and
`DeepseekV4ForCausalLM` registered; `Glm5NextForConditionalGeneration` **not** registered.

UNVERIFIED (2026-09-02), marked in the cells:

* `VLLM_PLE_CPU_OFFLOAD` — recipe env for Qwen3.8-Flash-Next; absent from the inspected
  g798544433 PLE implementation. **2026-09-05 correction:** GPU embedding weights are
  TP row-sharded. Native FP8 TP4+EP4 loaded at 44.35 GiB per GPU without PLE offload;
  the earlier 114 GB/GPU TP2 and replicated-table estimates were wrong. TP2 memory
  also depends on whether experts remain distributed across all four GPUs. See the
  [source and memory audit](../analysis/qwen_native_fp8_audit_20260905.md).
* Which MXFP4 MoE backend `auto` picks on sm_120 (`TRITON` vs `MARLIN`; the gpt-oss cells pin
  their backend explicitly, the DeepSeek cells use `auto` like the recipe) and whether
  `flashinfer_cutlass` / `b12x` pass their `is_supported_config()` on compute capability 12.0.
* `--quantization-config.moe.activation mxfp8` (recipe fix for an SM100 attention-sinks error)
  on sm_120 — comment only, not passed.
* `--cpu-offload-gb` as a fallback for the PLE table — comment only.
