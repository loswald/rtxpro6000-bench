# GLM baseline and FA2 diagnostic on the independent test node

All commands below target a dedicated test workspace. Root owns GPU execution. They do not use AIRR's `hardkill` or chain scripts and must not run on either existing shared host.

## Pins and downloads

- Checkpoint: `RedHatAI/GLM-5.3-Flash-NVFP4`, revision `36c184c6cda000a481711306df5adde42f63321a`.
- Vendor image: `vllm/vllm-openai@sha256:2e771fa615452282cc331eb418b3ef21636fce355bea0491fca89e6d362ab703`; vendor vLLM version `0.1.dev20051+g487ecf187`.
- Captured working dependencies: Torch `2.13.0+cu130`, Transformers `5.15.1`, FlashInfer `0.6.18`, compressed-tensors `0.17.0`, Triton `3.7.1`. B12X `1.3.0` was installed, but these GLM runs selected `FLASHINFER_CUTLASS` NVFP4 MoE.
- **Import only the vendor `vllm` package.** Its image carries FlashInfer `0.6.17`; the working environment's `0.6.18` supplies the NoPE FA2 support. Record the actual complete environment before benchmarking; these core version pins alone are not a full lockfile.

```bash
hf download RedHatAI/GLM-5.3-Flash-NVFP4 \
  --revision 36c184c6cda000a481711306df5adde42f63321a \
  --local-dir /workspace/models/GLM-5.3-Flash-NVFP4

python3 /workspace/bench/pull_image.py \
  vllm/vllm-openai@sha256:2e771fa615452282cc331eb418b3ef21636fce355bea0491fca89e6d362ab703 \
  /workspace/glmimg-pinned --want=vllm
```

The existing extractor requires `--want=vllm`, including the equals sign. Its `EXTRACTED` message is not sufficient proof that every layer succeeded; inspect the pull log and verify the package/source below. The digest comes from the vendor-image reproducer; the backend hash supplies the exact-code check for this experiment.

## Preserve a baseline and create the diagnostic copy

```bash
set -euo pipefail
glm_vendor=/workspace/glmimg-pinned/usr/local/lib/python3.12/dist-packages/vllm
glm_base=/workspace/glm-vendor-baseline
glm_audit=/workspace/glm-vendor-audit
glm_rel=v1/attention/backends/mla/flashinfer_mla_sparse_sm90.py
test -d "$glm_vendor"
test ! -e "$glm_base"
test ! -e "$glm_audit"
mkdir "$glm_base" "$glm_audit"
cp -a "$glm_vendor" "$glm_base/vllm"
python3 /workspace/bench/vllm_sm120_nope.py "$glm_base/vllm"
printf '%s  %s\n' \
  7a19dafb16f1a2f9ac58992ce78e4d27b8f52edf08059c387d4f32d70d0edab3 \
  "$glm_base/vllm/$glm_rel" | sha256sum -c -
cp -a "$glm_base/vllm" "$glm_audit/vllm"
python3 /workspace/bench/glm_fa2_plan_audit.py \
  --source "$glm_base/vllm/$glm_rel" \
  --output /workspace/glm_fa2_diagnostic.py
cp /workspace/glm_fa2_diagnostic.py "$glm_audit/vllm/$glm_rel"
```

If the baseline hash differs, stop and inspect the actual source difference. The generator deliberately cannot patch a different version. The original extracted package and baseline copy remain intact. The current emitted module is available locally at `analysis/glm_fa2_diagnostic/flashinfer_mla_sparse_sm90.py`, with its own manifest. The earlier `report/glm_fa2_diagnostic/` copy lacks the corrected current-metadata guard and must not be used. All 30 regressions, including 12 Torch CPU tensor tests, pass in an isolated Torch `2.8.0+cpu` environment. The command is `python -m unittest tests.test_glm_fa2_plan_audit -v` when Torch is available.

## Common launch and matched layouts

Set these in the test server's environment:

```bash
export VLLM_USE_DEEP_GEMM=0 VLLM_MOE_USE_DEEP_GEMM=0
export FLASHINFER_CUDA_ARCH_LIST=12.0f TORCH_CUDA_ARCH_LIST=12.0
export HF_HUB_OFFLINE=1 VLLM_ENGINE_READY_TIMEOUT_S=3600
export MAX_JOBS=6 NVCC_THREADS=2
```

Baseline candidate launch:

```bash
PYTHONPATH=/workspace/glm-vendor-baseline \
python3 -m vllm.entrypoints.openai.api_server \
  --model /workspace/models/GLM-5.3-Flash-NVFP4 --served-model-name m \
  --host 0.0.0.0 --port 8000 \
  --tensor-parallel-size 1 --data-parallel-size 4 --enable-expert-parallel \
  --attention-backend FLASHINFER_MLA_SPARSE_SM90 \
  --kv-cache-dtype auto --block-size 1024 --max-model-len 40960 \
  --max-num-seqs 192 --max-num-batched-tokens 16384 --gpu-memory-utilization 0.90 \
  --enable-prefix-caching --trust-remote-code --disable-custom-all-reduce \
  --reasoning-parser glm45 --tool-call-parser glm47 --enable-auto-tool-choice \
  --no-enable-flashinfer-autotune --disable-uvicorn-access-log
```

Use TP2, DP2, and `--max-num-seqs 384` for the historical failing layout, retaining EP. Its effective EP group is four ranks. Do not add speculation, change KV precision, reduce top-k, or change the checkpoint. The current matched chat experiment uses AIRR's corrected `glm45` reasoning parser on **both** layouts. The historical performance/smoke launch omitted parsers, so record this difference if comparing to the archived outputs. The original `box/quality20.py --mode chat --max-tokens 1024` is available for reproducing its exact old tripwire, but that tripwire can pass capped outputs and is not a quality acceptance gate.

## Activate count checking only after eager startup

For the check arm, select `/workspace/glm-vendor-audit`, append `--enforce-eager` to the same TP2/DP2 launch, and set these **before** starting workers:

```bash
mkdir -p /workspace/glm-fa2-markers
test ! -e /workspace/glm-fa2-markers/check.ready
export GLM_FA2_AUDIT_MARKER=/workspace/glm-fa2-markers/check.ready
export GLM_FA2_AUDIT_MODE=check
export GLM_FA2_AUDIT_LOG_DIR=/workspace/results/glm_fa2_check
export GLM_FA2_AUDIT_MAX_RECORDS=256
```

After the health check succeeds and initialization finishes:

```bash
touch /workspace/glm-fa2-markers/check.ready
python3 /workspace/bench/chat_probe.py \
  --model m --base-url http://127.0.0.1:8000 \
  --request-config /workspace/bench/glm53f_vendor_probe.json \
  --prompts-source /workspace/bench/quality20.py \
  --out-dir /workspace/results/glm_dp2_fa2_check_probe \
  --concurrency 10 --base-seed 1234
```

For the optional exact-count arm, restart the same eager configuration with `GLM_FA2_AUDIT_MODE=exact`, a new absent marker such as `exact.ready`, a fresh log directory, and a fresh probe output directory. Create that marker only after readiness. Never reuse an existing marker at startup; dummy profiling rows must remain outside the check.

Check mode records then stops before attention on the first mismatch. A pre-converter guard first validates current batch metadata, query capacity, cache geometry, and request-ID bounds. Exact mode replans using the exact converted live counts while preserving indices, top-k, masks, independently confirmed active/padded query counts, and values. It still rejects malformed prefixes, physical slot bounds, zero-count live rows, and metadata/geometry mismatches. Both arms add CPU synchronization and are unsuitable for throughput measurement. See `patches/glm_fa2_plan_audit.md` for interpretation. A recovered smoke result remains a candidate for the full paired 403-item gate.
