#!/usr/bin/env bash
# Provision a fresh 4x RTX PRO 6000 box (5 Sept 2026 recipe). Run ON the box as root, once, from a tmux session:
# apt basics, uv, vLLM nightly cu130 with the b12x extra, FlashInfer 0.6.18 with its jit-cache and the matching
# cubin from FlashInfer's ROOT index (the cu130 index does not carry 0.6.18 cubins), hf cli + hf_transfer, then
# the hardware truth that decides whether any throughput number from this box transfers: the power limit.
set +e
export DEBIAN_FRONTEND=noninteractive
log(){ echo "[$(date +%H:%M:%S)] $*"; }
mkdir -p /workspace/bench/lists /workspace/results/hwtruth /workspace/results/smoke /workspace/results/probe /workspace/results/eval /workspace/results/kl /workspace/models
log "apt basics"; apt-get update -qq >/dev/null 2>&1; apt-get install -y -qq tmux git jq rsync pciutils curl >/dev/null 2>&1
log "uv"; command -v uv >/dev/null || pip install -q uv >/dev/null 2>&1; uv --version
log "vLLM nightly cu130 [b12x] + FlashInfer 0.6.18"
IDX="--extra-index-url https://wheels.vllm.ai/nightly/cu130 --extra-index-url https://flashinfer.ai/whl/cu130 --index-strategy unsafe-best-match --torch-backend cu130"
uv pip install --system --pre "vllm[b12x]" "flashinfer-python==0.6.18" "flashinfer-jit-cache==0.6.18" $IDX > /workspace/results/upgrade.log 2>&1; echo "  vllm rc=$?"
FV=$(python3 -c "import importlib.metadata as m; print(m.version('flashinfer-python'))" 2>/dev/null)
uv pip install --system --no-deps "flashinfer-cubin==$FV" --index-url https://flashinfer.ai/whl/ >> /workspace/results/upgrade.log 2>&1; echo "  cubin $FV rc=$?"
uv pip install --system -q "huggingface_hub[cli]" hf_transfer >> /workspace/results/upgrade.log 2>&1
python3 - <<'PY'
import importlib
for m in ("torch","vllm","flashinfer","b12x"):
    try: mod=importlib.import_module(m); print("  ", m, getattr(mod,"__version__","(ok)"))
    except Exception as e: print("  ", m, "FAIL", type(e).__name__, str(e)[:120])
import torch; print("   cuda", torch.version.cuda, "available", torch.cuda.is_available(), "gpus", torch.cuda.device_count())
PY
log "hardware truth"
{
echo "== gpus =="; nvidia-smi --query-gpu=index,name,driver_version,pcie.link.gen.current,pcie.link.width.current,power.limit,power.max_limit,memory.total --format=csv
echo "== topology =="; nvidia-smi topo -m
echo "== host =="; nproc; free -g | head -2; df -h /workspace | tail -1; lscpu | grep -E "Model name|NUMA node\(s\)" | cut -c1-120
echo "== ACS on bridges =="; lspci -vvv 2>/dev/null | grep -E "^[0-9a-f:.]+ |ACSCtl" | grep -B1 ACSCtl | head -20
} > /workspace/results/hwtruth/hardware.txt 2>&1
head -8 /workspace/results/hwtruth/hardware.txt
log "download speed check (one 1 GB shard from the Hub)"
python3 -c "import time,urllib.request; t=time.time(); n=0
r=urllib.request.urlopen('https://huggingface.co/Qwen/Qwen3.8-27B-FP8/resolve/main/model-00001-of-00007.safetensors',timeout=60)
while n<1_000_000_000:
    b=r.read(16<<20)
    if not b: break
    n+=len(b)
print('   %.0f MB in %.1fs = %.0f MB/s'%(n/1e6,time.time()-t,n/1e6/(time.time()-t)))" 2>&1 | tail -1
log "PROVISION2 DONE"
