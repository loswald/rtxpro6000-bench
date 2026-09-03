#!/usr/bin/env bash
# Provision a fresh RTX PRO 6000 box: stack, patch, hardware truth. Run ON the box as root.
set +e
export DEBIAN_FRONTEND=noninteractive
log(){ echo "[$(date +%H:%M:%S)] $*"; }
mkdir -p /workspace/bench /workspace/results/hwtruth /workspace/results/smoke /workspace/results/probe /workspace/models

log "apt basics"
apt-get update -qq >/dev/null 2>&1
apt-get install -y -qq tmux git jq rsync pciutils curl >/dev/null 2>&1

log "uv"
command -v uv >/dev/null || pip install -q uv >/dev/null 2>&1
uv --version

log "vLLM nightly cu130 + flashinfer 0.6.18 + b12x"
uv pip install --system --prerelease=allow --index-strategy unsafe-best-match --torch-backend=cu130 \
  vllm --extra-index-url https://wheels.vllm.ai/nightly/cu130 > /workspace/results/upgrade.log 2>&1
echo "vllm rc=$?"
uv pip install --system -q "flashinfer-python==0.6.18" >> /workspace/results/upgrade.log 2>&1
uv pip install --system -q "flashinfer-jit-cache==0.6.18" --extra-index-url https://flashinfer.ai/whl/cu130 >> /workspace/results/upgrade.log 2>&1
uv pip install --system -q "flashinfer-cubin==0.6.18" --extra-index-url https://flashinfer.ai/whl >> /workspace/results/upgrade.log 2>&1 || uv pip uninstall --system -q flashinfer-cubin >> /workspace/results/upgrade.log 2>&1
uv pip install --system -q b12x >> /workspace/results/upgrade.log 2>&1
uv pip install --system -q "huggingface_hub[cli]" hf_transfer >> /workspace/results/upgrade.log 2>&1
python3 - <<'PY'
import importlib
for m in ("torch","vllm","flashinfer","b12x"):
    try:
        mod=importlib.import_module(m); print(m, getattr(mod,"__version__","(ok)"))
    except Exception as e: print(m, "FAIL", type(e).__name__, str(e)[:120])
import torch; print("cuda", torch.version.cuda, "nccl", torch.cuda.nccl.version(), "archs", torch.cuda.get_arch_list()[-2:])
PY

log "hardware truth"
{
echo "== gpus =="
nvidia-smi --query-gpu=index,name,driver_version,vbios_version,pcie.link.gen.max,pcie.link.gen.current,pcie.link.width.current,power.limit,power.max_limit,memory.total,ecc.mode.current --format=csv
echo "== topology =="; nvidia-smi topo -m
echo "== host =="; nproc; free -g | head -2; df -h /workspace / | tail -2; lscpu | grep -E "Model name|NUMA node\(s\)|Flags" | cut -c1-200
echo "== ACS on bridges =="; lspci -vvv 2>/dev/null | grep -E "^[0-9a-f:.]+ |ACSCtl" | grep -B1 ACSCtl | head -30
} > /workspace/results/hwtruth/hardware.txt 2>&1
head -12 /workspace/results/hwtruth/hardware.txt
