#!/usr/bin/env bash
# =============================================================================
# train/install_unsloth.sh
#
# Creates an ISOLATED virtualenv for Unsloth training so the vLLM container's
# torch / transformers / flashinfer pins are never touched (vLLM keeps serving on the
# co-tenancy cell's GPUs -- default 0,2 -- while this env trains on GPU 3).  Nothing here pip-installs into the
# image interpreter: uv is used if present (the box upgraded vLLM with uv), else the
# standalone uv installer into ~/.local/bin, else `python3 -m venv` + pip.
#
# Verified against Unsloth's Blackwell guide (unsloth.ai/docs/blog/fine-tuning-llms-
# with-blackwell-rtx-50-series-and-unsloth, read 2026-09-02):
#   * Blackwell (sm_120: RTX 50xx, RTX PRO 6000) needs torch wheels built with
#     CUDA 12.8+ -> `uv pip install ... --torch-backend=cu128` / pip `--extra-index-url
#     https://download.pytorch.org/whl/cu128`.  cu129/cu130 wheels also carry sm_120.
#   * `uv pip install unsloth unsloth_zoo bitsandbytes`, `uv pip install -U transformers`
#   * "triton>=3.3.1 is required for Blackwell support"
#   * xformers is OPTIONAL ("faster and uses less memory"); building it needs
#     TORCH_CUDA_ARCH_LIST="12.0" + ninja.  Skipped by default -> PyTorch SDPA.
#   * Reported (unsloth#2686, not in the guide): "CUDA driver error: unknown error" from
#     Unsloth's offloaded gradient checkpointing on RTX 6000 Pro with some torch/triton
#     combos -> lora_qwen8b.py exposes TRAIN_GRAD_CKPT=unsloth|true|false to route around it.
#
# Env: UNSLOTH_VENV=/opt/unsloth-venv  TORCH_BACKEND=cu128 (or cu129/cu130)
#      PYTHON=python3  BUILD_XFORMERS=0|1  UNSLOTH_EXTRA_PIP="<extra packages>"
#      FORCE_REINSTALL=0|1 (1 = recreate the venv)
# Exit: 0 ok, 1 install/verify failed
# =============================================================================
set -euo pipefail

UNSLOTH_VENV="${UNSLOTH_VENV:-/opt/unsloth-venv}"
TORCH_BACKEND="${TORCH_BACKEND:-cu128}"
PYTHON="${PYTHON:-python3}"
BUILD_XFORMERS="${BUILD_XFORMERS:-0}"
UNSLOTH_EXTRA_PIP="${UNSLOTH_EXTRA_PIP:-}"
FORCE_REINSTALL="${FORCE_REINSTALL:-0}"
PKGS=(unsloth unsloth_zoo bitsandbytes datasets trl transformers accelerate peft sentencepiece protobuf hf_transfer)

log() { printf '[install_unsloth %s] %s\n' "$(date -u +%H:%M:%S)" "$*" >&2; }
VPY="$UNSLOTH_VENV/bin/python"

venv_ok() {  # imports + sm_120 in torch's arch list
  [[ -x "$VPY" ]] && "$VPY" - <<'PY' >/dev/null 2>&1
import sys, torch, unsloth, trl, datasets  # noqa: F401
arch = torch.cuda.get_arch_list()
ok = torch.cuda.is_available() and any(a in ("sm_120", "compute_120", "sm_121") for a in arch)
sys.exit(0 if ok else 1)
PY
}

find_uv() {
  command -v uv 2>/dev/null && return 0
  for c in "$HOME/.local/bin/uv" "$HOME/.cargo/bin/uv" /usr/local/bin/uv; do [[ -x "$c" ]] && { echo "$c"; return 0; }; done
  return 1
}

if [[ "$FORCE_REINSTALL" == 1 ]]; then rm -rf "$UNSLOTH_VENV"; fi
if venv_ok; then
  log "existing env at $UNSLOTH_VENV already has unsloth + an sm_120-capable torch; verifying only"
else
  UV="$(find_uv || true)"
  if [[ -z "$UV" ]]; then
    log "uv not found; trying the standalone installer into ~/.local/bin (does not touch the image's Python)"
    if curl -LsSf https://astral.sh/uv/install.sh | env UV_NO_MODIFY_PATH=1 sh >/dev/null 2>&1; then UV="$(find_uv || true)"; fi
  fi
  ARCH_TARGET="${UNSLOTH_VENV}"
  if [[ -n "$UV" ]]; then
    log "uv: $UV ($("$UV" --version 2>/dev/null || echo '?'))"
    [[ -x "$VPY" ]] || "$UV" venv "$ARCH_TARGET" --python "$PYTHON" --seed
    log "torch + torchvision with --torch-backend=$TORCH_BACKEND (sm_120 requires cu128+)"
    "$UV" pip install --python "$VPY" -U torch torchvision --torch-backend="$TORCH_BACKEND"
    "$UV" pip install --python "$VPY" -U "triton>=3.3.1"
    log "unsloth + training stack (same --torch-backend so the resolver cannot swap torch for a PyPI cu12x wheel)"
    # shellcheck disable=SC2086
    "$UV" pip install --python "$VPY" -U --torch-backend="$TORCH_BACKEND" "${PKGS[@]}" ${UNSLOTH_EXTRA_PIP}
    PIPX=("$UV" pip install --python "$VPY")
  else
    log "no uv: python3 -m venv + pip with the PyTorch $TORCH_BACKEND index"
    [[ -x "$VPY" ]] || "$PYTHON" -m venv "$ARCH_TARGET" || { log "python3 -m venv failed (apt-get install -y python3-venv)"; exit 1; }
    "$VPY" -m pip install -q -U pip
    "$VPY" -m pip install -U torch torchvision --index-url "https://download.pytorch.org/whl/$TORCH_BACKEND"
    "$VPY" -m pip install -U "triton>=3.3.1"
    # shellcheck disable=SC2086
    "$VPY" -m pip install -U "${PKGS[@]}" ${UNSLOTH_EXTRA_PIP} --extra-index-url "https://download.pytorch.org/whl/$TORCH_BACKEND"
    PIPX=("$VPY" -m pip install)
  fi

  if [[ "$BUILD_XFORMERS" == "1" ]]; then
    log "building xformers from source for sm_120 (TORCH_CUDA_ARCH_LIST=12.0; needs nvcc + ~16 GB RAM)"
    "${PIPX[@]}" ninja
    export TORCH_CUDA_ARCH_LIST="12.0"
    tmp="$(mktemp -d)"
    git clone --depth=1 --recursive https://github.com/facebookresearch/xformers "$tmp/xformers"
    ( cd "$tmp/xformers" && "$VPY" setup.py install )
    rm -rf "$tmp"
  else
    # never let a wheel built without sm_120 shadow SDPA
    if [[ -n "${UV:-}" ]]; then "$UV" pip uninstall --python "$VPY" xformers >/dev/null 2>&1 || true
    else "$VPY" -m pip uninstall -y xformers >/dev/null 2>&1 || true; fi
  fi
fi

log "verifying $VPY"
"$VPY" - <<'PY'
import json, sys
import torch
info = {"python": sys.version.split()[0], "torch": torch.__version__, "cuda": torch.version.cuda, "arch_list": torch.cuda.get_arch_list()}
ok = torch.cuda.is_available()
info["cuda_available"] = ok
if ok:
    info["device"] = torch.cuda.get_device_name(0)
    info["capability"] = torch.cuda.get_device_capability(0)
for mod in ("unsloth", "unsloth_zoo", "trl", "transformers", "peft", "datasets", "triton", "bitsandbytes", "xformers"):
    try:
        m = __import__(mod)
        info[mod] = getattr(m, "__version__", "present")
    except Exception as e:  # noqa: BLE001
        info[mod] = f"MISSING ({type(e).__name__})"
print(json.dumps(info, indent=2))
if not ok:
    sys.exit("CUDA not available inside the unsloth venv")
if info["capability"][0] == 12 and not any(a in ("sm_120", "compute_120", "sm_121") for a in info["arch_list"]):
    sys.exit("torch build lacks sm_120 in get_arch_list(); use TORCH_BACKEND=cu128 or newer")
for mod in ("unsloth", "trl", "datasets"):
    if str(info[mod]).startswith("MISSING"):
        sys.exit(f"{mod} missing in the venv")
PY
log "done: $UNSLOTH_VENV  (python: $VPY)"
