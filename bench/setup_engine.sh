#!/usr/bin/env bash
# =============================================================================
# bench/setup_engine.sh — one-time, idempotent container prep (run as root inside
# the Vast.ai container).  Installs tmux/curl if missing, puts the `hf` CLI into an
# ISOLATED venv (/workspace/venv-tools -- never the engine interpreter), optionally
# performs the documented in-place vLLM upgrade, adds the `vllm[b12x]` extra (the
# `b12x` MXFP4 MoE backend for sm_120) pinned to the installed vLLM, and prints the
# engine/torch/flashinfer versions plus the models on disk.
#
# The campaign box runs vLLM main 0.28.1rc1.dev312+g41848caa6 (CUDA 13.0 build) installed
# IN PLACE with uv from https://wheels.vllm.ai/nightly/cu130 over the stale
# vllm/vllm-openai:cu130-nightly image (0.19.2), plus flashinfer-python 0.6.18 and vllm[b12x].
# Rules this script enforces:
#   * NOTHING here runs `pip install` into the engine interpreter: a plain PyPI resolve drags
#     torch/vllm back to PyPI (cu12x) builds.  Engine changes go through `uv pip install --system`
#     with the nightly wheel index, `--index-strategy unsafe-best-match` (the pinned nightly version
#     is searched across all indexes; extra indexes rank above PyPI in uv) and `--torch-backend cu130`
#     (torch/torchvision stay on the PyTorch cu130 index).  Without uv the engine is left untouched.
#   * The upgrade itself runs only with UPGRADE_VLLM=1 and is skipped when the installed version
#     already equals VLLM_TARGET_VERSION (FORCE_UPGRADE=1 re-runs it).
# Env: UPGRADE_VLLM=1  VLLM_TARGET_VERSION=0.28.1rc1.dev312+g41848caa6 (empty string = latest nightly)
#      FLASHINFER_VERSION=0.6.18  VLLM_WHEEL_INDEX=https://wheels.vllm.ai/nightly/cu130
#      UV_TORCH_BACKEND=cu130 (none = do not pass --torch-backend)  UV_INDEX_STRATEGY=unsafe-best-match
#      SKIP_B12X=1 (leave the b12x extra alone)  TOOLS_VENV=/workspace/venv-tools  FORCE_UPGRADE=1
# uv flags checked against docs.astral.sh (2026-09-02): --system --pre --extra-index-url
# --index-strategy {first-index,unsafe-first-match,unsafe-best-match} --torch-backend cuXXX|auto.
# =============================================================================
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
. "$HERE/env.sh"

VLLM_WHEEL_INDEX="${VLLM_WHEEL_INDEX:-https://wheels.vllm.ai/nightly/cu130}"
VLLM_TARGET_VERSION="${VLLM_TARGET_VERSION-0.28.1rc1.dev312+g41848caa6}"
FLASHINFER_VERSION="${FLASHINFER_VERSION:-0.6.18}"
UV_TORCH_BACKEND="${UV_TORCH_BACKEND:-cu130}"
UV_INDEX_STRATEGY="${UV_INDEX_STRATEGY:-unsafe-best-match}"
TOOLS_VENV="${TOOLS_VENV:-/workspace/venv-tools}"

# engine_uv <pkg-spec...>: the ONLY path that touches the engine interpreter (uv, never pip).
engine_uv() {
  if ! have uv; then
    log "WARN: uv not found -> engine left untouched (install: curl -LsSf https://astral.sh/uv/install.sh | sh). Refusing to run pip on the engine interpreter."
    return 1
  fi
  local extra=()
  if [ "$UV_TORCH_BACKEND" != none ]; then extra+=( --torch-backend "$UV_TORCH_BACKEND" ); fi
  log "uv pip install --system --pre $* --extra-index-url $VLLM_WHEEL_INDEX --index-strategy $UV_INDEX_STRATEGY ${extra[*]:-}"
  uv pip install --system --pre "$@" --extra-index-url "$VLLM_WHEEL_INDEX" --index-strategy "$UV_INDEX_STRATEGY" ${extra[@]+"${extra[@]}"}
}
vllm_version() { python3 -c 'import vllm; print(vllm.__version__)' 2>/dev/null || echo none; }

if ! have tmux || ! have curl; then
  log "installing tmux/curl"
  (apt-get update -qq && apt-get install -y -qq tmux curl >/dev/null) || log "WARN: apt-get failed; install tmux+curl manually"
fi

# ---- hf CLI: isolated venv, symlinked to /usr/local/bin/hf (same layout as vast/onstart.sh) ----
if ! have hf && ! have huggingface-cli; then
  log "installing the hf CLI into the isolated venv $TOOLS_VENV (never the engine interpreter)"
  if [ -x "$TOOLS_VENV/bin/python" ] || python3 -m venv "$TOOLS_VENV" 2>/dev/null || { have uv && uv venv -q --seed "$TOOLS_VENV"; }; then
    if "$TOOLS_VENV/bin/python" -m pip install -q -U 'huggingface_hub[cli]>=0.34' hf_transfer >/dev/null 2>&1 \
       || { have uv && uv pip install -q --python "$TOOLS_VENV/bin/python" 'huggingface_hub[cli]>=0.34' hf_transfer; }; then
      ln -sfn "$TOOLS_VENV/bin/hf" /usr/local/bin/hf 2>/dev/null || true
      log "hf CLI -> /usr/local/bin/hf ($("$TOOLS_VENV/bin/hf" version 2>/dev/null | head -1 || echo '?'))"
    else
      log "WARN: hf CLI install failed (prefetch.sh falls back to python snapshot_download)"
    fi
  else
    log "WARN: could not create $TOOLS_VENV (apt-get install -y python3-venv, or install uv)"
  fi
fi

# ---- the documented in-place upgrade (opt-in) ----------------------------------------------
cur_ver="$(vllm_version)"
if [ "${UPGRADE_VLLM:-0}" = 1 ]; then
  if [ -n "$VLLM_TARGET_VERSION" ] && [ "$cur_ver" = "$VLLM_TARGET_VERSION" ] && [ "${FORCE_UPGRADE:-0}" != 1 ]; then
    log "vLLM $cur_ver already == VLLM_TARGET_VERSION -> upgrade skipped (FORCE_UPGRADE=1 to redo)"
  else
    spec="vllm[b12x]${VLLM_TARGET_VERSION:+==$VLLM_TARGET_VERSION}"
    log "UPGRADE_VLLM=1: vLLM $cur_ver -> ${VLLM_TARGET_VERSION:-latest nightly} (+ flashinfer-python==$FLASHINFER_VERSION) from $VLLM_WHEEL_INDEX"
    engine_uv "$spec" "flashinfer-python==$FLASHINFER_VERSION" || log "WARN: upgrade did not complete; the engine may be unchanged -- check the versions below"
    cur_ver="$(vllm_version)"
  fi
fi

# ---- engine present? add the b12x extra (pinned, uv only) -------------------------------------
if [ "$cur_ver" != none ]; then
  log "vLLM $cur_ver detected"
  if [ "${SKIP_B12X:-0}" != 1 ] && ! python3 -c 'import b12x' 2>/dev/null; then
    # Pinned to the installed version so the resolver cannot move vllm/torch; nightly versions are
    # only on the wheel index (never PyPI), hence the index strategy in engine_uv.
    if engine_uv "vllm[b12x]==$cur_ver"; then
      log "vllm[b12x] extra installed (pinned to $cur_ver)"
    else
      log "WARN: could not add vllm[b12x]==$cur_ver with uv; the gptoss120b_x4_b12x cell will die at start and record it. SKIP_B12X=1 silences this."
    fi
  elif python3 -c 'import b12x' 2>/dev/null; then
    log "b12x $(python3 -c 'import b12x; print(getattr(b12x, "__version__", "?"))') present"
  fi
  if ! python3 -c 'import flashinfer' 2>/dev/null; then
    log "WARN: flashinfer not importable — FlashInfer attention/MoE backends unavailable (UPGRADE_VLLM=1 installs flashinfer-python==$FLASHINFER_VERSION)"
  fi
elif python3 -c 'import sglang' 2>/dev/null; then
  log "SGLang $(python3 -c 'import sglang; print(sglang.__version__)') detected (A/B instance)."
  log "The sweep client is 'vllm bench serve'; install a client-only vLLM in a separate venv (uv venv + uv pip install vllm) or set BENCHMARK_SERVING_PY"
else
  die "neither vllm nor sglang importable — wrong image?"
fi

# ---- report ------------------------------------------------------------------------------------
mkdir -p "$RESULTS_ROOT" "$MODELS_DIR"
log "MODELS_DIR=$MODELS_DIR ($(df -h "$MODELS_DIR" 2>/dev/null | awk 'NR==2{print $4" free of "$2}'))"
for d in "$MODELS_DIR"/*/; do
  [ -d "$d" ] || continue
  d="${d%/}"; log "  model dir: $(du -sh "$d" 2>/dev/null | cut -f1) $(model_dir_state "$d") $d"
done
log "GPUs: $(nvidia-smi --query-gpu=name,compute_cap --format=csv,noheader 2>/dev/null | sort -u | tr '\n' ';')"
log "host RAM: $(free -g 2>/dev/null | awk '/^Mem:/{print $2" GB total, "$7" GB available"}')"
python3 - <<'PY' || true
import importlib
for m in ("vllm", "sglang", "flashinfer", "b12x", "torch", "transformers", "triton", "huggingface_hub"):
    try:
        mod = importlib.import_module(m)
        print(f"  {m:16s} {getattr(mod, '__version__', '?')}")
    except Exception as e:  # noqa: BLE001
        print(f"  {m:16s} - ({type(e).__name__})")
try:
    import torch
    print("  torch.cuda:", torch.version.cuda, "arch list:", torch.cuda.get_arch_list())
except Exception:
    pass
PY
if [ -f "$HW_DECISIONS_FILE" ]; then
  log "decisions: $HW_DECISIONS_FILE -> P2P_OK=$P2P_OK CUSTOM_ALLREDUCE=$CUSTOM_ALLREDUCE NCCL_P2P_DISABLE=${NCCL_P2P_DISABLE:-unset} ACS_SUSPECTED=${ACS_SUSPECTED:-unknown} PESSIMISTIC_TP=${PESSIMISTIC_TP:-unknown} HOST_RAM_GB=$HOST_RAM_GB"
else
  log "no $HW_DECISIONS_FILE yet -> Next: bash vast/hardware_truth.sh   (writes results/hw/decisions.env)"
fi
log "setup done. Next: bash vast/hardware_truth.sh (writes results/hw/decisions.env), bash bench/collect_env.sh, then bench/launch.sh <cell> && bench/sweep.sh <cell>"
