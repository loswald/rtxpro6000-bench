#!/usr/bin/env bash
# =============================================================================
# patches/apply_dsv4_o_proj.sh [--check | --apply | --revert]
#
# Hand patch for DeepSeek-V4-Flash on sm_120 -- UNVERIFIED (2026-09-02).  Apply ONLY if a
# ds4flash_* launch dies inside deep_gemm_fp8_o_proj / fp8_einsum (DeepGEMM has no sm_120
# kernels; launch.json.error_excerpt shows it):
#   <site-packages>/vllm/models/deepseek_v4/nvidia/ops/o_proj.py  <-  patches/vllm_dsv4_nvidia_ops_o_proj.py
# What the patch changes (diff against the .orig kept next to it):
#   * compute_fp8_einsum_recipe(): on compute capability 12.x use the SM90 scale layout
#     ((1,128,128), no TMA-aligned scales) instead of the SM100 packed layout that DeepGEMM
#     lacks for sm_120.  VLLM_DSV4_OPROJ_SM120_RECIPE=sm100 restores the upstream choice (A/B).
#   * VLLM_DSV4_OPROJ_SM120_FALLBACK=1: bypass DeepGEMM fp8_einsum with a bf16 torch.einsum on
#     block-dequantised weights (cached per layer) -- slower; a correctness fallback only.
# --check (default): print the installed file and whether it matches the patch / the .orig; exit 0
#          when patched, 1 otherwise.
# --apply: back the installed file up to <file>.sqwish.orig (once) and copy the patch over it,
#          clearing __pycache__.  --revert restores the backup.
# PYTHON=python3 selects the ENGINE interpreter; nothing is pip-installed.  Export the env knobs in
# the shell (or a cell's `export VLLM_...`) before bench/launch.sh -- launch.sh snapshots VLLM_*.
# =============================================================================
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="${PYTHON:-python3}"
PATCH="$HERE/vllm_dsv4_nvidia_ops_o_proj.py"
ORIG="$HERE/vllm_dsv4_nvidia_ops_o_proj.py.orig"
ACTION="${1:---check}"
case "$ACTION" in --check|--apply|--revert) ;; *) sed -n '3,/^# =\{5,\}/p' "$0" | sed '$d' | sed 's/^# \{0,1\}//'; exit 2 ;; esac

target="$("$PY" - <<'PY' 2>/dev/null || true
import importlib.util
s = importlib.util.find_spec("vllm.models.deepseek_v4.nvidia.ops.o_proj")
print(s.origin if s and s.origin else "")
PY
)"
[ -n "$target" ] || { echo "vllm.models.deepseek_v4.nvidia.ops.o_proj is not importable with $PY (vLLM main >= 2026-08 needed)" >&2; exit 2; }
state="$("$PY" - "$target" "$PATCH" "$ORIG" <<'PY'
import filecmp, sys
t, p, o = sys.argv[1:4]
print("patched" if filecmp.cmp(t, p, shallow=False) else "upstream-known" if filecmp.cmp(t, o, shallow=False) else "upstream-other")
PY
)"
echo "installed: $target  ($state)"
case "$ACTION" in
  --check)
    [ "$state" = patched ] ;;
  --apply)
    [ -f "$target.sqwish.orig" ] || cp -p "$target" "$target.sqwish.orig"
    if [ "$state" = upstream-other ]; then
      echo "WARN: the installed file differs from the .orig this patch was made against; compare $target.sqwish.orig with $ORIG before trusting the result" >&2
    fi
    cp -p "$PATCH" "$target"
    find "$(dirname "$target")" -name '__pycache__' -type d -exec rm -rf {} + 2>/dev/null || true
    echo "applied. Knobs: VLLM_DSV4_OPROJ_SM120_RECIPE=sm90|sm100  VLLM_DSV4_OPROJ_SM120_FALLBACK=0|1  (backup: $target.sqwish.orig)" ;;
  --revert)
    [ -f "$target.sqwish.orig" ] || { echo "no backup at $target.sqwish.orig" >&2; exit 1; }
    cp -p "$target.sqwish.orig" "$target"
    find "$(dirname "$target")" -name '__pycache__' -type d -exec rm -rf {} + 2>/dev/null || true
    echo "reverted from $target.sqwish.orig" ;;
esac
