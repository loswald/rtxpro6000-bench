#!/usr/bin/env bash
# =============================================================================
# bench/prefetch.sh <cell-name | hf-model-id> [more...]            download weights
# bench/prefetch.sh --delete <cell-name | hf-model-id> [...] --yes  free disk
# bench/prefetch.sh --list                                          what is on disk
#
# Layout: $MODELS_DIR/<basename of the HF id>  (e.g. /workspace/models/DeepSeek-V4-Flash-0731,
# /workspace/models/Qwen3.8-27B-FP8, /workspace/models/gpt-oss-120b), written with
#   hf download <repo> --local-dir <dir> --max-workers N [--exclude ...]
# i.e. plain files, NOT the HF hub cache.  bench/env.sh:load_cell picks the directory up
# automatically (MODEL_PATH) for `vllm serve` and the bench client's --tokenizer.
# A finished download gets a .complete marker; a complete directory is never re-downloaded
# (FORCE=1 re-runs hf download, which only fetches missing/changed files).  Partial
# directories (*.incomplete under .cache/huggingface) are resumed.
#
# Disk on the campaign box is ~390 GB: benchmark one model, `bench/stop.sh <cell>`, then
#   bench/prefetch.sh --delete <cell> --yes
# (refuses while a running process still references the directory; FORCE=1 overrides;
# without --yes / CONFIRM=1 it only prints what it would remove).
# Repo-specific excludes: openai/gpt-oss-* skips original/* and metal/* (~130 GB of
# non-safetensors weights).  PREFETCH_EXCLUDE='pat1 pat2' adds patterns for any repo.
# Disk guard (390 GB box): a download is REFUSED when free space < size + 10 % + 5 GB
# (sizes from the table in model_size_gb(); PREFETCH_SIZE_GB=N for an unlisted repo,
# PREFETCH_FORCE_DISK=1 overrides) -- the same rule as vast/onstart.sh download_models.sh.
# Records seconds + size to results/prefetch.log.   Env: HF_MAX_WORKERS=16
# `hf download` options verified against huggingface_hub CLI docs (2026-09-02).
# =============================================================================
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
. "$HERE/env.sh"

usage() { sed -n '3,/^# =\{5,\}/p' "$0" | sed '$d' | sed 's/^# \{0,1\}//'; exit 1; }   # header block
ACTION=download; YES=0; ARGS=()
while [ $# -gt 0 ]; do
  case "$1" in
    --delete|--rm) ACTION=delete ;;
    --list)        ACTION=list ;;
    --yes|-y)      YES=1 ;;
    -h|--help)     usage ;;
    --*)           die "unknown option: $1" ;;
    *)             ARGS+=( "$1" ) ;;
  esac
  shift
done

# hf_transfer only when importable (HF_HUB_ENABLE_HF_TRANSFER=1 without the package breaks downloads)
if python3 -c 'import hf_transfer' 2>/dev/null; then export HF_HUB_ENABLE_HF_TRANSFER=1; else unset HF_HUB_ENABLE_HF_TRANSFER; fi

# resolve_repo <cell|repo> -> prints the HF id
resolve_repo() {
  local arg="$1" repo
  if [ -f "$CELLS_DIR/$arg.env" ]; then
    repo="$( (load_cell "$arg" >/dev/null 2>&1 && printf '%s' "$MODEL") || true )"
    [ -n "$repo" ] || die "could not resolve MODEL from cells/$arg.env"
    printf '%s\n' "$repo"
  else
    printf '%s\n' "$arg"
  fi
}
# repo_excludes <repo> -> space-separated glob patterns
repo_excludes() {
  local repo="$1" pats=""
  case "$repo" in
    openai/gpt-oss-*) pats="original/* metal/*" ;;   # safetensors only (~65 GB instead of ~196 GB)
  esac
  printf '%s\n' "${pats}${PREFETCH_EXCLUDE:+ $PREFETCH_EXCLUDE}"
}
model_dest() {  # <repo> -> directory, with a guard against deleting MODELS_DIR itself
  local repo="$1" base
  base="$(basename "$repo")"
  case "$base" in ''|.|..|/) die "refusing to work on '$repo' (bad basename)" ;; esac
  printf '%s/%s\n' "$MODELS_DIR" "$base"
}
# model_size_gb <repo> -> GB on disk (Hub usedStorage 2026-09-02; gpt-oss = safetensors only), 0 = unknown
model_size_gb() {
  case "$(basename "$1")" in
    Qwen3.8-27B-FP8)         echo 31 ;;
    gpt-oss-120b)            echo 65 ;;
    Qwen3-8B)                echo 16 ;;
    Qwen3.8-27B)             echo 56 ;;
    Qwen3.5-122B-A10B-FP8)   echo 127 ;;
    DeepSeek-V4-Flash-0731)  echo 167 ;;
    Qwen3.8-Flash-Next-FP8)  echo 186 ;;
    GLM-5.3-Flash)           echo 328 ;;
    *)                       echo "${PREFETCH_SIZE_GB:-0}" ;;
  esac
}
# disk_guard <repo> <dest>: refuse a download that cannot fit (remaining size + 10 % + 5 GB headroom)
disk_guard() {
  local repo="$1" dest="$2" size need avail already
  size="$(model_size_gb "$repo")"
  if [ "${size:-0}" -le 0 ]; then log "note: no size-table entry for $repo -> disk guard skipped (PREFETCH_SIZE_GB=N enables it)"; return 0; fi
  already="$(du -sBG "$dest" 2>/dev/null | cut -f1 | tr -d 'G')"; already="${already:-0}"
  need=$(( size - already + size / 10 + 5 )); [ "$need" -gt 0 ] || need=0
  avail="$(df -BG --output=avail "$MODELS_DIR" 2>/dev/null | tail -1 | tr -dc '0-9')"
  [ -n "$avail" ] || return 0
  if [ "$avail" -lt "$need" ]; then
    log "ERROR: $repo needs ~${need} GB more (${size} GB total, ${already} GB already present, +10 % +5 GB headroom) but $MODELS_DIR has only ${avail} GB free"
    log "       free disk first: bench/prefetch.sh --list ; bench/stop.sh <cell> ; bench/prefetch.sh --delete <cell|id> --yes   (PREFETCH_FORCE_DISK=1 overrides)"
    [ "${PREFETCH_FORCE_DISK:-0}" = 1 ] || return 1
  else
    log "disk: $repo ~${size} GB (${already} GB present), ${avail} GB free in $MODELS_DIR -> OK"
  fi
  return 0
}

dl() {
  local repo="$1" dest state t0 secs size rc=0
  dest="$(model_dest "$repo")"
  state="$(model_dir_state "$dest")"
  if [ "$state" = complete ] && [ "${FORCE:-0}" != 1 ]; then
    [ -f "$dest/.complete" ] || touch "$dest/.complete"
    log "present: $repo at $dest ($(du -sh "$dest" 2>/dev/null | cut -f1 || echo '?')) — FORCE=1 to re-check with hf download"
    return 0
  fi
  disk_guard "$repo" "$dest" || return 1
  mkdir -p "$dest" "$RESULTS_ROOT"
  local excl=() exargs=()
  read -r -a excl <<< "$(repo_excludes "$repo")"
  if [ ${#excl[@]} -gt 0 ]; then exargs=( --exclude "${excl[@]}" ); fi
  log "downloading $repo -> $dest (state=$state${excl[*]+, exclude: ${excl[*]}}; $(df -h "$MODELS_DIR" 2>/dev/null | awk 'NR==2{print $4" free"}'))"
  t0=$(date +%s)
  if have hf; then
    hf download "$repo" --local-dir "$dest" --max-workers "${HF_MAX_WORKERS:-16}" ${exargs[@]+"${exargs[@]}"} >/dev/null || rc=$?
  elif have huggingface-cli; then
    huggingface-cli download "$repo" --local-dir "$dest" --max-workers "${HF_MAX_WORKERS:-16}" ${exargs[@]+"${exargs[@]}"} >/dev/null || rc=$?
  else
    python3 - "$repo" "$dest" "${HF_MAX_WORKERS:-16}" ${excl[@]+"${excl[@]}"} <<'PY' || rc=$?
import sys
from huggingface_hub import snapshot_download
repo, dest, mw = sys.argv[1], sys.argv[2], int(sys.argv[3])
ignore = sys.argv[4:] or None
snapshot_download(repo, local_dir=dest, max_workers=mw, ignore_patterns=ignore)
PY
  fi
  secs=$(( $(date +%s) - t0 ))
  size="$(du -sh "$dest" 2>/dev/null | cut -f1 || echo '?')"
  if [ "$rc" = 0 ]; then
    touch "$dest/.complete"
    printf '%s\tdownload\t%s\t%ss\t%s\t%s\n' "$(date -Is)" "$repo" "$secs" "$size" "$dest" >> "$RESULTS_ROOT/prefetch.log"
    log "prefetched $repo in ${secs}s ($size at $dest)"
  else
    printf '%s\tdownload-failed(rc=%s)\t%s\t%ss\t%s\t%s\n' "$(date -Is)" "$rc" "$repo" "$secs" "$size" "$dest" >> "$RESULTS_ROOT/prefetch.log"
    log "WARN: download of $repo exited $rc after ${secs}s (resumable: re-run the same command)"
    return "$rc"
  fi
}

delete_model() {
  local repo="$1" dest size
  dest="$(model_dest "$repo")"
  case "$dest" in "$MODELS_DIR"|"$MODELS_DIR/"|/|"") die "refusing to delete '$dest'" ;; esac
  if [ ! -d "$dest" ]; then log "not on disk: $dest"; return 0; fi
  size="$(du -sh "$dest" 2>/dev/null | cut -f1 || echo '?')"
  if [ "${FORCE:-0}" != 1 ] && pgrep -f -- "$dest" >/dev/null 2>&1; then
    log "REFUSING to delete $dest ($size): a running process still references it — bench/stop.sh <cell> first (FORCE=1 overrides)"
    return 1
  fi
  if [ "$YES" != 1 ] && [ "${CONFIRM:-0}" != 1 ]; then
    log "would delete $dest ($size). Re-run with --yes (or CONFIRM=1) to remove it."
    return 0
  fi
  rm -rf -- "$dest"
  mkdir -p "$RESULTS_ROOT"
  printf '%s\tdeleted\t%s\t-\t%s\t%s\n' "$(date -Is)" "$repo" "$size" "$dest" >> "$RESULTS_ROOT/prefetch.log"
  log "deleted $dest ($size freed); now $(df -h "$MODELS_DIR" 2>/dev/null | awk 'NR==2{print $4" free of "$2" on "$6}')"
}

list_models() {
  log "models under $MODELS_DIR ($(df -h "$MODELS_DIR" 2>/dev/null | awk 'NR==2{print $4" free of "$2}')):"
  local d
  for d in "$MODELS_DIR"/*/; do
    [ -d "$d" ] || continue
    d="${d%/}"
    printf '  %-10s %-8s %s\n' "$(du -sh "$d" 2>/dev/null | cut -f1)" "$(model_dir_state "$d")" "$d" >&2
  done
}

case "$ACTION" in
  list)   list_models ;;
  delete) [ ${#ARGS[@]} -ge 1 ] || usage
          rc=0; for arg in "${ARGS[@]}"; do delete_model "$(resolve_repo "$arg")" || rc=1; done; exit $rc ;;
  *)      [ ${#ARGS[@]} -ge 1 ] || usage
          rc=0; for arg in "${ARGS[@]}"; do dl "$(resolve_repo "$arg")" || rc=1; done; exit $rc ;;
esac
