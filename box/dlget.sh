# Shared downloader, sourced by the dl*.sh chains.  `get <repo> <dirname>`
#
# Two rules learned the hard way on 4 Sept 2026:
#   * never `sed -i` the log this is appending to - sed replaces the inode and every later line goes to a
#     deleted file, so the campaign saw nine downloaded models as missing;
#   * a directory is not a model. A failed download leaves a config and a couple of small files behind,
#     which passed every heuristic while holding 0.7 GB of a 171 GB checkpoint. So on success we drop a
#     sentinel, and that is what the campaign trusts.
export HF_HUB_ENABLE_HF_TRANSFER=1
L=${L:-/workspace/results/dl6000.log}

_shards_present(){ # dir -> 0 when every shard the index names is on disk AND a tokenizer came with it
  local d="$1"
  [ -f "$d/config.json" ] || return 1
  find "$d" \( -name '*.incomplete' -o -name '*.part' \) 2>/dev/null | grep -q . && return 1
  # weights without a tokenizer is a half-formed download that looks complete: Laguna-S-2.1 arrived with
  # all ten shards and nothing to tokenise with, and only said so 20 s into a launch an hour later
  # Draft models are the exception: they load beside a target and use ITS tokenizer, so requiring one here
  # would mark every drafter on the box FAILED and delete it.
  case "$(basename "$d")" in
    *DSpark*|*DFlash*|*MTP*|*MTPv2*|*assistant*|*draft*|*Draft*|*EAGLE*|*eagle*|*speculator*) : ;;
    *)
      if ! compgen -G "$d/tokenizer*" >/dev/null && ! compgen -G "$d/*.model" >/dev/null \
         && ! compgen -G "$d/vocab*" >/dev/null && ! grep -qs '"tokenizer_mode"' "$d/config.json"; then
        return 1
      fi ;;
  esac
  if [ -f "$d/model.safetensors.index.json" ]; then
    python3 - "$d" <<'PY'
import json, os, sys
d = sys.argv[1]
try:
    wm = json.load(open(os.path.join(d, "model.safetensors.index.json")))["weight_map"]
except Exception:
    sys.exit(1)
files = set(wm.values())
sys.exit(0 if files and all(os.path.exists(os.path.join(d, f)) for f in files) else 1)
PY
    return $?
  fi
  compgen -G "$d/*.safetensors" >/dev/null || compgen -G "$d/*.bin" >/dev/null || compgen -G "$d/*.gguf" >/dev/null
}

get(){ # repo dirname
  local d=/workspace/models/$2
  if [ -f "$d/.dl_complete" ]; then echo "[$(date +%H:%M:%S)] have $2" | tee -a "$L"; return 0; fi
  echo "[$(date +%H:%M:%S)] downloading $1" | tee -a "$L"
  # Two passes. hf_transfer is much faster but its content-addressed path drops individual files without
  # failing the command: Laguna-S-2.1 arrived with all ten weight shards and no tokenizer.json, and the
  # same file downloaded first time over plain HTTP. So: fast path, then a plain-HTTP pass to fill gaps,
  # and only then judge the result.
  HF_HUB_ENABLE_HF_TRANSFER=1 hf download "$1" --local-dir "$d" > "/workspace/dl_$2.log" 2>&1
  if ! _shards_present "$d"; then
    echo "[$(date +%H:%M:%S)] retrying $2 without hf_transfer" | tee -a "$L"
    HF_HUB_ENABLE_HF_TRANSFER=0 hf download "$1" --local-dir "$d" >> "/workspace/dl_$2.log" 2>&1
  fi
  if _shards_present "$d"; then
    : > "$d/.dl_complete"
    echo "[$(date +%H:%M:%S)] done $2 ($(du -sh "$d" | cut -f1))" | tee -a "$L"
    return 0
  fi
  echo "[$(date +%H:%M:%S)] FAILED $2 ($(tail -1 "/workspace/dl_$2.log" 2>/dev/null | cut -c1-100))" | tee -a "$L"
  # a stub costs disk and, worse, looks like a model to anything that only checks for a config
  python3 -c "import shutil,sys; shutil.rmtree(sys.argv[1], ignore_errors=True)" "$d"
  return 1
}
