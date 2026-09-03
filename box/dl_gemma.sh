#!/usr/bin/env bash
export HF_HUB_ENABLE_HF_TRANSFER=1
try(){ # dest repo...
  d=/workspace/models/$1; shift
  [ -f "$d/config.json" ] && { echo "have $d"; return; }
  for r in "$@"; do
    echo "[$(date +%H:%M:%S)] trying $r"
    if hf download "$r" --local-dir "$d" > /workspace/dl_$(basename $d).log 2>&1; then echo "[$(date +%H:%M:%S)] done $d ($(du -sh $d|cut -f1)) from $r"; return; fi
    echo "  failed: $(grep -m1 -oE "401|403|404|GatedRepo|Repository Not Found|Access to model" /workspace/dl_$(basename $d).log)"
  done
  echo "[$(date +%H:%M:%S)] ALL SOURCES FAILED for $d"
}
try gemma-4-31B-it     unsloth/gemma-4-31B-it     google/gemma-4-31B-it
try gemma-4-26B-A4B-it unsloth/gemma-4-26B-A4B-it google/gemma-4-26B-A4B-it
echo GEMMA-DL-DONE
