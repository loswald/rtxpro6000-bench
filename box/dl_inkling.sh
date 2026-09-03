#!/usr/bin/env bash
export HF_HUB_ENABLE_HF_TRANSFER=1
get(){ d=/workspace/models/$2; [ -f "$d/config.json" ] && { echo "[$(date +%H:%M:%S)] have $2"; return; }
  echo "[$(date +%H:%M:%S)] downloading $1"
  hf download "$1" --local-dir "$d" > /workspace/dl_$2.log 2>&1 && echo "[$(date +%H:%M:%S)] done $2 ($(du -sh $d|cut -f1))" || echo "[$(date +%H:%M:%S)] FAILED $2"; }
get thinkingmachines/Inkling-Small-NVFP4 Inkling-Small-NVFP4
get RadixArk/Inkling-Small-DSpark       Inkling-Small-DSpark
echo INKLING-DL-DONE
