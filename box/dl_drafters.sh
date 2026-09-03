#!/usr/bin/env bash
export HF_HUB_ENABLE_HF_TRANSFER=1
get(){ d=/workspace/models/$2; [ -f "$d/config.json" ] && { echo "[$(date +%H:%M:%S)] have $2"; return; }
  echo "[$(date +%H:%M:%S)] downloading $1"
  hf download "$1" --local-dir "$d" > /workspace/dl_$2.log 2>&1 && echo "[$(date +%H:%M:%S)] done $2 ($(du -sh $d|cut -f1))" || echo "[$(date +%H:%M:%S)] FAILED $2"; }
get nvidia/MiniMax-M3-DSpark                                         MiniMax-M3-DSpark
get google/gemma-4-26B-A4B-it-assistant                              gemma-4-26B-A4B-it-assistant
get nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4-DSpark        Nemotron-3.5-Lightning-DSpark
get meta-models/Muse-Glimmer-30B-assistant                           Muse-Glimmer-30B-assistant
get incoai/Qwen3.8-27B-DFlash2                                       Qwen3.8-27B-DFlash2
echo DRAFTERS-DL-DONE
