#!/usr/bin/env bash
export HF_HUB_ENABLE_HF_TRANSFER=1
get(){
  d=/workspace/models/$2
  [ -f "$d/config.json" ] && { echo "[$(date +%H:%M:%S)] have $2"; return; }
  echo "[$(date +%H:%M:%S)] downloading $1"
  hf download "$1" --local-dir "$d" > /workspace/dl_$2.log 2>&1 \
    && echo "[$(date +%H:%M:%S)] done $2 ($(du -sh $d|cut -f1))" || echo "[$(date +%H:%M:%S)] FAILED $2"
}
# The sm_120 practitioner stack. RTX 5090 is the same silicon as RTX PRO 6000 Blackwell,
# so these are validated on our exact architecture, unlike the datacentre sm_100 builds.
get gittensor-model-hub/Qwen3.8-27B-NVFP4-RTX5090 Qwen27B-NVFP4-RTX5090   # 17.9 GB target
get gittensor-model-hub/Qwen3.8-27B-DSpark-NVFP4  Qwen27B-DSpark-NVFP4    # 1.4 GB drafter
get sakamakismile/Qwen3.8-27B-MTP-NVFP4           Qwen27B-MTP-NVFP4       # MTP-head variant
echo NVFP4-DL-DONE
