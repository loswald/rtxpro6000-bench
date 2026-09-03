#!/usr/bin/env bash
export HF_HUB_ENABLE_HF_TRANSFER=1
get(){
  d=/workspace/models/$2
  [ -f "$d/config.json" ] && { echo "[$(date +%H:%M:%S)] have $2"; return; }
  echo "[$(date +%H:%M:%S)] downloading $1"
  hf download "$1" --local-dir "$d" > /workspace/dl_$2.log 2>&1 \
    && echo "[$(date +%H:%M:%S)] done $2 ($(du -sh $d|cut -f1))" || echo "[$(date +%H:%M:%S)] FAILED $2"
}
# ordered by AA Intelligence Index, only models that fit 4x96GB
get RedHatAI/GLM-5.3-Flash-NVFP4   GLM-5.3-Flash-NVFP4     # index 57, highest that fits
get meta-models/Muse-Glimmer-30B   Muse-Glimmer-30B        # index 35, one card
echo FLEET-DL-DONE
