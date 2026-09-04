#!/usr/bin/env bash
R=/workspace/results
for i in $(seq 1 1440); do grep -q "CHAIN6000B DONE" $R/chain6000.log 2>/dev/null && break; sleep 60; done
source /workspace/bench/dlget.sh
# the rotation deletes each model once measured; both later passes need them back
while read -r repo dir; do
  [ -z "$repo" ] && continue
  [ -f "/workspace/models/$dir/config.json" ] || { sed -i -E "/\] (done|have) $dir( |$)/d" $R/dl6000.log; get "$repo" "$dir"; }
done <<LIST
olka-fi/Ling-3.0-flash-NVFP4 Ling-3.0-flash-NVFP4
stepfun-ai/Step-3.7-Flash-NVFP4 Step-3.7-Flash-NVFP4
nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4 Nemotron-3-Super-NVFP4
nvidia/Nemotron-3-Super-120B-A12B-BF16-MTPv2 Nemotron-3-Super-MTPv2
poolside/Laguna-S-2.1-NVFP4 Laguna-S-2.1-NVFP4
poolside/Laguna-S-2.1-DFlash-NVFP4 Laguna-S-2.1-DFlash
mitomtuna/MiMo-V2.5-0703-NVFP4 MiMo-V2.5-NVFP4
RedHatAI/Hy3-NVFP4-FP8 Hy3-NVFP4-FP8
LIST
echo "[$(date +%H:%M:%S)] chain3: ksweep retry6000 (throughput)"
bash /workspace/bench/ksweep.sh /workspace/bench/lists/retry6000.txt > $R/ksweep6000.log 2>&1
echo "[$(date +%H:%M:%S)] chain4: ksweep eval6000 (quality, same items and seed as the 5090 box)"
MODE=eval EVAL_BUDGET=900 bash /workspace/bench/ksweep.sh /workspace/bench/lists/eval6000.txt > $R/keval6000.log 2>&1
echo "[$(date +%H:%M:%S)] CHAIN6000C DONE"
