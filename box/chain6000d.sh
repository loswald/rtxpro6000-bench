#!/usr/bin/env bash
R=/workspace/results
for i in $(seq 1 2880); do grep -q "KLDIFF6000 DONE" $R/chain6000.log 2>/dev/null && break; sleep 60; done
source /workspace/bench/dlget.sh
while read -r repo dir; do
  [ -z "$repo" ] && continue
  [ -f "/workspace/models/$dir/config.json" ] || { sed -i -E "/\] (done|have) $dir( |$)/d" $R/dl6000.log; get "$repo" "$dir"; }
done <<LIST
protoLabsAI/Agents-A1-NVFP4 Agents-A1-NVFP4
sakamakismile/KAT-Coder-V2.5-Dev-NVFP4 KAT-Coder-V2.5-NVFP4
ibm-granite/granite-4.2-30b-nvfp4 granite-4.2-30b-nvfp4
inclusionAI/Ling-3.0-tiny Ling-3.0-tiny
ornith-ai/Ornith-1.5-35B-A3B-NVFP4 Ornith-1.5-35B-A3B-NVFP4
ornith-ai/Ornith-1.5-9B Ornith-1.5-9B
poolside/Laguna-XS-2.1-NVFP4 Laguna-XS-2.1-NVFP4
poolside/Laguna-XS-2.1-DFlash-NVFP4 Laguna-XS-2.1-DFlash
internlm/Intern-S2-Mobius-FP8 Intern-S2-Mobius-FP8
LIST
echo "[$(date +%H:%M:%S)] chain5: tier C throughput"
bash /workspace/bench/ksweep.sh /workspace/bench/lists/tierc6000.txt > $R/ksweep_tierc.log 2>&1
echo "[$(date +%H:%M:%S)] chain5: tier C quality"
MODE=eval EVAL_BUDGET=600 bash /workspace/bench/ksweep.sh /workspace/bench/lists/tierc6000.txt > $R/keval_tierc.log 2>&1
echo "[$(date +%H:%M:%S)] CHAIN6000D DONE"
