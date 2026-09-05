#!/usr/bin/env bash
# 600 W box, fourth chain: after chain600w3d finishes, the questions its results raised.
#   1. The suite's own noise floor - the same Qwen configuration evaluated twice - and the gittensor weights
#      under Qwen's official chat template, which the build does not ship (see lists/control600w.txt). Both
#      cheap, and between them they decide how the quantiser ladder is read.
#   2. The QAT rung at 600 W: throughput that counts, and a quality run pairable with this box's other rows.
#   3. DeepSeek-V4-Flash with a 98k window and doubled caps: the 403-item pass truncated 17% of code and 6% of
#      maths answers, and a truncated answer scores as wrong, so 0.801 / 0.831 are floors.
R=/workspace/results; B=/workspace/bench
step(){ echo "[$(date +%H:%M:%S)] 600W-4: $*"; }
for i in $(seq 1 2880); do grep -q "CHAIN600W3 DONE" $R/chain600w.log 2>/dev/null && break; sleep 60; done
step "noise floor (Qwen NVFP4 a second time) and the gittensor weights under the official chat template"
MODE=eval EVAL_BUDGET=3600 bash $B/ksweep.sh $B/lists/control600w.txt > $R/keval_control.log 2>&1
step "QUASAR-QAT NVFP4 at 600 W: fetch, throughput, quality"
source $B/dlget.sh
L=$R/dl600w.log get QUASAR-QAT/Qwen3.8-27B-QUASAR-NVFP4 Qwen27B-QUASAR-NVFP4
bash $B/ksweep.sh $B/lists/quant600w.txt > $R/ksweep_quant600w.log 2>&1
MODE=eval EVAL_BUDGET=3600 bash $B/ksweep.sh $B/lists/quant600w.txt > $R/keval_quant600w.log 2>&1
step "DeepSeek-V4-Flash with room to 65k tokens (window 98k, caps doubled, 32 streams)"
EXTRA_ENV="VLLM_DSV4_OPROJ_SM120_FALLBACK=1" MODE=eval FIRST_ONLY=1 EVAL_CONC=32 EVAL_BUDGET=10800 \
  EVAL_MAXTOK=65536 EVAL_CAPS="math=65536,code=40960,knowledge=40960,ifeval=32768,tools=16384,longctx=12288" \
  bash $B/ksweep.sh $B/lists/ds600w4_eval.txt > $R/keval_ds64k.log 2>&1
step "CHAIN600W4 DONE"
