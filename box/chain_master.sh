#!/usr/bin/env bash
# The campaign's remaining work, in one sequential chain.
#
# Written after a scheduling bug that is worth recording: the logit-divergence chain and the
# quantisation-ladder chain were each armed to start on the same marker ("CHAIN6000C DONE"), so both would
# have woken at once and fought over the GPUs. That collision had already cost Ornith-1.5-397B its
# speculation arm earlier the same day, when a second campaign started on a busy box and each run's
# kill_all took down the other's servers. Independent waiters on a shared resource are the bug; one
# sequential chain is the fix.
#
# Order is deliberate: the cheap, decisive measurements first, the long downloads last.
R=/workspace/results
step(){ echo "[$(date +%H:%M:%S)] MASTER: $*"; }
for i in $(seq 1 2880); do grep -q "CHAIN6000C DONE" $R/chain6000.log 2>/dev/null && break; sleep 60; done

# 1. Logit-level divergence. Resolves what 403 task items cannot: a control pair fixes the noise floor,
#    then NVFP4 against FP8, the KV dtype, kernel equivalence, and speculation against its own base.
step "logit-level divergence (control, NVFP4 vs FP8, KV dtype, kernel equivalence, speculation)"
POS=16 bash /workspace/bench/kldiff.sh > $R/kldiff.log 2>&1

# 2. The quantisation ladder. Task accuracy on the same weights through four quantisers plus the native
#    parent, all at TP1 with one recipe and one seed, to separate "four-bit is lossy" from "that upload is bad".
step "quantisation ladder: fetch the rungs"
source /workspace/bench/dlget.sh
get Qwen/Qwen3.8-27B                        Qwen3.8-27B
get QUASAR-QAT/Qwen3.8-27B-QUASAR-NVFP4     Qwen27B-QUASAR-NVFP4
get sakamakismile/Qwen3.8-27B-MTP-NVFP4     Qwen27B-MTP-NVFP4
step "quantisation ladder: throughput then quality"
bash /workspace/bench/ksweep.sh /workspace/bench/lists/quant6000.txt > $R/ksweep_quant.log 2>&1
MODE=eval EVAL_BUDGET=2400 bash /workspace/bench/ksweep.sh /workspace/bench/lists/quant6000.txt > $R/keval_quant.log 2>&1

# 3. GLM-5.3-Flash at the precision Z.AI actually trained it in. Everything else we have measured of this
#    model is a community NVFP4 re-quantisation of an already-quantised FP8 release - the configuration the
#    maths signal on Qwen predicts should hurt most, on the highest-intelligence model that fits the node.
#    ~330 GB of weights across 384 GB of VRAM: a poor throughput configuration and the right quality reference.
step "GLM-5.3-Flash at its NATIVE precision: free space, fetch the FP8 release, measure"
for m in Step-3.7-Flash-NVFP4 Nemotron-3-Super-NVFP4 Ling-3.0-flash-NVFP4 Qwen3.8-27B; do
  [ -d /workspace/models/$m ] && { python3 -c "import shutil,sys; shutil.rmtree(sys.argv[1], ignore_errors=True)" /workspace/models/$m; echo "  freed $m"; }
done
df -h / | tail -1 | awk '{print "  disk: " $4 " free (the FP8 release is ~330 GB)"}'
L=$R/dl6000.log get zai-org/GLM-5.3-Flash GLM-5.3-Flash-FP8
ARMS=fp8 bash /workspace/bench/glm_eval.sh > $R/glm_eval_fp8.log 2>&1

# 4. The five tensor-parallel arms the fleet tier lost, with their causes fixed rather than the flags repeated.
step "the tensor-parallel arms the fleet lost, causes fixed"
bash /workspace/bench/ksweep.sh /workspace/bench/lists/tp6000.txt > $R/ksweep_tp.log 2>&1
MODE=eval EVAL_BUDGET=1800 bash /workspace/bench/ksweep.sh /workspace/bench/lists/tp6000.txt > $R/keval_tp.log 2>&1

# 5. gemma-4 is the one model whose vendor default is thinking OFF; both modes get measured rather than assumed.
step "thinking on vs off, gemma-4 BF16"
MODE=eval EVAL_BUDGET=3600 bash /workspace/bench/ksweep.sh /workspace/bench/lists/thinkmode6000.txt > $R/keval_think.log 2>&1

# 6. Motif-3 runs only on its vendor fork, whose image was missing pydivsufsort (installed beside the tree).
step "Motif-3 on its vendor fork"
MD=/workspace/models/Motif-3-NVFP4 bash /workspace/bench/motif_vllm.sh > $R/motif2.log 2>&1

step "MASTER DONE"
