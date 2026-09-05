#!/usr/bin/env bash
# Third box (Vast 49977359, 4x RTX PRO 6000 Server Edition, 600 W, 2.5 TB disk). Takes over the queue the 400 W
# box lost when Vast stopped it at the balance threshold and could not restart it (its cards were re-rented).
# Being a 600 W box, its throughput counts, so every quality list here is also swept for throughput.
#
# Order: the quantisation ladder first (small downloads, the decisive result), then the logit ladder against the
# BF16 parent, then the big MoEs the fleet lost, then the thinking pairs, then the roster models that never got a
# quality run, then the soak tests. Every step is idempotent; a restart resumes.
R=/workspace/results; B=/workspace/bench; MD=/workspace/models
step(){ echo "[$(date +%H:%M:%S)] CHAINC: $*"; }
for i in $(seq 1 240); do grep -q "PROVISION2 DONE" $R/provision2.log 2>/dev/null && break; sleep 15; done
python3 -c "import vllm,flashinfer,b12x" 2>/dev/null || { step "stack import failed - stopping"; exit 1; }
source $B/dlget.sh
export L=$R/dl_c.log

step "downloads, ladder first"
get Qwen/Qwen3.8-27B                                    Qwen3.8-27B
get Qwen/Qwen3.8-27B-FP8                                Qwen3.8-27B-FP8
get gittensor-model-hub/Qwen3.8-27B-NVFP4-RTX5090       Qwen27B-NVFP4-RTX5090
get QUASAR-QAT/Qwen3.8-27B-QUASAR-NVFP4                 Qwen27B-QUASAR-NVFP4
get sakamakismile/Qwen3.8-27B-MTP-NVFP4                 Qwen27B-MTP-NVFP4
get incoai/Qwen3.8-27B-DFlash2                          Qwen3.8-27B-DFlash2
step "quantisation ladder at 600 W: throughput, then quality"
bash $B/ksweep.sh $B/lists/quant6000.txt > $R/ksweep_quant.log 2>&1
MODE=eval EVAL_BUDGET=5400 bash $B/ksweep.sh $B/lists/quant6000.txt > $R/keval_quant.log 2>&1
step "logit divergence: every rung against the BF16 parent; the DFlash2 drafter"
POS=16 bash $B/kldiff.sh > $R/kldiff.log 2>&1

step "downloads: the mixture-of-experts models the fleet lost, and the thinking pairs"
get olka-fi/MiniMax-M3-MXFP4                            MiniMax-M3-MXFP4
get nvidia/MiniMax-M3-DSpark                            MiniMax-M3-DSpark
get thinkingmachines/Inkling-Small-NVFP4                Inkling-Small-NVFP4
get RadixArk/Qwen3.8-Flash-Next-NVFP4                   Qwen3.8-Flash-Next-NVFP4
get meta-models/Muse-Glimmer-30B                        Muse-Glimmer-30B
get meta-models/Muse-Glimmer-30B-assistant              Muse-Glimmer-30B-assistant
get google/gemma-4-26B-A4B-it                           gemma-4-26B-A4B-it
get google/gemma-4-31B-it                               gemma-4-31B-it
step "the tensor-parallel arms with their kernel choices fixed: throughput, then quality"
bash $B/ksweep.sh $B/lists/tp6000.txt > $R/ksweep_tp.log 2>&1
MODE=eval EVAL_BUDGET=5400 bash $B/ksweep.sh $B/lists/tp6000.txt > $R/keval_tp.log 2>&1
step "gemma-4-26B thinking on vs off; gemma-4-31B both ways; Ling-3.0-flash"
get olka-fi/Ling-3.0-flash-NVFP4                        Ling-3.0-flash-NVFP4
MODE=eval EVAL_BUDGET=3600 bash $B/ksweep.sh $B/lists/thinkmode6000.txt > $R/keval_think.log 2>&1
MODE=eval EVAL_BUDGET=5400 bash $B/ksweep.sh $B/lists/tierb6000.txt > $R/keval_tierb.log 2>&1
bash $B/ksweep.sh $B/lists/tierb6000.txt > $R/ksweep_tierb.log 2>&1

step "downloads: Laguna, Nemotron-3-Super, Hy3, Ornith, MiMo"
get poolside/Laguna-S-2.1-NVFP4                         Laguna-S-2.1-NVFP4
get poolside/Laguna-S-2.1-DFlash-NVFP4                  Laguna-S-2.1-DFlash
get nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4      Nemotron-3-Super-NVFP4
get RedHatAI/Hy3-NVFP4-FP8                              Hy3-NVFP4-FP8
get littlecedar/Ornith-1.5-397B-NVFP4-MTP-Graft         Ornith-1.5-397B-NVFP4
get mitomtuna/MiMo-V2.5-0703-NVFP4                      MiMo-V2.5-NVFP4
step "Laguna thinking on vs off"
MODE=eval EVAL_BUDGET=3600 bash $B/ksweep.sh $B/lists/laguna6000.txt > $R/keval_laguna.log 2>&1
step "the roster models that never had a quality run: Nemotron-3-Super, Hy3, Ornith, MiMo - throughput too"
MODE=eval FIRST_ONLY=1 EVAL_BUDGET=5400 bash $B/ksweep.sh $B/lists/tierc_c.txt > $R/keval_tierc.log 2>&1
bash $B/ksweep.sh $B/lists/tierc_c.txt > $R/ksweep_tierc.log 2>&1

step "soak: Qwen3.8-27B NVFP4 (QAT build) and FP8, four replicas, 120 min each"
TAG=soak_q27_qat DIR=$MD/Qwen27B-QUASAR-NVFP4 LIN=b12x MINUTES=120 bash $B/soak.sh > $R/soak_q27_qat.log 2>&1
TAG=soak_q27_fp8 DIR=$MD/Qwen3.8-27B-FP8 LIN=b12x MINUTES=120 bash $B/soak.sh > $R/soak_q27_fp8.log 2>&1
step "CHAINC DONE"
