#!/usr/bin/env bash
# 600 W box, third chain, fourth issue. Same programme as chain600w3c.sh; DeepSeek-V4-Flash now launches with
# VLLM_DSV4_OPROJ_SM120_FALLBACK=1, which every launcher that ever served this model on this GPU exported and
# the sweep's launcher did not. Without it the attention output projection goes through DeepGEMM's fp8_einsum,
# which has no sm_120 path and asserts on tensor rank during memory profiling - the failure attempts two and
# three blamed on the MoE backend and then on the sequence budget. The o_proj fallback is the patch this
# repository already carries (box/patch_oproj.py); it is applied on the box and gated by that variable.
R=/workspace/results; B=/workspace/bench
step(){ echo "[$(date +%H:%M:%S)] 600W-3: $*"; }
export EXTRA_ENV="VLLM_DSV4_OPROJ_SM120_FALLBACK=1"
step "DeepSeek-V4-Flash, fourth attempt (o_proj fallback on): quality on the first MoE backend that serves, then throughput on every one that does"
MODE=eval FIRST_ONLY=1 EVAL_CONC=64 EVAL_BUDGET=7200 bash $B/ksweep.sh $B/lists/ds600w3_eval.txt > $R/keval_ds4.log 2>&1
bash $B/ksweep.sh $B/lists/ds600w3.txt > $R/ksweep_ds4.log 2>&1
unset EXTRA_ENV
step "finish the runs that lost items to the request timeout: Qwen FP8 (15 items), Muse-Glimmer (14)"
MODE=eval EVAL_RESUME=1 EVAL_BUDGET=3600 bash $B/ksweep.sh $B/lists/resume600w.txt > $R/keval_resume.log 2>&1
step "GLM-5.3-Flash: MTP with room to 65k tokens - does the model finish what it truncated at 32k?"
ARMS=mtp64k bash $B/glm_eval.sh > $R/glm_eval_mtp64k.log 2>&1
step "the five configurations the quality pass lost, plus RedHat and unsloth under the W4A4 kernel"
source $B/dlget.sh
L=$R/dl600w.log get Qwen/Qwen3.6-35B-A3B-FP8 Qwen3.6-35B-A3B-FP8
MODE=eval EVAL_BUDGET=5400 bash $B/ksweep.sh $B/lists/fix600w.txt > $R/keval_fix.log 2>&1
bash $B/ksweep.sh $B/lists/fix600w.txt > $R/ksweep_fix.log 2>&1
step "logit pairs: which four-bit release stays closest to FP8; W4A4 vs W4A16; the DFlash2 drafter"
POS=16 bash $B/kldiff600w.sh > $R/kldiff600w.log 2>&1
step "GLM-5.3-Flash at its native FP8: free space (measured checkpoints only), fetch, quality, throughput"
for m in GLM-5.3-Flash-NVFP4 Qwen3.8-Flash-Next-NVFP4 MiniMax-M3-DSpark Inkling-Small-DSpark gemma-4-31B-it; do
  free=$(df -BG --output=avail /workspace | tail -1 | tr -dc 0-9)
  [ "$free" -ge 360 ] && break
  [ -d /workspace/models/$m ] && { python3 -c "import shutil,sys; shutil.rmtree(sys.argv[1], ignore_errors=True)" /workspace/models/$m; echo "  freed $m"; }
done
df -h /workspace | tail -1 | awk '{print "  disk: " $4 " free (the FP8 release is ~330 GB)"}'
L=$R/dl600w.log get zai-org/GLM-5.3-Flash GLM-5.3-Flash-FP8
ARMS=fp8 bash $B/glm_eval.sh > $R/glm_eval_fp8.log 2>&1
step "CHAIN600W3 DONE"
