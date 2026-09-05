#!/usr/bin/env bash
# Logit-level pairs on the 600 W box, which holds all three community four-bit builds of Qwen3.8-27B.
#
# The task suite put RedHat's NVFP4 at parity with the official FP8 on the 388 items both scored (0.801 vs
# 0.802) and the gittensor build ten points lower on maths. The logit pass measured only the gittensor
# build (KL 0.145 against FP8, seven times the control). So: which four-bit release actually stays closest
# to FP8, at the distribution level, on the same box, against a control run the same way? And two
# questions the task suite cannot separate: does quantising the ACTIVATIONS (the W4A4 kernel, b12x) move
# the distribution beyond what quantising the weights already did (the W4A16 kernel, auto) - and does the
# DFlash2 drafter change the base model's distribution at all, which is the property speculation must have.
source /workspace/bench/kllib.sh
Q8=$MD/Qwen3.8-27B-FP8
Q4=$MD/Qwen27B-NVFP4-RTX5090
RH=$MD/Qwen27B-NVFP4-RedHat
US=$MD/Qwen27B-NVFP4-unsloth
B12=(--kernel-config.linear_backend b12x)

log "===== logit-level quality, 600 W box: three four-bit releases against FP8 ====="
# the floor, on the FP8 checkpoint this time (the first control was NVFP4 against itself)
pair kl600_control "$Q8" 4 "${B12[@]}" --kv-cache-dtype auto  "$Q8" 4 "${B12[@]}" --kv-cache-dtype auto
pair kl600_gittensor_vs_fp8 "$Q4" 4 "${B12[@]}" --kv-cache-dtype auto  "$Q8" 4 "${B12[@]}" --kv-cache-dtype auto
# the two community builds have only ever been served under the auto (W4A16) kernel here; try W4A4 first,
# and if that kernel refuses the checkpoint format, measure them as they were actually served
pair kl600_redhat_vs_fp8  "$RH" 4 "${B12[@]}" --kv-cache-dtype auto  "$Q8" 4 "${B12[@]}" --kv-cache-dtype auto || \
pair kl600_redhat_auto_vs_fp8  "$RH" 2 --kv-cache-dtype auto  "$Q8" 4 "${B12[@]}" --kv-cache-dtype auto
pair kl600_unsloth_vs_fp8 "$US" 4 "${B12[@]}" --kv-cache-dtype auto  "$Q8" 4 "${B12[@]}" --kv-cache-dtype auto || \
pair kl600_unsloth_auto_vs_fp8 "$US" 2 --kv-cache-dtype auto  "$Q8" 4 "${B12[@]}" --kv-cache-dtype auto
# activations: the same weights under W4A4 and under W4A16
pair kl600_redhat_w4a4_vs_w4a16 "$RH" 4 "${B12[@]}" --kv-cache-dtype auto  "$RH" 2 --kv-cache-dtype auto
pair kl600_gittensor_w4a4_vs_w4a16 "$Q4" 4 "${B12[@]}" --kv-cache-dtype auto  "$Q4" 2 --kv-cache-dtype auto
# speculation must leave the distribution alone; DFlash2 is the drafter that loads
if [ -f "$MD/Qwen3.8-27B-DFlash2/config.json" ]; then
  pair kl600_spec_dflash2 "$Q4" 4 "${B12[@]}" --kv-cache-dtype fp8 \
       "$Q4" 6 "${B12[@]}" --kv-cache-dtype fp8 \
       --speculative-config "{\"method\":\"dflash\",\"model\":\"$MD/Qwen3.8-27B-DFlash2\",\"num_speculative_tokens\":7}"
fi
log "KLDIFF600W DONE"
kill_all
