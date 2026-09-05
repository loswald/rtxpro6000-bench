#!/usr/bin/env bash
# Logit-level quality pass: the cascade-free way to ask what a serving choice costs.
#
# Task accuracy on 435 items answers "is this model still good". It cannot answer "did THIS FLAG change
# the model", because a single differing token cascades and because 400 items only resolve differences of
# several points. Comparing next-token distributions can: identical contexts, both servers, twenty
# log-probabilities per position, and a control pair that fixes the noise floor. Anything above the
# control is caused by the treatment.
#
# Two servers run side by side on separate cards, so each pair is measured in one session with nothing
# else moving. Every treatment is reported against a control launched the same way.
source /workspace/bench/kllib.sh
Q4=$MD/Qwen27B-NVFP4-RTX5090
Q8=$MD/Qwen3.8-27B-FP8
BF=$MD/Qwen3.8-27B          # the native parent; every other rung is a loss measured against this
B12=(--kernel-config.linear_backend b12x)

log "===== logit-level quality, Qwen3.8-27B ====="
if [ -z "${LADDER_ONLY:-}" ]; then   # the four pairs below were measured on the 400 W box on 4 Sept; a second box need not repeat them

# 1. The noise floor. Same checkpoint, same flags, two cards. Everything below must be read against this:
#    two identical servers do not agree perfectly, because reduction order differs with the device.
pair kl_control "$Q4" 4 "${B12[@]}" --kv-cache-dtype fp8 "$Q4" 4 "${B12[@]}" --kv-cache-dtype fp8

# 2. The decision this campaign turns on: four-bit weights and activations are 64% faster than FP8 on this
#    card. Do they change what the model predicts?
pair kl_nvfp4_vs_fp8 "$Q4" 4 "${B12[@]}" --kv-cache-dtype fp8 "$Q8" 4 "${B12[@]}" --kv-cache-dtype fp8

# 3. The KV cache dtype, measured on this model rather than inherited from the gpt-oss result.
pair kl_kv_fp8_vs_auto "$Q4" 4 "${B12[@]}" --kv-cache-dtype fp8 "$Q4" 4 "${B12[@]}" --kv-cache-dtype auto

# 4. Kernel equivalence. b12x and flashinfer_b12x tie on throughput; they should also be numerically
#    indistinguishable. If they are not, one of them is wrong.
pair kl_b12x_vs_fib12x "$Q4" 4 "${B12[@]}" --kv-cache-dtype fp8 \
     "$Q4" 4 --kernel-config.linear_backend flashinfer_b12x --kv-cache-dtype fp8

fi

# 5. Speculative decoding must not change the distribution at all: it proposes tokens and verifies them
#    against the same model. Divergence above the control is a bug in the speculator, not a trade-off.
# The DSpark drafter does not load against this target: "The size of tensor a (128) must match the size of
# tensor b (256)" in dspark/speculator.py. DFlash2 does load, and it is the drafter the fleet actually
# measured, so it is the one whose distribution matters. This is the pair that can answer what the greedy
# sequence test could not - speculation must leave the next-token distribution alone, and unlike a
# generated sequence, a distribution can be compared on a stack that is not bit-deterministic.
if [ -f "$MD/Qwen3.8-27B-DFlash2/config.json" ]; then
  pair kl_spec_dflash2 "$Q4" 4 "${B12[@]}" --kv-cache-dtype fp8 \
       "$Q4" 6 "${B12[@]}" --kv-cache-dtype fp8 \
       --speculative-config "{\"method\":\"dflash\",\"model\":\"$MD/Qwen3.8-27B-DFlash2\",\"num_speculative_tokens\":7}"
fi
if [ -f "$MD/Qwen27B-DSpark-NVFP4/config.json" ]; then
  pair kl_spec_vs_base "$Q4" 4 "${B12[@]}" --kv-cache-dtype fp8 \
       "$Q4" 8 "${B12[@]}" --kv-cache-dtype fp8 --speculative-draft-model-quantization modelopt_fp4 \
       --speculative-config "{\"method\":\"draft_model\",\"model\":\"$MD/Qwen27B-DSpark-NVFP4\",\"num_speculative_tokens\":5,\"draft_tensor_parallel_size\":1}"
fi

log "KLDIFF DONE"
kill_all

# ---- added 4 Sept: the question is loss FROM the native baseline, so measure against it directly ----
# Task accuracy resolves differences of several points across 403 items. Logit divergence resolves the
# ones it cannot see, and it is the only way to separate "this format loses information" from "this
# particular quantiser is bad" - the same weights, three quantisers, one baseline.
if [ -f "$BF/config.json" ]; then
  log "===== loss from the native BF16 parent ====="
  pair kl_bf16_control  "$BF" 2 --kv-cache-dtype auto            "$BF" 2 --kv-cache-dtype auto
  pair kl_bf16_vs_fp8   "$BF" 2 --kv-cache-dtype auto            "$Q8" 4 "${B12[@]}" --kv-cache-dtype auto
  pair kl_bf16_vs_nvfp4 "$BF" 2 --kv-cache-dtype auto            "$Q4" 4 "${B12[@]}" --kv-cache-dtype auto
  [ -f "$MD/Qwen27B-QUASAR-NVFP4/config.json" ] && \
  pair kl_bf16_vs_qat   "$BF" 2 --kv-cache-dtype auto            "$MD/Qwen27B-QUASAR-NVFP4" 4 "${B12[@]}" --kv-cache-dtype auto
  [ -f "$MD/Qwen27B-MTP-NVFP4/config.json" ] && \
  pair kl_bf16_vs_ptq2  "$BF" 2 --kv-cache-dtype auto            "$MD/Qwen27B-MTP-NVFP4" 4 "${B12[@]}" --kv-cache-dtype auto
else
  log "SKIP the BF16 ladder (native parent not downloaded yet)"
fi
log "KLDIFF-BF16 DONE"
kill_all
