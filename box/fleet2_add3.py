#!/usr/bin/env python3
"""Harden the Qwen3.8-Flash-Next arm in /workspace/bench/fleet2.sh (idempotent).

Stock vLLM 0.28.1rc1.dev332 on the node does register Qwen4ExpForConditionalGeneration and Qwen4ExpMTP,
so the arm can run on the working engine. Two RTX PRO 6000 reports from the roster audit apply:
the default GDN decode kernel deadlocks at ~32 concurrency (fix: VLLM_GDN_DECODE_KERNEL=triton) and the
built-in MTP head is worth ~+56% at 3 draft tokens while 1 token hangs. ENV is the launcher's export
string; a temporary assignment on the function call scopes the extra variable to this arm only."""
p = "/workspace/bench/fleet2.sh"
s = open(p).read()
old = 'tpm qwen38fn  "$MD/Qwen3.8-Flash-Next-NVFP4" 2 2 --kernel-config.linear_backend b12x\n'
new = ('ENV="$ENV VLLM_GDN_DECODE_KERNEL=triton" tpm qwen38fn     "$MD/Qwen3.8-Flash-Next-NVFP4" 2 2 --kernel-config.linear_backend b12x\n'
       'ENV="$ENV VLLM_GDN_DECODE_KERNEL=triton" tpm qwen38fn_mtp "$MD/Qwen3.8-Flash-Next-NVFP4" 2 2 --kernel-config.linear_backend b12x \\\n'
       '  --speculative-config \'{"method":"qwen4_exp_mtp","num_speculative_tokens":3}\'\n')
if "qwen38fn_mtp" in s:
    print("  fleet2.sh: qwen38fn arms already hardened")
elif old in s:
    open(p, "w").write(s.replace(old, new))
    print("  fleet2.sh: qwen38fn arm gets VLLM_GDN_DECODE_KERNEL=triton and an MTP=3 arm")
else:
    print("  fleet2.sh: qwen38fn line not found verbatim; nothing changed")
