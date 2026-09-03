#!/usr/bin/env python3
"""Append the late additions to /workspace/bench/fleet2.sh (idempotent).

Done from a file rather than an SSH heredoc because the block needs both quote styles and
the remote command string is itself single-quoted.
"""
p = "/workspace/bench/fleet2.sh"
s = open(p).read()
if "late additions" in s:
    print("  fleet2.sh already has the late additions")
    raise SystemExit(0)

add = r'''
log "===== FLEET, late additions ====="
# Inkling-Small (Thinking Machines, Apache-2.0, 266B MoE, multimodal incl. audio). vLLM 0.28.1 has
# native inkling_mm_model + inkling_mtp support. Runs only if the download finished in time.
if [ -f "$MD/Inkling-Small-NVFP4/config.json" ]; then
  tpm inkling     "$MD/Inkling-Small-NVFP4" 4 1 --kernel-config.linear_backend b12x
  tpm inkling_mtp "$MD/Inkling-Small-NVFP4" 4 1 --kernel-config.linear_backend b12x \
      --speculative-config '{"method":"inkling_mtp","num_speculative_tokens":2}'
else
  log "SKIP inkling (download not finished)"
fi
# GLM-5.3-Flash MTP arm. The first attempt lost its JSON quotes in the launcher; glm_vllm.sh now
# routes the config through SPEC, and ARMS=mtp skips the already-measured base arm.
kill_all
ARMS=mtp bash /workspace/bench/glm_vllm.sh >> /workspace/results/glm_vllm_full.log 2>&1
'''
i = s.rfind('log "FLEET2 DONE"')
s = (s[:i] + add + "\n" + s[i:]) if i > 0 else (s + add)
open(p, "w").write(s)
print("  fleet2.sh: late additions appended before FLEET2 DONE" if i > 0 else "  fleet2.sh: appended at end (no FLEET2 DONE marker)")
