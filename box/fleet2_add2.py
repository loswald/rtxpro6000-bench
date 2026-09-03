#!/usr/bin/env python3
"""Second batch of late additions to /workspace/bench/fleet2.sh: speculative-decoding arms
for models already on the box, using the official/community drafters the roster audit found.
Each arm runs only if its drafter finished downloading. Idempotent."""
p = "/workspace/bench/fleet2.sh"
s = open(p).read()
if "drafter arms" in s:
    print("  fleet2.sh already has the drafter arms")
    raise SystemExit(0)

add = r'''
log "===== FLEET, drafter arms (speculation on models already measured) ====="
D=/workspace/models
# MiniMax-M3 + NVIDIA's DSpark drafter (native MTP is unusable: no checkpoint ships the MTP tensors).
if [ -f "$D/MiniMax-M3-DSpark/config.json" ]; then
  tpm minimaxm3_dspark "$MD/MiniMax-M3-MXFP4" 4 1 --moe-backend flashinfer_cutlass --quantization-config.moe.activation mxfp8 \
      --block-size 128 --speculative-config "{\"method\":\"dspark\",\"model\":\"$D/MiniMax-M3-DSpark\",\"num_speculative_tokens\":8}"
else log "SKIP minimaxm3_dspark (drafter absent)"; fi
# gemma-4-26B-A4B + Google's official MTP assistant (0.4 GB).
if [ -f "$D/gemma-4-26B-A4B-it-assistant/config.json" ]; then
  rep gemma26_mtp "$MD/gemma-4-26B-A4B-it" \
      --speculative-config "{\"method\":\"gemma4_mtp\",\"model\":\"$D/gemma-4-26B-A4B-it-assistant\",\"num_speculative_tokens\":3}"
else log "SKIP gemma26_mtp (assistant absent)"; fi
# Nemotron-3.5-Lightning: built-in MTP head (free), then NVIDIA's DSpark drafter.
rep nemo35_mtp "$MD/Nemotron-3.5-Lightning-30B" --kernel-config.linear_backend b12x \
    --speculative-config "{\"method\":\"nemotron_h_mtp\",\"num_speculative_tokens\":3}"
if [ -f "$D/Nemotron-3.5-Lightning-DSpark/config.json" ]; then
  rep nemo35_dspark "$MD/Nemotron-3.5-Lightning-30B" --kernel-config.linear_backend b12x \
      --speculative-config "{\"method\":\"dspark\",\"model\":\"$D/Nemotron-3.5-Lightning-DSpark\",\"num_speculative_tokens\":3}"
else log "SKIP nemo35_dspark (drafter absent)"; fi
# Muse-Glimmer-30B + Meta's official DFlash assistant.
if [ -f "$D/Muse-Glimmer-30B-assistant/config.json" ]; then
  rep muse30_dflash "$MD/Muse-Glimmer-30B" \
      --speculative-config "{\"method\":\"dflash\",\"model\":\"$D/Muse-Glimmer-30B-assistant\",\"num_speculative_tokens\":15}"
else log "SKIP muse30_dflash (assistant absent)"; fi
# Qwen3.8-27B NVFP4 (b12x) + the incoai DFlash2 block-diffusion drafter (vs DSpark measured in nvtier2).
if [ -f "$D/Qwen3.8-27B-DFlash2/config.json" ]; then
  rep q27_dflash2 "$MD/Qwen27B-NVFP4-RTX5090" --kernel-config.linear_backend b12x \
      --speculative-config "{\"method\":\"dflash\",\"model\":\"$D/Qwen3.8-27B-DFlash2\",\"num_speculative_tokens\":7}"
else log "SKIP q27_dflash2 (drafter absent)"; fi
'''
i = s.rfind('log "===== FLEET, late additions =====")')
if i < 0:
    i = s.rfind('log "===== FLEET, late additions')
s = (s[:i] + add + "\n" + s[i:]) if i > 0 else s.replace('log "FLEET2 DONE"', add + '\nlog "FLEET2 DONE"')
open(p, "w").write(s)
print("  fleet2.sh: drafter arms inserted before the late additions")
