#!/usr/bin/env bash
source /workspace/bench/dlget.sh
# wait for the control-model batch to finish first (same log, same tmux session lineage)
for i in $(seq 1 60); do grep -q DL6000A-DONE /workspace/results/dl6000.log && break; sleep 30; done
# fleet2 replica tier, in run order
get nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4   Nemotron-3.5-Lightning-30B
get Qwen/Qwen3.6-35B-A3B-FP8                             Qwen3.6-35B-A3B-FP8
get meta-models/Muse-Glimmer-30B                         Muse-Glimmer-30B
get openai/gpt-oss-20b                                   gpt-oss-20b
get google/gemma-4-26B-A4B-it                            gemma-4-26B-A4B-it
get google/gemma-4-31B-it                                gemma-4-31B-it
# fleet2 TP tier
get olka-fi/MiniMax-M3-MXFP4                             MiniMax-M3-MXFP4
get RadixArk/Qwen3.8-Flash-Next-NVFP4                    Qwen3.8-Flash-Next-NVFP4
# drafters (fleet2_add2)
get nvidia/MiniMax-M3-DSpark                             MiniMax-M3-DSpark
get google/gemma-4-26B-A4B-it-assistant                  gemma-4-26B-A4B-it-assistant
get nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4-DSpark Nemotron-3.5-Lightning-DSpark
get meta-models/Muse-Glimmer-30B-assistant               Muse-Glimmer-30B-assistant
get incoai/Qwen3.8-27B-DFlash2                           Qwen3.8-27B-DFlash2
# late additions (fleet2_add): Inkling +- MTP drafter, GLM-5.3-Flash
get thinkingmachines/Inkling-Small-NVFP4                 Inkling-Small-NVFP4
get RadixArk/Inkling-Small-DSpark                        Inkling-Small-DSpark
get RedHatAI/GLM-5.3-Flash-NVFP4                         GLM-5.3-Flash-NVFP4
T=/workspace/models/glm53f_tok; mkdir -p $T; cp /workspace/models/GLM-5.3-Flash-NVFP4/tok* /workspace/models/GLM-5.3-Flash-NVFP4/*.jinja $T/ 2>/dev/null; ls $T | paste -sd" " | sed "s/^/[glm53f_tok] /"
echo DL6000B-DONE
