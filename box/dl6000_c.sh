#!/usr/bin/env bash
source /workspace/bench/dlget.sh
L=/workspace/results/dl6000.log
sed -i -E "/\] (done|have) (Nemotron-3.5-Lightning-30B|Qwen3.6-35B-A3B-FP8|Muse-Glimmer-30B|gpt-oss-20b|gemma-4-26B-A4B-it|gemma-4-31B-it|MiniMax-M3-MXFP4|Qwen3.8-Flash-Next-NVFP4|Inkling-Small-NVFP4)( |$)/d" $L
echo "[$(date +%H:%M:%S)] re-queue of the fleet2 set (deleted by the rotation prologue)"
get nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4   Nemotron-3.5-Lightning-30B
get Qwen/Qwen3.6-35B-A3B-FP8                             Qwen3.6-35B-A3B-FP8
get meta-models/Muse-Glimmer-30B                         Muse-Glimmer-30B
get openai/gpt-oss-20b                                   gpt-oss-20b
get google/gemma-4-26B-A4B-it                            gemma-4-26B-A4B-it
get google/gemma-4-31B-it                                gemma-4-31B-it
get olka-fi/MiniMax-M3-MXFP4                             MiniMax-M3-MXFP4
get RadixArk/Qwen3.8-Flash-Next-NVFP4                    Qwen3.8-Flash-Next-NVFP4
get thinkingmachines/Inkling-Small-NVFP4                 Inkling-Small-NVFP4
echo DL6000C-DONE
