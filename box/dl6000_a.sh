#!/usr/bin/env bash
source /workspace/bench/dlget.sh
get gittensor-model-hub/Qwen3.8-27B-NVFP4-RTX5090   Qwen27B-NVFP4-RTX5090     # control vs the first box (5,161 out tok/s)
get Qwen/Qwen3.8-27B-FP8                            Qwen3.8-27B-FP8           # FP8 control (3,146)
get gittensor-model-hub/Qwen3.8-27B-DSpark-NVFP4    Qwen27B-DSpark-NVFP4      # drafter arm still owed from nvtier2
echo DL6000A-DONE
