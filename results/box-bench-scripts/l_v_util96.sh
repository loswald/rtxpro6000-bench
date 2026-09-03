#!/usr/bin/env bash
exec bash /workspace/bench/launch_x4.sh /workspace/models/gpt-oss-120b gptoss --gpu-memory-utilization 0.96 --max-num-batched-tokens 16384
