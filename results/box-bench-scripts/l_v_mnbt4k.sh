#!/usr/bin/env bash
exec bash /workspace/bench/launch_x4.sh /workspace/models/gpt-oss-120b gptoss --max-num-batched-tokens 4096
