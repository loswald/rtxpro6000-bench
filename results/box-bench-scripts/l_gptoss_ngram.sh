#!/usr/bin/env bash
exec bash /workspace/bench/launch_x4.sh /workspace/models/gpt-oss-120b gptoss \
  --speculative-config '{"method":"ngram","num_speculative_tokens":5,"prompt_lookup_max":4,"prompt_lookup_min":2}'
