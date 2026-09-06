# Explicit-profile chat tripwire

`chat_probe.py` reads the exact 20 literal prompt/expected-substring pairs from `box/quality20.py`, without importing or executing that script. Supply an explicit JSON request configuration; `gates/profiles/qwen38fn_vendor.json` sets temperature 1, top-p 0.95, top-k 20, min-p 0, presence penalty 0, thinking enabled, and an 8192-token budget. The GLM probe profile retains AIRR's temperature 0.95/top-p 0.95/min-p 0/max-reasoning settings with the same explicit probe budget.

```bash
python3 gates/chat_probe.py \
  --model m --base-url http://127.0.0.1:8000 \
  --request-config gates/profiles/qwen38fn_vendor.json \
  --out-dir results/qwen_native_vendor_probe \
  --concurrency 10 --base-seed 1234
```

The output directory must be new. Existing greedy results cannot be overwritten. Every item gets a stable seed derived from the base seed, its index, and exact prompt. Each item's directory contains the exact serialized HTTP request, raw response body even on HTTP errors, and a parsed result with response usage, status, timestamps, and content hashes. The run manifest records the explicit configuration and source/prompt hashes, and remains partial until all 20 items are recorded.

Final-answer and reasoning repetition are measured separately. Capped completions and empty final answers cannot pass. Expected substrings are checked only in the final answer; a correct phrase in reasoning cannot rescue a wrong final answer. Repetition in reasoning is reported separately and is not by itself a failure when the final answer is valid, complete, and nonrepetitive. The original expected-substring checks remain lightweight corruption checks, not a rigorous task-accuracy scorer.

Exit code 0 means all 20 probe items passed; code 1 means a completed probe had failures; code 2 means invalid configuration or an output-path error. A smoke pass does not establish capability preservation. Compare matched runtime/model/prompt/budget profiles and then run the full paired capability gate for any optimization candidate.
