#!/usr/bin/env python3
"""Give SGLang's DSA backend a dense-prefill path on sm_120.

On H100/B200 any prompt shorter than index_topk (2048 for GLM-5.3-Flash) skips the sparse
kernels entirely: `use_mha` routes it to one-shot dense attention. That path is gated to
SM90/SM100 because it calls FlashAttention-3 or TRT-LLM ragged attention, neither of which
exists on sm_120, so on the RTX PRO 6000 every prompt goes through the indexer and the
sparse kernel with a 2048-slot top-k padded far beyond the sequence. FlashInfer's generic
ragged prefill (FA2 template, head_dim 256) does run on sm_120, so we (1) open the gate
for device_sm == 120 and (2) add a FlashInfer branch to `_forward_standard_mha`.

usage: python3 dsa_sm120.py /path/to/sglang/srt/layers/attention/dsa_backend.py
"""
import os, re, sys, ast

p = sys.argv[1]
if not os.path.exists(p + ".orig"):
    open(p + ".orig", "w").write(open(p).read())
s = open(p + ".orig").read()

# (1) the gate: accept sm_120 alongside SM90/SM100
gate = re.compile(r"(or \(device_sm >= 100 and device_sm < 110\))")
assert len(gate.findall(s)) == 1, f"gate anchor count {len(gate.findall(s))}"
s = gate.sub(r"\1 or device_sm == 120", s, count=1)

# (2) the branch: insert before the `>= 10` branch inside _forward_standard_mha
m = re.search(r"\n([ \t]+)if self\.device_sm_major >= 10:\n[ \t]+import flashinfer\n", s)
assert m, "branch anchor not found"
ind = m.group(1)
body = [
    "if self.device_sm_major == 12:",
    "    # sm_120 has neither FA3 nor TRT-LLM ragged attention; FlashInfer's generic",
    "    # ragged prefill (FA2 template) supports head_dim 256 here. (patched)",
    "    import flashinfer",
    "",
    "    wrapper = getattr(self, \"_sm120_ragged_wrapper\", None)",
    "    if wrapper is None:",
    "        ws = self.workspace_buffer",
    "        if ws is None:",
    "            ws = torch.empty(256 * 1024 * 1024, dtype=torch.uint8, device=q.device)",
    "            self.workspace_buffer = ws",
    "        wrapper = flashinfer.prefill.BatchPrefillWithRaggedKVCacheWrapper(ws, \"NHD\")",
    "        self._sm120_ragged_wrapper = wrapper",
    "    wrapper.plan(",
    "        cu_seqlens_q.to(torch.int32),",
    "        cu_seqlens_k.to(torch.int32),",
    "        layer.tp_q_head_num,",
    "        layer.tp_k_head_num,",
    "        layer.head_dim,",
    "        head_dim_vo=layer.v_head_dim,",
    "        causal=causal,",
    "        sm_scale=layer.scaling,",
    "        q_data_type=q.dtype,",
    "        kv_data_type=k.dtype,",
    "    )",
    "    return wrapper.run(q, k, v)",
]
branch = "\n" + "\n".join((ind + line) if line else "" for line in body)
s = s[: m.start()] + branch + s[m.start():]
ast.parse(s)
open(p, "w").write(s)
print("dsa_backend.py patched: gate opened for sm_120, FlashInfer ragged branch added; parses")
