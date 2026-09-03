# 4× RTX PRO 6000 Blackwell — throughput benchmark for the Scan decision

Sqwish Labs, 2 September 2026. Measured on a rented Vast.ai machine before committing to a 24‑month Scan node.

## 1. What was measured and how

**Machine.** Vast.ai verified host, 4× NVIDIA RTX PRO 6000 Blackwell Server Edition (96 GB GDDR7 each, compute capability 12.0 / sm_120), PCIe Gen5 x16 per GPU, no NVLink, 192 CPU threads, 1.5 TB RAM, 390 GB disk. $4.00/hr. Instance 49663078.

**Stack.** vLLM nightly `0.28.1rc1.dev312` (CUDA 13.0 build, main as of 2 Sept 2026), FlashInfer 0.6.18, b12x 1.3.0, torch 2.13.0+cu130, driver from the host. The image tag `vllm/vllm-openai:cu130-nightly` is stale (April 2026, vLLM 0.19); the stack was upgraded in place with uv.

**Model.** `deepseek-ai/DeepSeek-V4-Flash-0731` (MoE, ~284B total / 13B active, MXFP4 experts + FP8 attention/dense, 167 GB on disk, MIT). Served natively, no re-quantisation.

**Workload shapes** (random-token prompts, fixed output length, `--ignore-eos`):

| shape | input | output | models |
|---|---|---|---|
| short | 256 | 64 | classification, routing, short judges |
| router | 1,024 | 128 | routing, extraction |
| promptopt | 3,072 shared prefix + 512 unique | 256 | prompt‑optimisation loops (prefix caching intended) |
| judge | 4,096 | 512 | LLM‑as‑judge, counterfactual rewrites |
| rollout | 8,192 | 2,048 | agentic trajectory rollouts |

Concurrency C = requests in flight (16 → 256). Each run: 64–256 requests. Client: `vllm bench serve`, request rate ∞. A warm‑up pass preceded every sweep (JIT/CUDA‑graph shapes), and every measured run used a unique random seed so no prompt could hit the prefix cache from a previous run (the one dataset that did not do this is labelled contaminated below).

**Quality guard.** 20 fixed prompts at temperature 0 per layout, all coherent, no corruption signatures. No accuracy benchmark (GSM8K etc.) was run in this session.

## 2. Hardware truth (measured before any model ran)

| property | measured |
|---|---|
| PCIe link | Gen5 x16 on all four GPUs (Gen1 at idle) |
| topology | GPUs 0–1 and 2–3 share a PCIe switch (PIX); cross pairs are NODE |
| peer access | enabled on all pairs; NCCL transport `P2P/CUMEM` (true P2P, not host‑staged) |
| unidirectional peer copy | ~52 GB/s (≈82% of Gen5 x16) |
| NCCL all_reduce bus bandwidth | same‑switch pair ~21 GB/s · cross‑switch pair ~38 GB/s · all four GPUs ~19 GB/s (unchanged by `NCCL_P2P_LEVEL`/`NCHANNELS`/`PROTO`) |
| interpretation | PCIe **ACS is enabled** on the host: switch‑local traffic is bounced through the root complex, so the two GPUs behind one switch share one upstream link. Not changeable from a container. |
| power/thermals under load | 265–315 W per GPU of the 600 W limit at 99–100% utilisation, SM clock at max, no throttle reasons |
| other | HMM enabled (`uvm_disable_hmm=N`), ReBAR off, ECC on |

Consequences: tensor‑parallel layouts pay more per step here than on a BIOS‑tuned box (TP numbers are pessimistic); TP2 replicas must pair cross‑switch (GPUs 0+2 and 1+3); data‑parallel layouts are less sensitive.

## 3. Software findings on sm_120 (what it took to get DeepSeek‑V4‑Flash running)

1. **FP8 block‑scaled linear kernel.** vLLM's chooser selects the CUTLASS c3x kernel, which fails at dispatch on sm_120 (`dispatch_scaled_mm ... scaled_mm_helper.hpp:17`). Fix: `--kernel-config.linear_backend b12x`. The Triton alternative fails on this checkpoint's UE8M0 scale format (`KeyError: float8_e8m0fnu`).
2. **Attention output projection.** The FlashInfer sparse‑MLA backend for DeepSeek‑V4 routes `o_proj` through a DeepGEMM FP8 einsum, which asserts on sm_120 (`layout.hpp:40: t.dim() == N`). Patched with a cached bf16 dequantised einsum fallback (`patches/vllm_dsv4_nvidia_ops_o_proj.py`, enabled by `VLLM_DSV4_OPROJ_SM120_FALLBACK=1`). Correct output; small compute cost.
3. **MoE kernel.** `--moe-backend b12x` (native MXFP4 on sm_120) refuses expert‑parallel deployments; with EP the Marlin kernel is used. Without EP, b12x runs and measured the same throughput as Marlin, so the MoE GEMM is not the bottleneck at these shapes.
4. **KV layout.** `--kv-cache-dtype fp8_ds_mla` is rejected for this path; plain `fp8` works (43.9 GiB KV per GPU ≈ 282k tokens at TP4).
5. **Speculative decoding.** DSpark (k=7, probabilistic) runs but *reduces* throughput at every measured point; mean acceptance length ≈1.8.
6. **GLM‑5.3‑Flash** was not attempted: vLLM issue #53963 documents no sm_120 sparse‑MLA path and FP8 failing for this model.
7. DeepEP is not importable in this image (compiled against another torch ABI) — harmless here, it is an NVLink/IB library.

Load time from page cache: ~11 s for 167 GB; startup incl. CUDA‑graph capture ≈2 min.

<!-- SECTIONS FROM THE ANALYSIS WORKFLOW ARE INSERTED BELOW -->
