# rtxpro6000-bench — Blackwell node serving benchmark for Sqwish Labs

Everything behind the decision on a two-year 4×/7× RTX PRO 6000 Blackwell (sm_120) commitment for dedicated LLM inference:
the benchmark harness, the per-model serving recipes that actually work on this silicon, the raw
measurements, the roster audit, and the running report.

Report (kept current): https://claude.ai/code/artifact/60fe58ea-4220-4e4d-8d53-71495220864f

## Layout

| path | what |
|---|---|
| `bench/` | local harness: `launch.sh`, `sweep.sh`, `summarise.py`, `rr_proxy.py`, `collect_env.sh`, `setup_engine.sh` |
| `box/` | scripts that run on the rented node: campaign runners (`nvtier*.sh`, `fleet2.sh`, `roster3.sh`, `glm_vllm.sh`, `qtier.sh`), `agg.py` (per-port aggregation), `quality20.py` (corruption tripwire), `logit_diff.py` (KL / top-k quantization metric), `hardkill.sh` / `cleanup.sh` (safe teardown), `pull_image.py` (lift a vendor Docker image without Docker), `vllm_sm120_nope.py` (the GLM-5.3-Flash port) |
| `cells/`, `gates/`, `train/`, `vast/`, `patches/` | sweep cells, quality gates, training probes, Vast.ai provisioning, source patches (DeepSeek-V4 o_proj on sm_120) |
| `results/` | raw `vllm bench serve` JSON per port, aggregated `summary_full.tsv`, server logs, tripwire verdicts |
| `report/` | the HTML report and the roster audit |
| `econ/` | cost stack, runway and make-vs-buy models |

## Headline numbers (4 × RTX PRO 6000, steady state, 3 Sept 2026)

Per node, per minute. Shapes: router 1,024 in / 128 out; promptopt 3,072 shared prefix + 512 / 256;
judge 4,096 / 512; short 256 / 64. Artificial Analysis Intelligence Index v4.1.1 in brackets.

| model · configuration | shape | conc. | req/min | input tok/min | output tok/min |
|---|---|---|---|---|---|
| Qwen3.8-27B NVFP4 (52) · 4 replicas · `b12x` W4A4 | router | 1,024 | 2,419 | 2,477,000 | 310,000 |
| | promptopt | 1,024 | 1,126 | 4,036,000 | 288,000 |
| | judge | 512 | 516 | 2,112,000 | 264,000 |
| Qwen3.8-27B FP8 (52) · 4 replicas · `b12x` | router | 1,024 | 1,475 | 1,510,000 | 189,000 |
| Qwen3.8-27B NVFP4 · vLLM auto kernel (W4A16) | router | 1,024 | 783 | 802,000 | 100,000 |
| gpt-oss-120b (24) · 4 replicas · FlashInfer CUTLASS mxfp8 | short | 1,024 | 13,826 | 3,539,000 | 885,000 |
| | router | 1,024 | 5,115 | 5,238,000 | 655,000 |
| | promptopt | 1,024 | 3,468 | 12,431,000 | 888,000 |
| DeepSeek-V4-Flash · DP4 + EP | promptopt | 512 | 863 | 3,094,000 | 221,000 |
| GLM-5.3-Flash (57) · ported vLLM · TP4 | router | 256 | 431 | 442,000 | 55,000 |
| | promptopt | 256 | 235 | 842,000 | 60,000 |

## The rules that cost the most to learn

1. **Force the kernel; never trust the default.** On sm_120 the only true 4-bit-compute NVFP4 paths are
   `--kernel-config.linear_backend b12x` and `flashinfer_b12x` (+64% over FP8 on Qwen3.8-27B). vLLM's
   auto pick and the CuTe-DSL backend are W4A16 dequant paths and run *slower* than FP8. For MXFP4 MoE
   use `--moe-backend flashinfer_cutlass --quantization-config.moe.activation mxfp8` (+26.5% on gpt-oss).
2. **NVFP4 beats FP8 on both speed and accuracy on Blackwell.** Prefer RTX-5090 / `sm120` community builds;
   sm_100 datacentre builds fail here.
3. **Speculation must be measured by acceptance counters**, never inferred. A flat loss across shapes is
   a fixed per-step tax, not a finding.
4. **New architectures ship as vendor Docker images before they land upstream.** `box/pull_image.py` lifts
   them without Docker; GLM-5.3-Flash needed that plus an sm_120 port of its NoPE sparse-MLA backend.
5. **`--no-enable-flashinfer-autotune` on every TP launch**, or a rank with a cached tuning result deadlocks
   the ranks that are still profiling.
6. **Steady state needs eight prompts per concurrent slot and a unique seed per run**, and every A/B needs
   an in-session control.
7. An NVFP4 MoE checkpoint is **1.1–1.25 bytes per listed parameter**, not 0.55. Sum the file tree.
8. **TP=7 is impossible** on a 7-GPU node (prime). It runs pipeline-parallel or TP2×PP3 on six cards.

## Reproducing

Every server launch in `box/*.sh` writes the exact command it ran to `l_*.sh` and the kernel the engine
selected to the server log; `results/kernels_by_server.tsv` collects those kernel lines, one per server.
`box/agg.py` writes one `summary.tsv` per run under `results/probe/<tag>/` with one row per
(shape, concurrency): req/s, in/out/total tok/s, TTFT/TPOT/ITL/E2E at p50/p99. `results/summary_all.tsv`
is the concatenation of all of them (213 rows across 39 runs at the time of writing). Raw per-port
`vllm bench serve` JSON sits beside each summary; the multi-GB server logs stay on the node.
