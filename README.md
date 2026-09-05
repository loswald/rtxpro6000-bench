# rtxpro6000-bench

Throughput benchmark of open-weight LLMs on a rented **4x NVIDIA RTX PRO 6000 Blackwell Server Edition** box
(96 GB GDDR7 each, PCIe Gen5 x16, no NVLink, compute capability 12.0 = `sm_120`) on Vast.ai, run by Sqwish Labs
(Cambridge, UK) before committing to a 2-year Scan-hosted node.

**Goal: maximum throughput** -- output tok/s, total tok/s and requests/s at saturation. TTFT/TPOT are logged for the
record only. Everything runs *inside* one Vast.ai container as root (SSH, GPUs exposed; no nested Docker, no BIOS;
tmux available). Engine: vLLM, upgraded **in place** with `uv` from the stale image build (`vllm/vllm-openai:cu130-nightly`
= vLLM 0.19.2) to **vLLM `0.28.1rc1.dev312+g41848caa6`** (vLLM main as of 2026-09-02, CUDA 13.0 wheels from
`https://wheels.vllm.ai/nightly/cu130`) + `flashinfer-python 0.6.18` + the `vllm[b12x]` extra. Optional SGLang A/B on a
second instance (`lmsysorg/sglang:v0.5.18-cu130`). Bash + Python 3 stdlib only; `curl`, `python3`, `tmux` are all the
harness needs (`jq` may be missing on the box).

---

## Measured on the Vast box (2026-09-02)

Collected once by hand and by `vast/hardware_truth.sh`; the harness does **not** re-derive these, it reads them from
`results/hw/decisions.env`.

| what | measured | consequence |
|---|---|---|
| GPUs | 4x RTX PRO 6000 Blackwell **Server Edition**, 96 GB each, cc 12.0 (`sm_120`), PCIe Gen5 x16, **no NVLink** | TP is PCIe-bound on this box and on the Scan node alike |
| `nvidia-smi topo -m` | GPUs 0-1 `PIX` (same PCIe switch), 2-3 `PIX`, every cross pair `NODE` | two switch groups {0,1} and {2,3} |
| peer access | `torch.cuda.can_device_access_peer` = True for **all** pairs; unidirectional peer copy ~52 GB/s | `P2P_OK=1` |
| NCCL transport | `P2P/CUMEM` (real P2P, **not** SHM) | `NCCL_P2P_DISABLE` stays **0**; never set automatically |
| NCCL `all_reduce` busbw | pair 0-1 (same switch) **~21 GB/s**; pair 0-2 (cross switch) **~38 GB/s**; 4-GPU ring **~19 GB/s** -- regardless of `NCCL_P2P_LEVEL` / `NCCL_MIN_NCHANNELS` / `NCCL_PROTO` tuning | same-switch pairs are the *slow* ones |
| interpretation | PCIe **ACS is enabled on the host**: switch-local P2P is redirected through the root complex. Not changeable from the container | `ACS_SUSPECTED=1`, `PESSIMISTIC_TP=1`: **TP2/TP4 rows carry a dagger**, TP1 replica rows do not |
| TP2 pairing | a TP2 pair should span both switches | DP2xTP2 cells use **`GPU_IDS=0,2,1,3`** (vLLM gives DP rank *i* the *i*-th TP-sized slice of `CUDA_VISIBLE_DEVICES`: rank 0 -> physical 0,2; rank 1 -> 1,3; verified in `vllm/v1/engine/utils.py:get_physical_gpu_ids_for_local_dp_rank`). TP4 cells stay on `0,1,2,3` and are pessimistic |
| custom all-reduce | untested win on a PCIe/ACS box | **off by default** (`--disable-custom-all-reduce`); `CUSTOM_ALLREDUCE=1 bench/launch.sh <cell>` for an A/B |
| host | 192 CPU threads, **1.5 TB RAM**, HMM enabled (`uvm_disable_hmm=N`), ReBAR off | Qwen3.8-Flash-Next DP2 (~51 GB host RAM per DP rank for the N-gram table) fits easily |
| disk | **~390 GB total, 373 GB free at start** | models are benchmarked **one at a time and deleted** (order below) |
| price | ~$4/h | `bench/summarise.py --cost-per-hour 4` |

## Decisions the harness implements (they override anything older in the files)

1. **Never** auto-set `NCCL_P2P_DISABLE=1`. The old contract "P2P latency > 5 us -> NCCL_P2P_DISABLE=1" was wrong for this
   box and is gone. Only a human sets it (in `decisions.env` with `HUMAN_DECISION=1`) and only if peer access is unsupported.
2. Custom all-reduce **off** by default; `CUSTOM_ALLREDUCE=1` per launch for an A/B.
3. TP2 replicas pair **cross-switch**: DP2xTP2 cells set `GPU_IDS=0,2,1,3` (comment in the cell says why). TP4 on `0,1,2,3`, pessimistic.
4. One hardware/decision contract: `results/hw/decisions.env` (`P2P_OK CUSTOM_ALLREDUCE NCCL_P2P_DISABLE ACS_SUSPECTED
   PESSIMISTIC_TP HOST_RAM_GB NOTES`), written by `vast/hardware_truth.sh`, sourced by `bench/env.sh`; every sweep meta JSON
   and `bench/summarise.py` row carries `acs_suspected` / `pessimistic_tp` so TP2/TP4 rows get a dagger and TP1 rows do not.
   Without the file the two flags are recorded as `null` ("unknown", never a silent 0) and TP>1 rows show `?`.
5. Models are plain directories: `hf download <repo> --local-dir /workspace/models/<basename>`; `load_cell` derives
   `MODEL_PATH=${MODELS_DIR:-/workspace/models}/$(basename "$MODEL")` and uses it for `vllm serve` and the bench
   `--tokenizer` when it exists (else the HF id). `bench/prefetch.sh <cell|id>` downloads into that layout (with a free-disk
   guard), `bench/prefetch.sh --list` shows what is there, `bench/prefetch.sh --delete <cell|id> --yes` frees the disk.
6. Throughput-first sweep with a time budget: per-shape concurrency lists and prompt counts (section 6), spec decoding OFF
   everywhere, `--max-num-seqs 256`, `--max-num-batched-tokens 8192` default + a `RUN_TAG=mnbt16k` 16384 variant, estimated
   minutes per run logged.
7. Correctness gates drive the **real** launcher: `KV_CACHE_DTYPE=<kv> RUN_TAG=<tag> bench/launch.sh <cell>` (port = the
   cell's `BASE_PORT`); the filenames in this README are the real ones.
8. Host RAM is 1.5 TB: the Qwen3.8-Flash-Next DP2 cell fits; `launch.sh` warns (never dies) when free RAM < needed, using
   `HOST_RAM_GB` from `decisions.env` or `free -g`.
9. No new dependencies beyond the vLLM image (bash, python3 stdlib, curl, tmux).

---

## Layout

```
rtxpro6000-bench/
├── README.md                  this file: measured truth -> decisions -> layout -> flow -> disk-constrained order -> results
├── vast/                      laptop-side provisioning + in-container hardware truth (vast track)
│   ├── README.md              vastai CLI flow, image-tag truth, onstart guard, decisions.env contract, 390 GB disk budget
│   ├── search_offers.sh       vastai search offers with the campaign filters
│   ├── create_instance.sh     vastai create instance --image ... --ssh --direct --onstart onstart.sh   (new box only)
│   ├── onstart.sh             in-container boot: env, apt, isolated hf-CLI venv, optional cuda-samples/nccl-tests builds,
│   │                          optional downloads (--local-dir layout, disk guard), waits for the harness, runs hardware_truth.sh.
│   │                          NEVER pip-installs into the engine interpreter (ALLOW_ENGINE_PIP=0)
│   ├── hardware_truth.sh      nvidia-smi -q/topo, PCIe link, peer-access + P2P copy matrix (torch), NCCL all_reduce busbw for the
│   │                          4-GPU ring / same-switch pair / cross-switch pair (nccl-tests or torch fallback)
│   │                          -> results/hw/decisions.env (THE contract) + results/hw/hardware.json + results/hw/machine.env (facts only)
│   ├── sync.sh                push harness to /workspace/rtxpro6000-bench, pull results/ (incl. results/hw/), ssh/tmux helpers
│   └── COST.md                $/h ranges, the ~$4/h actual, campaign estimate
├── bench/                     engine launch + sweeps (bench track)
│   ├── env.sh                 shared defaults: sm_120 vLLM env, NCCL env, sources results/hw/decisions.env, load_cell() (MODEL_PATH), shapes
│   ├── setup_engine.sh        UPGRADE_VLLM=1: engine upgrade in place with uv ONLY (vLLM main cu130 nightly wheels pinned to
│   │                          VLLM_TARGET_VERSION, flashinfer-python 0.6.18, vllm[b12x]; never pip into the engine); hf CLI -> /workspace/venv-tools
│   ├── prefetch.sh <cell|id>  hf download <repo> --local-dir $MODELS_DIR/<basename> (the only layout the harness uses) with a
│   │                          free-disk guard; --list; --delete <cell|id> --yes frees the 390 GB disk between models
│   ├── launch.sh <cell>       tmux session bench_<cell>, one window per server, waits for /health -> results/<cell>[__RUN_TAG]/launch.json
│   ├── stop.sh <cell>|--all   tear down and wait for GPU memory release
│   ├── sweep.sh <cell> [shapes] [concurrencies]   vllm bench serve sweep + nvidia-smi dmon -> results/<cell>/
│   ├── _run_server.sh         internal tmux runner (env file, log, exit code)
│   ├── rr_proxy.py            round-robin proxy (:8080) in front of x4 replica cells
│   ├── summarise.py           results/**/<run>.json + dmon -> results/summary.md/.csv/.json (dagger on pessimistic TP rows)
│   └── collect_env.sh         driver/CUDA/engine versions, GPU edition/vBIOS/power, image id, decisions -> results/env.json
├── cells/                     one .env per cell (model, TP/DP/replicas, GPU_IDS, KV dtype, extra args)
│   ├── README.md              cell table and conventions; _template.env documents every key
│   └── *.env                  see the cells table below
├── gates/                     correctness gates (gates track)
│   ├── hwdecisions.py         read-only helper for results/hw/decisions.env: hw_decisions(), pessimistic_flags(tp=...) -> the
│   │                          acs_suspected / pessimistic_tp fields every gate/co-tenancy/sleep-wake JSON carries (CLI: --json | --shell)
│   ├── gsm8k.sh <cell> [port] lm-eval GSM8K 200-question subset -> results/<cell>/gsm8k.json
│   ├── kv_diff.py             50 fixed prompts, FP8-KV vs bf16-KV capture/compare, sm_120 corruption flags
│   ├── run_kv_diff.sh <cell> [kvA] [kvB]   launches the cell twice via bench/launch.sh (KV_CACHE_DTYPE/RUN_TAG) -> kv_diff.json
│   └── gates_summary.py       Markdown of all gate / co-tenancy / sleep-wake results -> results/gates_summary.md
├── train/                     training co-tenancy (train track)
│   ├── install_unsloth.sh     isolated Unsloth venv for Blackwell (cu128+ torch, triton>=3.3.1, no xformers)
│   ├── lora_qwen8b.py         Unsloth LoRA r=16 bf16 on Qwen/Qwen3-8B, fixed wall clock, tokens/s logging
│   ├── lora_cotenant.sh       before/during serving benchmark around a 15 min training run on GPU 3 -> results/cotenancy.json
│   └── sleep_wake.sh [cell|port]   vLLM sleep-mode level-1 timing -> results/sleep_wake.json
├── patches/                   hand patch, UNVERIFIED (2026-09-02): DeepSeek-V4 o_proj FP8-einsum scale recipe for sm_120 (DeepGEMM has
│   ├── apply_dsv4_o_proj.sh   no sm_120 kernels) -- --check | --apply | --revert; apply only if a ds4flash launch dies in deep_gemm_fp8_o_proj
│   └── vllm_dsv4_nvidia_ops_o_proj.py (+ .orig)   patched vllm/models/deepseek_v4/nvidia/ops/o_proj.py and the upstream file it came from
└── results/                   one JSON per run + gate outputs + hw/ (never edit by hand; `vast/sync.sh pull`)
```

Scripts are authored on Windows; `vast/sync.sh push` strips CRLF and sets `+x`. For a hand copy:
`sed -i 's/\r$//' */*.sh && chmod +x bench/*.sh gates/*.sh train/*.sh vast/*.sh`.

On the box: `/workspace/rtxpro6000-bench` = this harness; `/workspace/models` = weights (plain directories);
`/workspace/bench` = the box's own scratch scripts (not ours); `/workspace/tools`, `/workspace/venv-*` = helpers.

---

## Flow

```
(box exists) -> vast/sync.sh push -> vast/hardware_truth.sh (results/hw/decisions.env) -> bench/setup_engine.sh (engine upgrade)
 -> bench/collect_env.sh -> per model, disk permitting: bench/prefetch.sh -> bench/launch.sh -> bench/sweep.sh (8192, then mnbt16k)
 -> gates/gsm8k.sh + gates/run_kv_diff.sh -> bench/stop.sh -> rm -rf /workspace/models/<name> -> next model ...
 -> train/lora_cotenant.sh (+ train/sleep_wake.sh) -> bench/collect_env.sh again -> bench/summarise.py + gates/gates_summary.py
 -> vast/sync.sh pull -> vastai destroy instance
```

### 1. Provision (laptop) -- the current instance already exists; details in `vast/README.md`

```bash
pip install -U vastai && vastai set api-key <KEY>            # key lives only in the CLI config
./vast/sync.sh push <ID>                                      # harness -> /workspace/rtxpro6000-bench (CRLF stripped, +x)
./vast/sync.sh ssh <ID>                                       # shell on the box (cd /workspace/rtxpro6000-bench)
# new box only:  ./vast/search_offers.sh   ->   ./vast/create_instance.sh <OFFER_ID>   (DOWNLOAD_SET=none, ALLOW_ENGINE_PIP=0)
```

`onstart.sh` (for a new box; safe to re-run on this one) never touches the engine interpreter, puts the `hf` CLI in
`/workspace/venv-tools`, and creates `/workspace/venv-eval` (lm-eval) and `/workspace/venv-train` (unsloth) with
`--system-site-packages`; the gates and trainer pick those up and fall back to `/opt/lmeval-venv` / `/opt/unsloth-venv`.

### 2. Hardware truth (FIRST, before any engine touches the GPUs)

```bash
bash vast/hardware_truth.sh                 # ~5 min; nccl-tests/cuda-samples optional (torch fallbacks)
cat results/hw/decisions.env
```

Writes `results/hw/decisions.env` (the contract), `results/hw/hardware.json` (+ copy `results/hardware.json`) and
`results/hw/machine.env` (machine facts only: GPU name/vBIOS/driver, switch pairs, TP2 cross-switch order -- no decision keys,
sourced by nothing). Nothing is copied to `./env.sh` any more and login shells no longer source such a file (a stale `./env.sh`
from an older run is reported at the end; `rm -f` it). Expected on this box:

```
P2P_OK=1                # peer access supported on all pairs (bandwidth/latency recorded, never a gate)
CUSTOM_ALLREDUCE=0      # --disable-custom-all-reduce by default; CUSTOM_ALLREDUCE=1 for the A/B
NCCL_P2P_DISABLE=0      # NEVER set to 1 by a script
ACS_SUSPECTED=1         # same-switch pair 21 GB/s < cross-switch pair 38 GB/s
PESSIMISTIC_TP=1        # dagger on TP2/TP4 rows
HOST_RAM_GB=1510
NOTES="..."             # + informational SAME_SWITCH_PAIRS=0-1,2-3  TP2_CROSS_SWITCH_GPU_IDS=0,2,1,3
```

`bench/env.sh` sources it (values already in the environment win, so launch-time overrides such as `CUSTOM_ALLREDUCE=1`
work); `gates/hwdecisions.py` is the read-only Python view of the same file (`pessimistic_flags(hw_decisions(), tp=TP)`);
every `launch.json`, sweep `*.meta.json`, `bench/summarise.py` row and gate JSON carries `acs_suspected` and
`pessimistic_tp`. **Dagger rule:** `pessimistic_tp == 1` and `TP > 1` (TP2, TP4, DP-over-TP) -> dagger; TP1 replica cells
(27B x4, gpt-oss x4) are never daggered. If `decisions.env` is missing when a run is made, both flags are `null`
("unknown") and `summarise.py` / `gates_summary.py` mark TP>1 rows with `?` -- run `vast/hardware_truth.sh` first.

### 3. Engine + environment snapshot

```bash
bash bench/setup_engine.sh                  # uv upgrade in place: vLLM main cu130 nightly + flashinfer-python 0.6.18 + vllm[b12x]; tmux/curl
IMAGE=vllm/vllm-openai:cu130-nightly IMAGE_DIGEST=sha256:<from the Vast instance page> bash bench/collect_env.sh
```

-> `results/env.json` (+ `nvidia-smi-q.txt`, `nvidia-smi-topo.txt`, `pip-list.txt`) with the real `vllm.__version__`
(`0.28.1rc1.dev312+g41848caa6`), torch/CUDA/NCCL/flashinfer/b12x versions and the decisions. Re-run before summarising.

### 4. Models on disk (390 GB: one big model at a time)

Layout (decision 5): `hf download <repo> --local-dir /workspace/models/<basename-of-repo>` -> e.g.
`/workspace/models/DeepSeek-V4-Flash-0731`, `/workspace/models/Qwen3.8-27B-FP8`, `/workspace/models/gpt-oss-120b`
(plain directories, **not** the HF cache layout). `load_cell` uses the directory for `vllm serve` and for the bench client's
`--tokenizer` when it exists, else the HF id.

```bash
bench/prefetch.sh qwen38_27b_fp8_x4         # hf download ... --local-dir $MODELS_DIR/Qwen3.8-27B-FP8 (cell name or HF id);
                                            # refuses when free disk < size + 10 % + 5 GB (PREFETCH_FORCE_DISK=1 overrides)
bench/prefetch.sh --list                    # size + complete/partial per directory, df
bench/prefetch.sh --delete gptoss120b_x4_marlin --yes   # = rm -rf /workspace/models/gpt-oss-120b; refuses while a process uses it
df -h /workspace; du -sh /workspace/models/*
rm -rf /workspace/models/gpt-oss-120b       # the same by hand, once the cell's sweeps + gates are pulled (vast/sync.sh pull)
```

Already on disk on 2026-09-02: DeepSeek-V4-Flash-0731 (167 GB), Qwen3.8-27B-FP8 (31), gpt-oss-120b (65) -> ~110 GB free.

### 5. Cells

| cell | model | layout / `GPU_IDS` | KV | dagger | notes |
|---|---|---|---|---|---|
| `gptoss120b_x4_marlin` / `gptoss120b_x4_ficutlass` / `gptoss120b_x4_b12x` | openai/gpt-oss-120b (MXFP4, 65 GB) | 4x TP1, ports 8000-8003, `0,1,2,3` (`REPLICAS`/`GPU_IDS` env-overridable: `gates/run_kv_diff.sh` launches one replica) | fp8 | no | `--moe-backend marlin` baseline vs `flashinfer_cutlass` (expected rejected on sm_120) vs `b12x` (the sm_12x MXFP4 kernels from `vllm[b12x]`; `flashinfer_b12x` is a different backend and is rejected for MXFP4) |
| `qwen38_27b_bf16_x4` / `qwen38_27b_fp8_x4` | Qwen/Qwen3.8-27B / -FP8 | 4x TP1, `0,1,2,3` (env-overridable, as above) | fp8 | no | dense; FP8 cell is the kv_diff target, bf16 the GSM8K reference |
| `ds4flash_tp2x2` | deepseek-ai/DeepSeek-V4-Flash-0731 (167 GB) | DP2xTP2, **`0,2,1,3`** (cross-switch pairs) | `fp8_ds_mla` (fallback `fp8`, the recipe's sm_120 value) | yes | native MXFP4 experts + FP8; DSpark draft OFF; memory-tight (~84 GB weights/GPU) |
| `ds4flash_tp4` | same | TP4, `0,1,2,3` | same | yes | recipe flags for sm_120: `--block-size 256 --moe-backend auto --attention_config.use_fp4_indexer_cache False --tokenizer-mode deepseek_v4` |
| `qwen38flashnext_fp8_tp2x2` | Qwen/Qwen3.8-Flash-Next-FP8 (186 GB) | DP2xTP2, **`0,2,1,3`** | **bfloat16** | yes | `VLLM_PLE_CPU_OFFLOAD=1` (51B N-gram table in host RAM, ~51 GB per DP rank -> ~102 GB of 1.5 TB), `--no-enable-flashinfer-autotune` (recipe) |
| `qwen38flashnext_fp8_tp4` | same | TP4, `0,1,2,3` | bfloat16 | yes | the recipe's 4-GPU row |
| `qwen35_122b_fp8_tp2x2` | Qwen/Qwen3.5-122B-A10B-FP8 (127 GB) | DP2xTP2, **`0,2,1,3`** | fp8 | yes | `ENABLE_EP=1` A/B (`RUN_TAG=ep`) |
| `glm53flash_fp8_tp4_loadtest` | zai-org/GLM-5.3-Flash (328 GB FP8) | TP4, `LOADTEST_ONLY=1` | fp8 -> bf16 retry | yes | attempt-to-load only; needs the disk otherwise empty; `launch.sh` records status + error excerpt in `loadtest.json` |
| `cotenant_tp2_gpu01` | Qwen/Qwen3.5-122B-A10B-FP8 (override `COTENANT_MODEL`) | TP2 on **`0,2`** (cross-switch pair; `COTENANT_GPU_IDS=0,1` for the same-switch A/B), sleep mode on; GPU 3 trains, GPU 1 idle | fp8 | yes | serving half of the co-tenancy cell; the cell keeps its historical name |

Common sm_120 baseline (from `bench/env.sh`): `VLLM_USE_DEEP_GEMM=0 FLASHINFER_CUDA_ARCH_LIST=12.0f TORCH_CUDA_ARCH_LIST=12.0
NCCL_P2P_LEVEL=PHB NCCL_IB_DISABLE=1 NCCL_MIN_NCHANNELS=8 VLLM_SERVER_DEV_MODE=1`, and
`--compilation-config '{"cudagraph_mode":"FULL_AND_PIECEWISE"}' --no-enable-flashinfer-autotune --enable-prefix-caching
--enable-chunked-prefill --max-num-seqs 256 --max-num-batched-tokens 8192 --disable-custom-all-reduce` (the last one
dropped only with `CUSTOM_ALLREDUCE=1`). No `--speculative-config` anywhere (decision 6). Override per launch:
`MAX_NUM_BATCHED_TOKENS=16384 RUN_TAG=mnbt16k bench/launch.sh ds4flash_tp4` -> `results/ds4flash_tp4__mnbt16k/`.

```bash
bench/launch.sh qwen38_27b_fp8_x4 --dry-run   # print the exact vllm serve line(s) + proxy; no tmux, nothing under results/
bench/launch.sh qwen38_27b_fp8_x4          # blocks until /health on all ports; launch.json has KV capacity, time-to-ready, decisions
bench/sweep.sh  qwen38_27b_fp8_x4 --dry-run   # plan (prompts + est. minutes per point) and the bench commands, no server needed
bench/sweep.sh  qwen38_27b_fp8_x4          # 3 shapes x their concurrency lists (x4 cells: C split across replicas)
bench/stop.sh   qwen38_27b_fp8_x4
```

### 6. Sweeps (throughput-first, time-budgeted)

`bench/sweep.sh <cell> [shapes] [concurrencies] [--dataset-path real.jsonl]` runs `vllm bench serve --request-rate inf
--max-concurrency C --num-prompts N --ignore-eos --random-input-len IN --random-output-len OUT --random-range-ratio 0
--percentile-metrics ttft,tpot,itl,e2el --metric-percentiles 50,90,99 --save-result --metadata ...` per point, sampling
`nvidia-smi dmon -s pucm` once per second. Defaults (all overridable per shape via env / positional args):

| shape | in / out | concurrencies | prompts per point | est. minutes per point |
|---|---|---|---|---|
| router | 1024 / 128 | 1 4 8 16 32 64 128 256 | max(4C, 64) | ~1 |
| judge | 4096 / 512 | 1 4 16 64 128 256 | max(4C, 64) | 1-5 |
| agent | 32768 / 2048 | 1 4 16 64 | **max(2C, 16)** | 1-6 |

Why: the old agent-shape rule (`C=256`, `4C` prompts) would prefill 1024 x 32768 = ~33.5M tokens per point (~30 min) for
a number nobody needs -- the agent shape saturates KV long before C=64 on 96 GB GPUs. `bench/sweep.sh` logs the estimated
minutes per point before it starts. Real prompts: `--dataset-path real.jsonl` selects `--dataset-name custom`, one
`{"prompt": "...", "output_tokens": N}` per line -- the per-line key is **`output_tokens`** (vLLM main
`vllm/benchmarks/datasets/datasets.py`, 2026-09-02); `--custom-output-len N` (default 256) fixes the output length for every
prompt, `-1` honours the per-line key. Each point leaves `<run_id>.json` (bench output incl. `--metadata`: engine version,
model, TP/DP, kv dtype, concurrency, shape, gpu_ids, `acs_suspected`, `pessimistic_tp`), `<run_id>.meta.json` and
`<run_id>.dmon.csv` in `results/<cell>[__RUN_TAG]/`. Repeat with `MAX_NUM_BATCHED_TOKENS=16384 RUN_TAG=mnbt16k` (launch +
sweep). A full three-shape sweep of one cell is ~30-45 min per `RUN_TAG` at ~$4/h. `MAX_RUN_MINUTES=N` skips points estimated
longer than N (recorded as `<run_id>.skipped.json`); `BENCH_DROP_ARGS="--flag ..."` removes a client flag the box's exact build
rejects, without editing the script.

### 7. Correctness gates (real launcher, real ports)

**Gate 1 -- GSM8K 200-question subset** (`gates/gsm8k.sh <cell> [port]`, server from `bench/launch.sh <cell>` still up):

```bash
bash gates/gsm8k.sh qwen38_27b_bf16_x4                     # x4 cells default to the rr_proxy :8080 -> 32 concurrent requests spread
REF_JSON=results/qwen38_27b_bf16_x4/gsm8k.json bash gates/gsm8k.sh qwen38_27b_fp8_x4   # pass = within 3 pts of the bf16 reference
MODE=chat TASK=gsm8k_cot MAX_GEN_TOKS=2048 bash gates/gsm8k.sh ds4flash_tp4            # chat template for reasoning models
```

`lm_eval[api]` (v0.4.13) runs `--model local-completions --model_args model=<served id>,base_url=...,num_concurrent=32,
max_retries=3,tokenized_requests=False,tokenizer=<local model dir or HF id> --tasks gsm8k --limit 200 --seed 1234
--gen_kwargs temperature=0 --log_samples` (chat: `local-chat-completions --apply_chat_template --fewshot_as_multiturn`).
Output `results/<cell>/gsm8k.json` with `exact_match_strict`, `exact_match_flexible`, engine version, `launch.json`
excerpt, `acs_suspected`, `pessimistic_tp`. Absolute pass/fail exists only against a `REF_JSON`. Knobs: `NUM_CONCURRENT`
(floor 32), `THINK_END_TOKEN='</think>'` for `MODE=chat` reasoning models (`--think_end_token`), `EXTRA_GEN_KWARGS='k=v,...'`
appended to `--gen_kwargs`, `TOKENIZER=<dir>` (default: the local model directory).

**Gate 2 -- FP8-KV vs bf16-KV diff** (`gates/run_kv_diff.sh <cell> [kvA] [kvB]`):

```bash
bash gates/run_kv_diff.sh qwen38_27b_fp8_x4                  # fp8 vs auto (bf16)
bash gates/run_kv_diff.sh ds4flash_tp4 fp8_ds_mla auto       # MLA cells
SELF_CHECK=1 bash gates/run_kv_diff.sh gptoss120b_x4_marlin  # + same-server noise floor (kv_diff_selfcheck.json)
```

It launches the cell twice through the real launcher -- `KV_CACHE_DTYPE=<kv> RUN_TAG=kvdiff_<kv> bench/launch.sh <cell>
--no-smoke`, port = the cell's `BASE_PORT` (first replica; `VIA_PROXY=1` for the rr_proxy), `bench/stop.sh <cell> --quiet`
between sides -- and `gates/kv_diff.py` sends a fixed 50-prompt set (20 judge-style rubric prompts, 20 code prompts, 10
long multi-step reasoning prompts; temperature 0, seed 1234, max_tokens 512, thinking disabled) and reports exact-match
rate, mean normalised token edit distance, char similarity and **corruption flags** (runs of `!!!!!!`, a token repeated >= 10x,
an n-gram repeated >= 6x, replacement/control characters, empty output -- the known sm_120 signature of a broken FP8
KV / attention path). Any corrupted output on either side fails the gate, as does mean normalised edit distance >
`MAX_NED` (0.30). A launch that dies (unsupported kv dtype) yields `status: launch_failed` with `launch.json`'s error
excerpt in `results/<cell>__kvdiff_<kv>/`. `ENDPOINT=completions` (default, raw prompt, no chat template) or `ENDPOINT=chat`
(sends `chat_template_kwargs {"enable_thinking": false, "thinking": false}` unless `ALLOW_THINKING=1`).
`KVDIFF_SINGLE_REPLICA=1` (default) launches an x4 cell as one server on its first GPU for the capture (the cells declare
`REPLICAS=${REPLICAS:-4}`), saving three model loads per side; `0` keeps the full cell. Run it inside tmux (two model loads).

### 8. Training co-tenancy (`train/lora_cotenant.sh`)

```bash
CELL=cotenant_tp2_gpu01 SHAPE=judge CONC=64 TRAIN_MINUTES=15 bash train/lora_cotenant.sh
COTENANT_MODEL=Qwen/Qwen3.8-27B-FP8 COTENANT_ALIAS=qwen38_27b_fp8 bash train/lora_cotenant.sh   # lighter server
```

1. Launches `cells/cotenant_tp2_gpu01.env` (TP2 on `GPU_IDS=0,2`, `--enable-sleep-mode`) via `bench/launch.sh` if not healthy
   (`AUTO_LAUNCH=1`); `STOP_AFTER=auto|0|1` stops it at the end (auto = only if launched here; never `stop.sh --all`, which
   would kill the trainer). The trainer runs in its own tmux session: `tmux attach -t train_lora`.
2. Training env: `/workspace/venv-train` from onstart, else `train/install_unsloth.sh` builds `/opt/unsloth-venv`
   (uv, torch `--torch-backend=cu128`, `triton>=3.3.1`, `unsloth unsloth_zoo bitsandbytes trl datasets`; xformers skipped).
3. BEFORE: ~5 min `vllm bench serve` window (judge shape, C64). 4. `train/lora_qwen8b.py` on `CUDA_VISIBLE_DEVICES=3`
   (Unsloth LoRA r=16/alpha=32, bf16, batch 8 x seq 2048, `yahma/alpaca-cleaned`, wall-clock stop; `Qwen/Qwen3-8B` --
   `Qwen/Qwen3.8-8B` does not exist on the Hub). 5. DURING: identical window. 6. `results/cotenancy.json`. 7.
   `train/sleep_wake.sh cotenant_tp2_gpu01` -> `results/sleep_wake.json` (standalone: `AUTO_LAUNCH=1 STOP_AFTER=auto` launch
   the cell with `ENABLE_SLEEP_MODE=1` through `bench/launch.sh` when no server answers, and stop it again afterwards).

If Unsloth's offloaded checkpointing hits the sm_120 "CUDA driver error: unknown error" (unsloth#2686), rerun with
`TRAIN_GRAD_CKPT=false` or `TRAIN_GRAD_CKPT=true`.

### 9. Summarise and archive

```bash
bash bench/collect_env.sh
python3 bench/summarise.py --cost-per-hour 4          # results/summary.md: tok/s per cell x concurrency per shape, $/M tok, daggers
python3 gates/gates_summary.py                        # results/gates_summary.md: GSM8K, kv_diff, load tests, co-tenancy, sleep/wake
./vast/sync.sh pull <ID> && vastai destroy instance <ID>
```

---

## Disk-constrained execution order (390 GB total, 373 GB free at start)

Rule: at most ~330 GB of weights resident, >= 20 GB free for logs/JIT caches; delete a model only after its cell
results are pulled. Start the next `prefetch.sh` while the current sweep runs whenever both fit.

| step | on disk (GB) | cells | then |
|---|---|---|---|
| 1 | Qwen3.8-27B-FP8 (31) + gpt-oss-120b (65) + DeepSeek-V4-Flash (167) = 263 -- **already there** | `qwen38_27b_fp8_x4` (8192 + mnbt16k, gsm8k, kv_diff); `gptoss120b_x4_marlin` / `_ficutlass` / `_b12x` (sweeps, gsm8k, kv_diff on marlin) | `rm -rf gpt-oss-120b`; prefetch Qwen3.8-27B (56) |
| 2 | 27B-FP8 31 + 27B 56 + DS-V4 167 = 254 | `qwen38_27b_bf16_x4` (sweep, gsm8k = `REF_JSON`) | `rm -rf Qwen3.8-27B Qwen3.8-27B-FP8` |
| 3 | DS-V4 167 (+ prefetch Qwen3.5-122B 127 = 294) | `ds4flash_tp4` then `ds4flash_tp2x2` (`GPU_IDS=0,2,1,3`); `fp8_ds_mla` vs `auto` kv_diff; gsm8k (chat) | `rm -rf DeepSeek-V4-Flash-0731` |
| 4 | Qwen3.5-122B 127 (+ Qwen3-8B 16) | `qwen35_122b_fp8_tp2x2` (+ `ENABLE_EP=1 RUN_TAG=ep`), gsm8k, kv_diff; **co-tenancy** `cotenant_tp2_gpu01` + LoRA on GPU 3; sleep/wake | `rm -rf Qwen3.5-122B-A10B-FP8`; prefetch Qwen3.8-Flash-Next-FP8 (186) |
| 5 | Flash-Next 186 (+ Qwen3-8B 16) | `qwen38flashnext_fp8_tp4`, `qwen38flashnext_fp8_tp2x2` (`0,2,1,3`, `VLLM_PLE_CPU_OFFLOAD=1`); gsm8k; no kv_diff (bf16 KV) | `rm -rf Qwen3.8-Flash-Next-FP8 Qwen3-8B` |
| 6 (optional) | GLM-5.3-Flash 328 alone | `glm53flash_fp8_tp4_loadtest` (fp8, then `KV_CACHE_DTYPE=bfloat16`) -> `loadtest.json` | `rm -rf GLM-5.3-Flash` |
| 7 | -- | reruns, real-prompt JSONL (`--dataset-path`) on the two leading cells, `bench/collect_env.sh`, `bench/summarise.py`, `gates/gates_summary.py`, `vast/sync.sh pull`, `vastai destroy instance` | |

Time budget at ~$4/h: each cell is ~30-45 min per `RUN_TAG` of sweeps + ~10 min of gates + launch time; downloads of the
>150 GB models are ~25-40 min each at 1 Gbit/s and bill GPU time, hence the overlap rule above. The whole matrix is
~30-36 h of instance time (~$120-150) plus co-tenancy and reruns.

## Reading the results

- Decision metric: **output tok/s and total tok/s at the saturating concurrency** per shape and cell, from
  `results/summary.md`; `$/M output tokens` uses `--cost-per-hour` (4).
- **Dagger (`pessimistic_tp == 1`, TP > 1)**: the row is a pessimistic lower bound. ACS on this host slows same-switch
  traffic; the Scan node (same GPUs, no NVLink either) should be checked for ACS/switch topology before reading TP2/TP4
  numbers as its ceiling. TP1 replica cells are unaffected and carry no dagger. A `?` marks a TP>1 row recorded before
  `results/hw/decisions.env` existed (flag unknown): re-read it as pessimistic once `hardware_truth.sh` reports `ACS_SUSPECTED=1`.
- A cell is admissible in the final table only if `gsm8k.json` is within tolerance of its bf16 reference **and**
  `kv_diff.json` has `pass: true`. FP8-KV cells that fail kv_diff are re-swept with `KV_CACHE_DTYPE=auto` and reported as bf16-KV.
- `cotenancy.json`: `serving_delta_during_vs_before_pct.output_throughput` is the interference cost of training on the
  spare GPU; `training_tok_s_steady` is the LoRA throughput bought for it.
- `sleep_wake.json`: `wake_call_s_mean` + `first_request_after_wake_s_mean` is the cost of parking a model; compare with
  `launch.json.seconds_to_ready` for a cold start.

## Known sm_120 pitfalls (why the baseline looks the way it does)

- DeepGEMM has no sm_120 kernels -> `VLLM_USE_DEEP_GEMM=0` (the DeepSeek-V4-Flash recipe: sm_120 cannot use the SM100
  FP4 indexer cache or `deep_gemm_mega_moe` -> `--attention_config.use_fp4_indexer_cache False --moe-backend auto`). If a
  ds4flash launch still dies inside `deep_gemm_fp8_o_proj`, `patches/apply_dsv4_o_proj.sh --apply` installs the hand patch
  (SM90 scale layout on cc 12.x; `VLLM_DSV4_OPROJ_SM120_FALLBACK=1` = bf16 einsum fallback) -- UNVERIFIED (2026-09-02).
- FlashInfer must be JIT-built for 12.0 (`FLASHINFER_CUDA_ARCH_LIST=12.0f`, needs `nvcc`; onstart installs it when missing);
  autotune is slow and occasionally picks bad configs -> `--no-enable-flashinfer-autotune` (also what the Qwen3.8-Flash-Next
  recipe uses on every row).
- Garbage output (`!!!!!!!!`, repeated tokens) is the symptom of an unsupported FP8 KV / attention kernel combination on
  sm_120 -- exactly what `kv_diff.py` flags.
- No NVLink and ACS on: TP scaling is PCIe-bound and same-switch pairs are the slow ones -> cross-switch TP2 pairing,
  custom all-reduce off, **P2P left on** (host staging measured ~8 GB/s ring busbw vs ~19 GB/s with P2P).
- Only cu128+ torch wheels carry `sm_120`; the training venv is separate from vLLM's interpreter, and nothing may
  `pip install` into the engine interpreter after the uv upgrade (it drags torch/vllm back to PyPI builds).
- `vllm/vllm-openai:cu130-nightly` was last pushed 2026-04-23 (vLLM 0.19.2, predates the August models) -- hence the
  in-place upgrade to vLLM main; `vast/README.md` section 8.

## Flag verification against vLLM main (2026-09-02)

Checked in `vllm/engine/arg_utils.py`, `vllm/config/cache.py`, `vllm/config/kernel.py`, `vllm/benchmarks/serve.py`,
`vllm/v1/engine/utils.py`, docs.vllm.ai `vllm serve` reference and the recipes (DeepSeek-V4-Flash, Qwen3.8-Flash-Next,
gpt-oss-120b):

- `vllm serve`: `--disable-custom-all-reduce`, `--enable-flashinfer-autotune` (negated as `--no-enable-flashinfer-autotune`,
  used verbatim by the Qwen3.8-Flash-Next recipe), `--compilation-config`, `--kv-cache-dtype` (`auto bfloat16 fp8 fp8_e4m3
  fp8_ds_mla ...`; the DeepSeek-V4-Flash recipe's sm_120 row uses `fp8`), `--tensor-parallel-size`, `--data-parallel-size`,
  `--enable-expert-parallel`, `--moe-backend` (`MoEBackend` literal: `auto marlin flashinfer_cutlass flashinfer_trtllm
  flashinfer_b12x b12x triton ...`; for MXFP4/gpt-oss the oracle accepts `b12x` but **rejects `flashinfer_b12x`** -- "not
  supported for MXFP4 MoE" -- hence the `_b12x` cell pins `b12x`),
  `--block-size` (int), `--tokenizer-mode deepseek_v4`, `--attention-config` + the dotted `--attention_config.use_fp4_indexer_cache
  False` form (recipe, verbatim), `--speculative-config` (not used), `--enable-sleep-mode`, `--language-model-only`,
  `--max-num-seqs`, `--max-num-batched-tokens`, `--gpu-memory-utilization`, `--max-model-len`, `--enable-prefix-caching`,
  `--enable-chunked-prefill`, `--trust-remote-code`, `--served-model-name`, `--reasoning-parser`, `--tool-call-parser`
  (`deepseek_v4`, `qwen3_xml`, `openai` per the recipes), `--enable-auto-tool-choice`, `--disable-uvicorn-access-log`: **verified**.
- Env: `VLLM_USE_DEEP_GEMM`, `VLLM_SERVER_DEV_MODE`, `VLLM_ENGINE_READY_TIMEOUT_S`, `VLLM_LOGGING_LEVEL`,
  `VLLM_LOG_STATS_INTERVAL`: in `vllm/envs.py`. `VLLM_PLE_CPU_OFFLOAD=1`: documented by the Qwen3.8-Flash-Next recipe but
  **UNVERIFIED (2026-09-02)** in vLLM main -- absent from `envs.py` and from `vllm/models/qwen4_exp/` (the PLE cache is
  TP-replicated there); the cells export it anyway, `launch.sh` sizes the host-RAM warning as PLE_TABLE_GB x TP x DP, and the
  first launch's server log decides (grep `PLE` / `offload`).
- `vllm bench serve`: `--backend --endpoint --base-url --model --tokenizer --request-rate inf --max-concurrency --ignore-eos
  --percentile-metrics --metric-percentiles --save-result --save-detailed --result-dir --result-filename --metadata KEY=VALUE
  --disable-tqdm --ready-check-timeout-sec --num-warmups --dataset-name --dataset-path`: **verified** in `serve.py`; result keys
  `request_throughput output_throughput total_token_throughput completed num_prompts duration mean/median/p99_{ttft,tpot,itl,e2el}_ms`.
- `vllm bench serve --num-prompts` (default 1000) `--seed` (default 0) `--trust-remote-code --no-oversample --random-input-len
  --random-output-len --random-range-ratio --random-prefix-len --custom-output-len` (default 256) `--skip-chat-template` and
  `--dataset-name` choices incl. `random`, `custom`: **verified** in `vllm/benchmarks/datasets/datasets.py` (`add_dataset_parser`).
  The custom JSONL key is `output_tokens` (there is no `--custom-skip-chat-template`; the flag is `--skip-chat-template`).
  `vllm bench serve --tokenizer-mode` is a plain `type=str` argument (no argparse choices; help lists auto/hf/slow/mistral/
  deepseek_v32) resolved by `vllm/tokenizers/registry.py`, where `deepseek_v4` is registered too -- the DS4 cells' optional
  `DS4_BENCH_TOKENIZER_MODE=deepseek_v4` is therefore accepted, though its tokenizer load in the client is UNVERIFIED.
  `--metadata KEY=VALUE` splits on the first `=` and accepts an empty value.
- Not verified (and not used by the harness): `--reasoning-config` (appears in the DeepSeek-V4-Flash recipe), SGLang
  `sglang.launch_server` flags for the optional A/B image. Anything else the cells pass that is not listed above should be
  treated as **UNVERIFIED (2026-09-02)** and tolerated at launch (documented env override) rather than assumed.
- `--data-parallel-size` + `CUDA_VISIBLE_DEVICES` slicing (rank *i* -> slice *i*): **verified** (`get_physical_gpu_ids_for_local_dp_rank`).
