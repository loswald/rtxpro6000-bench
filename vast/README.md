# Vast.ai provisioning and host truth -- 4x RTX PRO 6000 Blackwell

Everything here runs from your laptop (Git Bash / WSL / macOS / Linux) or inside the rented container. There is no
nested Docker, no BIOS access and no host access: the Vast container is the machine. CLI flow verified as of
**2026-09-02** against the `vastai` CLI source (github.com/vast-ai/vast-cli `vast.py`) and Vast docs.

**State on 2026-09-02:** the instance already exists and is running (verified 4x RTX PRO 6000 Blackwell **Server
Edition**, ~$4/h, ~390 GB container disk with 373 GB free at start, 192 CPU threads, 1.5 TB RAM). Its image
`vllm/vllm-openai:cu130-nightly` is stale (vLLM 0.19.2) and is being upgraded **in place** with `uv` to vLLM
`0.28.1rc1.dev312+g41848caa6` (vLLM main, CUDA 13.0 wheels from `https://wheels.vllm.ai/nightly/cu130`) plus
`flashinfer-python 0.6.18` and the `vllm[b12x]` extra. The models are already on disk as plain directories
(`hf download <repo> --local-dir /workspace/models/<basename>`). Nothing in `vast/` re-installs the engine or
re-downloads those models; `onstart.sh` is safe to re-run on this box.

| file | role |
|---|---|
| `README.md` | this flow + exact CLI commands |
| `search_offers.sh` | `vastai search offers` with the campaign filters, pretty table |
| `create_instance.sh` | `vastai create instance ...` with the image, disk, `--ssh --direct`, `--onstart onstart.sh`, waits for ssh (only for a NEW box) |
| `onstart.sh` | runs as root inside the container at boot: apt, isolated tool venv (hf CLI), optional cuda-samples/nccl-tests builds, optional model downloads with a disk guard, waits for the harness, runs `hardware_truth.sh`. **Never pip-installs into the engine interpreter** (`ALLOW_ENGINE_PIP=0`) |
| `hardware_truth.sh` | nvidia-smi -q / topo, PCIe link per GPU, peer-access + P2P copy matrix (torch), NCCL all_reduce busbw for the 4-GPU ring, a same-switch pair and a cross-switch pair (nccl-tests or torch fallback) -> **`results/hw/decisions.env`** (the contract `bench/env.sh` reads) + `results/hw/hardware.json` + `results/hw/machine.env` (facts only, sourced by nothing) |
| `sync.sh` | push harness to `/workspace/rtxpro6000-bench`, pull `results/` (incl. `results/hw/`) back, ssh/tmux helpers |
| `COST.md` | $/h ranges with sources, campaign estimate |

## 0. Prerequisites (once, on the laptop)

```bash
# CLI. Windows: pip (or, with no system Python, 'uv tool install vastai'). Linux/macOS/WSL: pip or the isolated installer.
pip install -U vastai              # or:  uv tool install vastai      or:  curl -fsSL https://vast.ai/install.sh | bash
vastai --version

# API key: console.vast.ai -> Account -> API Keys -> create. Paste it ONLY into this command:
vastai set api-key <KEY>
#   The CLI stores it in its own config file (Linux/macOS: ~/.config/vastai/vast_api_key, legacy ~/.vast_api_key).
#   Never put the key in this repo, in onstart.sh, in --env, or in a shell script. Nothing in ./vast reads it -- the
#   CLI does. Check:  vastai show user

# SSH public key on the account (used for every instance):
vastai create ssh-key "$(cat ~/.ssh/id_ed25519.pub)"     # or paste it in console.vast.ai -> Keys
```

Optional but nice: `rsync` locally (Git Bash lacks it -> `sync.sh` falls back to tar-over-ssh automatically), `jq`.
The helpers need a working `python3`/`python` **or** `uv` for JSON formatting (on Windows the Store `python3` alias is
not a Python; `uv` is picked up automatically).

## 1. Flow at a glance

| step | command | wall time |
|---|---|---|
| (new box only) search verified 4x boxes | `./vast/search_offers.sh` | 1 min |
| (new box only) rent one | `./vast/create_instance.sh <OFFER_ID>` | image pull 2-6 min |
| push the harness | `./vast/sync.sh push <ID>` -> `/workspace/rtxpro6000-bench` | 10 s |
| hardware truth (before any engine touches the GPUs) | `./vast/sync.sh tmux <ID> hwtruth` or `./vast/sync.sh ssh <ID> 'bash /workspace/rtxpro6000-bench/vast/hardware_truth.sh'` | ~5 min |
| read the contract | `cat results/hw/decisions.env` (`P2P_OK=1 CUSTOM_ALLREDUCE=0 NCCL_P2P_DISABLE=0 ACS_SUSPECTED=1 PESSIMISTIC_TP=1 HOST_RAM_GB=...`) | - |
| engine upgrade / check | `bash bench/setup_engine.sh` (bench track; uv, vLLM main cu130 nightly, flashinfer, `vllm[b12x]`) | minutes |
| run the campaign, one model at a time (disk!) | `./vast/sync.sh ssh <ID>` then `bench/prefetch.sh -> launch -> sweep -> gates -> stop -> rm -rf /workspace/models/<name>` (top-level README) | hours per model |
| pull results as you go | `./vast/sync.sh watch <ID> 600` | continuous |
| destroy | `vastai destroy instance <ID>` | immediate; billing stops |

## 2. Search offers (new box only)

Exact command the helper runs (query terms are space-separated `field op value`; GPU names use `_` for spaces;
`in [a,b]` lists; the CLI adds `external=false rentable=true verified=true` by default unless you pass `-n`):

```bash
vastai search offers 'num_gpus=4 gpu_name in [RTX_PRO_6000_WS,RTX_PRO_6000_S] verified=true rentable=true rented=any gpu_frac=1 cuda_vers>=13.0 driver_version>=595.0.0 cpu_ram>=256 disk_space>=1200 inet_down>=1000 direct_port_count>=4 duration>=3' -o 'dph_total'
```

| filter | why |
|---|---|
| `num_gpus=4 gpu_frac=1` | whole machine (all its GPUs), no co-tenant on the PCIe fabric |
| `gpu_name in [RTX_PRO_6000_WS,RTX_PRO_6000_S]` | Workstation (600 W) or Server edition. **Max-Q (`RTX_PRO_6000_MAX-Q`, 300 W) is excluded** -- it would understate throughput |
| `verified=true` | Vast-verified host (`verification` column also shows `verified`) |
| `cuda_vers>=13.0 driver_version>=595.0.0` | `cuda_vers` = max CUDA the host driver supports (alias of `cuda_max_good`); dotted driver strings compare numerically |
| `cpu_ram>=256` | GB (the CLI multiplies by 1000 internally) -- the Qwen3.8-Flash-Next N-gram table CPU offload needs >= 51 GB per DP rank; the rented box has 1.5 TB |
| `disk_space>=1200` | GB container disk wanted. **The box actually rented has ~390 GB** -- the filters are wishes; with less disk the campaign runs one model at a time (section 9) |
| `inet_down>=1000` | Mb/s; ~1 TB of weights over the campaign |
| `direct_port_count>=4` | host router has open ports -> `--direct` ssh + optional direct :8000 |
| `duration>=3` | host allows rentals >= 3 days |
| `-o 'dph_total'` | cheapest first; `-o 'pcie_bw-,dph_total'` = fastest CPU<->GPU PCIe first (trailing `-` = descending) |

Columns to read in the helper's table: `$/h` (all 4 GPUs), `W` (gpu_max_power: expect 600), `cpu`, `ram_GB`, `disk_GB`,
`inet_dn/up` (Mb/s), `pcie_GBs` (measured CPU->GPU), `gen` (5 wanted), `rel` (reliability, want > 0.98), `ports`,
`cuda`, `driver`, `stor$/GBmo` (storage billed on top), `dl$/GB` (ingress cost -- prefer 0), `days`, `geo`.

```bash
./vast/search_offers.sh                                           # table
./vast/search_offers.sh 'geolocation in [GB,DE,NL,FR,SE,FI,NO]'   # add terms
./vast/search_offers.sh --json | jq '.[] | {id,dph_total,gpu_name,cpu_name,cpu_ram,disk_space,inet_down,pcie_bw,pci_gen,reliability2,direct_port_count,storage_cost,inet_down_cost,driver_version,cuda_max_good,gpu_max_power,mobo_name}'
MIN_INET_MBPS=500 MIN_PORTS=2 MIN_DISK_GB=380 ./vast/search_offers.sh   # relax when nothing matches
```

No `RTX_PRO_6000_*` results at all? List what the market calls the card: `vastai search offers 'num_gpus=4 gpu_ram>=90 verified=true' --raw | jq -r '.[].gpu_name' | sort | uniq -c`.

## 3. Create the instance (new box only)

```bash
./vast/create_instance.sh <OFFER_ID>
# which runs (paths resolved):
vastai create instance <OFFER_ID> \
  --image vllm/vllm-openai:cu130-nightly \
  --disk 1200 --ssh --direct --label rtxpro6000-bench --cancel-unavail \
  --onstart ./vast/onstart.sh \
  --env '-e DOWNLOAD_SET=none -e ENGINE=auto -e HARNESS_WAIT_MIN=90 -e INSTALL_TRAIN=1 -e INSTALL_EVAL=1 -e ALLOW_ENGINE_PIP=0 -p 8000:8000 -p 30000:30000'
```

| flag | meaning (from `vastai create instance --help`) |
|---|---|
| `--image` | docker image to launch. See section 8: the tag is stale and gets upgraded in place |
| `--disk 1200` | container disk in GB (must be <= the offer's `disk_space`; `DISK_GB=` to change) |
| `--ssh --direct` | ssh launch mode over a direct host port (runtype `ssh_direc ssh_proxy`); proxy ssh stays available as fallback |
| `--onstart FILE` | the CLI **reads the file and uploads its contents** as the onstart script (Vast stores it as `/root/onstart.sh`, runs it as root on every start). `--onstart-cmd "string"` is the inline alternative |
| `--env '...'` | Docker-options string: `-e KEY=VAL` env vars and `-p PORT:PORT` port publishes (values are parsed, `-p` must be digits/`:`/tcp/udp) |
| `--label` | free text shown in `vastai show instances` |
| `--cancel-unavail` | fail immediately if the offer was taken instead of leaving a stopped instance |
| `--bid_price X` | *interruptible* instance at bid $X/h (machine total). Not for the main run; fine for an SGLang A/B |

Important: in SSH launch mode Vast **replaces the image ENTRYPOINT**, so the vLLM API server does not auto-start --
the harness launches `vllm serve` itself with the campaign flags. Vast env inside the container: `CONTAINER_ID`,
`PUBLIC_IPADDR`, `GPU_COUNT`, `DATA_DIRECTORY`, `VAST_TCP_PORT_22`, `VAST_TCP_PORT_8000` (external port mapped to :8000),
`CONTAINER_API_KEY` (per-instance key; onstart keeps it out of `/etc/environment`).

Second instance for an SGLang A/B (same offer class, fewer models):

```bash
ENGINE=sglang IMAGE=lmsysorg/sglang:v0.5.18-cu130 LABEL=rtxpro6000-sglang DOWNLOAD_SET=qwen38_27b_fp8,gptoss120b INSTALL_TRAIN=0 ./vast/create_instance.sh <OFFER_ID_2>
```

## 4. Wait, ssh, logs

```bash
vastai show instances                       # table: ID, status, SSH Addr, $/h, label
vastai show instance <ID> --raw | jq '{actual_status,ssh_host,ssh_port,public_ipaddr,ports,gpu_name,num_gpus,dph_total,disk_space,label}'
vastai ssh-url <ID>                         # ssh://root@HOST:PORT
ssh -p <PORT> root@<HOST>                   # or: ./vast/sync.sh ssh <ID>
vastai logs <ID>                            # container stdout incl. onstart output
./vast/sync.sh ssh <ID> 'tail -50 /workspace/onstart.log; tmux ls; df -h /workspace; du -sh /workspace/models/*'
```

`create_instance.sh` waits for `actual_status == running` and prints the ssh line. First boot: image pull
(~9 GB) 2-6 min, then onstart. Direct ssh not answering after 5 min -> `vastai show instance <ID> --raw | jq .status_msg`;
if the host's direct ports are blocked, destroy and recreate without `--direct` (proxy ssh via `ssh5.vast.ai`).

## 5. Push the harness; what onstart does

```bash
./vast/sync.sh push <ID>        # rtxpro6000-bench/ -> /workspace/rtxpro6000-bench (results/, models excluded; CRLF stripped; +x set)
./vast/sync.sh tmux <ID> hwtruth
```

Remote layout: `/workspace/rtxpro6000-bench` = this harness (`REMOTE_ROOT`, `BENCH_ROOT`); `/workspace/models` = weights as
plain directories (`hf download --local-dir` layout, e.g. `/workspace/models/DeepSeek-V4-Flash-0731`); `/workspace/bench` = the
box's own scratch scripts (not ours, never overwritten); `/workspace/tools` = built helpers; `/workspace/venv-tools` = isolated
venv for the `hf` CLI.

`onstart.sh` (idempotent, log `/workspace/onstart.log`):

1. exports the Vast env + baseline env to `/etc/environment` and `/etc/profile.d/10-bench.sh`
   (`VLLM_USE_DEEP_GEMM=0 FLASHINFER_CUDA_ARCH_LIST=12.0f TORCH_CUDA_ARCH_LIST=12.0 NCCL_P2P_LEVEL=PHB NCCL_IB_DISABLE=1 NCCL_MIN_NCHANNELS=8 HF_HOME=/workspace/hf MODELS_DIR=/workspace/models BENCH_ROOT=/workspace/rtxpro6000-bench HW_DIR=$BENCH_ROOT/results/hw TOOLS_DIR=/workspace/tools`).
   **`NCCL_P2P_DISABLE` is never exported** (decision 1), and the profile no longer sources any `$BENCH_ROOT/env.sh`: the
   decision contract stays in `results/hw/decisions.env` and is read by `bench/env.sh` at run time, so a stale exported
   `P2P_OK` / `ACS_SUSPECTED` cannot shadow the file.
2. apt: tmux git jq rsync pciutils cmake build-essential numactl python3-venv ...; installs `cuda-nvcc-13-x` + `libnccl-dev` if the image lacks nvcc (optional -- `hardware_truth.sh` has torch fallbacks)
3. **engine interpreter: look, don't touch.** Prints vLLM/torch/flashinfer/b12x versions. The `hf` CLI, if missing, goes into
   `/workspace/venv-tools` (symlinked to `/usr/local/bin/hf`). Only `ALLOW_ENGINE_PIP=1` allows the guarded `b12x` pip install
   (dry-run first; refuses if it would touch torch/vllm/flashinfer). The real engine upgrade is `bench/setup_engine.sh`.
4. tmux `build`: cuda-samples `p2pBandwidthLatencyTest` (cmake, `CMAKE_CUDA_ARCHITECTURES=120`, nvcc fallback) and `nccl-tests` (`MPI=0`, `NVCC_GENCODE=sm_120`) -> `/workspace/tools` (skipped with `SKIP_BUILDS=1` or without nvcc)
5. tmux `downloads` (only when `DOWNLOAD_SET != none`): sequential `hf download <repo> --local-dir /workspace/models/<basename>` (resumable, 6 retries, `.complete` markers, `status.tsv`), **refusing any model that does not fit the free disk** (`skipped_disk`). `/workspace/models/models.json` maps key -> repo -> path. Default is `none`: the bench track's `bench/prefetch.sh <cell>` fetches per cell in the same layout.
6. tmux `hwtruth`: clones `HARNESS_REPO` if given, else waits up to `HARNESS_WAIT_MIN` for your `sync.sh push`, waits for the builds, runs `hardware_truth.sh` unless `results/hw/decisions.env` already exists
7. tmux `evalenv` (`/workspace/venv-eval`, lm-eval) and `trainenv` (`/workspace/venv-train`, unsloth) -- both `--system-site-packages` so they reuse the image's torch and cannot break the engine install

`DOWNLOAD_SET` keys: `qwen38_27b_fp8 gptoss120b qwen3_8b qwen38_27b qwen35_122b_fp8 dsv4flash qwen38flashnext glm53flash`;
sets: `none` (default), `small` (27B-FP8, gpt-oss, Qwen3-8B), `core` (all but GLM -- does NOT fit 390 GB at once), `all`, or a
comma list. Re-run any time: `/workspace/tools/download_models.sh dsv4flash`.

## 6. Hardware truth output and the decision contract

`bash /workspace/rtxpro6000-bench/vast/hardware_truth.sh` (default `OUT=$HW_DIR=results/hw`) writes:

- **`results/hw/decisions.env`** -- plain `KEY=VALUE`, sourced by `bench/env.sh`; every sweep meta JSON and summary carries
  `acs_suspected` / `pessimistic_tp` from it:

  | key | rule | value on this box (2026-09-02) |
  |---|---|---|
  | `P2P_OK` | 1 iff peer access is supported on **all** GPU pairs (`torch.cuda.can_device_access_peer`, or cuda-samples `CAN Access Peer`). Bandwidth/latency are recorded, **never** used to disable P2P | `1` |
  | `CUSTOM_ALLREDUCE` | always `0` = vLLM `--disable-custom-all-reduce`; A/B per launch with `CUSTOM_ALLREDUCE=1 bench/launch.sh <cell>` | `0` |
  | `NCCL_P2P_DISABLE` | **always `0`; the script never writes `1`.** P2P works and beats host staging. Only a human sets it (edit the file, add `HUMAN_DECISION=1`) and only when peer access is genuinely unsupported | `0` |
  | `ACS_SUSPECTED` | 1 when a same-switch (topo `PIX`/`PXB`) pair all-reduces slower than a cross-switch pair (busbw ratio < `ACS_RATIO`=0.8) or `lspci` shows `ACSCtl` enabled | `1` (21 vs 38 GB/s) |
  | `PESSIMISTIC_TP` | `ACS_SUSPECTED=1 or P2P_OK=0` -> TP2/TP4 rows carry a dagger; TP1 replica rows do not | `1` |
  | `HOST_RAM_GB` | `free -g` total (the Qwen3.8-Flash-Next DP2 cell needs ~51 GB per DP rank in host RAM) | `~1510` |
  | `NOTES` | one line: GPUs, pairs, peer-access source, bandwidths, NCCL transports, ACS evidence, TP2 pairing advice | |
  | informational | `GENERATED_UTC`, `HW_JSON`, `SAME_SWITCH_PAIRS=0-1,2-3`, `TP2_CROSS_SWITCH_GPU_IDS=0,2,1,3`, `HUMAN_DECISION=0` | |

- `results/hw/hardware.json` (+ `raw/*`): GPU name/edition/vBIOS/driver/power limits/ECC/BAR1 (ReBAR guess), PCIe max vs idle
  link per GPU (sysfs + lspci), HMM parameter, `nvidia-smi topo -m` pair matrix + switch groups, torch peer-access matrix with
  unidirectional copy GB/s and 8 B copy latency per pair, p2pBandwidthLatencyTest matrices when the binary exists, NCCL
  all_reduce busbw per label (`ring4_baseline` with the harness env, `ring4_default_level`, `pair_same`, `pair_cross`,
  `ring4_p2p_disabled` as a comparison only), transports NCCL picked (`P2P/CUMEM` vs `SHM`), engine versions, models on disk,
  and the `decisions` block. Copied to `results/hardware.json` for `bench/collect_env.sh` and the gates.
- `results/hw/machine.env`: machine FACTS only (GPU name/edition/vBIOS/driver/power limit, `SAME_SWITCH_PAIRS`,
  `TP2_CROSS_SWITCH_GPU_IDS`, the recommended sm_120 baseline as comments). It carries **no decision keys and no `NCCL_*`
  exports** and is sourced by nothing; `bench/env.sh` reads `decisions.env` directly at run time and the caller's own
  environment wins over it (`CUSTOM_ALLREDUCE=1 bench/launch.sh ...` is the A/B). Older versions wrote `results/hw/env.sh` and
  copied it to `<repo>/env.sh` -- no longer; a stale `<repo>/env.sh` is reported at the end of the run (delete it), and a
  stale `results/hw/env.sh` is renamed `env.sh.legacy`.

Measured on this box (2026-09-02) and what it means: topo `0-1 PIX`, `2-3 PIX`, all cross pairs `NODE`; peer access
`True` for every pair, unidirectional peer copy ~52 GB/s, NCCL transport `P2P/CUMEM`; all_reduce busbw pair 0-1 ~21 GB/s,
pair 0-2 ~38 GB/s, 4-GPU ring ~19 GB/s regardless of `NCCL_P2P_LEVEL` / `NCHANNELS` / `PROTO` -> PCIe **ACS is enabled on the
host** (switch-local P2P is redirected through the root complex) and cannot be changed from the container. Hence: TP2 replicas
pair **cross-switch** (`GPU_IDS=0,2,1,3` for DP2xTP2 cells -- vLLM gives DP rank *i* the *i*-th TP-sized slice of
`CUDA_VISIBLE_DEVICES`, so rank 0 -> physical 0,2 and rank 1 -> 1,3), TP4 stays on `0,1,2,3` and is marked pessimistic,
custom all-reduce off by default, and `NCCL_P2P_DISABLE=1` is **never** set (it was ~8 GB/s vs 19 GB/s here).

Knobs: `SKIP_P2P=1 SKIP_NCCL=1 SKIP_TORCH=1` skip the slow parts; `PARSE_ONLY=1` re-parses `raw/` without touching the GPUs;
`TORCH_NCCL=1` forces the torch all_reduce microbench even when nccl-tests exists; `ACS_SUSPECTED_OVERRIDE=0|1`; `ACS_RATIO`;
`NCCL_TIMEOUT`. A previous `decisions.env` is kept as `decisions.env.prev`; its `NCCL_P2P_DISABLE`/`CUSTOM_ALLREDUCE` survive a
re-run only when it contains `HUMAN_DECISION=1`.

## 7. Pull results, destroy

```bash
./vast/sync.sh pull <ID>            # /workspace/rtxpro6000-bench/results/ -> ./results/  (incl. results/hw/, + onstart log)
./vast/sync.sh watch <ID> 600       # every 10 min
vastai destroy instance <ID>        # billing stops; 'stop' does NOT (storage keeps billing)
vastai show instances               # confirm it is gone
```

## 8. Image tag truth (Docker Hub, checked 2026-09-02) and the in-place upgrade

| tag | last pushed | note |
|---|---|---|
| `vllm/vllm-openai:cu130-nightly` | **2026-04-23** | **what the running instance was created from: vLLM 0.19.2**, predates DeepSeek-V4-Flash (Aug 1), Qwen3.8 (Aug 14/31), GLM-5.3-Flash (Aug 31) support |
| `vllm/vllm-openai:nightly` / `cu129-nightly` | 2026-09-02 | current nightlies (CUDA 12.9 build) |
| `vllm/vllm-openai:v0.28.0` (= `latest`) | 2026-08-26 | newest release, CUDA 12.9 |
| `vllm/vllm-openai:v0.20.0-cu130` / `latest-cu130` | 2026-04-28 | newest generic cu130 *release* |
| `vllm/vllm-openai:deepseekv4-flash-vision-x86_64-cu130` | 2026-09-01 | model-launch cu130 builds also exist: `qwen38-flash-next-x86_64-cu130` (08-26), `glm53-flash-x86_64-cu130` (08-26), `qwen38-x86_64-cu130` (08-12) |
| `lmsysorg/sglang:v0.5.18-cu130` | 2026-08-21 | exists (also `latest-cu130`) |

What we do instead of recreating the instance: upgrade the container's interpreter **in place with `uv`** to vLLM
`0.28.1rc1.dev312+g41848caa6` (vLLM main as of 2026-09-02, CUDA 13.0 build from `https://wheels.vllm.ai/nightly/cu130`),
`flashinfer-python 0.6.18` and `vllm[b12x]` -- `bench/setup_engine.sh` (bench track) owns that. `onstart.sh` never pip-installs
into that interpreter (`ALLOW_ENGINE_PIP=0`), `bench/collect_env.sh` records the resulting versions in `results/env.json`, and
every `launch.json`/sweep meta carries `engine_version`. `FLASHINFER_CUDA_ARCH_LIST=12.0f` needs the FlashInfer JIT to find `nvcc`
-- onstart installs `cuda-nvcc-<major>-<minor>` matching the image toolkit when missing.

## 9. Model set and disk budget (Hub sizes, 2026-09-02) -- 390 GB means one model at a time

| key | repo | GB on disk | cell(s) |
|---|---|---|---|
| `qwen38_27b_fp8` | Qwen/Qwen3.8-27B-FP8 | 31 | 4x TP1 replicas (kv_diff target) |
| `gptoss120b` | openai/gpt-oss-120b | 65 (safetensors only; `original/` + `metal/` excluded, 196 otherwise) | 4x TP1, marlin vs flashinfer_cutlass vs b12x |
| `qwen3_8b` | Qwen/Qwen3-8B | 16 | Unsloth LoRA co-tenancy (`Qwen/Qwen3.8-8B` does not exist on the Hub) |
| `qwen38_27b` | Qwen/Qwen3.8-27B | 56 | 4x TP1 bf16 (GSM8K reference) |
| `qwen35_122b_fp8` | Qwen/Qwen3.5-122B-A10B-FP8 | 127 | DP2xTP2 (`GPU_IDS=0,2,1,3`), co-tenancy server |
| `dsv4flash` | deepseek-ai/DeepSeek-V4-Flash-0731 | 167 | DP2xTP2 / TP4 |
| `qwen38flashnext` | Qwen/Qwen3.8-Flash-Next-FP8 | 186 | TP4 / DP2xTP2, bf16 KV, `VLLM_PLE_CPU_OFFLOAD=1` |
| `glm53flash` | zai-org/GLM-5.3-Flash | 328 | attempt-to-load only; needs the disk otherwise empty |
| **all** | | **~976** | does **not** fit: 390 GB total, 373 GB free at start (image + venvs are on the same disk) |

Rules: benchmark sequentially, delete after the cell's sweep + gates are pulled (`./vast/sync.sh pull`), keep at most ~330 GB
of weights resident, leave >= 20 GB free for logs/JIT caches. Layout is always `hf download <repo> --local-dir
/workspace/models/<basename-of-repo>` (plain directory, no HF cache blobs; `bench/prefetch.sh` and `download_models.sh` both
do this; `bench/env.sh:load_cell` uses the directory when it exists, else the HF id). Delete with
`rm -rf /workspace/models/<name>` and check `df -h /workspace`. The suggested order is in the top-level README
("Disk-constrained execution order"); on 2026-09-02 `DeepSeek-V4-Flash-0731` (167), `Qwen3.8-27B-FP8` (31) and
`gpt-oss-120b` (65) were already on disk (~110 GB free).

## 10. Troubleshooting

- **Offer gone / "no such ask"**: `--cancel-unavail` makes it fail fast; search again.
- **`manifest unknown` in `vastai logs`**: image tag missing -> section 8; recreate with `IMAGE=...`.
- **onstart says "waiting for harness"**: run `./vast/sync.sh push <ID>`; it polls every 20 s for `HARNESS_WAIT_MIN`.
- **`bash: /bin/bash^M`**: CRLF from a Windows editor -- `sync.sh push` strips it; for a hand-copied file `sed -i 's/\r$//' file`.
- **rsync missing (Git Bash)**: `sync.sh` falls back to tar-over-ssh automatically (no `--delete`).
- **nvcc missing / p2p build failed**: optional. `hardware_truth.sh` falls back to torch for peer access, copy bandwidth and the NCCL
  all_reduce microbench. To build anyway: `apt-get install cuda-nvcc-13-0 cuda-cudart-dev-13-0 cuda-cccl-13-0` then `bash /workspace/tools/build_tools.sh`.
- **hardware truth says `ACS_SUSPECTED=1`**: expected on this host; nothing to fix from the container. Pair TP2 cross-switch, read
  TP2/TP4 rows as pessimistic (dagger). Do **not** set `NCCL_P2P_DISABLE=1` -- it is slower (8 vs 19 GB/s ring busbw).
- **Someone set `NCCL_P2P_DISABLE=1` anyway**: only via `decisions.env` with `HUMAN_DECISION=1`; delete the file and re-run
  `hardware_truth.sh` to get back to the measured default.
- **HF 429 / slow**: downloads retry 6x with 30 s backoff; a token is not required for these public repos -- if you add one, pass it as `-e HF_TOKEN=...` in `--env`, never in a file.
- **Disk full / `skipped_disk` in `status.tsv`**: `df -h /workspace; du -sh /workspace/models/*`; delete a finished model
  (`rm -rf /workspace/models/<name>`), re-run `bench/prefetch.sh <cell>`. Never store two of the >150 GB models plus GLM at once.
- **Engine got downgraded (`vllm.__version__` back to 0.19/0.20)**: something ran `pip install` into the engine interpreter. Re-run
  `bench/setup_engine.sh`; keep `ALLOW_ENGINE_PIP=0`.
- **Host clock/power**: `hardware.json.gpus[].power_limit_w` should read 600 for WS/Server; 300 means a Max-Q slipped through -- destroy and re-search.

Cost expectations: see `COST.md`.
