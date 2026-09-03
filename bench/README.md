# bench/ — serving cells and the throughput sweep

Runs inside the Vast.ai container (root, 4x RTX PRO 6000 Blackwell Server Edition exposed, no
nested Docker). Bash + Python 3 stdlib only. Goal: **maximum throughput at saturation**
(tokens/s, requests/s); TTFT/TPOT are logged, not optimised.

Engine on the box: vLLM main `0.28.1rc1.dev312+g41848caa6` (CUDA 13.0 build from
`https://wheels.vllm.ai/nightly/cu130`, installed in place with uv over the stale
`vllm/vllm-openai:cu130-nightly` image) + `flashinfer-python 0.6.18` + the `vllm[b12x]` extra.
Every `vllm serve` / `vllm bench serve` flag used here was checked against vLLM main on
2026-09-02 (`vllm/engine/arg_utils.py`, `vllm/benchmarks/serve.py`,
`vllm/benchmarks/datasets/datasets.py`, docs.vllm.ai CLI reference, recipes.vllm.ai).

## Order of operations

```bash
cd /workspace/bench                        # harness root (vast/sync.sh push)
bash bench/setup_engine.sh                 # tmux, curl, hf CLI, vllm[b12x] (best effort), versions, models on disk
bash vast/hardware_truth.sh                # (vast track) writes results/hw/decisions.env  <- READ BY EVERY bench SCRIPT
bash bench/collect_env.sh                  # results/env.json snapshot (versions, GPUs, decisions, models on disk)
export COST_PER_HOUR=4.00                  # machine $/hr for the cost column

bash bench/prefetch.sh --list              # what is under /workspace/models (size + complete/partial)
bash bench/launch.sh ds4flash_tp4          # tmux session bench_ds4flash_tp4, waits on /health, writes launch.json
bash bench/sweep.sh  ds4flash_tp4          # router/judge/agent x per-shape concurrency lists, dmon sampling
bash bench/stop.sh   ds4flash_tp4          # kills the session + every process of the cell, waits for GPU memory
bash bench/prefetch.sh --delete ds4flash_tp4 --yes   # free the ~167 GB before the next model (390 GB disk)

MAX_NUM_BATCHED_TOKENS=16384 RUN_TAG=mnbt16k bash bench/launch.sh ds4flash_tp4   # the 8192/16384 A/B
RUN_TAG=mnbt16k bash bench/sweep.sh ds4flash_tp4 router,judge 32,64,128,256
python3 bench/summarise.py                 # results/summary.md (+ .csv, .json)
```

`bench/launch.sh <cell> --dry-run` and `bench/sweep.sh <cell> --dry-run` print the exact
commands without a server and without touching `results/`.

Cells (in `cells/`): `ds4flash_tp4`, `ds4flash_tp2x2`, `qwen38flashnext_fp8_tp2x2`
(+ `qwen38flashnext_fp8_tp4` fallback), `qwen38_27b_bf16_x4`, `qwen38_27b_fp8_x4`,
`gptoss120b_x4_marlin`, `gptoss120b_x4_ficutlass` (+ `gptoss120b_x4_b12x` third arm),
`qwen35_122b_fp8_tp2x2`, `glm53flash_fp8_tp4_loadtest` (attempt-to-load only),
`cotenant_tp2_gpu01`. `cells/_template.env` documents every key.

## Hardware decisions this harness implements (measured 2026-09-02)

4x RTX PRO 6000 Blackwell Server Edition, 96 GB each, sm_120, PCIe Gen5 x16, **no NVLink**.
`nvidia-smi topo`: GPUs 0-1 PIX (same switch), 2-3 PIX, cross pairs NODE. Peer access works on
every pair (`can_device_access_peer=True`, ~52 GB/s unidirectional copy), NCCL transport
P2P/CUMEM. But all_reduce busbw is ~21 GB/s on the same-switch pair, ~38 GB/s cross-switch,
~19 GB/s for the 4-GPU ring, whatever the NCCL tuning: **PCIe ACS is enabled on the host**
(switch-local P2P is bounced through the root complex) and cannot be changed from the container.

| decision | where |
|---|---|
| **Never** set `NCCL_P2P_DISABLE=1` automatically. P2P is on; host staging would be slower. Only `results/hw/decisions.env` `NCCL_P2P_DISABLE=1` (a documented human decision) or `FORCE_NCCL_P2P_DISABLE=1` (A/B) sets it; anything inherited from a shell is dropped with a note. | `env.sh` |
| Custom all-reduce **off** by default (`--disable-custom-all-reduce`); `CUSTOM_ALLREDUCE=1 bench/launch.sh <cell>` for an A/B. | `env.sh`, `launch.sh` |
| DP2xTP2 cells pair **cross-switch**: `GPU_IDS=0,2,1,3` (vLLM gives DP rank i the i-th TP-sized slice of `CUDA_VISIBLE_DEVICES`, so rank0 -> GPUs 0,2 and rank1 -> 1,3). `launch.sh` warns when a DP2xTP2 cell still uses `0,1,2,3` on an ACS box. TP4 cells stay on `0,1,2,3` and are marked pessimistic. | cells, `launch.sh` |
| Every `launch.json`, `*.meta.json` and bench JSON carries `p2p_ok`, `p2p_disabled`, `custom_allreduce`, `acs_suspected`, `pessimistic_tp`; `summarise.py` daggers (†) rows with `pessimistic_tp=1` = TP>1 on the ACS box (lower bound). TP1 replica rows are never daggered. | `launch.sh`, `sweep.sh`, `summarise.py` |
| Host RAM is 1.5 TB: the Qwen3.8-Flash-Next DP2 cell (~51 GB N-gram table per DP rank with `VLLM_PLE_CPU_OFFLOAD=1`) fits; `launch.sh` still **warns** (never dies) when `free -g` available < needed (+10 % + 4 GB), using `HOST_RAM_GB` from decisions.env. | `launch.sh` |
| Spec decoding **off** everywhere (`--speculative-config` only when `SPEC_CONFIG` is set; then logged as an A/B). `max-num-seqs 256`, `max-num-batched-tokens 8192`, `RUN_TAG=mnbt16k` variant at 16384. | `env.sh`, `launch.sh` |

### `results/hw/decisions.env` (written by `vast/hardware_truth.sh`, read by `bench/env.sh`)

```
P2P_OK=1            # peer access supported on ALL pairs (bandwidth/latency recorded, not used to disable P2P)
CUSTOM_ALLREDUCE=0  # default off; CUSTOM_ALLREDUCE=1 on the command line = A/B
NCCL_P2P_DISABLE=0  # 1 only by explicit human decision when peer access is unsupported
ACS_SUSPECTED=1
PESSIMISTIC_TP=1    # TP>1 rows get the dagger
HOST_RAM_GB=1500
NOTES="all_reduce busbw 21/38/19 GB/s; ACS on host"
```

Plain `KEY=VALUE` lines, `#` comments, quotes optional. If the file is missing every script
warns once, assumes `P2P_OK=1 CUSTOM_ALLREDUCE=0` and leaves `ACS_SUSPECTED` / `PESSIMISTIC_TP`
**unknown** (empty in the shell, `null` in `launch.json` / `*.meta.json`, `"unknown"` in the bench
`--metadata`): TP>1 rows then show `?` instead of the dagger until `vast/hardware_truth.sh` has run.
`ACS_SUSPECTED=1 PESSIMISTIC_TP=1` on the command line override the file for a one-off. `P2P_OK`
is only ever read from the file: older boxes exported a `P2P_OK=false` verdict from a legacy
`$BENCH_ROOT/env.sh` into login shells. That file is no longer written by `vast/hardware_truth.sh`
nor sourced by onstart; if a stale copy exists, `bench/env.sh` reads it in a subshell and imports
only machine facts / `COST_PER_HOUR` -- never a decision key or `NCCL_*` (`BENCH_SKIP_ROOT_ENV=1`
skips it).

## Models on disk

Weights are plain directories `/workspace/models/<basename of the HF id>` (`MODELS_DIR`), written
with `hf download <repo> --local-dir <dir>` — e.g. `/workspace/models/DeepSeek-V4-Flash-0731`,
`/workspace/models/Qwen3.8-27B-FP8`, `/workspace/models/gpt-oss-120b`. `load_cell` derives
`MODEL_PATH` from the cell's `MODEL`; when the directory exists (config.json present) it is used
for `vllm serve` and for the bench client's `--tokenizer`, otherwise the HF id is used and
`launch.sh` calls `prefetch.sh` first. `MODEL_SOURCE=local|hub` and the path land in every JSON.

```bash
bench/prefetch.sh <cell|hf-id> [...]         # hf download --local-dir (gpt-oss: --exclude original/* metal/*), .complete marker
bench/prefetch.sh --list                     # size + complete/partial per directory, df
bench/prefetch.sh --delete <cell|hf-id> --yes # rm -rf the directory (refuses while a process references it; FORCE=1)
```

Disk is ~390 GB total (373 GB free at start): benchmark models **sequentially** and delete each
directory before downloading the next. `prefetch.sh` refuses a download when free space is below
size + 10 % + 5 GB (size table: 27B-FP8 31, gpt-oss 65, Qwen3-8B 16, 27B 56, Qwen3.5-122B 127,
DeepSeek-V4-Flash 167, Flash-Next 186, GLM-5.3-Flash 328 GB; `PREFETCH_SIZE_GB=N` for another repo,
`PREFETCH_FORCE_DISK=1` overrides). `prefetch.log` in `results/` records every download and
deletion with sizes.

## Scripts

| script | what it does |
|---|---|
| `env.sh` | sm_120 baseline env (`VLLM_USE_DEEP_GEMM=0`, `FLASHINFER_CUDA_ARCH_LIST=12.0f`, NCCL PCIe settings), the decisions.env contract, common `vllm serve` flags, shape / concurrency / prompt-count tables, `load_cell` (MODEL_PATH, PORTS, RESULTS_DIR, CELL_PESSIMISTIC_TP), `model_dir_state`. |
| `launch.sh <cell>` | starts server(s) in tmux (`bench_<cell>`), one window per process (+ `rr_proxy.py` for x4 cells), resolves local weights, host-RAM and GPU-pairing warnings, waits on `/health` with timeout, snapshots versions, writes `results/<cell>/launch.json` (incl. hw decisions and the exact argv), smoke request. `LOADTEST_ONLY=1` cells record `loadtest.json` and tear down. `--dry-run` prints the commands. |
| `_run_server.sh` | internal tmux runner: sources `server.env`, tees the log, writes `.pid` / `.exit`, forwards TERM/HUP to the server. `BENCH_CELL=<cell>` is exported to the whole process tree. |
| `sweep.sh <cell> [shapes] [concs]` | `vllm bench serve` per shape x concurrency with `nvidia-smi dmon -s pucm` in the background; per-shape concurrency lists and prompt counts, estimated minutes logged (and refined from the last run), `MAX_RUN_MINUTES` budget, optional `--dataset-path real.jsonl`, `BENCH_DROP_ARGS="--flag ..."` to remove a client flag a build rejects. Ends by running `summarise.py`. |
| `stop.sh [<cell>\|--all]` | TERMs the API server(s), kills the tmux session, then finds every process of the cell (`BENCH_CELL` marker in `/proc/<pid>/environ`, `--served-model-name <alias>`, or vllm-like processes whose `CUDA_VISIBLE_DEVICES` touches the cell's GPUs: EngineCore / TP workers that outlived the parent), TERM -> KILL, waits until the cell's GPUs report < 1.5 GB used. `--all` also kills dmon samplers, bench clients and any co-tenant GPU job. |
| `summarise.py` | aggregates result JSONs + dmon CSVs -> `results/summary.md`, `.csv`, `.json`, with the † dagger for pessimistic TP rows and `?` for TP>1 rows whose flag is unknown (no decisions.env at run time). |
| `rr_proxy.py` | stdlib round-robin proxy (`:8080`) in front of the four replica ports; streams SSE; 502 on backend errors; `/health`, `/proxy_stats`. Used for gates/ad-hoc; the sweep uses per-port fan-out by default. |
| `prefetch.sh <cell\|model>` | `hf download --local-dir` into `$MODELS_DIR/<basename>` behind a free-disk guard; `--list`, `--delete <cell\|id> --yes`. |
| `setup_engine.sh` | idempotent container prep; uv ONLY for the engine interpreter (never pip): `UPGRADE_VLLM=1` performs the documented in-place upgrade pinned to `VLLM_TARGET_VERSION` from the cu130 nightly index (`--index-strategy unsafe-best-match --torch-backend cu130`), adds `vllm[b12x]`; the `hf` CLI goes into `/workspace/venv-tools`. |
| `collect_env.sh` | driver/CUDA/engine versions, GPU edition/vBIOS/power, decisions, models on disk -> `results/env.json`. |

## Shapes, concurrency and prompt counts

| shape | input | output | concurrencies (default) | prompts per run |
|---|---|---|---|---|
| router | 1024 | 128 | 1 4 8 16 32 64 128 256 | max(4C, 64) |
| judge | 4096 | 512 | 1 4 16 64 128 256 | max(4C, 64) |
| agent | 32768 | 2048 | 1 4 16 64 | max(2C, 16) |

`--request-rate inf`, `--ignore-eos` (exact output length), `--random-range-ratio 0` (fixed
lengths). Percentiles saved: p50/p90/p99 of TTFT, TPOT, ITL, E2EL. The agent shape at C=256 with
4C prompts would prefill ~33M tokens (~30 min per point at ~20k tok/s), hence the shorter list
and 2C prompts. Overrides: `ROUTER_CONCS`, `JUDGE_CONCS`, `AGENT_CONCS` (or a concurrency list on
the command line, applied to every shape), `ROUTER_NP_MULT`/`_MIN` etc., `NUM_PROMPTS_CAP`,
`EST_TOTAL_TOK_S` (planning assumption for the logged estimate), `MAX_RUN_MINUTES` (skip longer
runs, recorded as `<run_id>.skipped.json`).

**x4 replica cells**: concurrency C is split over `min(C, 4)` servers, one `vllm bench serve`
per port run concurrently (C=1 -> one replica at c=1; C=256 -> four at c=64). `summarise.py`
sums req/s and tok/s across the ports of a run, takes completion-weighted mean p50s and max
p99s. `--via-proxy` instead drives `rr_proxy.py` with exact C (single Python process, may cap
very high aggregate tok/s — use for sanity checks, not headline numbers).

## Real-prompt JSONL (`--dataset-path`)

One JSON object per line; `prompt` is required, `output_tokens` is the per-line output length
that vLLM's custom dataset reads:

```json
{"prompt": "You are a router. Classify the following request ...", "output_tokens": 128}
{"prompt": "Judge the two answers below ...", "output_tokens": 512}
```

`--output-len N` (default 256) overrides every line; `--output-len -1` honours the per-line
`output_tokens` (`--custom-output-len -1`; every line must carry it). The run is labelled
`custom` (`--shape-label NAME` to change). With the completions backend the chat template is
skipped (`--skip-chat-template`); set `BENCH_BACKEND=openai-chat BENCH_ENDPOINT=/v1/chat/completions`
in the cell to benchmark through the chat endpoint instead.

## What each run records

`results/<cell>[__$RUN_TAG]/`

| file | content |
|---|---|
| `<run_id>.meta.json` | cell, engine + version, model + `model_path`/`model_source`, TP/DP/replicas, kv dtype, mnbt, shape, in/out len, C, prompts, ports, per-replica concurrencies, `p2p_ok` `p2p_disabled` `custom_allreduce` `acs_suspected` `pessimistic_tp`, estimate, timestamps, bench exit code, GPU memory after the run |
| `<run_id>[__pPORT].json` | `vllm bench serve --save-result` output incl. `--metadata` (same keys) |
| `<run_id>.dmon.csv` | `nvidia-smi dmon -s pucm -d 1 -o DT`: power, temp, SM/mem util, clocks, FB memory per GPU per second |
| `<run_id>.skipped.json` | runs skipped by `MAX_RUN_MINUTES` |
| `launch.json` | seconds to `/health`, `GPU KV cache size` / `Maximum concurrency` lines, hw decisions, host-RAM check, exact `server_argv`, image hint |
| `server_p<port>.log/.sh/.pid/.exit` | full server log, the exact command as a runnable script, pid, exit code |
| `versions.txt gpus.txt server.env` | pip versions, GPU inventory, exact server environment |

`run_id = <cell>[__tag]__<shape>__c<C>__<YYYYmmddTHHMMSS>`.

## Caveats worth knowing before the meter runs

* `ds4flash_tp2x2`: ~84 GB weights per GPU; may OOM at cudagraph capture. Reference is `ds4flash_tp4`.
* `qwen38flashnext_fp8_tp2x2`: `VLLM_PLE_CPU_OFFLOAD=1` needs >= 51 GB host RAM **per DP rank**;
  fine with 1.5 TB, `launch.sh` logs the check. The recipe wants vLLM >= 0.29.0 (main has it).
* DeepSeek KV dtype: cells default to `fp8_ds_mla` (valid `--kv-cache-dtype` on main); the recipe's
  RTX PRO 6000 row uses plain `fp8`. `KV_CACHE_DTYPE=fp8 bench/launch.sh ds4flash_tp4` if rejected.
  `--attention_config.use_fp4_indexer_cache False` (recipe) is accepted by main but the field is
  deprecated for removal in v0.29 (successor: `indexer_kv_dtype`).
* The bench client loads the tokenizer from the local model directory with `--tokenizer-mode auto`.
  That option is a plain string (no argparse choices; the help lists auto/hf/slow/mistral/deepseek_v32)
  resolved by `vllm/tokenizers/registry.py`, where `deepseek_v4` is registered as well, so the DS4
  cells' `DS4_BENCH_TOKENIZER_MODE=deepseek_v4` is accepted (its load path is UNVERIFIED);
  `DS4_BENCH_TOKENIZER=<dir>` points `--tokenizer` at another local directory instead.
* gpt-oss at the agent shape: ~37 KB/token KV at fp8 -> only ~18 concurrent 34.8K-token
  sequences per GPU; C>=64 saturates the KV cache (queueing, huge TTFT) — expected.
* Per-port fan-out sums assume the four bench processes overlap fully (equal prompt counts,
  started together); the summary uses the longest duration.
* `stop.sh --all` kills the co-tenant Unsloth job on GPU 3 too — use `stop.sh <cell>` during the training cell.
