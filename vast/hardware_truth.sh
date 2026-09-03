#!/usr/bin/env bash
# hardware_truth.sh -- ground truth for the 4x RTX PRO 6000 Blackwell box.
# Runs INSIDE the Vast.ai container (root). No Docker, no BIOS, no host access. Run it BEFORE any engine touches
# the GPUs (it allocates up to ~1 GB per GPU for the NCCL microbench).
#
# Produces (default OUT = $HW_DIR = <repo>/results/hw -- the same directory bench/env.sh reads):
#   $OUT/decisions.env   THE shared hardware/decision contract, plain KEY=VALUE, sourced by bench/env.sh:
#                          P2P_OK  CUSTOM_ALLREDUCE  NCCL_P2P_DISABLE  ACS_SUSPECTED  PESSIMISTIC_TP  HOST_RAM_GB  NOTES
#                        (+ informational GENERATED_UTC HW_JSON SAME_SWITCH_PAIRS TP2_CROSS_SWITCH_GPU_IDS)
#   $OUT/hardware.json   everything parsed: GPU edition/vBIOS/power/BAR1, PCIe link per GPU, topo matrix, peer-access
#                        matrix, P2P copy bandwidth + latency, NCCL all_reduce busbw (4-GPU ring, same-switch pair,
#                        cross-switch pair, P2P-disabled comparison), ACS evidence, the decisions above
#   $OUT/machine.env     machine FACTS only (GPU name/edition/vBIOS/driver/power, switch pairs, TP2 cross-switch order,
#                        recommended sm_120 baseline env) for humans and ad-hoc scripts.  It carries NO decision keys
#                        and is NOT sourced by bench/env.sh or by login shells: decisions.env is the single contract.
#                        (Older versions wrote $OUT/env.sh with P2P_OK/NCCL_P2P_DISABLE exports and copied it to
#                        <repo>/env.sh -- gone; a stale <repo>/env.sh is reported at the end so you can delete it.)
#   $OUT/raw/*           every raw tool output untouched (+ the two torch helper scripts that were run)
#   <repo>/results/hardware.json   convenience copy (bench/collect_env.sh and the gates look here)
#
# Decision rules (agreed 2026-09-02 after measuring this box; see README "Measured on the Vast box"):
#   P2P_OK            1 iff peer access is supported on ALL GPU pairs (torch.cuda.can_device_access_peer, or the
#                     cuda-samples p2pBandwidthLatencyTest "CAN Access Peer" lines). Bandwidth and latency are
#                     RECORDED only -- they never disable P2P. The old rule "P2P latency > 5 us -> NCCL_P2P_DISABLE=1"
#                     was wrong for this box (P2P works, transport P2P/CUMEM, ~52 GB/s unidirectional) and is gone.
#   NCCL_P2P_DISABLE  0. This script NEVER writes 1. P2P beats host staging; only a human flips it (edit decisions.env
#                     and add HUMAN_DECISION=1 so a re-run keeps it) and only when peer access is genuinely unsupported.
#   CUSTOM_ALLREDUCE  0 (vLLM gets --disable-custom-all-reduce by default). A/B per launch: CUSTOM_ALLREDUCE=1 bench/launch.sh ...
#   ACS_SUSPECTED     1 when a same-PCIe-switch pair (topo PIX/PXB) all-reduces SLOWER than a cross-switch pair
#                     (busbw ratio < ACS_RATIO, default 0.8), or lspci shows ACS enabled on a bridge port. PCIe ACS on
#                     the host redirects switch-local P2P through the root complex; nothing in the container can change it.
#   PESSIMISTIC_TP    1 when ACS_SUSPECTED=1 or P2P_OK=0 -> TP2/TP4 rows carry a dagger in every summary; TP1 replica
#                     rows do not.
#   TP2 pairing       DP2 x TP2 cells pair CROSS-switch (GPU_IDS=0,2,1,3 on this box: vLLM gives DP rank i the i-th
#                     TP-sized slice of CUDA_VISIBLE_DEVICES). The recommendation is derived from the topo matrix and
#                     written as TP2_CROSS_SWITCH_GPU_IDS.
#
# Usage:  bash vast/hardware_truth.sh [OUT_DIR]
# Env:    TOOLS_DIR=/workspace/tools     optional cuda-samples p2pBandwidthLatencyTest + nccl-tests binaries (onstart.sh
#                                        builds them only when nvcc exists). Torch fallbacks run when they are missing:
#                                        torch peer-access/copy matrix and a torch.distributed NCCL all_reduce microbench.
#         SKIP_P2P=1 SKIP_NCCL=1 SKIP_TORCH=1     skip the slow parts
#         PARSE_ONLY=1                             re-parse raw/ and rewrite the outputs without touching the GPUs
#         TORCH_NCCL=1                             run the torch all_reduce microbench even when nccl-tests exists
#         ACS_RATIO=0.8  ACS_SUSPECTED_OVERRIDE=0|1  NCCL_TIMEOUT=900 (s per NCCL run)
#         BENCH_ROOT / RESULTS_ROOT / HW_DIR       repo root / results dir / output dir (defaults derived from this file)
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${BENCH_ROOT:-$(cd "$HERE/.." && pwd)}"
RESULTS_ROOT="${RESULTS_ROOT:-$REPO_ROOT/results}"
OUT="${1:-${HW_DIR:-$RESULTS_ROOT/hw}}"
RAW="$OUT/raw"
TOOLS="${TOOLS_DIR:-/workspace/tools}"
NCCL_TIMEOUT="${NCCL_TIMEOUT:-900}"
ACS_RATIO="${ACS_RATIO:-0.8}"
PARSE_ONLY="${PARSE_ONLY:-0}"
mkdir -p "$RAW" "$RESULTS_ROOT"
LOG="$OUT/hardware_truth.log"
exec > >(tee -a "$LOG") 2>&1

have() { command -v "$1" >/dev/null 2>&1; }
hdr()  { printf '\n==== %s  (%s) ====\n' "$1" "$(date -u +%FT%TZ)"; }

hdr "hardware_truth start"
echo "OUT=$OUT  REPO_ROOT=$REPO_ROOT  TOOLS=$TOOLS  host=$(hostname)  container_id=${CONTAINER_ID:-?}  PARSE_ONLY=$PARSE_ONLY"

# ---- helper scripts (written to raw/ so the record shows exactly what ran) -----------------------------------------
cat > "$RAW/torch_p2p.py" <<'PY'
#!/usr/bin/env python3
# torch_p2p.py OUT_JSON -- peer-access matrix (torch.cuda.can_device_access_peer), unidirectional device->device copy
# bandwidth (256 MiB, 10 reps) and small-copy latency (8 B, 200 reps, includes launch overhead) for every GPU pair.
# torch enables peer access lazily on the first cross-device copy, so the copies below ride P2P when it is supported.
import json, sys, time
import torch
n = torch.cuda.device_count()
res = {"tool": "torch", "torch": torch.__version__, "cuda": torch.version.cuda, "device_count": n,
       "names": [torch.cuda.get_device_name(i) for i in range(n)],
       "capability": [list(torch.cuda.get_device_capability(i)) for i in range(n)],
       "can_access_peer": [[True if i == j else bool(torch.cuda.can_device_access_peer(i, j)) for j in range(n)] for i in range(n)],
       "unidir_copy_gbps": [[None] * n for _ in range(n)], "copy_latency_us": [[None] * n for _ in range(n)], "errors": []}
NB = 256 << 20
for i in range(n):
    for j in range(n):
        if i == j or not res["can_access_peer"][i][j]:
            continue
        try:
            src = torch.empty(NB, dtype=torch.uint8, device=f"cuda:{i}")
            dst = torch.empty(NB, dtype=torch.uint8, device=f"cuda:{j}")
            for _ in range(3):
                dst.copy_(src, non_blocking=True)
            torch.cuda.synchronize(i); torch.cuda.synchronize(j)
            t0 = time.perf_counter()
            for _ in range(10):
                dst.copy_(src, non_blocking=True)
            torch.cuda.synchronize(i); torch.cuda.synchronize(j)
            res["unidir_copy_gbps"][i][j] = round(NB * 10 / (time.perf_counter() - t0) / 1e9, 2)
            s8 = torch.zeros(2, dtype=torch.float32, device=f"cuda:{i}")
            d8 = torch.zeros(2, dtype=torch.float32, device=f"cuda:{j}")
            for _ in range(20):
                d8.copy_(s8); torch.cuda.synchronize(j)
            t0 = time.perf_counter()
            for _ in range(200):
                d8.copy_(s8); torch.cuda.synchronize(j)
            res["copy_latency_us"][i][j] = round((time.perf_counter() - t0) / 200 * 1e6, 2)
            del src, dst, s8, d8
        except Exception as e:  # noqa: BLE001
            res["errors"].append(f"{i}->{j}: {e!r}")
res["all_pairs_peer_access"] = all(res["can_access_peer"][i][j] for i in range(n) for j in range(n)) if n else None
off = [res["unidir_copy_gbps"][i][j] for i in range(n) for j in range(n) if i != j and res["unidir_copy_gbps"][i][j]]
res["unidir_copy_gbps_min"] = min(off) if off else None
res["unidir_copy_gbps_max"] = max(off) if off else None
lat = [res["copy_latency_us"][i][j] for i in range(n) for j in range(n) if i != j and res["copy_latency_us"][i][j]]
res["copy_latency_us_max"] = max(lat) if lat else None
with open(sys.argv[1], "w") as f:
    json.dump(res, f, indent=2)
print(json.dumps({k: res[k] for k in ("device_count", "all_pairs_peer_access", "unidir_copy_gbps_min", "unidir_copy_gbps_max", "copy_latency_us_max")}))
PY
cat > "$RAW/torch_allreduce.py" <<'PY'
#!/usr/bin/env python3
# torch_allreduce.py WORLD OUT_JSON [MASTER_PORT] -- NCCL all_reduce busbw via torch.distributed; fallback for
# nccl-tests' all_reduce_perf. Uses the GPUs in CUDA_VISIBLE_DEVICES in order (rank r -> cuda:r).
# busbw = algbw * 2*(n-1)/n (nccl-tests convention), so numbers are comparable with all_reduce_perf.
import json, os, sys, time
import torch, torch.distributed as dist, torch.multiprocessing as mp

SIZES = [8, 1 << 20, 8 << 20, 128 << 20, 1 << 30]

def worker(rank, world, port, out):
    os.environ["MASTER_ADDR"] = "127.0.0.1"; os.environ["MASTER_PORT"] = str(port)
    torch.cuda.set_device(rank)
    dev = torch.device("cuda", rank)
    try:
        dist.init_process_group("nccl", rank=rank, world_size=world, device_id=dev)
    except TypeError:
        dist.init_process_group("nccl", rank=rank, world_size=world)
    rows = []
    for nbytes in SIZES:
        n = max(nbytes // 4, 2)
        chk = torch.ones(n, device=dev); dist.all_reduce(chk); torch.cuda.synchronize()
        wrong = int((chk != float(world)).sum().item())
        x = torch.ones(n, device=dev)
        for _ in range(5):
            dist.all_reduce(x)
        torch.cuda.synchronize(); dist.barrier()
        iters = 20 if nbytes >= (128 << 20) else 100
        t0 = time.perf_counter()
        for _ in range(iters):
            dist.all_reduce(x)
        torch.cuda.synchronize()
        dt = (time.perf_counter() - t0) / iters
        algbw = nbytes / dt / 1e9
        rows.append({"size_bytes": nbytes, "time_us": round(dt * 1e6, 2), "algbw_gbps": round(algbw, 3),
                     "busbw_gbps": round(algbw * 2 * (world - 1) / world, 3), "wrong": wrong})
        del x, chk
    dist.barrier()
    if rank == 0:
        with open(out, "w") as f:
            json.dump({"tool": "torch.distributed all_reduce", "world": world,
                       "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"), "torch": torch.__version__,
                       "nccl": ".".join(map(str, torch.cuda.nccl.version())), "rows": rows}, f, indent=2)
    dist.destroy_process_group()

if __name__ == "__main__":
    world = int(sys.argv[1]); out = sys.argv[2]; port = int(sys.argv[3]) if len(sys.argv) > 3 else 29517
    mp.spawn(worker, args=(world, port, out), nprocs=world, join=True)
PY

# pick_pairs: first same-switch (PIX/PXB) and first cross-switch (PHB/NODE/SYS) GPU pair from raw/topo.txt
pick_pairs() {
  python3 - "$RAW/topo.txt" <<'PY'
import re, sys, os
p = sys.argv[1]
txt = open(p, errors="replace").read() if os.path.exists(p) else ""
hdr = None
for l in txt.splitlines():
    t = l.split()
    if len(t) > 1 and t[0] == "GPU0" and t[1].startswith("GPU"):
        hdr = t; break
same = cross = None
if hdr:
    for l in txt.splitlines():
        t = l.split()
        if len(t) < 2 or not re.match(r"^GPU\d+$", t[0]) or t[1].startswith("GPU"):
            continue
        i = int(t[0][3:])
        for j, h in enumerate(hdr):
            if not h.startswith("GPU") or h == t[0] or j + 1 >= len(t):
                continue
            k = int(h[3:]); link = t[j + 1]
            if i < k:
                if link in ("PIX", "PXB") and same is None: same = (i, k)
                if link in ("PHB", "NODE", "SYS") and cross is None: cross = (i, k)
print("SAME=%s CROSS=%s" % ("%d,%d" % same if same else "", "%d,%d" % cross if cross else ""))
PY
}

collect() {
# ---------------------------------------------------------------- 1. nvidia-smi
hdr "nvidia-smi"
if ! have nvidia-smi; then echo "ERROR: nvidia-smi not found -- is this the GPU container?"; fi
nvidia-smi | tee "$RAW/nvidia-smi.txt"
nvidia-smi -q > "$RAW/nvidia-smi-q-full.txt" 2>&1
nvidia-smi -q -d POWER,CLOCK,MEMORY,ECC > "$RAW/nvidia-smi-q-power-clock-memory-ecc.txt" 2>&1
# Core fields (all supported since 5xx drivers). If one field name is unknown the whole query fails,
# so keep an extended query separate.
nvidia-smi --query-gpu=index,name,uuid,pci.bus_id,vbios_version,driver_version,compute_cap,memory.total,power.limit,power.default_limit,power.max_limit,power.min_limit,clocks.max.sm,clocks.max.memory,pcie.link.gen.max,pcie.link.gen.current,pcie.link.width.max,pcie.link.width.current,ecc.mode.current,persistence_mode,temperature.gpu \
  --format=csv,nounits > "$RAW/gpu-query.csv" 2>"$RAW/gpu-query.err" || echo "WARN: gpu-query failed: $(cat "$RAW/gpu-query.err")"
nvidia-smi --query-gpu=index,pcie.link.gen.gpumax,pcie.link.gen.hostmax,fan.speed,clocks.sm,clocks.mem,power.draw \
  --format=csv,nounits > "$RAW/gpu-query-ext.csv" 2>/dev/null || echo "(extended pcie fields not supported by this driver; fine)"
cat "$RAW/gpu-query.csv"

hdr "nvidia-smi topo"
nvidia-smi topo -m | tee "$RAW/topo.txt"
nvidia-smi topo -p2p r > "$RAW/topo-p2p-read.txt" 2>&1 || true
nvidia-smi topo -p2p w > "$RAW/topo-p2p-write.txt" 2>&1 || true
nvidia-smi topo -p2p n > "$RAW/topo-p2p-nvlink.txt" 2>&1 || true
cat "$RAW/topo-p2p-read.txt"

# ---------------------------------------------------------------- 2. PCIe link per GPU (sysfs + lspci), ACS, HMM
hdr "PCIe link width/speed per GPU"
: > "$RAW/pcie-links.txt"; : > "$RAW/lspci-links.txt"
while IFS= read -r bus; do
  bus="$(echo "$bus" | tr -d ' ')"; [ -z "$bus" ] && continue
  sys="/sys/bus/pci/devices/$(echo "$bus" | sed 's/^0000//' | tr 'A-F' 'a-f')"
  if [ -d "$sys" ]; then
    printf 'GPU %s cur_speed=%s cur_width=%s max_speed=%s max_width=%s numa_node=%s\n' \
      "$bus" "$(cat "$sys/current_link_speed" 2>/dev/null)" "$(cat "$sys/current_link_width" 2>/dev/null)" \
      "$(cat "$sys/max_link_speed" 2>/dev/null)" "$(cat "$sys/max_link_width" 2>/dev/null)" "$(cat "$sys/numa_node" 2>/dev/null)" \
      | tee -a "$RAW/pcie-links.txt"
  else
    echo "GPU $bus sysfs path missing ($sys)" | tee -a "$RAW/pcie-links.txt"
  fi
  if have lspci; then
    short="$(echo "$bus" | sed 's/^0000://')"
    { echo "--- lspci -vv -s $short"; lspci -vv -s "$short" 2>/dev/null | grep -E 'LnkCap:|LnkSta:|LnkCap2|LnkSta2|NUMA node|Subsystem|Region 1'; } >> "$RAW/lspci-links.txt"
  fi
done < <(nvidia-smi --query-gpu=pci.bus_id --format=csv,noheader 2>/dev/null)
cat "$RAW/lspci-links.txt"
echo "NOTE: current_link_speed drops to 2.5 GT/s when the GPU is idle (ASPM); max_link_speed/LnkCap is the truth for Gen5."
# PCIe ACS: visible only if the container may read extended config space (usually it cannot -> empty file, fine).
if have lspci; then
  lspci -vvv 2>/dev/null | awk '/^[0-9a-f][0-9a-f]*:[0-9a-f][0-9a-f]\.[0-9a-f]/{dev=$0} /ACSCap:|ACSCtl:/{print dev; print}' > "$RAW/acs-lspci.txt" || true
  echo "lspci ACS capability lines: $(grep -c 'ACSCtl' "$RAW/acs-lspci.txt" 2>/dev/null || echo 0) (0 = extended config space not readable from the container; the NCCL pair test decides instead)"
fi
# HMM / UVM parameters (best effort; sysfs may not be exposed)
{ cat /sys/module/nvidia_uvm/parameters/uvm_disable_hmm 2>/dev/null || true; } > "$RAW/uvm-hmm.txt"
cat /proc/driver/nvidia/params > "$RAW/nvidia-params.txt" 2>/dev/null || true
echo "uvm_disable_hmm=$(cat "$RAW/uvm-hmm.txt" 2>/dev/null || echo '?')  (N = HMM enabled)"

# ---------------------------------------------------------------- 3. CPU / RAM / disk / NUMA
hdr "CPU / RAM / disk"
{ echo "nproc=$(nproc)"; grep -m1 'model name' /proc/cpuinfo; lscpu 2>/dev/null | grep -E 'Socket|NUMA|Thread|Core|Model name|Flags' | grep -v Flags; free -g; df -h /workspace / 2>/dev/null; } | tee "$RAW/cpu-mem-disk.txt"
df -h --output=size,used,avail,pcent,target /workspace 2>/dev/null | tail -1 > "$RAW/disk-workspace.txt" || df -h /workspace | tail -1 > "$RAW/disk-workspace.txt"
have numactl && numactl -H > "$RAW/numactl.txt" 2>&1 || true
lscpu > "$RAW/lscpu.txt" 2>&1 || true
cat /proc/meminfo > "$RAW/meminfo.txt"
du -sh "${MODELS_DIR:-/workspace/models}"/* > "$RAW/models-disk.txt" 2>/dev/null || true

# ---------------------------------------------------------------- 4. engine versions
hdr "engine versions"
python3 - <<'PY' > "$RAW/engine-versions.json" 2>/dev/null || echo '{}' > "$RAW/engine-versions.json"
import json, importlib
out = {}
for mod, key in [("vllm","vllm"),("sglang","sglang"),("torch","torch"),("flashinfer","flashinfer"),("transformers","transformers"),("b12x","b12x"),("triton","triton")]:
    try:
        m = importlib.import_module(mod); out[key] = getattr(m, "__version__", "present")
    except Exception:
        out[key] = None
try:
    import torch
    out["torch_cuda"] = torch.version.cuda
    out["torch_cudnn"] = torch.backends.cudnn.version()
    out["torch_nccl"] = ".".join(map(str, torch.cuda.nccl.version())) if torch.cuda.is_available() else None
    out["cuda_device_count"] = torch.cuda.device_count()
    out["cuda_capability"] = [list(torch.cuda.get_device_capability(i)) for i in range(torch.cuda.device_count())]
except Exception as e:
    out["torch_error"] = str(e)
print(json.dumps(out, indent=2))
PY
cat "$RAW/engine-versions.json"
{ nvcc --version 2>/dev/null | tail -2; cat /usr/local/cuda/version.json 2>/dev/null | head -5; } > "$RAW/cuda-toolkit.txt" || true

# ---------------------------------------------------------------- 5. torch peer-access + copy bandwidth/latency matrix
hdr "torch peer access / P2P copy matrix"
if [ "${SKIP_TORCH:-0}" = "1" ]; then
  echo "SKIP_TORCH=1"
elif python3 -c 'import torch' 2>/dev/null; then
  timeout 600 python3 "$RAW/torch_p2p.py" "$RAW/torch-p2p.json" 2>"$RAW/torch-p2p.err" || echo "WARN: torch_p2p exit=$? ($(tail -1 "$RAW/torch-p2p.err" 2>/dev/null))"
else
  echo "WARN: torch not importable -> peer access falls back to p2pBandwidthLatencyTest only"
fi

# ---------------------------------------------------------------- 6. cuda-samples p2pBandwidthLatencyTest (optional)
hdr "p2pBandwidthLatencyTest"
P2P_BIN="$(find "$TOOLS" -type f -name p2pBandwidthLatencyTest 2>/dev/null | head -1)"
if [ "${SKIP_P2P:-0}" = "1" ]; then
  echo "SKIP_P2P=1"
elif [ -n "$P2P_BIN" ]; then
  echo "using $P2P_BIN"
  timeout 900 "$P2P_BIN" > "$RAW/p2pBandwidthLatencyTest.txt" 2>&1 || echo "WARN: p2pBandwidthLatencyTest exit=$?"
  grep -E 'Device=|Unidirectional|Bidirectional|Latency' "$RAW/p2pBandwidthLatencyTest.txt" | head -20
else
  echo "p2pBandwidthLatencyTest not found under $TOOLS (optional; the torch matrix above covers peer access + bandwidth)."
fi

# ---------------------------------------------------------------- 7. NCCL all_reduce: 4-GPU ring, same-switch pair, cross-switch pair
hdr "NCCL all_reduce busbw"
NGPU="$(nvidia-smi --query-gpu=index --format=csv,noheader 2>/dev/null | wc -l | tr -d ' ')"; NGPU="${NGPU:-0}"
SAME=""; CROSS=""
eval "$(pick_pairs 2>/dev/null)" || true
[ -z "$SAME" ] && [ "$NGPU" -ge 2 ] && SAME="0,1" && echo "(no PIX/PXB pair in topo -> pair_same uses 0,1 as a stand-in)"
[ -z "$CROSS" ] && [ "$NGPU" -ge 3 ] && CROSS="0,2" && echo "(no PHB/NODE/SYS pair in topo -> pair_cross uses 0,2 as a stand-in)"
echo "same-switch pair: ${SAME:-none}   cross-switch pair: ${CROSS:-none}   gpus: $NGPU"
NCCL_COMMON=(NCCL_IB_DISABLE=1 "NCCL_MIN_NCHANNELS=${NCCL_MIN_NCHANNELS:-8}" NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT,GRAPH)
NCCL_BIN="$(find "$TOOLS" -type f -name all_reduce_perf 2>/dev/null | head -1)"
[ -f "$TOOLS/nccl_env.sh" ] && . "$TOOLS/nccl_env.sh"
run_nccl_tests() {  # label ngpu [ENV=VAL ...]   (NCCL_P2P_LEVEL / NCCL_P2P_DISABLE are cleared first, then set from the args)
  local label="$1" ng="$2"; shift 2
  env -u NCCL_P2P_LEVEL -u NCCL_P2P_DISABLE "${NCCL_COMMON[@]}" "$@" timeout "$NCCL_TIMEOUT" "$NCCL_BIN" -b 8 -e 1G -f 2 -g "$ng" -n 20 -w 5 \
    > "$RAW/nccl-$label.txt" 2> "$RAW/nccl-$label.debug" || echo "WARN: nccl-tests ($label) exit=$?"
  echo "--- $label ($*)"; grep -E '^\s+[0-9]+\s+[0-9]+\s+float|Avg bus' "$RAW/nccl-$label.txt" | tail -3
}
run_torch_nccl() {  # label ngpu port [ENV=VAL ...]
  local label="$1" ng="$2" port="$3"; shift 3
  env -u NCCL_P2P_LEVEL -u NCCL_P2P_DISABLE "${NCCL_COMMON[@]}" "$@" timeout "$NCCL_TIMEOUT" python3 "$RAW/torch_allreduce.py" "$ng" "$RAW/torch-nccl-$label.json" "$port" \
    > "$RAW/torch-nccl-$label.log" 2> "$RAW/torch-nccl-$label.debug" || echo "WARN: torch all_reduce ($label) exit=$? ($(grep -m1 -iE 'error|exception' "$RAW/torch-nccl-$label.debug" | cut -c1-200))"
  echo "--- $label ($*)"
  python3 -c 'import json,sys; d=json.load(open(sys.argv[1])); print("  " + "  ".join("%dB:%.1f GB/s" % (r["size_bytes"], r["busbw_gbps"]) for r in d["rows"]))' "$RAW/torch-nccl-$label.json" 2>/dev/null || true
}
if [ "${SKIP_NCCL:-0}" = "1" ]; then
  echo "SKIP_NCCL=1"
elif [ "$NGPU" -lt 2 ]; then
  echo "fewer than 2 GPUs visible -> no NCCL test"
else
  if [ -n "$NCCL_BIN" ] && [ "${TORCH_NCCL:-0}" != "1" ]; then
    echo "using $NCCL_BIN"
    run_nccl_tests ring4_baseline      "$NGPU" "NCCL_P2P_LEVEL=${NCCL_P2P_LEVEL:-PHB}"          # the harness baseline env
    run_nccl_tests ring4_default_level "$NGPU"                                                    # NCCL's own default P2P level
    [ -n "$SAME" ]  && run_nccl_tests pair_same  2 "NCCL_P2P_LEVEL=${NCCL_P2P_LEVEL:-PHB}" "CUDA_VISIBLE_DEVICES=$SAME"
    [ -n "$CROSS" ] && run_nccl_tests pair_cross 2 "NCCL_P2P_LEVEL=${NCCL_P2P_LEVEL:-PHB}" "CUDA_VISIBLE_DEVICES=$CROSS"
    echo "--- comparison only (never a decision input): host-staged path"
    run_nccl_tests ring4_p2p_disabled  "$NGPU" NCCL_P2P_DISABLE=1
  elif python3 -c 'import torch, torch.distributed' 2>/dev/null; then
    [ -z "$NCCL_BIN" ] && echo "all_reduce_perf not found under $TOOLS -> torch.distributed all_reduce microbench (same busbw convention)"
    run_torch_nccl ring4_baseline      "$NGPU" 29517 "NCCL_P2P_LEVEL=${NCCL_P2P_LEVEL:-PHB}"
    run_torch_nccl ring4_default_level "$NGPU" 29518
    [ -n "$SAME" ]  && run_torch_nccl pair_same  2 29519 "NCCL_P2P_LEVEL=${NCCL_P2P_LEVEL:-PHB}" "CUDA_VISIBLE_DEVICES=$SAME"
    [ -n "$CROSS" ] && run_torch_nccl pair_cross 2 29520 "NCCL_P2P_LEVEL=${NCCL_P2P_LEVEL:-PHB}" "CUDA_VISIBLE_DEVICES=$CROSS"
    echo "--- comparison only (never a decision input): host-staged path"
    run_torch_nccl ring4_p2p_disabled  "$NGPU" 29521 NCCL_P2P_DISABLE=1
  else
    echo "WARN: neither nccl-tests nor torch available -> no NCCL numbers; ACS_SUSPECTED will be undetermined"
  fi
fi

# ---------------------------------------------------------------- 8. short idle dmon baseline (10 s)
hdr "nvidia-smi dmon idle baseline"
timeout 12 nvidia-smi dmon -s pucm -d 1 -c 10 > "$RAW/dmon-idle.txt" 2>&1 || true
tail -3 "$RAW/dmon-idle.txt"
}

if [ "$PARSE_ONLY" = "1" ]; then
  echo "PARSE_ONLY=1 -> skipping collection, parsing $RAW"
else
  collect
fi

# ---------------------------------------------------------------- 9. parse -> hardware.json + decisions.env + machine.env
hdr "parse -> hardware.json / decisions.env / machine.env"
OUT="$OUT" RAW="$RAW" REPO_ROOT="$REPO_ROOT" RESULTS_ROOT="$RESULTS_ROOT" ACS_RATIO="$ACS_RATIO" \
ACS_SUSPECTED_OVERRIDE="${ACS_SUSPECTED_OVERRIDE:-}" NCCL_P2P_LEVEL_BASELINE="${NCCL_P2P_LEVEL:-PHB}" python3 - <<'PY'
import csv, datetime, io, json, os, re

OUT, RAW = os.environ["OUT"], os.environ["RAW"]
REPO_ROOT, RESULTS_ROOT = os.environ["REPO_ROOT"], os.environ["RESULTS_ROOT"]
ACS_RATIO = float(os.environ.get("ACS_RATIO") or 0.8)
ACS_OVERRIDE = (os.environ.get("ACS_SUSPECTED_OVERRIDE") or "").strip()
NOW = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

def rd(name):
    p = os.path.join(RAW, name)
    return open(p, errors="replace").read() if os.path.exists(p) else ""
def rj(name):
    try:
        return json.loads(rd(name) or "null")
    except ValueError:
        return None
def f(v):
    try: return float(v)
    except (TypeError, ValueError): return None

hw = {"generated_utc": NOW, "container_id": os.environ.get("CONTAINER_ID"), "public_ip": os.environ.get("PUBLIC_IPADDR"),
      "decision_rules": {"p2p_ok": "peer access supported on all GPU pairs (bandwidth/latency recorded, never used to disable P2P)",
                         "nccl_p2p_disable": "never set automatically; human decision only (HUMAN_DECISION=1 in decisions.env)",
                         "custom_allreduce": "0 by default; A/B with CUSTOM_ALLREDUCE=1 at launch",
                         "acs_suspected": f"same-switch pair busbw / cross-switch pair busbw < {ACS_RATIO}, or lspci ACSCtl enabled",
                         "pessimistic_tp": "acs_suspected or not p2p_ok"}}

# --- GPUs -----------------------------------------------------------------------------------------------------------
gpus = []
q = rd("gpu-query.csv")
if q.strip():
    for r in csv.DictReader(io.StringIO(q)):
        r = {k.strip().split(" [")[0]: (v.strip() if isinstance(v, str) else v) for k, v in r.items() if k}
        name = r.get("name", "")
        ed = ("Max-Q" if "max-q" in name.lower() else "Server Edition" if "server" in name.lower()
              else "Workstation Edition" if "workstation" in name.lower() else "unknown")
        gpus.append({"index": int(r.get("index", len(gpus)) or len(gpus)), "name": name, "edition_guess": ed, "uuid": r.get("uuid"),
                     "pci_bus_id": r.get("pci.bus_id"), "vbios": r.get("vbios_version"), "driver": r.get("driver_version"),
                     "compute_cap": r.get("compute_cap"), "memory_total_mib": f(r.get("memory.total")),
                     "power_limit_w": f(r.get("power.limit")), "power_default_limit_w": f(r.get("power.default_limit")),
                     "power_max_limit_w": f(r.get("power.max_limit")), "power_min_limit_w": f(r.get("power.min_limit")),
                     "clocks_max_sm_mhz": f(r.get("clocks.max.sm")), "clocks_max_mem_mhz": f(r.get("clocks.max.memory")),
                     "pcie_gen_max": r.get("pcie.link.gen.max"), "pcie_gen_current_idle": r.get("pcie.link.gen.current"),
                     "pcie_width_max": r.get("pcie.link.width.max"), "pcie_width_current_idle": r.get("pcie.link.width.current"),
                     "ecc_mode": r.get("ecc.mode.current"), "persistence_mode": r.get("persistence_mode")})
hw["gpus"] = gpus; hw["gpu_count"] = len(gpus)
bar1 = [int(x) for x in re.findall(r"BAR1 Memory Usage\s*\n\s*Total\s*:\s*(\d+)\s*MiB", rd("nvidia-smi-q-power-clock-memory-ecc.txt"))]
for g, b in zip(gpus, bar1):
    g["bar1_total_mib"] = b; g["rebar_enabled_guess"] = b > 1024
links = {}
for line in rd("pcie-links.txt").splitlines():
    m = re.match(r"GPU (\S+) cur_speed=(.*?) cur_width=(\S*) max_speed=(.*?) max_width=(\S*) numa_node=(\S*)", line)
    if m:
        links[m.group(1)] = {"cur_speed_idle": m.group(2).strip(), "cur_width": m.group(3), "max_speed": m.group(4).strip(),
                             "max_width": m.group(5), "numa_node": m.group(6)}
for g in gpus:
    g["sysfs_link"] = links.get(g["pci_bus_id"], {})
    ms = g["sysfs_link"].get("max_speed", "")
    g["pcie_gen5_capable"] = ("32" in ms) or (str(g.get("pcie_gen_max")) == "5")
hmm = rd("uvm-hmm.txt").strip()
hw["hmm"] = {"uvm_disable_hmm": hmm or None, "hmm_enabled_guess": (hmm == "N") if hmm in ("Y", "N") else None}

# --- topo matrix -> pairwise link types, same-switch groups, TP2 pairing recommendation --------------------------------
topo = rd("topo.txt"); hw["topo_raw"] = topo
pair, hdrs = {}, None
for l in topo.splitlines():
    toks = l.split()
    if len(toks) > 1 and toks[0] == "GPU0" and toks[1].startswith("GPU"):
        hdrs = toks; break
if hdrs:
    for l in topo.splitlines():
        toks = l.split()
        if len(toks) < 2 or not re.match(r"^GPU\d+$", toks[0]) or toks[1].startswith("GPU"):
            continue
        src = toks[0]
        for j, h in enumerate(hdrs):
            if h.startswith("GPU") and j + 1 < len(toks) and h != src:
                pair[f"{src}->{h}"] = toks[j + 1]
hw["topo_gpu_pairs"] = pair
hw["topo_link_types_present"] = sorted(set(pair.values()))
SAME_LINKS, CROSS_LINKS = ("PIX", "PXB"), ("PHB", "NODE", "SYS")
def gid(s): return int(s[3:])
same_pairs = sorted({tuple(sorted((gid(a), gid(b)))) for k, v in pair.items() for a, b in [k.split("->")] if v in SAME_LINKS})
cross_pairs = sorted({tuple(sorted((gid(a), gid(b)))) for k, v in pair.items() for a, b in [k.split("->")] if v in CROSS_LINKS})
nvlink_pairs = sorted({tuple(sorted((gid(a), gid(b)))) for k, v in pair.items() for a, b in [k.split("->")] if v.startswith("NV")})
hw["same_switch_pairs"] = same_pairs; hw["cross_switch_pairs"] = cross_pairs; hw["nvlink_pairs"] = nvlink_pairs
# connected components over same-switch links = PCIe switch groups
ids = sorted({gid(k.split("->")[0]) for k in pair}) or list(range(len(gpus)))
groups, seen = [], set()
for i in ids:
    if i in seen: continue
    comp, stack = [], [i]
    while stack:
        x = stack.pop()
        if x in seen: continue
        seen.add(x); comp.append(x)
        stack.extend(b for a, b in same_pairs if a == x); stack.extend(a for a, b in same_pairs if b == x)
    groups.append(sorted(comp))
hw["switch_groups"] = groups
tp2_ids = None
if len(groups) == 2 and len(groups[0]) == 2 and len(groups[1]) == 2:
    tp2_ids = f"{groups[0][0]},{groups[1][0]},{groups[0][1]},{groups[1][1]}"   # rank0 -> one GPU from each switch, rank1 -> the other two
    tp2_note = "DP2xTP2: each TP pair spans both switches (rank i gets slice i of CUDA_VISIBLE_DEVICES)"
elif len(ids) >= 2:
    tp2_ids = ",".join(str(i) for i in ids); tp2_note = "no 2x2 switch layout detected; natural order kept"
else:
    tp2_note = "n/a"
hw["tp2_cross_switch_gpu_ids"] = tp2_ids; hw["tp2_pairing_note"] = tp2_note

# --- torch peer-access / copy matrix ----------------------------------------------------------------------------------
tp = rj("torch-p2p.json") or {}
hw["torch_p2p"] = tp or {"available": False}

# --- cuda-samples p2pBandwidthLatencyTest (optional) -----------------------------------------------------------------
p2p_txt = rd("p2pBandwidthLatencyTest.txt")
p2p = {"available": bool(p2p_txt.strip())}
if p2p["available"]:
    p2p["cannot_access_peer_lines"] = len(re.findall(r"CANNOT Access Peer", p2p_txt))
    p2p["can_access_peer_lines"] = len(re.findall(r"\bCAN Access Peer", p2p_txt))
    mats, key, sub = {}, None, None
    for line in p2p_txt.splitlines():
        if re.search(r"P2P=(Enabled|Disabled) Bandwidth.*Matrix", line):
            key = re.sub(r"\s+", " ", line.strip()); sub = "D"; mats[key] = {}; continue
        if re.search(r"P2P=(Enabled|Disabled) Latency.*Matrix", line):
            key = re.sub(r"\s+", " ", line.strip()); sub = None; mats[key] = {}; continue
        if key is None: continue
        t = line.split()
        if not t: continue
        if t[0] in ("D\\D", "GPU", "CPU"):
            sub = t[0]; mats[key][sub] = []; continue
        if re.match(r"^\d+$", t[0]) and sub is not None:
            try: mats[key].setdefault(sub, []).append([float(x) for x in t[1:]])
            except ValueError: pass
    def offdiag(m): return [x for i, row in enumerate(m) for j, x in enumerate(row) if i != j]
    summary = {}
    for k, subs in mats.items():
        for s, m in subs.items():
            od = offdiag(m)
            if od: summary[f"{k} [{s}]"] = {"min_offdiag": min(od), "max_offdiag": max(od), "mean_offdiag": sum(od) / len(od), "n": len(m)}
    p2p["matrices"] = mats; p2p["summary"] = summary
    def pick(pattern, sub, field):
        for k, v in summary.items():
            if re.search(pattern, k) and k.endswith(f"[{sub}]"): return v[field]
        return None
    p2p["unidir_p2p_enabled_min_gbps"] = pick(r"Unidirectional P2P=Enabled", "D\\D", "min_offdiag")
    p2p["unidir_p2p_enabled_max_gbps"] = pick(r"Unidirectional P2P=Enabled", "D\\D", "max_offdiag")
    p2p["unidir_p2p_disabled_min_gbps"] = pick(r"Unidirectional P2P=Disabled", "D\\D", "min_offdiag")
    p2p["bidir_p2p_enabled_min_gbps"] = pick(r"Bidirectional P2P=Enabled", "D\\D", "min_offdiag")
    p2p["latency_p2p_enabled_gpu_max_us"] = pick(r"P2P=Enabled Latency", "GPU", "max_offdiag")
    p2p["latency_p2p_enabled_gpu_min_us"] = pick(r"P2P=Enabled Latency", "GPU", "min_offdiag")
    p2p["latency_p2p_disabled_gpu_max_us"] = pick(r"P2P=Disabled Latency", "GPU", "max_offdiag")
hw["p2p_test"] = p2p

# --- NCCL all_reduce (nccl-tests or torch fallback), per label ---------------------------------------------------------
def parse_nccl_tests(txt, dbg):
    rows = []
    for line in txt.splitlines():
        m = re.match(r"^\s*(\d+)\s+(\d+)\s+(\w+)\s+(\w+)\s+(-?\d+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+(\S+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+(\S+)", line)
        if m:
            rows.append({"size_bytes": int(m.group(1)), "time_us": float(m.group(6)), "algbw_gbps": float(m.group(7)),
                         "busbw_gbps": float(m.group(8)), "wrong": m.group(9), "time_us_ip": float(m.group(10)),
                         "algbw_gbps_ip": float(m.group(11)), "busbw_gbps_ip": float(m.group(12)), "wrong_ip": m.group(13)})
    avg = re.search(r"Avg bus bandwidth\s*:\s*([\d.]+)", txt)
    return finish_nccl(rows, dbg, "nccl-tests all_reduce_perf", float(avg.group(1)) if avg else None,
                       any(r["wrong"] not in ("0", "N/A") or r["wrong_ip"] not in ("0", "N/A") for r in rows))
def parse_torch_nccl(js, dbg):
    rows = (js or {}).get("rows") or []
    return finish_nccl(rows, dbg, "torch.distributed all_reduce", None, any(int(r.get("wrong", 0)) != 0 for r in rows), js)
def finish_nccl(rows, dbg, source, avg, wrong, js=None):
    if not rows: return {"available": False, "source": source}
    def at(sz):
        for r in rows:
            if r["size_bytes"] == sz: return r["busbw_gbps"]
        return None
    transports = sorted(set(re.findall(r"via (P2P/CUMEM|P2P/IPC|P2P/direct pointer|P2P/[A-Za-z_]+|SHM/[A-Za-z_/]+|NET/[A-Za-z_/]+)", dbg)))
    nccl_ver = re.search(r"NCCL version ([\d.]+)", dbg)
    return {"available": True, "source": source, "rows": rows, "avg_busbw_gbps": avg,
            "busbw_gbps_8MB": at(8388608), "busbw_gbps_128MB": at(134217728), "busbw_gbps_1GB": at(1073741824),
            "busbw_gbps_largest": rows[-1]["busbw_gbps"], "latency_us_8B": rows[0]["time_us"] if rows[0]["size_bytes"] == 8 else None,
            "any_wrong": bool(wrong), "transports_seen": transports, "nccl_version": nccl_ver.group(1) if nccl_ver else (js or {}).get("nccl"),
            "cuda_visible_devices": (js or {}).get("cuda_visible_devices")}
LABELS = ["ring4_baseline", "ring4_default_level", "pair_same", "pair_cross", "ring4_p2p_disabled"]
nccl = {}
for lab in LABELS:
    if os.path.exists(os.path.join(RAW, f"nccl-{lab}.txt")):
        nccl[lab] = parse_nccl_tests(rd(f"nccl-{lab}.txt"), rd(f"nccl-{lab}.debug"))
    elif os.path.exists(os.path.join(RAW, f"torch-nccl-{lab}.json")):
        nccl[lab] = parse_torch_nccl(rj(f"torch-nccl-{lab}.json"), rd(f"torch-nccl-{lab}.debug"))
    else:
        nccl[lab] = {"available": False}
hw["nccl"] = nccl
hw["nccl_env_baseline"] = {"NCCL_P2P_LEVEL": os.environ.get("NCCL_P2P_LEVEL_BASELINE"), "NCCL_IB_DISABLE": "1", "NCCL_MIN_NCHANNELS": os.environ.get("NCCL_MIN_NCHANNELS", "8")}
hw["nccl_all_reduce_p2p_default"] = nccl["ring4_baseline"]        # legacy keys kept for collect_env.sh
hw["nccl_all_reduce_p2p_disabled"] = nccl["ring4_p2p_disabled"]

# --- lspci ACS ----------------------------------------------------------------------------------------------------------
acs_txt = rd("acs-lspci.txt")
acs_ctl = re.findall(r"ACSCtl:\s*(.*)", acs_txt)
acs_enabled_ports = sum(1 for l in acs_ctl if re.search(r"(SrcValid|ReqRedir|CmpltRedir|UpstreamFwd)\+", l))
hw["acs_lspci"] = {"ctl_lines": len(acs_ctl), "enabled_ports": acs_enabled_ports, "readable": bool(acs_ctl)}

# --- engine versions, cpu/mem/disk -------------------------------------------------------------------------------------
try: hw["engine_versions"] = json.loads(rd("engine-versions.json") or "{}")
except ValueError: hw["engine_versions"] = {}
cm = rd("cpu-mem-disk.txt")
m = re.search(r"model name\s*:\s*(.*)", cm); hw["cpu_model"] = m.group(1).strip() if m else None
m = re.search(r"nproc=(\d+)", cm); hw["cpu_threads_visible"] = int(m.group(1)) if m else None
m = re.search(r"Mem:\s+(\d+)", cm); ram_gb = int(m.group(1)) if m else None
if ram_gb is None:
    m = re.search(r"MemTotal:\s+(\d+) kB", rd("meminfo.txt")); ram_gb = int(int(m.group(1)) / 1048576) if m else None
hw["ram_total_gib"] = ram_gb
dw = rd("disk-workspace.txt").split()
hw["workspace_disk"] = {"size": dw[0], "used": dw[1], "avail": dw[2], "use_pct": dw[3], "mount": dw[4]} if len(dw) >= 5 else (rd("disk-workspace.txt").strip() or None)
hw["models_on_disk"] = [l.split(None, 1) for l in rd("models-disk.txt").splitlines() if l.strip()]
hw["cuda_toolkit"] = rd("cuda-toolkit.txt").strip()

# --- decisions ---------------------------------------------------------------------------------------------------------
reasons = []
peer_all, peer_src = None, None
if tp.get("all_pairs_peer_access") is not None:
    peer_all, peer_src = bool(tp["all_pairs_peer_access"]), "torch.cuda.can_device_access_peer"
elif p2p.get("available") and (p2p.get("cannot_access_peer_lines", 0) + p2p.get("can_access_peer_lines", 0)) > 0:
    peer_all, peer_src = p2p["cannot_access_peer_lines"] == 0, "p2pBandwidthLatencyTest"
if peer_all is True:
    reasons.append(f"peer access supported on all {len(gpus) or '?'} GPUs pairwise ({peer_src})")
elif peer_all is False:
    reasons.append(f"peer access NOT supported on at least one pair ({peer_src}) -> P2P_OK=0; NCCL_P2P_DISABLE still 0 (human decision only)")
else:
    reasons.append("peer access not measured (no torch, no p2pBandwidthLatencyTest) -> P2P_OK=0 conservatively; NCCL_P2P_DISABLE still 0")
P2P_OK = 1 if peer_all else 0
p2p_status = "ok" if peer_all else ("unsupported" if peer_all is False else "unknown")
bw_min, bw_max = tp.get("unidir_copy_gbps_min") or p2p.get("unidir_p2p_enabled_min_gbps"), tp.get("unidir_copy_gbps_max") or p2p.get("unidir_p2p_enabled_max_gbps")
lat_max = tp.get("copy_latency_us_max") or p2p.get("latency_p2p_enabled_gpu_max_us")
if bw_min: reasons.append(f"unidirectional P2P copy {bw_min:.0f}-{bw_max:.0f} GB/s (recorded, not a gate)")
if lat_max: reasons.append(f"max P2P copy latency {lat_max:.1f} us (recorded, not a gate)")
tr = nccl["ring4_baseline"].get("transports_seen") or []
if tr: reasons.append("NCCL transports: " + ", ".join(tr))

def bw_of(lab):
    n = nccl.get(lab) or {}
    return n.get("busbw_gbps_128MB") or n.get("busbw_gbps_largest")
same_bw, cross_bw, ring_bw = bw_of("pair_same"), bw_of("pair_cross"), bw_of("ring4_baseline")
acs, acs_evidence = None, []
if same_bw and cross_bw:
    ratio = same_bw / cross_bw
    acs = ratio < ACS_RATIO
    acs_evidence.append(f"same-switch pair {same_pairs[0] if same_pairs else '?'} all_reduce busbw {same_bw:.1f} GB/s vs cross-switch pair "
                        f"{cross_pairs[0] if cross_pairs else '?'} {cross_bw:.1f} GB/s (ratio {ratio:.2f}, threshold {ACS_RATIO})")
if acs_enabled_ports:
    acs = True; acs_evidence.append(f"lspci: ACSCtl enabled on {acs_enabled_ports} bridge port(s)")
if not same_pairs and acs is None:
    acs = False; acs_evidence.append("no same-switch (PIX/PXB) GPU pair in topo -> ACS test not applicable")
if ACS_OVERRIDE in ("0", "1"):
    acs = ACS_OVERRIDE == "1"; acs_evidence.append(f"ACS_SUSPECTED_OVERRIDE={ACS_OVERRIDE}")
if acs is None:
    acs_evidence.append("undetermined: no pairwise NCCL numbers and no readable ACS capability -> ACS_SUSPECTED=0; re-run without SKIP_NCCL or set ACS_SUSPECTED_OVERRIDE=1")
ACS = 1 if acs else 0
if ring_bw: acs_evidence.append(f"4-GPU ring all_reduce busbw {ring_bw:.1f} GB/s")
PESS = 1 if (ACS == 1 or P2P_OK == 0) else 0

# human decisions persist across re-runs only when the previous decisions.env says HUMAN_DECISION=1
dec_path = os.path.join(OUT, "decisions.env")
prev, human = {}, False
if os.path.exists(dec_path):
    for line in open(dec_path, errors="replace"):
        m = re.match(r'^([A-Z][A-Z0-9_]*)=(.*)$', line.strip())
        if m: prev[m.group(1)] = m.group(2).strip().strip('"')
    human = prev.get("HUMAN_DECISION") == "1"
    try: os.replace(dec_path, dec_path + ".prev")
    except OSError: pass
NCCL_P2P_DISABLE = "1" if (human and prev.get("NCCL_P2P_DISABLE") == "1") else "0"
CUSTOM_ALLREDUCE = "1" if (human and prev.get("CUSTOM_ALLREDUCE") == "1") else "0"
if human: reasons.append("HUMAN_DECISION=1 in previous decisions.env -> NCCL_P2P_DISABLE/CUSTOM_ALLREDUCE kept from it")

g0 = gpus[0] if gpus else {}
notes = []
notes.append(f"{len(gpus)}x {g0.get('name', '?')} ({g0.get('edition_guess', '?')}), cc {g0.get('compute_cap', '?')}, {int((g0.get('memory_total_mib') or 0) / 1024)} GB each, PCIe gen{g0.get('pcie_gen_max', '?')} x{g0.get('pcie_width_max', '?')}, NVLink pairs: {len(nvlink_pairs)}")
notes.append(f"same-switch pairs {','.join('%d-%d' % p for p in same_pairs) or 'none'}; cross-switch pairs {','.join('%d-%d' % p for p in cross_pairs) or 'none'}")
notes += reasons + acs_evidence
if ACS: notes.append(f"ACS suspected -> TP2 replicas pair cross-switch (GPU_IDS={tp2_ids}); TP4 pessimistic")
notes.append("NCCL_P2P_DISABLE is never set automatically; CUSTOM_ALLREDUCE=1 only as an explicit A/B")
NOTES = " | ".join(notes).replace('"', "'").replace("\n", " ")

decisions = {"P2P_OK": P2P_OK, "CUSTOM_ALLREDUCE": int(CUSTOM_ALLREDUCE), "NCCL_P2P_DISABLE": int(NCCL_P2P_DISABLE),
             "ACS_SUSPECTED": ACS, "PESSIMISTIC_TP": PESS, "HOST_RAM_GB": ram_gb if ram_gb is not None else 0, "NOTES": NOTES}
hw["decisions"] = decisions
hw["decisions_env_path"] = dec_path
hw["p2p_ok"] = bool(P2P_OK); hw["p2p_status"] = p2p_status; hw["p2p_reasons"] = reasons; hw["p2p_source"] = peer_src
hw["acs_suspected"] = bool(ACS); hw["acs_evidence"] = acs_evidence
hw["pessimistic_tp"] = bool(PESS)
hw["custom_allreduce_default"] = int(CUSTOM_ALLREDUCE)
hw["nccl_p2p_disable"] = int(NCCL_P2P_DISABLE)
hw["host_ram_gb"] = ram_gb
hw["run_flag"] = ""                      # legacy field; no 'p2p_disabled' flag exists any more
hw["recommended_env"] = {"NCCL_P2P_LEVEL": os.environ.get("NCCL_P2P_LEVEL_BASELINE") or "PHB", "NCCL_IB_DISABLE": "1", "NCCL_MIN_NCHANNELS": os.environ.get("NCCL_MIN_NCHANNELS", "8"),
                         "VLLM_USE_DEEP_GEMM": "0", "FLASHINFER_CUDA_ARCH_LIST": "12.0f", "TORCH_CUDA_ARCH_LIST": "12.0"}

with open(os.path.join(OUT, "hardware.json"), "w") as fh:
    json.dump(hw, fh, indent=2)

lines = [f"# decisions.env -- written by vast/hardware_truth.sh {NOW}. Plain KEY=VALUE; bench/env.sh sources it (HW_DIR={OUT}).",
         "# Rules: P2P_OK = peer access on all pairs. NCCL_P2P_DISABLE is NEVER set to 1 by the script -- P2P works and beats host",
         "# staging. CUSTOM_ALLREDUCE=0 by default (A/B with CUSTOM_ALLREDUCE=1 at launch). PESSIMISTIC_TP marks TP2/TP4 rows (dagger).",
         "# To keep a hand edit across re-runs add the line HUMAN_DECISION=1 (only NCCL_P2P_DISABLE and CUSTOM_ALLREDUCE are preserved).",
         f"P2P_OK={P2P_OK}",
         f"CUSTOM_ALLREDUCE={CUSTOM_ALLREDUCE}",
         f"NCCL_P2P_DISABLE={NCCL_P2P_DISABLE}",
         f"ACS_SUSPECTED={ACS}",
         f"PESSIMISTIC_TP={PESS}",
         f"HOST_RAM_GB={decisions['HOST_RAM_GB']}",
         f'NOTES="{NOTES}"',
         "# --- informational (not part of the contract) ---",
         f"GENERATED_UTC={NOW}",
         f"HW_JSON={os.path.join(OUT, 'hardware.json')}",
         f"SAME_SWITCH_PAIRS={','.join('%d-%d' % p for p in same_pairs)}",
         f"TP2_CROSS_SWITCH_GPU_IDS={tp2_ids or ''}",
         f"HUMAN_DECISION={'1' if human else '0'}"]
with open(dec_path, "w") as fh:
    fh.write("\n".join(lines) + "\n")

def q(s): return '"' + str(s if s is not None else "").replace('"', "'") + '"'
# machine.env: FACTS only.  No decision keys (P2P_OK, CUSTOM_ALLREDUCE, NCCL_P2P_DISABLE, ACS_SUSPECTED,
# PESSIMISTIC_TP, HOST_RAM_GB live in decisions.env and are read by bench/env.sh at run time), no NCCL_* exports:
# a file sourced by login shells must never be able to shadow decisions.env or inject NCCL_P2P_DISABLE.
env_lines = ["# machine.env generated by vast/hardware_truth.sh " + NOW + " -- machine FACTS for humans / ad-hoc scripts.",
             "# Not sourced by bench/env.sh or by login shells; the decision contract is decisions.env next to this file.",
             f"HW_DIR={OUT}",
             f"DECISIONS_ENV={dec_path}",
             f"HW_JSON={os.path.join(OUT, 'hardware.json')}",
             f"GPU_COUNT={len(gpus)}",
             f"GPU_NAME={q(g0.get('name'))}",
             f"GPU_EDITION={q(g0.get('edition_guess'))}",
             f"GPU_POWER_LIMIT_W={int(g0['power_limit_w']) if g0.get('power_limit_w') else 0}",
             f"GPU_VBIOS={q(g0.get('vbios'))}",
             f"NVIDIA_DRIVER={q(g0.get('driver'))}",
             f"SAME_SWITCH_PAIRS={q(','.join('%d-%d' % p for p in same_pairs))}",
             f"TP2_CROSS_SWITCH_GPU_IDS={q(tp2_ids or '')}",
             "# recommended sm_120 baseline env (bench/env.sh exports the same defaults; informational here):"]
for k, v in hw["recommended_env"].items():
    env_lines.append(f"# {k}={v}")
env_lines.append("# NCCL_P2P_DISABLE / NCCL_SHM_DISABLE: never set by this script (decision 1)")
with open(os.path.join(OUT, "machine.env"), "w") as fh:
    fh.write("\n".join(env_lines) + "\n")
for stale in (os.path.join(OUT, "env.sh"),):
    if os.path.exists(stale):
        try: os.replace(stale, stale + ".legacy")
        except OSError: pass

print(json.dumps({"gpu_count": len(gpus), "decisions": decisions, "same_switch_pairs": same_pairs, "cross_switch_pairs": cross_pairs,
                  "tp2_cross_switch_gpu_ids": tp2_ids, "p2p_source": peer_src}, indent=2))
print("GPU0:", json.dumps({k: g0.get(k) for k in ("name", "edition_guess", "vbios", "driver", "power_limit_w", "memory_total_mib", "pcie_gen_max", "pcie_width_max", "bar1_total_mib", "sysfs_link")}))
print("torch p2p:", json.dumps({k: tp.get(k) for k in ("all_pairs_peer_access", "unidir_copy_gbps_min", "unidir_copy_gbps_max", "copy_latency_us_max")}))
for lab in LABELS:
    n = nccl[lab]
    print(f"nccl {lab:20s}", json.dumps({k: n.get(k) for k in ("source", "busbw_gbps_128MB", "busbw_gbps_1GB", "latency_us_8B", "any_wrong", "transports_seen")}))
PY
rc=$?
echo
echo "wrote: $OUT/decisions.env  $OUT/hardware.json  $OUT/machine.env  (parse rc=$rc)"
echo "--- decisions.env"; cat "$OUT/decisions.env"
# convenience copy: results/hardware.json is where collect_env.sh / gates look.  NOTHING is copied to <repo>/env.sh
# any more (decision contract = decisions.env, read by bench/env.sh at run time).
cp -f "$OUT/hardware.json" "$RESULTS_ROOT/hardware.json" 2>/dev/null || true
if [ -f "$REPO_ROOT/env.sh" ]; then
  echo "NOTE: legacy $REPO_ROOT/env.sh exists (written by an older hardware_truth.sh). bench/env.sh ignores its decision keys and"
  echo "      onstart no longer sources it; delete it to avoid confusion:  rm -f $REPO_ROOT/env.sh"
fi
hdr "hardware_truth done"
